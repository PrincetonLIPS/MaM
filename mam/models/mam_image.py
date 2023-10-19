import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from tqdm import tqdm

from utils.constants import BIT_UNKNOWN_VAL
from utils.mar_utils_mol import sample_order
from utils.data_utils import image_float_to_int, image_int_to_float


LOG_2 = math.log(2.0)

class MAM(nn.Module):
    def __init__(self, net, cfg, init_samples=None, mean=None):
        super().__init__()
        self.unet = net
        self.net = net
        self.cfg = cfg
        self.LogZ = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        allbits_unknown = torch.ones(1, cfg.L) * BIT_UNKNOWN_VAL
        allbits_unknown_mask = torch.zeros(1, cfg.L).bool()
        self.register_buffer('allbits_unknown_mask', allbits_unknown_mask)
        self.register_buffer('allbits_unknown', allbits_unknown)
        if cfg.include_onpolicy and cfg.mode == 'train':
            self.register_buffer('samples', init_samples)
        self.mseloss = nn.MSELoss()

    def sum_except_batch(self, x):
        return x.reshape(x.shape[0], -1).sum(-1)

    def net_forward(self, x, mask):
        return self.net(x, self.sum_except_batch(mask), mask)

    def sample_mask(self, batch, device):
        sigma = torch.rand(size=(batch, self.cfg.L), device=device)
        sigma = torch.argsort(sigma, dim=-1)
        t = torch.randint(high=self.cfg.K, size=(batch,), device=device)
        twrap = t.reshape(batch, 1)
        previous_selection = sigma < twrap
        current_selection = sigma == twrap
        return previous_selection, current_selection

    def eval_ll(self, x, mask):     
        x_aug = torch.cat((x, self.allbits_unknown), dim=0)
        mask_aug = torch.cat((mask, self.allbits_unknown_mask), dim=0)
        logp_x_aug, _ = self.net_forward(x_aug, mask_aug) # (B, L, K)
        logp_x = logp_x_aug[:x.shape[0]] # (B,)
        log_z = logp_x_aug[-1]
        return logp_x, log_z

    def eval_mar_bl_x_begin(self):
        logp_x, _ = self.net_forward(self.allbits_unknown, self.allbits_unknown_mask) # (1)
        # logp_diff = F.relu(logp_x - 0.0)
        mar_bl_loss = self.mseloss(logp_x, self.LogZ)
        return mar_bl_loss

    def eval_mar_bl_loss(self, x, rand_order_tj):
        mask_prev, mask_select_prev, mask_curr, mask_select, is_prev_full_mask = self.sample_mb_mask(
            x.shape[0], rand_order_tj, x.device, sample_method='random')
        x_mask = x * mask_curr + self.allbits_unknown * (~mask_curr)
        logp_x_mask, _ = self.net_forward(x_mask, mask_curr) # (B, L, K)
        x_mask_prev = x * mask_prev + self.allbits_unknown * (~mask_prev)
        logp_x_mask_prev, logits_prev = self.net_forward(x_mask_prev, mask_prev) # (B, L, K)

        distout = torch.distributions.categorical.Categorical(logits=logits_prev)
        ll = distout.log_prob(image_float_to_int(x, self.cfg.binary)) # (B, L)
        ll = self.sum_except_batch(ll*mask_select_prev) # (B,)
        logp_x_mask_from_prev = ll + logp_x_mask_prev # (B,)
        log_sum_ineq_thresh_diff = (logp_x_mask - logp_x_mask_from_prev)
        mar_bl_loss = self.mseloss(log_sum_ineq_thresh_diff, torch.zeros_like(log_sum_ineq_thresh_diff))
        return mar_bl_loss

    def sample_mb_mask(self, batch_size, rand_order, device, sample_method=None):
        if sample_method == 'random':
            ind = torch.randint(1, self.cfg.L+1, (batch_size,1), device=device) # (B, 1) of {1, ..., L}
            mask_select = rand_order == ind
            mask_curr = rand_order < ind
            mask_select_prev = rand_order == (ind - 1)
            mask_prev = rand_order < (ind - 1)
            is_prev_full_mask = (ind == 1)
        else:
            raise NotImplementedError
        return mask_prev, mask_select_prev, mask_curr, mask_select, is_prev_full_mask.squeeze(-1)

    def sum_except_batch(self, x):
        return x.reshape(x.shape[0], -1).sum(-1)

    def forward(self, x):
        rand_order_tj = sample_order(x, gen_order=self.cfg.gen_order)
        mask_full = torch.ones_like(x, dtype=torch.bool)
        logp_real, log_z = self.eval_ll(x, mask_full) # allow log_z to be learned
        logp_real = logp_real.mean()
        mb_loss = self.eval_mar_bl_loss(x, rand_order_tj)
        mb_loss_begin = self.eval_mar_bl_x_begin() # maybe not needed
        if self.cfg.include_onpolicy:
            rand_order_tj_samples = sample_order(self.samples, gen_order=self.cfg.gen_order)
            with torch.no_grad():
                self.samples, _ = self.censor_and_sample(self.samples, self.cfg.gibbs_steps, rand_order_tj_samples)
            mask_full_fake = torch.ones_like(self.samples, dtype=torch.bool)
            logp_fake, _ = self.eval_ll(self.samples, mask_full_fake)
            logp_fake = logp_fake.mean()
            mb_loss_fake = self.eval_mar_bl_loss(self.samples, rand_order_tj_samples)
            mb_loss = (mb_loss + mb_loss_fake)/2
        loss = self.cfg.L * mb_loss + mb_loss_begin
        return loss, logp_real, log_z, mb_loss, mb_loss_begin
    
    def sample(self, rand_order, num_samples, greedy=False):
        X_i = torch.ones(num_samples, self.cfg.L).to(rand_order.device) * BIT_UNKNOWN_VAL #(B, L)
        for i in range(self.cfg.L):
            mask_select = rand_order == i # (B, L)
            mask_curr = rand_order < i # (B, L)
            _, logits = self.net_forward(X_i, mask_curr) # (B, L, K)
            logits = logits[mask_select] # (B, K)
            if greedy:
                X_i_sample_ind = torch.argmax(logits, dim=-1) # (B,)
            else:
                X_i_distr = torch.distributions.categorical.Categorical(logits=logits)
                X_i_sample_ind = X_i_distr.sample() #(B,)
                X_i_sample = image_int_to_float(X_i_sample_ind.unsqueeze(-1), self.cfg.binary)
            X_i = X_i + mask_select * X_i_sample # (B, L)
        return X_i  # (B, L)
    
    def sample_and_eval(self, rand_order, num_samples):
        X_i = torch.ones(num_samples, self.cfg.L).to(rand_order.device) * BIT_UNKNOWN_VAL #(B, L)
        logp_X_i_ls = []
        logp_X_i_arm_ls = []
        X_i_ls = []
        logp_cond_i = torch.zeros(num_samples).to(rand_order.device)
        for i in range(self.cfg.L):
            mask_select = rand_order == i # (B, L)
            mask_curr = rand_order < i # (B, L)
            logp_X_i, logits = self.net_forward(X_i, mask_curr) # (B, L, K)
            # record logp marg and arm
            X_i_ls.append(X_i.unsqueeze(1))
            logp_X_i_ls.append(logp_X_i.unsqueeze(1))
            logp_X_i_arm_ls.append(logp_cond_i.unsqueeze(1))
            logits = logits[mask_select] # (B, K)
            X_i_distr = torch.distributions.categorical.Categorical(logits=logits)
            X_i_sample_ind = X_i_distr.sample() #(B,)
            X_i_sample = image_int_to_float(X_i_sample_ind.unsqueeze(-1), self.cfg.binary)
            logp_cond_i = X_i_distr.log_prob(X_i_sample_ind) + logp_cond_i # (B,)
            X_i = X_i + mask_select * X_i_sample # (B, L)
        mask_curr = rand_order < self.cfg.L # (B, L)
        logp_X_i, _ = self.net_forward(X_i, mask_curr) # (B,)
        X_i_ls.append(X_i.unsqueeze(1))
        logp_X_i_ls.append(logp_X_i.unsqueeze(1))
        logp_X_i_arm_ls.append(logp_cond_i.unsqueeze(1))
        # concat
        logp_X_i_marg = torch.cat(logp_X_i_ls, dim=1) # (B, L)
        logp_X_i_arm = torch.cat(logp_X_i_arm_ls, dim=1) # (B, L)
        X_i_tj = torch.cat(X_i_ls, dim=1) # (B, L, L)
        return X_i, X_i_tj, logp_X_i_marg, logp_X_i_arm # (B, L)

    def censor_and_sample(self, X, num_steps, rand_order, greedy=False):
        if num_steps > self.cfg.L:
            raise ValueError('num_steps must be <= L')
        # rand_order = torch.argsort(torch.rand_like(X).to(self.device)) # (B, D)
        X_i = X.clone() # (B, D)
        # backward censor
        for i in range(num_steps):
            mask_censor = rand_order == i
            X_i = (~mask_censor) * X_i + mask_censor * BIT_UNKNOWN_VAL
        X_i_censored = X_i.clone()
        # forward sample
        for i in reversed(range(num_steps)):
            mask_select = rand_order == i
            mask_curr = rand_order < i
            _, logits = self.net_forward(X_i, mask_curr) # (B, L, K)
            logits = logits[mask_select] # (B, K)
            if greedy:
                X_i_sample_ind = torch.argmax(logits, dim=-1) # (B,)
            else:
                X_i_distr = torch.distributions.categorical.Categorical(logits=logits)
                X_i_sample_ind = X_i_distr.sample() #(B,)
                X_i_sample = image_int_to_float(X_i_sample_ind.unsqueeze(-1), self.cfg.binary)
            X_i = X_i + mask_select * X_i_sample # (B, L)
        return X_i, X_i_censored # (B, L)
    
    def censor_and_sample_new(self, X, backward_steps, forward_steps, rand_order, greedy=False):
        if backward_steps > self.cfg.L:
            raise ValueError('num_steps must be <= L')
        # rand_order = torch.argsort(torch.rand_like(X).to(self.device)) # (B, D)
        X_i = X.clone() # (B, D)
        # backward censor
        mask_censor = rand_order < backward_steps
        X_i = (~mask_censor) * X_i + mask_censor * BIT_UNKNOWN_VAL
        X_i_censored = X_i.clone()
        # forward sample
        for i in reversed(range(forward_steps)):
            mask_select = rand_order == i
            mask_curr = rand_order < i
            _, logits = self.net_forward(X_i, mask_curr) # (B, L, K)
            logits = logits[mask_select] # (B, K)
            if greedy:
                X_i_sample_ind = torch.argmax(logits, dim=-1) # (B,)
            else:
                X_i_distr = torch.distributions.categorical.Categorical(logits=logits)
                X_i_sample_ind = X_i_distr.sample() #(B,)
                X_i_sample = image_int_to_float(X_i_sample_ind.unsqueeze(-1), self.cfg.binary)
            X_i = X_i + mask_select * X_i_sample # (B, L)
        return X_i, X_i_censored # (B, L)
    
    def corrupt_and_flip(self, X, num_steps, rand_order, random=False):
        if num_steps > self.cfg.L:
            raise ValueError('num_steps must be <= L')
        mask_censor = rand_order < num_steps
        X_i_censored = (~mask_censor) * X + mask_censor * BIT_UNKNOWN_VAL
        if random:
            X_i_rand = torch.randint_like(X, low=0, high=self.cfg.K)
            X_i_rand = image_int_to_float(X_i_rand, self.cfg.binary)
            X_i = (~mask_censor) * X_i_rand + mask_censor * X_i_rand
        else:
            X_i = (~mask_censor) * X + mask_censor * (-X)
        return X_i, X_i_censored

    def cond_sample(self, x_cond, rand_order, num_samples, greedy=False, thresh=-1.0):
        unknown_mask = x_cond.squeeze(0) == BIT_UNKNOWN_VAL
        # find number of unknowns in x_cond
        num_steps = unknown_mask.sum().item()
        mask_select_full = torch.zeros(num_samples, self.cfg.L).bool().to(x_cond.device) # (B, L)
        # repeat x_cond num_samples times
        X_i = x_cond.repeat(num_samples, 1) # (1, L) -> (B, L)
        for i in reversed(range(num_steps)):
            mask_select = rand_order == i
            # set subset of mask_full to mask_select
            mask_select_full[:,unknown_mask] = mask_select 
            mask_curr = ~(X_i == BIT_UNKNOWN_VAL) # (B, L)
            _, logits = self.net_forward(X_i, mask_curr) # (B, L, K)
            logits = logits[mask_select_full] # (B, K)
            if greedy:
                X_i_sample_ind = torch.argmax(logits, dim=-1) # (B,)
            else:
                X_i_distr = torch.distributions.categorical.Categorical(logits=logits)
                X_i_sample_ind = X_i_distr.sample() #(B,)
                X_i_sample = image_int_to_float(X_i_sample_ind.unsqueeze(-1), self.cfg.binary)
            X_i = X_i + mask_select_full * X_i_sample # (B, L)
        return X_i # (B, L) in values [1:K]

    def est_logp(self, X, mc_ll, gen_order, one_step=False):
        logp_ls = []
        if one_step:
            rand_order_tj = sample_order(X, gen_order=self.cfg.gen_order)
            mask_prev, mask_select_prev, _, _, _ = self.sample_mb_mask(
                X.shape[0], rand_order_tj, X.device, sample_method='random')
            x_mask_prev = X * mask_prev + self.allbits_unknown * (~mask_prev)
            _, logits_prev = self.net_forward(x_mask_prev, mask_prev) # (B, L, K)
            distout = torch.distributions.categorical.Categorical(logits=logits_prev)
            ll = distout.log_prob(image_float_to_int(X, self.cfg.binary)) # (B, L)
            ll = self.sum_except_batch(ll*mask_select_prev) # (B,)
            batch_logp = ll * self.cfg.L
        else:
            for _ in range(mc_ll):
                if gen_order == 'random':
                    rand_order = torch.argsort(torch.rand_like(X), dim=-1).to(X.device)
                elif gen_order == 'backward':
                    rand_order = torch.arange(X.shape[-1]-1, -1, step=-1).expand(X.shape[0],-1).to(X.device)
                elif gen_order == 'forward':
                    rand_order = torch.arange(X.shape[-1]).expand(X.shape[0],-1).to(X.device)
                X_mask = torch.ones_like(X).to(X.device) * BIT_UNKNOWN_VAL # (B, L), all unknown
                logp_step = torch.zeros(X.shape[0]).to(X.device)
                for j in range(self.cfg.L):
                    current_selection = rand_order == j # (B,L)
                    current_mask = rand_order < j # (B,L)
                    _ , logits_given_x_mask = self.net_forward(X_mask, current_mask) # (B, L, K)
                    distout = torch.distributions.categorical.Categorical(logits=logits_given_x_mask)
                    ll = distout.log_prob(image_float_to_int(X, self.cfg.binary)) # (B, L)
                    ll = self.sum_except_batch(ll*current_selection) # (B,)
                    logp_step = logp_step + ll
                    X_mask = X_mask + current_selection * X # (B, L)
                logp_ls.append(logp_step.unsqueeze(1))
            batch_logp = torch.logsumexp(torch.cat(logp_ls, dim=1), dim=1) - torch.tensor(mc_ll).log()  # (B,)
        return batch_logp