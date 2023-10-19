import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import math
from tqdm import tqdm

from utils.constants import BIT_UNKNOWN_VAL
from utils.mar_utils_mol import gen_order

eps = np.finfo(np.float32).eps.item()

class MAM(nn.Module):
    def __init__(self, net, score_model, init_samples, cfg):
        super().__init__()
        self.net = net
        self.score_model = score_model
        self.cfg = cfg
        if cfg.LogZ is None:
            init_z = math.log(2**cfg.L)
            if cfg.objective == 'FM':
                self.LogZ = nn.Parameter(torch.tensor([init_z]), requires_grad=True)
            elif cfg.objective == 'KL':
                self.LogZ = nn.Parameter(torch.tensor([init_z]), requires_grad=False) # fix logZ
        else:
            self.LogZ = nn.Parameter(torch.tensor([cfg.LogZ]), requires_grad=False)
        allbits_unknown = torch.ones(1, cfg.L) * BIT_UNKNOWN_VAL
        self.register_buffer('allbits_unknown', allbits_unknown)
        if cfg.mode != 'generate':
            self.register_buffer('samples', init_samples)
        self.mseloss = nn.MSELoss()

    def sample_mask(self, batch, device):
        sigma = torch.rand(size=(batch, self.cfg.L), device=device)
        sigma = torch.argsort(sigma, dim=-1)
        t = torch.randint(high=self.cfg.K, size=(batch,), device=device)
        twrap = t.reshape(batch, 1)
        previous_selection = sigma < twrap
        current_selection = sigma == twrap
        return previous_selection, current_selection

    def eval_mar_bl_x_end(self, x, f):
        logp_x, _ = self.net(x) # (B)
        mar_bl_loss = self.mseloss(logp_x, f)
        return mar_bl_loss

    def eval_mar_bl_x_begin(self):
        logp_x, _ = self.net(self.allbits_unknown) # (1)
        mar_bl_loss = self.mseloss(logp_x, self.LogZ)
        return mar_bl_loss

    def eval_mar_bl_loss(self, x, rand_order_tj):
        mask_prev, mask_select_prev, mask_curr = self.sample_mb_mask(
            x.shape[0], rand_order_tj, x.device, sample_method='random')
        x_mask = x * mask_curr + self.allbits_unknown * (~mask_curr)
        logp_x_mask, _ = self.net(x_mask) # (B, L, K)
        x_mask_prev = x * mask_prev + self.allbits_unknown * (~mask_prev)
        logp_x_mask_prev, logits_prev = self.net(x_mask_prev) # (B, L, K)
        logits_prev = logits_prev[mask_select_prev] # (B, K)
        x_select_prev = x[mask_select_prev] # (B,)
        logps_given_x_mask_prev = F.log_softmax(logits_prev, dim=-1)
        logp_x_mask_from_prev = logps_given_x_mask_prev[torch.arange(x.shape[0]), (x_select_prev-1).long()] + logp_x_mask_prev # (B,)
        mar_bl_loss = self.mseloss(logp_x_mask, logp_x_mask_from_prev)
        return mar_bl_loss

    def sample_mb_mask(self, batch_size, rand_order, device, sample_method=None):
        if sample_method == 'random':
            ind = torch.randint(1, self.cfg.L+1, (batch_size,1), device=device) # (B, 1) of {1, ..., L}
            mask_select = rand_order == ind
            mask_curr = rand_order < ind
            mask_select_prev = rand_order == (ind - 1)
            mask_prev = rand_order < (ind - 1)
        else:
            raise NotImplementedError
        return mask_prev, mask_select_prev, mask_curr
    
    def eval_logp(self, x, rand_order_tj):
        batch_size = x.shape[0]
        logp_x = torch.zeros((batch_size,), dtype=torch.float32).to(x.device)
        for i in range(self.cfg.L):
            mask_curr = rand_order_tj < i # (B, L)
            mask_select = rand_order_tj == i
            x_mask = x * mask_curr + self.allbits_unknown * (~mask_curr)
            _, logits = self.net(x_mask) # (B, L, K)
            logits = logits[mask_select] # (B, K)
            x_select = x[mask_select] # (B,)
            logps_given_x_mask = F.log_softmax(logits, dim=-1)
            logp_x = logp_x + logps_given_x_mask[torch.arange(x.shape[0]), (x_select-1).long()] # (B,)
        return logp_x
    
    def eval_logp_mc(self, x, rand_order_tj, parallel=False):
        batch_size = x.shape[0]
        i = torch.randint(0, self.cfg.L, (batch_size,1), device=x.device) # (B, 1) of [0:L-1]
        mask_curr = rand_order_tj < i # (B, L)
        mask_select = rand_order_tj == i
        x_mask = x * mask_curr + self.allbits_unknown * (~mask_curr)
        _, logits = self.net(x_mask) # (B, L, K)
        if parallel:
            mask_future = ~mask_curr
            distout = torch.distributions.categorical.Categorical(logits=logits)
            ll = distout.log_prob( x.long() - 1 )
            logp_x_mc = self.sum_except_batch(ll * mask_future) / self.sum_except_batch(mask_future)
        else:
            logits = logits[mask_select] # (B, K)
            x_select = x[mask_select] # (B,)
            logps_given_x_mask = F.log_softmax(logits, dim=-1)
            logp_x_mc = logps_given_x_mask[torch.arange(x.shape[0]), (x_select-1).long()] # (B,)
        return logp_x_mc
    
    def eval_ll(self, x):     
        x_aug = torch.cat((x, self.allbits_unknown), dim=0)
        logp_x_aug, _ = self.net(x_aug) # (B, L, K)
        logp_x = logp_x_aug[:x.shape[0]] # (B,)
        log_z = logp_x_aug[-1]
        return logp_x, log_z

    def sum_except_batch(self, x):
        return x.reshape(x.shape[0], -1).sum(-1)

    def forward(self, x, y):
        if self.cfg.objective == 'FM':
            rand_order_tj = gen_order(x.shape[0], self.cfg.L, x.device, gen_order=self.cfg.gen_order)
            rand_order_tj_samples = gen_order(self.cfg.batch_size, self.cfg.L, x.device, gen_order=self.cfg.gen_order)
            with torch.no_grad():
                self.samples = self.censor_and_sample(self.samples, self.cfg.gibbs_steps, rand_order_tj_samples)
                samples_y = self.score_model(self.samples - 1.0) # convert back to [0:K-1] first
            mb_loss = self.eval_mar_bl_loss(x, rand_order_tj) + self.eval_mar_bl_loss(self.samples, rand_order_tj_samples)
            mb_loss_begin = self.eval_mar_bl_x_begin()
            logp_x, _ = self.net(x) # (B)
            logp_samples, _ = self.net(self.samples) # (B)
            mb_loss_end = self.mseloss(logp_x, y) 
            # + self.mseloss(logp_samples, samples_y)
            mb_loss_total = self.cfg.alpha * (self.cfg.L * (0.5*mb_loss) + mb_loss_begin) + (0.5*mb_loss_end)
            loss = mb_loss_total
            return loss, mb_loss, mb_loss_begin, mb_loss_total, y, logp_x
        elif self.cfg.objective == 'KL':
            rand_order_tj = gen_order(self.cfg.batch_size, self.cfg.L, self.samples.device, gen_order=self.cfg.gen_order)
            with torch.no_grad():
                if self.cfg.gen_order == 'random':
                    self.samples = self.censor_and_sample(self.samples, self.cfg.gibbs_steps, rand_order_tj)
                else:
                    self.samples = self.sample(rand_order_tj, self.cfg.batch_size)
                energies_x = - self.score_model(self.samples - 1.0) # convert back to [0:K-1] first
            logp_x, _ = self.net(self.samples) # (B)
            f_diff = logp_x.detach() + energies_x
            # use baselines for REINFORCE
            f_mean = f_diff.mean()
            f_std = f_diff.std()
            r = (f_diff - f_mean)/(f_std + eps)
            if self.cfg.logp_mc:
                logp_mc = self.eval_logp_mc(self.samples, rand_order_tj, parallel=True)
                kl_loss = r * logp_mc
            else:
                kl_loss = r * logp_x
            mb_loss = self.eval_mar_bl_loss(self.samples, rand_order_tj)
            mb_loss_begin = self.eval_mar_bl_x_begin()
            rand_order_tj_2 = gen_order(x.shape[0], self.cfg.L, self.samples.device, gen_order=self.cfg.gen_order)
            mb_loss_2 = self.eval_mar_bl_loss(x, rand_order_tj_2)
            mb_loss_total = self.cfg.L * mb_loss + mb_loss_begin
            loss = self.cfg.alpha * mb_loss_total + kl_loss.mean() # mb_loss_begin
            return loss, mb_loss, mb_loss_begin, mb_loss_total, - energies_x, logp_x
    
    def sample(self, rand_order, num_samples, greedy=False):
        X_i = torch.ones(num_samples, self.cfg.L).to(rand_order.device) * BIT_UNKNOWN_VAL #(B, L)
        for i in range(self.cfg.L):
            mask_select = rand_order == i # (B, L)
            _, logits = self.net(X_i) # (B, L, K)
            logits = logits[mask_select] # (B, K)
            if greedy:
                X_i_sample_ind = torch.argmax(logits, dim=-1) # (B,)
            else:
                X_i_distr = torch.distributions.categorical.Categorical(logits=logits)
                X_i_sample_ind = X_i_distr.sample() #(B,)
            X_i = X_i + mask_select * (X_i_sample_ind.unsqueeze(-1) + 1.0) # (B, L), add 1 to convert to values [1:K]
        return X_i  # (B, L) in values [1:K]

    def censor_and_sample(self, X, num_steps, rand_order, greedy=False):
        if num_steps > self.cfg.L:
            raise ValueError('num_steps must be <= L')
        # rand_order = torch.argsort(torch.rand_like(X).to(self.device)) # (B, D)
        X_i = X.clone() # (B, D)
        # backward censor
        mask_erase = rand_order >= self.cfg.L - num_steps
        X_i = (~mask_erase) * X_i + mask_erase * BIT_UNKNOWN_VAL
        # forward sample
        for i in reversed(range(num_steps)):
            mask_select = rand_order == (self.cfg.L - 1 - i)
            _, logits = self.net(X_i) # (B, L, K)
            logits = logits[mask_select] # (B, K)
            if greedy:
                X_i_sample_ind = torch.argmax(logits, dim=-1) # (B,)
            else:
                X_i_distr = torch.distributions.categorical.Categorical(logits=logits)
                X_i_sample_ind = X_i_distr.sample() #(B,)
            X_i = X_i + mask_select * (X_i_sample_ind.unsqueeze(-1) + 1.0) # (B, L), add 1 to convert to values [1:K]
        return X_i # (B, L) in values [1:K]

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
            _, logits = self.net(X_i) # (B, L, K)
            logits = logits[mask_select_full] # (B, K)
            if greedy:
                X_i_sample_ind = torch.argmax(logits, dim=-1) # (B,)
            else:
                X_i_distr = torch.distributions.categorical.Categorical(logits=logits)
                X_i_sample_ind = X_i_distr.sample() #(B,)
            X_i = X_i + mask_select_full * (X_i_sample_ind.unsqueeze(-1) + 1.0) # (B, L), add 1 to convert to values [1:K]
        return X_i # (B, L) in values [1:K]

    def est_logp(self, X, mc_ll, gen_order):
        logp_ls = []
        for _ in range(mc_ll):
            if gen_order == 'random':
                rand_order = torch.argsort(torch.rand_like(X), dim=-1).to(X.device)
            elif gen_order == 'backward':
                rand_order = torch.arange(X.shape[-1]-1,-1, step=-1).expand(X.shape[0],-1).to(X.device)
            elif gen_order == 'forward':
                rand_order = torch.arange(X.shape[-1]).expand(X.shape[0],-1).to(X.device)
            # X_mask = torch.ones_like(X).to(X.device) * BIT_UNKNOWN_VAL # (B, L), all unknown
            logp_step = torch.zeros(X.shape[0]).to(X.device)
            for j in range(self.cfg.L):
                mask_curr = rand_order < j # (B, L)
                current_selection = rand_order == j # (B, L)
                X_mask = X * mask_curr + BIT_UNKNOWN_VAL * (~mask_curr)
                _ , logits_given_x_mask = self.net(X_mask) # (B, L, K)
                x_next = X[current_selection] #(B,) of {1, 2, ..., K}
                # select logits for current selection
                logits_given_x_mask = logits_given_x_mask[current_selection] # (B, K)
                logps_given_x_mask = torch.log_softmax(logits_given_x_mask, dim=-1) # (B, K)
                logp_step = logp_step + logps_given_x_mask[torch.arange(X.shape[0]), (x_next-1).long()]
                # X_mask = X_mask + current_selection * X # (B, L)
            logp_ls.append(logp_step.unsqueeze(1))
        batch_logp = torch.logsumexp(torch.cat(logp_ls, dim=1), dim=1) - torch.tensor(mc_ll).log()  # (B,)
        return batch_logp
    
    def eval_loss(self, samples, use_marg):
        energies_x = - self.score_model(samples - 1.0) # convert back to [0:K-1] first
        rand_order_tj = gen_order(self.cfg.batch_size, self.cfg.L, samples.device, gen_order=self.cfg.gen_order)
        if use_marg:
            logp_x, _ = self.net(self.samples) # (B)
            f_diff = logp_x.detach() + energies_x
            f_mean = 0 # f_diff.mean()
            f_std = 1 #f_diff.std()
            r = (f_diff - f_mean)/(f_std + eps)
        else:
            logp_x = self.eval_logp(samples, rand_order_tj)
            f_diff = logp_x.detach() + energies_x
            f_mean = 0 # f_diff.mean()
            f_std = 1 #f_diff.std()
            r = (f_diff - f_mean)/(f_std + eps)
        loss = (r * logp_x).mean()
        return loss