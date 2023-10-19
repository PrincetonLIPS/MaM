import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from tqdm import tqdm
import numpy as np

from utils.constants import BIT_UNKNOWN_VAL
from utils.mar_utils_mol import gen_order

eps = np.finfo(np.float32).eps.item()

class ARMModel(nn.Module):
    def __init__(self, net, score_model, init_samples, cfg):
        super().__init__()
        self.net = net
        self.score_model = score_model
        self.cfg = cfg
        if cfg.LogZ is None:
            init_z = math.log(2**cfg.L) 
            if cfg.objective == 'MB':
                self.LogZ = nn.Parameter(torch.tensor([init_z]), requires_grad=True)
            elif cfg.objective == 'KL':
                self.LogZ = nn.Parameter(torch.tensor([init_z]), requires_grad=False)
        else:
            self.LogZ = nn.Parameter(torch.tensor([cfg.score_model.LogZ]), requires_grad=False)
        allbits_unknown = torch.ones(1, cfg.L) * BIT_UNKNOWN_VAL
        self.register_buffer('allbits_unknown', allbits_unknown)
        if cfg.mode != 'generate':
            self.register_buffer('samples', init_samples)
        self.mseloss = nn.MSELoss()

    def eval_logp(self, x, rand_order_tj):
        batch_size = x.shape[0]
        logp_x = torch.zeros((batch_size,), dtype=torch.float32).to(x.device)
        for i in range(self.cfg.L):
            mask_curr = rand_order_tj < i # (B, L)
            mask_select = rand_order_tj == i
            x_mask = x * mask_curr + self.allbits_unknown * (~mask_curr)
            logits = self.net(x_mask) # (B, L, K)
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
        logits = self.net(x_mask) # (B, L, K)
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

    def sum_except_batch(self, x):
        return x.reshape(x.shape[0], -1).sum(-1)

    def forward(self, x, y, training=True):
        if self.cfg.objective == 'FM':
            rand_order_tj = gen_order(x.shape[0], self.cfg.L, x.device, gen_order=self.cfg.gen_order)
            if self.cfg.logp_mc and training:
                logp_x = self.cfg.L * self.eval_logp_mc(x, rand_order_tj)
            else:
                logp_x = self.eval_logp(x, rand_order_tj)
            f_diff = logp_x - y
            loss = self.mseloss(logp_x + self.LogZ, y)
            energies_x = - y
        elif self.cfg.objective == 'KL':
            rand_order_tj = gen_order(self.cfg.batch_size, self.cfg.L, self.samples.device, gen_order=self.cfg.gen_order)
            with torch.no_grad():
                # self.samples = self.censor_and_sample(self.samples, self.cfg.gibbs_steps, rand_order_tj)
                self.samples = self.sample(rand_order_tj, self.cfg.batch_size)
                energies_x = - self.score_model(self.samples - 1.0) # convert back to [0:K-1] first
            if self.cfg.logp_mc and training:
                with torch.no_grad():
                    logp_x_mc_1 = self.cfg.L * self.eval_logp_mc(self.samples, rand_order_tj)
                    # logp_x_full = self.eval_logp(self.samples, rand_order_tj)
                rand_order_tj_2 = gen_order(self.cfg.batch_size, self.cfg.L, self.samples.device, gen_order=self.cfg.gen_order)
                logp_x = self.cfg.L * self.eval_logp_mc(self.samples, rand_order_tj_2)
                f_diff = logp_x_mc_1 + energies_x
                f_mean = f_diff.mean()
                f_std = f_diff.std()
                r = (f_diff - f_mean)/(f_std + eps)
            else:
                logp_x = self.eval_logp(self.samples, rand_order_tj)
                f_diff = logp_x.detach() + energies_x
                f_mean = f_diff.mean()
                f_std = f_diff.std()
                r = (f_diff - f_mean)/(f_std + eps)
            loss = (r * logp_x).mean()
            
        return loss, - energies_x, logp_x

    def sample(self, rand_order, num_samples, greedy=False):
        X_i = torch.ones(num_samples, self.cfg.L).to(rand_order.device) * BIT_UNKNOWN_VAL #(B, L)
        for i in range(self.cfg.L):
            mask_select = rand_order == i # (B, L)
            logits = self.net(X_i) # (B, L, K)
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
            logits = self.net(X_i) # (B, L, K)
            logits = logits[mask_select] # (B, K)
            if greedy:
                X_i_sample_ind = torch.argmax(logits, dim=-1) # (B,)
            else:
                X_i_distr = torch.distributions.categorical.Categorical(logits=logits)
                X_i_sample_ind = X_i_distr.sample() #(B,)
            X_i = X_i + mask_select * (X_i_sample_ind.unsqueeze(-1) + 1.0) # (B, L), add 1 to convert to values [1:K]
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
                logits_given_x_mask = self.net(X_mask) # (B, L, K)
                x_next = X[current_selection] #(B,) of {1, 2, ..., K}
                # select logits for current selection
                logits_given_x_mask = logits_given_x_mask[current_selection] # (B, K)
                logps_given_x_mask = torch.log_softmax(logits_given_x_mask, dim=-1) # (B, K)
                logp_step = logp_step + logps_given_x_mask[torch.arange(X.shape[0]), (x_next-1).long()]
                # X_mask = X_mask + current_selection * X # (B, L)
            logp_ls.append(logp_step.unsqueeze(1))
        batch_logp = torch.logsumexp(torch.cat(logp_ls, dim=1), dim=1) - torch.tensor(mc_ll).log()  # (B,)
        return batch_logp
    
    def eval_loss(self, samples, logp_mc):
        energies_x = - self.score_model(samples - 1.0) # convert back to [0:K-1] first
        rand_order_tj = gen_order(self.cfg.batch_size, self.cfg.L, samples.device, gen_order=self.cfg.gen_order)
        if logp_mc:
            with torch.no_grad():
                logp_x_mc_1 = self.cfg.L * self.eval_logp_mc(samples, rand_order_tj)
                # logp_x_full = self.eval_logp(samples, rand_order_tj)
            rand_order_tj_2 = gen_order(self.cfg.batch_size, self.cfg.L, samples.device, gen_order=self.cfg.gen_order)
            logp_x = self.cfg.L * self.eval_logp_mc(samples, rand_order_tj_2)
            f_diff = logp_x_mc_1 + energies_x
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
