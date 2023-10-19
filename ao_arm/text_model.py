import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from arch.transformer import TransformerNet

class ARM(nn.Module):
    def __init__(self, obs_dims, cfg):
        super(ARM, self).__init__()

        self.D = 26 + 1 + 1 # characters + space + mask_token
        self.obs_dims = obs_dims
        self.xdim = np.prod(np.array(obs_dims)).item()
        self.cfg = cfg

        assert(self.cfg.arch in ['Transformer'])
        
        self.net = TransformerNet(
            num_src_vocab=self.D, # add one for mask token
            num_tgt_vocab=self.D - 1, # no mask token in target prediction
            embedding_dim=768,
            hidden_size=3072,
            nheads=12,
            n_layers=12,
            max_src_len=250,
            dropout=0.0
        )

    def sum_except_batch(self, x):
        return x.reshape(x.shape[0], -1).sum(-1)

    def net_forward(self, x, mask):
        logits = self.net(x)
        return logits
    
    def sample_mask(self, batch, device):
        sigma = torch.rand(size=(batch, self.xdim), device=device)
        sigma = torch.argsort(sigma, dim=-1).reshape(batch, *self.obs_dims)  
        t = torch.randint(high=self.xdim, size=(batch,), device=device)
        twrap = t.reshape(batch, 1)
        previous_selection = sigma < twrap
        current_selection = sigma == twrap
        return previous_selection, current_selection

    def mask_to_order(self, mask):
        '''
        mask is bitmask of size (batch, *self.obs_dims)
        we will mask an img x by doing x*img
        so we call 1-bits 'unmasked', and 0-bits 'masked'
        '''
        batch = mask.shape[0]
        # we want to place all the ones before zeros, but randomize the ordering in each bucket
        # to do so just add a large constant to all the ones, then add random noise to every value, and sort
        large_constant = (int)(1e8)
        flat_mask = mask.long().reshape(batch, self.xdim) * large_constant
        flat_noise_mask = flat_mask + torch.randint(high=self.xdim, size=(batch, self.xdim), device=mask.device)
        flat_unmasked_first_order = flat_noise_mask.argsort(descending=True).argsort()
        unmasked_first_order = flat_unmasked_first_order.reshape(*mask.shape)
        return unmasked_first_order


    def likelihood(self, x, mask, order=None, full=True):
        # mask should have cardinality at least one

        if mask is None: mask = torch.ones(*x.shape, device=x.device).long()

        batch = x.shape[0]
        zeroimg = torch.zeros(batch, *self.obs_dims, device=x.device) + 0

        if order is not None:
            sigma = order
        else:
            sigma = self.mask_to_order(mask)
        T = self.sum_except_batch(mask)

        total_ll = 0

        if not full:
            # instead of doing the full likelihood, just use one timestep as an approximation
            t = T
            # sample an intermediate prefix by taking a random int from [0, t)
            batch_arange = torch.arange(self.xdim, device=x.device).reshape(1, self.xdim).repeat(batch, 1)
            nonzero_weights = batch_arange < t.reshape(batch, 1)
            weights = torch.ones(batch, self.xdim, device=x.device).float()
            weights = weights * nonzero_weights
            tpre = torch.multinomial(weights.float(), num_samples=1)[:,0]
            twrap = tpre.reshape(batch, 1)

            previous_selection = sigma < twrap
            current_selection = sigma == twrap

            xin = x * previous_selection + zeroimg * (~previous_selection)

            logits = self.net_forward(xin, previous_selection).reshape(batch, *self.obs_dims, self.D-1)
            distout = torch.distributions.categorical.Categorical(logits=logits)

            ll = distout.log_prob( x - 1 )
            ll = self.sum_except_batch(ll * current_selection)

            # importance weight
            ll = ll * t / self.xdim

            return ll.mean()


        for t in range(self.xdim):
            if t > T.max(): break
            #print("%u out of %u steps" % (t, T.max()))
            previous_selection = (sigma < t)
            current_selection = (sigma == t)

            xin = x * previous_selection + zeroimg * (~previous_selection)

            logits = self.net_forward(xin, previous_selection).reshape(batch, *self.obs_dims, self.D-1)
            distout = torch.distributions.categorical.Categorical(logits=logits)

            ll = distout.log_prob( x - 1 )
            ll = self.sum_except_batch(ll * current_selection)
            ll = ll * (T > t) # stop if we're done with all the unmasked inputs

            total_ll += ll

            if t % 300 == 0:
                print(t)
                print(total_ll.mean() / (t+1))

        return total_ll.mean()

    def forward(self, x):
        batch = x.shape[0]
        zeroimg = torch.zeros(batch, *self.obs_dims, device=x.device) + 0

        previous_selection, current_selection = self.sample_mask(batch, x.device)
        future_selection = ~previous_selection
        xin = x * previous_selection + zeroimg * (~previous_selection)
        
        logits = self.net_forward(xin, previous_selection).reshape(batch, *self.obs_dims, self.D-1)
        distout = torch.distributions.categorical.Categorical(logits=logits)

        ll = distout.log_prob( x - 1 )
        ll_final = self.sum_except_batch(ll * future_selection) / self.sum_except_batch(future_selection)
        return ll_final.mean()

    def conditional_sample(self, X, mask, sharpness=1):
        batch = X.shape[0]
        zeroimg = torch.zeros(batch, *self.obs_dims, device=X.device) + 0
        mask = mask.bool()
        xin = X * mask

        sigma = self.mask_to_order(mask)

        start_t = self.sum_except_batch(mask).min()
        for t in range(start_t, self.xdim):
            if t % 10 == 0:
                print("%u out of %u steps" % (t, self.xdim))
            
            previous_selection = (sigma < t)
            current_selection = (sigma == t)

            logits = self.net_forward(xin, previous_selection).reshape(batch, *self.obs_dims, self.D-1)

            probs = F.softmax(logits * sharpness, dim=-1)
            probs = (probs * current_selection.unsqueeze(dim=-1)).sum(dim=1)

            sample = torch.multinomial(probs, num_samples=1).squeeze()
            sample = sample.reshape(batch, 1)

            xin = xin * previous_selection + sample * current_selection + zeroimg * (~(previous_selection | current_selection))
            xin = X * mask + xin * (~mask) # make sure each time we reinstate the evidence

        return xin

    def sample(self, batch, device='cuda:0', sharpness=1):
        xin = torch.zeros(batch, *self.obs_dims, device=device)
        mask = xin.bool()
        return self.conditional_sample(xin, mask, sharpness)
