import torch
import torch.utils.data as data_utils
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import torch.distributions as dists
from tqdm import tqdm
import igraph as ig
import numpy as np
import torch.nn.functional as F
import seaborn as sns
from matplotlib import pyplot as plt


class LatticeIsingModel(nn.Module):
    def __init__(self, dim, init_sigma=.15, init_bias=0., learn_G=False, learn_sigma=False, learn_bias=False,
                 lattice_dim=2, n_samples=2000):
        super().__init__()
        g = ig.Graph.Lattice(dim=[dim] * lattice_dim, circular=True)  # Boundary conditions
        A = np.asarray(g.get_adjacency().data)  # g.get_sparse_adjacency()
        self.G = nn.Parameter(torch.tensor(A).float(), requires_grad=learn_G)
        self.sigma = nn.Parameter(torch.tensor(init_sigma).float(), requires_grad=learn_sigma)
        self.bias = nn.Parameter(torch.ones((dim ** lattice_dim,)).float() * init_bias, requires_grad=learn_bias)
        self.init_dist = dists.Bernoulli(logits=2 * self.bias)
        self.data_dim = dim ** lattice_dim
        self.n_samples = n_samples

    def init_sample(self, n):
        return self.init_dist.sample((n,))

    @property
    def J(self):
        return self.G * self.sigma
    
    def get_scores(self, x):
        return self(x)
    
    def plot_scores(self, scores, save_path):
        sns.kdeplot(scores, fill=True)
        plt.savefig(save_path)
        plt.close()

    def forward(self, x):
        if len(x.size()) > 2:
            x = x.view(x.size(0), -1)

        x = (2 * x) - 1 # from {0,1} to {-1,1}

        xg = x @ self.J
        xgx = (xg * x).sum(-1)
        b = (self.bias[None, :] * x).sum(-1)
        return xgx + b
    
    def generate_samples(self, n_samples, gt_steps=1000000):
        sampler = PerDimGibbsSampler(self.data_dim, rand=False)
        samples = self.init_sample(n_samples)
        print("Generating {:d} samples from {:s}".format(n_samples, str(self)))
        for _ in tqdm(range(gt_steps)):
            samples = sampler.step(samples, self).detach()
        return samples.detach().cpu()
    
class PerDimGibbsSampler(nn.Module):
    def __init__(self, dim, rand=False):
        super().__init__()
        self.dim = dim
        self.changes = torch.zeros((dim,))
        self.change_rate = 0.
        self.p = nn.Parameter(torch.zeros((dim,)))
        self._i = 0
        self._ar = 0.
        self._hops = 0.
        self._phops = 1.
        self.rand = rand

    def step(self, x, model):
        sample = x.clone()
        lp_keep = model(sample).squeeze()
        if self.rand:
            changes = dists.OneHotCategorical(logits=torch.zeros((self.dim,))).sample((x.size(0),)).to(x.device)
        else:
            changes = torch.zeros((x.size(0), self.dim)).to(x.device)
            changes[:, self._i] = 1.

        sample_change = (1. - changes) * sample + changes * (1. - sample)

        lp_change = model(sample_change).squeeze()

        lp_update = lp_change - lp_keep
        update_dist = dists.Bernoulli(logits=lp_update)
        updates = update_dist.sample()
        sample = sample_change * updates[:, None] + sample * (1. - updates[:, None])
        self.changes[self._i] = updates.mean()
        self._i = (self._i + 1) % self.dim
        self._hops = (x != sample).float().sum(-1).mean().item()
        self._ar = self._hops
        return sample

    def logp_accept(self, xhat, x, model):
        # only true if xhat was generated from self.step(x, model)
        return 0


def prepare_ising_data(model, config, distributed=False):
    def binary(x, bits):
        mask = 2**torch.arange(bits).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()
    Z = 0
    # if config.ising_model.dim == 2:
    #     for i in range(2 ** (config.ising_model.dim ** 2)):
    #         sample = torch.tensor([i])
    #         energy = - model(binary(sample, config.ising_model.dim ** 2).float())
    #         Z += torch.exp(-energy)
    #     print("LogZ", torch.log(Z))
        # config.ising_model.LogZ = torch.log(Z).item()
    x_train = model.init_sample(config.n_train)
    y_train = model(x_train)
    x_val = model.init_sample(config.n_val)
    y_val = model(x_val)
    x_test = model.init_sample(config.n_test)
    y_test = model(x_test)
    dataset_train = data_utils.TensorDataset(x_train.float()+1.0, y_train) # 0 is reserved for ? mask
    dataset_val = data_utils.TensorDataset(x_val.float()+1.0, y_val)
    dataset_test = data_utils.TensorDataset(x_test.float()+1.0, y_test)

    if distributed:  
        train_sampler = DistributedSampler(dataset_train, num_replicas=config.world_size, rank=config.local_rank, shuffle=True, drop_last=False)
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=config.batch_size, sampler=train_sampler)
    else:
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True, num_workers=4)
        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=config.test_batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=config.test_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader