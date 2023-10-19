from tqdm import tqdm

import math
import numpy as np
import torch
import torch.nn.functional as F

import time

from utils.constants import BIT_UNKNOWN_VAL

def convert_val_to_onehot(samples, K):
    samples = F.one_hot(samples.long(), num_classes=K+1).to(torch.float32) # (B, L) -> (B, L, K+1)
    return samples

def convert_onehot_to_val(samples):
    samples = samples.argmax(-1).to(torch.float32)
    return samples

def sample_order(x_batch, gen_order='random'):
    if gen_order == 'random':
        rand_order = torch.argsort(torch.rand_like(x_batch), dim=-1)
    elif gen_order == 'forward':
        rand_order = torch.arange(x_batch.shape[-1]-1,-1, step=-1).expand(x_batch.shape[0],-1).to(x_batch.device)
    elif gen_order == 'backward':
        rand_order = torch.arange(x_batch.shape[-1]).expand(x_batch.shape[0],-1).to(x_batch.device)
    else:
        raise ValueError("order must be either 'random' or 'forward' or 'backward'")
    return rand_order

def gen_order(batch_size, data_dim, device, gen_order='random'):
    if gen_order == 'random':
        rand_order = torch.argsort(torch.rand(batch_size, data_dim), dim=-1).to(device)
    elif gen_order == 'backward':
        rand_order = torch.arange(data_dim-1,-1,step=-1).expand(batch_size,-1).to(device)
    elif gen_order == 'forward':
        rand_order = torch.arange(data_dim).expand(batch_size,-1).to(device)
    else:
        raise ValueError("order must be either 'random' or 'forward' or 'backward'")
    return rand_order