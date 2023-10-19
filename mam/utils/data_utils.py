import torch
import torch.utils.data as data_utils
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import os
import pathlib
import pickle

from utils.text8 import Text8
from utils.load_mols import load_moses


def image_int_to_float(x, binary=False):
    if binary:
        return x.float() * 2 - 1.0 # from {0, 1} to {-1.0, 1.0}
    else:
        return x / 127.5 - 1.

def image_float_to_int(x, binary=False):
    if binary:
        return torch.round( (x+1.0)/2.0 ).long() # from {-1.0, 1.0} to {0, 1}
    else:
        return torch.round( (x+1) * 127.5 ).long()

def load_static_mnist(data_dir):
    def lines_to_np_array(lines):
        return np.array([[int(i) for i in line.split()] for line in lines])
    data_folder_dir = os.path.join(data_dir, 'MNIST_static')
    if not os.path.exists(data_folder_dir):
        os.makedirs(data_folder_dir)
        print('Downloading static binarized MNIST')
        os.system('wget -P {} http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat'.format(data_folder_dir))
        os.system('wget -P {} http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat'.format(data_folder_dir))
        os.system('wget -P {} http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat'.format(data_folder_dir))
    with open(os.path.join(data_folder_dir, 'binarized_mnist_train.amat')) as f:
        lines = f.readlines()
    x_train = lines_to_np_array(lines).astype('float32').reshape(-1, 1, 28, 28)
    with open(os.path.join(data_folder_dir, 'binarized_mnist_valid.amat')) as f:
        lines = f.readlines()
    x_val = lines_to_np_array(lines).astype('float32').reshape(-1, 1, 28, 28)
    with open(os.path.join(data_folder_dir, 'binarized_mnist_test.amat')) as f:
        lines = f.readlines()
    x_test = lines_to_np_array(lines).astype('float32').reshape(-1, 1, 28, 28)
    # shuffle train data
    np.random.shuffle(x_train)
    # idle y's
    y_train = np.zeros( (x_train.shape[0], 1) )
    y_val = np.zeros( (x_val.shape[0], 1) )
    y_test = np.zeros( (x_test.shape[0], 1) )
    # convert data from {0, 1} to {-1, 1}
    x_train = 2 * x_train - 1
    x_val = 2 * x_val - 1
    x_test = 2 * x_test - 1
    # pytorch data loader
    train_dataset = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_dataset = data_utils.TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    test_dataset = data_utils.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    return train_dataset, val_dataset, test_dataset

def load_ising_gt_samples(config, score_model):
    samples_dir = os.path.join(config.data_dir, config.dataset)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    samples_path = os.path.join(samples_dir, '{}d_gt_samples.pkl'.format(config.ising_model.dim))
    if os.path.exists(samples_path):
        with open(samples_path, 'rb') as f:
            ising_samples = pickle.load(f)
    else:
        ising_samples = score_model.generate_samples(config.ising_model.n_samples, config.ising_model.gt_steps)
        with open(samples_path, 'wb') as f:
            pickle.dump(ising_samples, f)
    return ising_samples

def load_dataset(config, distributed=False, **kwargs):
    if config.dataset == 'TEXT8':
        data = Text8(root=config.data_dir, seq_len=config.L)
        data_shape = (1,config.L)
        num_classes = 27
        #train_dataset = torch.utils.data.ConcatDataset([data.train, data.valid])
        dataset_train = data.train
        dataset_val = data.valid
        dataset_test = data.test
    elif config.dataset == 'MOSES':
        dataset_train, dataset_val, dataset_test = load_moses(config)
    elif config.dataset == 'molecule':
        dataset_train, dataset_val, dataset_test = load_moses(config) # load data but not for MLE training
    elif config.dataset == 'MNIST_bin':
        dataset_train, dataset_val, dataset_test = load_static_mnist(config.data_dir)
    else:
        raise ValueError('Unknown dataset')
    # data loader for training 
    if distributed:  
        train_sampler = DistributedSampler(dataset_train, num_replicas=config.world_size, rank=config.local_rank, shuffle=True, drop_last=False)
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=config.batch_size, sampler=train_sampler, **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=config.test_batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=config.test_batch_size, shuffle=True, **kwargs)
    
    return train_loader, val_loader, test_loader
 
