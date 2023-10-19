import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data_utils
from torch.utils.data.distributed import DistributedSampler
import numpy as np

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

def get_dataset(config, distributed=False):
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': True}

    if config.dataset == 'MNIST_bin':
        train_dataset, _, test_dataset = load_static_mnist(config.data_dir)

    elif config.dataset == 'TEXT8':
        from .dataset_text8 import Text8
        data = Text8(root=config.data_dir, seq_len=config.seqlen)
        data_shape = (1,config.seqlen)
        num_classes = 27
        #train_dataset = torch.utils.data.ConcatDataset([data.train, data.valid])
        train_dataset = data.train
        test_dataset = data.test

    elif config.dataset == 'MOSES':
        from .dataset_mol import load_moses
        train_dataset, test_dataset, _ = load_moses(config)

    if distributed:  
        train_sampler = DistributedSampler(train_dataset, num_replicas=config.world_size, rank=config.local_rank, shuffle=True, drop_last=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler, **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader