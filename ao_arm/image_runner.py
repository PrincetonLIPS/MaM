import os
import logging
import numpy as np
import torch
import torch.optim as optim
import torchvision
from data.dataset import get_dataset
from image_model import ARM

class Runner(object):
    def __init__(self, cfg):
        self.cfg = cfg

        self.device_id = 'cuda:{}'.format(cfg.local_rank)
        self.master_node = (self.cfg.local_rank == 0)
        self.distributed = (self.cfg.world_size > 1)

        self.train_loader, self.test_loader = get_dataset(cfg, distributed=self.distributed)

        self.obs = (1, 28, 28) if self.cfg.dataset in ['MNIST', 'MNIST_bin']  else (3, 32, 32)
        self.cfg.binary = True if 'MNIST_bin' in self.cfg.dataset else False
        xdim = np.prod(self.obs)
        self.epoch = 0

        self.net = ARM(image_dims=self.obs, cfg=self.cfg)
        self.net.to(self.device_id)

        if self.distributed:
            self.net = torch.nn.parallel.DistributedDataParallel(self.net, device_ids=[cfg.local_rank], output_device=cfg.local_rank)
            self.net_module = self.net.module
        else:
            self.net_module = self.net
        
        self.clip_grad = 100.
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.cfg.lr)

        if self.cfg.loadpath is not None:
            self.load(self.cfg.loadpath)

        self.save_every = 200
        self.eval_every = 5

    def load(self, path):
        map_location = {"cuda:0": self.device_id}
        checkpoint = torch.load(path, map_location=map_location)
        self.net_module.load_state_dict(checkpoint['net'])        
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epoch = checkpoint['epoch']
        logging.info("loaded")

    def train(self):
        logging.info("training rank %u" % self.cfg.local_rank)
        self.net.train()
        dataloader = self.train_loader

        while self.epoch < self.cfg.n_epochs:
            
            epoch_metrics = {
                'log_ll': 0,
                'count': 0,
            }

            bsz = 0
            accum, accumll = 0, 0.0
            self.net.train()

            for it, (X, y) in enumerate(dataloader):
                X = X.cuda(device=self.device_id, non_blocking=True)

                log_ll = self.net(X)
                (-log_ll).backward()

                count = X.shape[0]
                epoch_metrics['log_ll'] += log_ll * count
                epoch_metrics['count'] += count

                bsz += X.shape[0]
                accum += X.shape[0]
                accumll += log_ll * count

                if bsz >= 128 // self.cfg.world_size:
                    total_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_grad)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    bsz = 0

                if accum >= 5120 // self.cfg.world_size:
                    if self.master_node:
                        logging.info("Iter %u, log-ll: %.2f" % ( (it + 1 + len(dataloader)*self.epoch) * self.cfg.batch_size, log_ll))
                        accum = 0
                        accumll = 0.0

            if self.master_node:
                states = {
                    'net': self.net_module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': self.epoch + 1,
                }

                torch.save(states, os.path.join(self.cfg.model_dir, 'checkpoint.pth'))
                if self.epoch % self.save_every == 0:
                    torch.save(states, os.path.join(self.cfg.model_dir, 'checkpoint_{}.pth'.format(self.epoch)))

            if self.epoch % 5 == 0:
                with torch.no_grad():
                    metric_tensor = torch.tensor( [epoch_metrics['log_ll'], epoch_metrics['count'] ] )
                    if self.distributed:
                        torch.distributed.reduce(metric_tensor, dst=0)

                test_epoch_metric_tensor = self.test_marginal()

                if self.master_node:
                    metric_tensor[0] /= metric_tensor[1]
                    logging.info("Epoch %u out of %u, train log_ll: %.2f, test log_ll: %.2f" % (self.epoch, self.cfg.n_epochs, metric_tensor[0], test_epoch_metric_tensor[0]))

            self.epoch += 1

    def test_marginal(self):
        logging.info("testing")
        self.net.eval()
        dataloader = self.test_loader
        mode = 'test'

        epoch_metrics = {
            'log_ll': 0,
            'count': 0,
        }

        for X, _ in dataloader:
            X = X.cuda(device=self.device_id, non_blocking=True)

            with torch.no_grad():
                log_ll = self.net_module.likelihood(X, mask=None, full=False)

            count = X.shape[0]
            epoch_metrics['log_ll'] += log_ll * count
            epoch_metrics['count'] += count

        with torch.no_grad():
            metric_tensor = torch.tensor( [epoch_metrics['log_ll'], epoch_metrics['count']] )
            if self.distributed:
                torch.distributed.reduce(metric_tensor, dst=0)

            if self.master_node:
                metric_tensor[0] /= metric_tensor[1]
                logging.info("%s count %u log_ll: %.4f" % (mode, metric_tensor[1], metric_tensor[0]))
        return metric_tensor
    
    def generate(self):
        num_samples = self.cfg.gen_num_samples
        batch_size = self.cfg.gen_batch_size
        num_batches = num_samples // batch_size
        self.net.eval()
        with torch.no_grad():
            for i in range(num_batches):
                samples = self.net_module.sample(batch_size)
                file_name = os.path.join(self.cfg.log_dir, 'samples_{}.png'.format(i))
                torchvision.utils.save_image(samples, file_name, normalize=True, nrow=int(batch_size ** .5))