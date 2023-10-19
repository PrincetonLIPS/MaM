import os
import logging
import time
from tqdm import tqdm
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tensorboardX import SummaryWriter

import selfies as sf

from models.unet import ARM_UNet
from models.mam_image import MAM
from utils.data_utils import load_dataset, image_float_to_int
from utils.mar_utils_mol import gen_order
from utils.eval_utils import compare_logp, preprocess_logp, create_epoch_metrics, update_epoch_metrics

from utils.constants import BIT_UNKNOWN_VAL

class Runner(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.writer = SummaryWriter(os.getcwd())

        self.device_id = 'cuda:{}'.format(cfg.local_rank)
        self.master_node = (self.cfg.local_rank == 0)
        self.distributed = (self.cfg.world_size > 1)
        self.train_loader, self.val_loader, self.test_loader = load_dataset(cfg, distributed=self.distributed)
        self.epoch = 0
        self.image_dims = (1, 28, 28) if self.cfg.dataset in ['MNIST_bin']  else (3, 32, 32)
        self.cfg.L = np.prod(np.array(self.image_dims)).item() # L, 1*28*28 for MNIST or 3*32*32 for CIFAR
        self.cfg.K = 2 if self.cfg.dataset in ['MNIST_bin'] else 256 # 2**bits in each dimension
        self.cfg.binary = True if self.cfg.dataset in ['MNIST_bin'] else False

        if cfg.arch == 'unet':
            self.ch_mult = [1]
            self.xdim = np.prod(np.array(self.image_dims)).item()
            self.nn = ARM_UNet(
                image_dims=self.image_dims,
                pixels=self.cfg.K,
                num_classes=256,
                ch=256,
                out_ch= self.cfg.K * self.image_dims[0],
                input_channels = self.image_dims[0],
                ch_mult = self.ch_mult,
                num_res_blocks=self.cfg.num_res_blocks,
                full_attn_resolutions=[32, 16, 14, 8, 7, 4],
                num_heads=1,
                dropout=0.,
                max_time=1000.,
                weave_attn=self.cfg.weave_attn)
        else:
            raise ValueError("Unknown model {}".format(cfg.nn.model))
        self.nn.to(self.device_id)
        logging.info(self.nn)


        self.marnet = MAM(self.nn, cfg)
        self.marnet.to(self.device_id)

        if self.distributed:
            self.marnet = torch.nn.parallel.DistributedDataParallel(self.marnet, device_ids=[cfg.local_rank], output_device=cfg.local_rank)
            self.marnet_module = self.marnet.module
        else:
            self.marnet_module = self.marnet

        self.clip_grad = self.cfg.clip_grad

        param_list = [{'params': self.marnet.net.parameters(), 'lr': self.cfg.lr},
                      {'params': self.marnet.LogZ, 'lr': self.cfg.zlr}]

        self.optimizer = optim.Adam(param_list)

        if self.cfg.mode == 'train':
            if self.cfg.load_pretrain and self.cfg.loadpath_pretrain is not None:
                self.load(self.cfg.loadpath_pretrain)
                # only finetune an additional MLP block after the transformer encoder
                for name, param in self.marnet.net.named_parameters():
                    if name.startswith('marg'):
                        param.requires_grad = True
                        print("param {} requires grad".format(name))
                    else:
                        param.requires_grad = False
            else:
                raise ValueError("Training marginals need to load trained conditionals from ao-arm first")
        else:
            if self.cfg.loadpath is not None:
                self.load(self.cfg.loadpath)
            else:
                raise ValueError("Model path is not specified")

        self.save_every = 200
        self.eval_every = 5

    def load(self, path):
        map_location = {"cuda:0": self.device_id}
        checkpoint = torch.load(path, map_location=map_location)
        if self.cfg.mode == 'train':
            self.marnet_module.load_state_dict(checkpoint['net'], strict=False)
        else:
            self.marnet_module.load_state_dict(checkpoint['net'], strict=False)
        print("loaded", flush=True)

    def train(self):
        print("training rank %u" % self.cfg.local_rank, flush=True)
        self.marnet.train()
        dataloader = self.train_loader

        it = 0
        while self.epoch < self.cfg.n_epochs:
            
            epoch_metrics = {
                'log_ll': 0,
                'mb_loss': 0,
                'mb_loss_begin': 0,
                'likelihood': 0,
                'count': 0,
            }
            bsz = 0
            accum, accumll = 0, 0.0
            if self.cfg.include_onpolicy:
                self.marnet.eval()
                with torch.no_grad():
                    init_samples = next(iter(self.train_loader))[0]
                    init_samples = init_samples.reshape(init_samples.shape[0], -1)
                    self.marnet_module.samples =init_samples.cuda(device=self.device_id, non_blocking=True)
            self.marnet.train()

            pbar = tqdm(dataloader)
            pbar.set_description("Epoch {}: Training".format(self.epoch))
            for x, _ in pbar:
                x = x.cuda(device=self.device_id, non_blocking=True)
                x = x.reshape(x.shape[0], -1) # (B, L)
                loss, logp_real, log_z, mb_loss, mb_loss_begin = self.marnet(x)
                loss.backward()

                count = x.shape[0]
                epoch_metrics['log_ll'] += logp_real.item() * count
                epoch_metrics['mb_loss'] += mb_loss.item() * count
                epoch_metrics['mb_loss_begin'] += mb_loss_begin.item() * count
                epoch_metrics['count'] += count

                bsz += x.shape[0]
                accum += x.shape[0]
                accumll += logp_real.item() * count

                if bsz >= 128 // self.cfg.world_size:
                    total_norm = torch.nn.utils.clip_grad_norm_(self.marnet.parameters(), self.clip_grad)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    bsz = 0

                if accum >= 5120 // self.cfg.world_size:
                    if self.master_node:
                        logging.info("Iter %u out of %u, log-ll: %.2f, log-z: %.2f, mb-loss: %.2f, mb-loss-begin: %.2f, loss: %.2f"
                            % (it, len(dataloader), logp_real, log_z, mb_loss, mb_loss_begin, loss))
                        self.writer.add_scalar('log_ll', logp_real, it + 1)
                        self.writer.add_scalar('log_z', log_z, it + 1) # self.marnet_module.LogZ
                        self.writer.add_scalar('mb_loss', mb_loss, it + 1)
                        self.writer.add_scalar('mb_loss_begin', mb_loss_begin, it + 1)
                        self.writer.add_scalar('loss', loss, it + 1)
                        accum = 0
                        accumll = 0.0

                pbar.set_postfix({"log_ll": f"{logp_real.item():.2f}", "log_z": f"{log_z.item():.2f}",\
                    "mb": f"{mb_loss.item():.2e}", "mb_begin": f"{mb_loss_begin.item():.2e}",\
                    "loss": f"{loss.item():.2e}"})
                it += 1

            if self.master_node:
                if self.cfg.save_model:
                    states = {
                        'net': self.marnet_module.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'epoch': self.epoch + 1,
                        'L': self.cfg.L,
                        'K': self.cfg.K,
                    }
                    torch.save(states, os.path.join(self.cfg.model_dir, 'checkpoint.pth'))
                    if self.epoch % self.save_every == 0:
                        torch.save(states, os.path.join(self.cfg.model_dir, 'checkpoint_{}.pth'.format(self.epoch)))

            if self.epoch % self.eval_every == 0:
                with torch.no_grad():
                    metric_tensor = torch.tensor( [epoch_metrics['log_ll'], epoch_metrics['mb_loss'],
                        epoch_metrics['likelihood'], epoch_metrics['count'] ] )
                    if self.distributed:
                        torch.distributed.reduce(metric_tensor, dst=0)

                test_epoch_metric_tensor = self.test_ll()

                if self.master_node:
                    metric_tensor[0] /= metric_tensor[-1]
                    self.writer.add_scalar('train_log_ll_ebm', metric_tensor[0], self.epoch)
                    self.writer.add_scalar('val_log_ll_ebm', test_epoch_metric_tensor[0], self.epoch)
                    self.writer.add_scalar('val_log_ll', test_epoch_metric_tensor[1], self.epoch)
                    self.writer.add_scalar('val_log_ll_err', test_epoch_metric_tensor[2], self.epoch)
                    self.writer.add_scalar('val_log_ll_err_var', test_epoch_metric_tensor[3], self.epoch)
                    logging.info("Epoch %u out of %u, train log_ll: %.2f,"
                        "val log_ll_ebm: %.2f, val log_ll: %.2f val log_ll_err: %.2f val log_ll_err_var: %.2f" % (
                        self.epoch, self.cfg.n_epochs, metric_tensor[0], test_epoch_metric_tensor[0], \
                        test_epoch_metric_tensor[1], test_epoch_metric_tensor[2], test_epoch_metric_tensor[3]))
                
            self.epoch += 1

    def test_ll(self):
        self.marnet.eval()
        dataloader = self.val_loader
        mode = 'test'

        epoch_metrics = {
            'log_ll_ebm': 0,
            'log_ll': 0,
            'log_ll_err': 0,
            'log_ll_err_var': 0,
            'count': 0,
        }
        pbar = tqdm(dataloader)
        pbar.set_description("Testing calculating likelihood")
        it = 0
        for x, _ in pbar:
            x = x.cuda(device=self.device_id, non_blocking=True)
            x = x.reshape(x.shape[0], -1) # (B, L)
            mask = torch.ones_like(x).cuda(device=self.device_id, non_blocking=True).bool()
            file_name = os.path.join(self.cfg.log_dir, 'samples_{}.png'.format(it))
            torchvision.utils.save_image(x.float().reshape(x.shape[0], *self.image_dims), file_name, normalize=True, nrow=int(self.cfg.test_batch_size ** .5))
            with torch.no_grad():
                logp_ebm, log_z = self.marnet_module.eval_ll(x, mask)
                logp = self.marnet_module.est_logp(x, self.cfg.eval.mc_ll, self.cfg.gen_order) # (B,)
            logp_err = (logp - logp_ebm + log_z).abs().mean()
            logp_err_var = (logp - logp_ebm + log_z).var()
            logp_ebm = logp_ebm.mean()
            logp = logp.mean()
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix({"log_ll": f"{logp:.2f}", "log_ll_ebm": f"{logp_ebm:.2f}",\
                    "log_ll_err": f"{logp_err:.2f}", "log_ll_err_var": f"{logp_err_var:.2f}"})
            count = x.shape[0]
            epoch_metrics['log_ll_ebm'] += logp_ebm.item() * count
            epoch_metrics['log_ll'] += logp.item() * count
            epoch_metrics['log_ll_err'] += logp_err.item() * count
            epoch_metrics['log_ll_err_var'] += logp_err_var.item() * count
            epoch_metrics['count'] += count
            it += 1
            if it==self.cfg.eval.num_batches:
                break

        with torch.no_grad():
            metric_tensor = torch.tensor( [epoch_metrics['log_ll_ebm'], epoch_metrics['log_ll'],\
            epoch_metrics['log_ll_err'], epoch_metrics['log_ll_err_var'], epoch_metrics['count'] ] )
            if self.distributed:
                torch.distributed.reduce(metric_tensor, dst=0)

            if self.master_node:
                metric_tensor[0] /= metric_tensor[-1]
                metric_tensor[1] /= metric_tensor[-1]
                metric_tensor[2] /= metric_tensor[-1]
                metric_tensor[3] /= metric_tensor[-1]
                logging.info("%s count %u log_ll_ebm: %.4f log_ll: %.4f log_ll_err: %.4f log_ll_err_var %.4f" % (
                    mode, metric_tensor[-1], metric_tensor[0], metric_tensor[1], metric_tensor[2], metric_tensor[3]))

        return metric_tensor

    def generate(self):
        self.marnet.eval()
        mode = 'generate'
        num_samples = self.cfg.gen_num_samples
        iters = num_samples // self.cfg.generate_batch_size
        pbar = tqdm(range(iters))
        x_gen_list = []
        for i in pbar:
            with torch.no_grad():
                if self.cfg.conditional:
                    # get a batch from test_loader
                    x_cond, _ = next(iter(self.test_loader))
                    x_cond = x_cond[i:i+1,].cuda(device=self.device_id, non_blocking=True)
                    x_cond = x_cond.reshape(x_cond.shape[0], -1) # (B, L)
                    x_cond[:, :14*28] = BIT_UNKNOWN_VAL                       
                    rand_order_gen = gen_order(
                        self.cfg.generate_batch_size, 14*28, self.device_id, gen_order=self.cfg.gen_order
                    )
                    x_gen = self.marnet_module.cond_sample(x_cond, rand_order_gen, self.cfg.generate_batch_size)
                else:
                    rand_order_gen = gen_order(
                        self.cfg.generate_batch_size, self.cfg.L, self.device_id, gen_order=self.cfg.gen_order
                    )
                    x_gen = self.marnet_module.sample(rand_order_gen, self.cfg.generate_batch_size)
                mask = torch.ones_like(x_gen).cuda(device=self.device_id, non_blocking=True).bool()
                logp_ebm, log_z = self.marnet_module.eval_ll(x_gen, mask=mask)
                logp = self.marnet_module.est_logp(x_gen, self.cfg.eval.mc_ll, self.cfg.gen_order) # (B,)
                logging.info("logp_ebm: %.4f logp: %.4f" % ((logp_ebm - log_z).mean(), logp.mean()))
            x_gen = image_float_to_int(x_gen, self.cfg.binary)
            x_gen_list.append(x_gen)
            file_name = os.path.join(self.cfg.log_dir, 'samples_{}.pdf'.format(i))
            torchvision.utils.save_image(x_gen.float().reshape(x_gen.shape[0], *self.image_dims), file_name, normalize=True, nrow=int(self.cfg.generate_batch_size ** .5))
            pbar.set_postfix({"generated samples": f"{(i+1)*self.cfg.generate_batch_size}"})
        x_gen_all = torch.cat(x_gen_list, dim=0)
        file_name = os.path.join(self.cfg.log_dir, 'samples_all.pdf')
        torchvision.utils.save_image(x_gen_all.float().reshape(x_gen_all.shape[0], *self.image_dims), file_name, normalize=True, nrow=int(self.cfg.gen_num_samples ** .5))

    def eval_mam_quality(self):
        self.marnet.eval()
        max_iters = self.cfg.eval_num_samples // self.cfg.test_batch_size
        itr = 0
        epoch_metrics = create_epoch_metrics()
        pbar = tqdm(self.test_loader)
        instance = 0
        for x, _ in pbar:
            x = x.cuda(device=self.device_id, non_blocking=True)
            x_orig_all = x.reshape(x.shape[0], -1) # (B, L)
            rand_order_gen = gen_order(
                self.cfg.test_batch_size, self.cfg.L, self.device_id, gen_order=self.cfg.gen_order
            )
            x_censored_list = []
            x_gen_list = []
            for steps in self.cfg.mask_steps:
                with torch.no_grad():
                    x_gen, x_censored = self.marnet_module.censor_and_sample(x_orig_all, steps, rand_order_gen)
                x_censored_list.append(x_censored.unsqueeze(1))
                x_gen_list.append(x_gen.unsqueeze(1))
            x_censored_all = torch.cat(x_censored_list, dim=1) # (B, M, L)
            x_gen_all = torch.cat(x_gen_list, dim=1) # (B, M, L)
            for i in range(x_orig_all.shape[0]):
                x_gen = x_gen_all[i,] # (M, L) of each instance
                x_censored = x_censored_all[i,]
                x_orig = x_orig_all[i:i+1,]
                with torch.no_grad():
                    mask = torch.ones_like(x_gen).cuda(device=self.device_id, non_blocking=True).bool()
                    logp_ebm, log_z = self.marnet_module.eval_ll(x_gen, mask)
                    logp = self.marnet_module.est_logp(x_gen, 5, self.cfg.gen_order) # (M,) # logp from ensemble
                    logp_2 = self.marnet_module.est_logp(x_gen, 1, self.cfg.gen_order) # (M,) # logp from single random ordering
                selected_ind = preprocess_logp(logp, self.cfg.eval.threshold)
                logp, logp_2, logp_ebm = logp[selected_ind], logp_2[selected_ind], logp_ebm[selected_ind]
                x_gen, x_censored = x_gen[selected_ind], x_censored[selected_ind]
                count = logp.shape[0]
                if count > 1:
                    cmp_results = compare_logp(logp, logp_2, logp_ebm, log_z)
                    x_all = torch.cat([x_orig, x_censored, x_gen], dim=0)
                    save_data_path = os.path.join(self.cfg.log_dir, 'samples_generate_cmp_{}.npz'.format(instance))
                    np.savez(save_data_path,
                             x_orig=x_orig.cpu().numpy(), 
                             x_censored=x_censored.cpu().numpy(), 
                             x_gen=x_gen.cpu().numpy())
                    file_name = os.path.join(self.cfg.log_dir, 'samples_{}.png'.format(instance))
                    torchvision.utils.save_image(x_all.float().reshape(x_all.shape[0], *self.image_dims), file_name, normalize=False, nrow=int(x_all.shape[0] ** .5))
                    with open(os.path.join('generated_samples_logp_{}.txt'.format(instance)), 'w') as f:
                        f.write("logp_e: {}".format(logp.cpu().numpy()) + '\n')
                        f.write("logp_r: {}".format(logp_2.cpu().numpy()) + '\n')
                        f.write("logp_ebm: {}".format(logp_ebm.cpu().numpy()) + '\n')
                        f.write("iter %u"
                            "sp: %.4f sp_p: %.4f sp_self: %.4f sp_self_p %.4f "
                            "pr: %.4f pr_p: %.4f pr_self: %.4f pr_self_p %.4f"
                            % (i, cmp_results['spearman'].correlation, cmp_results['spearman'].pvalue, cmp_results['spearman_self'].correlation, cmp_results['spearman_self'].pvalue,
                            cmp_results['pearson'][0], cmp_results['pearson'][1], cmp_results['pearson_self'][0], cmp_results['pearson_self'][1]))
                    update_epoch_metrics(epoch_metrics, cmp_results, count)
                instance += 1
            
            itr += 1
            if itr == max_iters:
                break
        # compute the average of the metrics
        metric_tensor = torch.tensor( [
            epoch_metrics['spearman'], epoch_metrics['spearman_pvalue'],\
            epoch_metrics['spearman_s'], epoch_metrics['spearman_s_pvalue'],\
            epoch_metrics['pearson'], epoch_metrics['pearson_pvalue'],\
            epoch_metrics['pearson_s'], epoch_metrics['pearson_s_pvalue'],\
            epoch_metrics['log_ll_e'], epoch_metrics['log_ll_s'],\
            epoch_metrics['log_ll_err'], epoch_metrics['log_ll_err_var'], epoch_metrics['count'] ] )
        # loop through the whole metric tensor by count
        for i in range(metric_tensor.shape[0]-1):
            metric_tensor[i] /= metric_tensor[-1]

        logging.info("count %u"
            "sp: %.4f sp_p: %.4f sp_self: %.4f sp_self_p %.4f "
            "pr: %.4f pr_p: %.4f pr_self: %.4f pr_self_p %.4f "
            "logp_e %.4f logp_s %.4f "        
            "logp_err %.4f logp_var %.4f" % (
            metric_tensor[-1], metric_tensor[0], metric_tensor[1], metric_tensor[2],\
            metric_tensor[3], metric_tensor[4], metric_tensor[5], metric_tensor[6],\
            metric_tensor[7], metric_tensor[8], metric_tensor[9], metric_tensor[10],\
            metric_tensor[11]))
        pbar.set_postfix({"generated samples": f"{(itr+1)*self.cfg.test_batch_size}"})