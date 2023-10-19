import os
import logging
from tqdm import tqdm
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import selfies as sf
from models.transformer import TransformerNet
from models.mam import MAM
from utils.data_utils import load_dataset
from utils.mar_utils_mol import gen_order
from utils.mol_utils import multiple_indices_to_string, string_to_int
from utils.constants import BIT_UNKNOWN_VAL
from utils.eval_utils import compare_logp, preprocess_logp, create_epoch_metrics, update_epoch_metrics

import time


class Runner(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.writer = SummaryWriter(os.getcwd())

        self.device_id = 'cuda:{}'.format(cfg.local_rank)
        self.master_node = (self.cfg.local_rank == 0)
        self.distributed = (self.cfg.world_size > 1)
        self.train_loader, self.val_loader, self.test_loader = load_dataset(cfg, distributed=self.distributed)
        self.epoch = 0

        if cfg.arch == 'transformer':
            self.nn = TransformerNet(
                num_src_vocab=(cfg.K + 1), # add one for mask token
                num_tgt_vocab= cfg.K, 
                embedding_dim=768,
                hidden_size=3072,
                nheads=12,
                n_layers=12,
                max_src_len=self.cfg.L,
            )
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
                    if name.startswith('dense_cls') or name == 'cls_token':
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
        self.marnet_module.net.load_state_dict(checkpoint['net'], strict=False)        
        print("loaded", flush=True)

    def eval_mam_quality(self):
        print("testing rank %u" % self.cfg.local_rank, flush=True)
        self.marnet.eval()
        dataloader = self.test_loader
        epoch_metrics = create_epoch_metrics()
        pbar = tqdm(dataloader)
        it = 0
        time_marnet = []
        time_ardm_5 = []
        time_ardm_1 = []
        for x, _ in pbar:
            x = x.cuda(device=self.device_id, non_blocking=True)
            with torch.no_grad():
                logp_mam, log_z = self.marnet_module.eval_ll(x)
                logp = self.marnet_module.est_logp(x, 5, self.cfg.gen_order) # (B,)
                logp_2 = self.marnet_module.est_logp(x, 1, self.cfg.gen_order) # (B,)
            
            selected_ind = preprocess_logp(logp, self.cfg.eval.threshold)
            logp, logp_2, logp_mam = logp[selected_ind], logp_2[selected_ind], logp_mam[selected_ind]
            count = logp.shape[0]

            if logp.shape[0] > 0:
                cmp_results = compare_logp(logp, logp_2, logp_mam, log_z)
                logging.info("iter %u logp: %.4f logp_mam: %.4f "
                    "log_ll_err: %.4f log_ll_err_var: %.4f "
                    "sp: %.4f sp_p: %.4f sp_self: %.4f sp_self_p %.4f "
                    "pr: %.4f pr_p: %.4f pr_self: %.4f pr_self_p %.4f"
                    % (it, cmp_results['logp'], cmp_results['logp_mam'], cmp_results['logp_err'], cmp_results['logp_err_var'],
                        cmp_results['spearman'].correlation, cmp_results['spearman'].pvalue, cmp_results['spearman_self'].correlation, cmp_results['spearman_self'].pvalue,
                        cmp_results['pearson'][0], cmp_results['pearson'][1], cmp_results['pearson_self'][0], cmp_results['pearson_self'][1]))
            update_epoch_metrics(epoch_metrics, cmp_results, count)
            it += 1
            if it==self.cfg.eval.num_batches:
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
                    rand_order = gen_order(self.cfg.batch_size, self.cfg.L, self.device_id, gen_order=self.cfg.gen_order)
                    self.marnet_module.samples = self.marnet.sample(rand_order, self.cfg.batch_size)
            self.marnet.train()

            pbar = tqdm(dataloader)
            pbar.set_description("Epoch {}: Training".format(self.epoch))
            for x, _ in pbar:
                x = x.cuda(device=self.device_id, non_blocking=True)
                x = x.squeeze(dim=1)
                if self.epoch == 5:
                    loss, logp_real, log_z, mb_loss, mb_loss_begin = self.marnet(x)
                else:
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

                if bsz >= 10 // self.cfg.world_size:
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
                states = {
                    'net': self.marnet_module.net.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': self.epoch + 1,
                    'alphabet': self.cfg.alphabet,
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

                val_epoch_metric_tensor = self.eval_ll(self.val_loader)

                if self.master_node:
                    metric_tensor[0] /= metric_tensor[-1]
                    self.writer.add_scalar('train_log_ll_mam', metric_tensor[0], self.epoch)
                    self.writer.add_scalar('val_log_ll_mam', val_epoch_metric_tensor[0], self.epoch)
                    self.writer.add_scalar('val_log_ll', val_epoch_metric_tensor[1], self.epoch)
                    self.writer.add_scalar('val_log_ll_err', val_epoch_metric_tensor[2], self.epoch)
                    self.writer.add_scalar('val_log_ll_err_var', val_epoch_metric_tensor[3], self.epoch)
                    logging.info("Epoch %u out of %u, train log_ll: %.2f,"
                        "val log_ll_mam: %.2f, val log_ll_e: %.2f, val log_ll_s: %.2f, val log_ll_err: %.2f, val log_ll_err_var: %.2f" % (
                        self.epoch, self.cfg.n_epochs, metric_tensor[0], val_epoch_metric_tensor[0], \
                        val_epoch_metric_tensor[1], val_epoch_metric_tensor[2], val_epoch_metric_tensor[3],val_epoch_metric_tensor[4]))
                
            self.epoch += 1

    def eval_test_ll(self):
        _ = self.eval_ll(self.test_loader)
        return 

    def eval_ll(self, dataloader):
        self.marnet.eval()
        mode = 'test'

        epoch_metrics = {
            'log_ll_mam': 0,
            'log_ll_e': 0,
            'log_ll_s': 0,
            'log_ll_err': 0,
            'log_ll_err_var': 0,
            'count': 0,
        }
        pbar = tqdm(dataloader)
        pbar.set_description("Testing calculating likelihood")
        it = 0
        for x, _ in pbar:
            x = x.cuda(device=self.device_id, non_blocking=True)
            with torch.no_grad():
                rand_order = torch.argsort(torch.rand_like(x), dim=-1).to(x.device)
                logp_mam, log_z = self.marnet_module.eval_ll(x) 
            with torch.no_grad():
                logp = self.marnet_module.est_logp(x, 5, self.cfg.gen_order) # (B,)
                logp_2 = self.marnet_module.est_logp(x, 1, self.cfg.gen_order)
            logp_err = (logp - logp_mam + log_z).abs().mean()
            logp_err_var = (logp - logp_mam + log_z).var()
            logp_mam = logp_mam.mean()
            logp = logp.mean()
            logp_2 = logp_2.mean()
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix({"log_ll": f"{logp:.2f}", "log_ll_mam": f"{logp_mam:.2f}",\
                    "log_ll_err": f"{logp_err:.2f}", "log_ll_err_var": f"{logp_err_var:.2f}"})
            it += 1
            if it==self.cfg.eval.num_batches:
                break

            count = x.shape[0]
            epoch_metrics['log_ll_mam'] += logp_mam * count
            epoch_metrics['log_ll_e'] += logp * count
            epoch_metrics['log_ll_s'] += logp_2 * count
            epoch_metrics['log_ll_err'] += logp_err * count
            epoch_metrics['log_ll_err_var'] += logp_err_var * count
            epoch_metrics['count'] += count

        with torch.no_grad():
            metric_tensor = torch.tensor( [epoch_metrics['log_ll_mam'], epoch_metrics['log_ll_e'],\
                epoch_metrics['log_ll_s'], epoch_metrics['log_ll_err'], epoch_metrics['log_ll_err_var'], epoch_metrics['count'] ] )
            if self.distributed:
                torch.distributed.reduce(metric_tensor, dst=0)

            if self.master_node:
                metric_tensor[0] /= metric_tensor[-1]
                metric_tensor[1] /= metric_tensor[-1]
                metric_tensor[2] /= metric_tensor[-1]
                metric_tensor[3] /= metric_tensor[-1]
                metric_tensor[4] /= metric_tensor[-1]
                logging.info("%s count %u log_ll_mam: %.4f log_ll_e: %.4f log_ll_s: %.4f log_ll_err: %.4f log_ll_err_var %.4f" % (
                    mode, metric_tensor[-1], metric_tensor[0], metric_tensor[1], metric_tensor[2], metric_tensor[3], metric_tensor[4]))

        return metric_tensor

    def generate(self):
        self.marnet.eval()
        mode = 'generate'
        num_samples = self.cfg.gen_num_samples
        iters = num_samples // self.cfg.generate_batch_size
        # open a file to write generated samples
        os.makedirs(self.cfg.gen_dir, exist_ok=True)
        with open(os.path.join(self.cfg.gen_dir, 'generated_samples_smiles.txt'), 'w') as f:
            pbar = tqdm(range(iters))
            for i in pbar:
                with torch.no_grad():
                    if self.cfg.conditional:
                        # convert from smiles to indices
                        x_cond = string_to_int(self.cfg.string_example, self.cfg.string_type, self.cfg.L, self.cfg.alphabet)
                        x_cond = torch.tensor(x_cond, dtype=torch.long, device=self.device_id).unsqueeze(0) + 1 # add 1 to include unknown as 0                       
                        x_cond[:, 0:7] = BIT_UNKNOWN_VAL                       
                        rand_order_gen = gen_order(
                            self.cfg.generate_batch_size, 7, self.device_id, gen_order=self.cfg.gen_order
                        )
                        x_gen = self.marnet_module.cond_sample(x_cond, rand_order_gen, self.cfg.generate_batch_size)
                    else:
                        rand_order_gen = gen_order(
                            self.cfg.generate_batch_size, self.cfg.L, self.device_id, gen_order=self.cfg.gen_order
                        )
                        x_gen = self.marnet_module.sample(rand_order_gen, self.cfg.generate_batch_size)
                    logp_mam, log_z = self.marnet_module.eval_ll(x_gen)
                    logp = self.marnet_module.est_logp(x_gen, self.cfg.eval.mc_ll, self.cfg.gen_order) # (B,)
                    logging.info("logp_mam: %.4f logp: %.4f" % ((logp_mam - log_z).mean(), logp.mean()))
                x_gen = (x_gen-1.0).int().cpu().numpy().tolist()
                if self.cfg.string_type == 'SELFIES':
                    x_gen_selfies = multiple_indices_to_string(x_gen, self.cfg.alphabet)
                    x_gen_smiles = list(map(sf.decoder, x_gen_selfies))
                elif self.cfg.string_type == 'SMILES':
                    x_gen_smiles = multiple_indices_to_string(x_gen, self.cfg.alphabet)
                for j in range(len(x_gen_smiles)):
                    # write the string to file with line break
                    f.write(x_gen_smiles[j] + '\n')
                pbar.set_postfix({"generated samples": f"{(i+1)*self.cfg.generate_batch_size}"})
        