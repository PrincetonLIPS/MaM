import os
import logging
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import pickle

import selfies as sf
from models.nn import MLPResDual
from models.transformer import TransformerNet
from models.mam_eb import MAM
from utils.ising_utils import prepare_ising_data, LatticeIsingModel
from utils.data_utils import load_dataset, load_ising_gt_samples
from utils.mar_utils_mol import gen_order
from utils.mol_utils import multiple_indices_to_string, string_to_int, indices_to_string
from utils.constants import BIT_UNKNOWN_VAL
from utils.eval_mol import MolEvalModel

class Runner(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.writer = SummaryWriter(os.getcwd())
        self.device_id = 'cuda:{}'.format(cfg.local_rank)
        self.master_node = (self.cfg.local_rank == 0)
        self.distributed = (self.cfg.world_size > 1)
        self.fig1 = plt.figure()
        self.fig2 = plt.figure()
        if self.cfg.dataset == 'ising':
            self.score_model = LatticeIsingModel(cfg.ising_model.dim, cfg.ising_model.sigma, cfg.ising_model.bias)
            if cfg.eval_reverse_kl:
                # load ground truth samples generated from MCMC
                # note: ground truthe samples are not avaialable during energy-based training
                ising_samples = load_ising_gt_samples(cfg, self.score_model)
                self.ising_samples = ising_samples.to(self.device_id)
            # dataloader for training, different from the MLE setting, x contains configurations, y contains the energies
            self.train_loader, self.val_loader, self.test_loader = prepare_ising_data(self.score_model, cfg, distributed=self.distributed)
        elif self.cfg.dataset == 'molecule':
            self.train_loader, self.val_loader, self.test_loader = load_dataset(cfg, distributed=self.distributed)
            self.score_model = MolEvalModel(cfg.alphabet, cfg.string_type, cfg.metric_name, cfg.target_value, self.cfg.tau)
        else:
            raise ValueError("Unknown dataset {}".format(cfg.dataset))
        self.epoch = 0

        if cfg.arch == 'mlp_dual':
            self.nn = MLPResDual(cfg.nn.hidden_dim, cfg.K, cfg.L, cfg.nn.n_layers, cfg.nn.res)
        elif cfg.arch == 'transformer':
            self.nn = TransformerNet(
                num_src_vocab=(cfg.K + 1), # add one for mask token
                num_tgt_vocab= cfg.K, 
                embedding_dim=768,
                hidden_size=3072,
                nheads=12,
                n_layers=12,
                max_src_len=self.cfg.L,
                is_cls_token=self.cfg.nn.is_cls_token,
            )
        else:
            raise ValueError("Unknown model {}".format(cfg.arch))
        self.nn.to(self.device_id)
        logging.info(self.nn)

        init_samples = torch.randint(low=1, high=cfg.K+1, size=(cfg.batch_size, cfg.L)).float()
        self.marnet = MAM(self.nn, self.score_model, init_samples, cfg)
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

        if self.cfg.load_model: 
            if self.cfg.loadpath is not None:
                self.load(self.cfg.loadpath)
            else:
                raise ValueError("load_model is True but loadpath is None")

        self.save_every = self.cfg.save_every
        self.eval_every = 5

    def load(self, path):
        map_location = {"cuda:0": self.device_id}
        checkpoint = torch.load(path, map_location=map_location)
        self.marnet_module.load_state_dict(checkpoint['net'], strict=False)
        # self.optimizer.load_state_dict(checkpoint['optimizer'])
        print("loaded", flush=True)
    
    def train(self):
        print("training rank %u" % self.cfg.local_rank, flush=True)
        self.marnet.train()
        dataloader = self.train_loader

        it = 0
        best_kl_div = None
        while self.epoch < self.cfg.n_epochs:
            epoch_metrics = {
                'mb_loss': 0,
                'mb_loss_begin': 0,
                'mb_loss_total': 0,
                'count': 0,
            }

            bsz = 0
            accum, accumll = 0, 0.0

            if self.cfg.objective == 'KL':
                rand_order = gen_order(self.cfg.batch_size, self.cfg.L, self.device_id, gen_order=self.cfg.gen_order)
                with torch.no_grad():
                    self.marnet_module.samples = self.marnet.sample(rand_order, self.cfg.batch_size)

            self.marnet.train()

            pbar = tqdm(dataloader)
            pbar.set_description("Epoch {}: Training".format(self.epoch))
            for x, _ in pbar:
                x = x.cuda(device=self.device_id, non_blocking=True)
                y = self.score_model(x-1.0)
                x = x.squeeze(dim=1)
                loss, mb_loss, mb_loss_begin, mb_loss_total, logf_t, logp_x = self.marnet(x, y)
                loss.backward()

                count = x.shape[0]
                epoch_metrics['mb_loss'] += mb_loss.item() * count
                epoch_metrics['mb_loss_begin'] += mb_loss_begin.item() * count
                epoch_metrics['mb_loss_total'] += mb_loss_total.item() * count
                epoch_metrics['count'] += count

                bsz += x.shape[0]
                accum += x.shape[0]

                if bsz >= 32 // self.cfg.world_size:
                    if self.clip_grad > 0:
                        total_norm = torch.nn.utils.clip_grad_norm_(self.marnet.parameters(), self.clip_grad)
                        if total_norm > 1e4:
                            print("grad_norm is {}".format(total_norm))
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    bsz = 0

                if accum >= 5120 // self.cfg.world_size:
                    if self.master_node:
                        logging.info("Iter %u out of %u, mb-loss-begin: %.2f, mb-loss: %.2f, mb-loss-end: %.2f, loss: %.2f, logz: %.2f"
                            % (it, len(dataloader), mb_loss_begin.item(), mb_loss.item(), mb_loss_total.item(), loss.item(), self.marnet_module.LogZ.item()))
                        logging.info("Iter %u out of %u, f_t_mean: %.2f, f_t_std: %.2f, alpha: %.2e"
                            % (it, len(dataloader), logf_t.mean().item(), logf_t.std().item(), self.cfg.alpha))
                        self.writer.add_scalar('Obj/mb_loss_begin', mb_loss_begin.item(), it + 1)
                        self.writer.add_scalar('Obj/mb_loss_total', mb_loss_total.item(), it + 1)
                        self.writer.add_scalar('Obj/mb_loss', mb_loss.item(), it + 1)
                        self.writer.add_scalar('Obj/loss', loss.item(), it + 1)
                        self.writer.add_scalar('Obj/logZ', self.marnet_module.LogZ.item(), it + 1)
                        self.writer.add_scalar('Obj/f_t_mean', logf_t.mean().item(), it + 1)
                        self.writer.add_scalar('Obj/f_t_std', logf_t.std().item(), it + 1)
                        accum = 0
                        accumll = 0.0

                pbar.set_postfix({"mb_loss_begin": f"{mb_loss_begin.item():.2f}",\
                    "mb": f"{mb_loss.item():.2e}", "mb_loss_total": f"{mb_loss_total.item():.2f}",\
                    "loss": f"{loss.item():.2e}", "logZ": f"{self.marnet_module.LogZ.item():.2f}",\
                    "f_t_mean": f"{logf_t.mean().item():.2f}", "f_t_std": f"{logf_t.std().item():.2f}"})
                it += 1

            if self.epoch % self.eval_every == 0:
                with torch.no_grad():
                    metric_tensor = torch.tensor([  epoch_metrics['mb_loss'], epoch_metrics['mb_loss_begin'],\
                        epoch_metrics['mb_loss_total'], epoch_metrics['count'] ] )
                    if self.distributed:
                        torch.distributed.reduce(metric_tensor, dst=0)

                if self.master_node:
                    if self.cfg.save_model:
                        states = {
                            'net': self.marnet_module.state_dict(),
                            # 'optimizer': self.optimizer.state_dict(),
                            'epoch': self.epoch + 1,
                            'L': self.cfg.L,
                            'K': self.cfg.K,
                        }
                        torch.save(states, os.path.join(self.cfg.model_dir, 'checkpoint.pth'))
                test_epoch_metric_tensor = self.test()

                if self.master_node:
                    for i in range(metric_tensor.shape[0]-1):
                        metric_tensor[i] /= metric_tensor[-1]
                    self.writer.add_scalar('Loss/train_mb_loss', metric_tensor[0], self.epoch)
                    self.writer.add_scalar('Loss/train_mb_loss_begin', metric_tensor[1], self.epoch)
                    self.writer.add_scalar('Loss/train_mb_loss_total', metric_tensor[2], self.epoch)
                    self.writer.add_scalar('Loss/test_mb_loss', test_epoch_metric_tensor[0], self.epoch)
                    self.writer.add_scalar('Loss/test_mb_loss_begin', test_epoch_metric_tensor[1], self.epoch)
                    self.writer.add_scalar('Loss/test_mb_loss_total', test_epoch_metric_tensor[2], self.epoch)
                    self.writer.add_scalar('Loss/test_mb_diff', test_epoch_metric_tensor[3], self.epoch)
                    self.writer.add_scalar('Loss/test_mb_diff_var', test_epoch_metric_tensor[4], self.epoch)
                    logging.info("Epoch %u out of %u, test mb_loss: %.2f, test mb_loss_begin: %.2f, test mb_loss_total: %.2f" % (
                        self.epoch, self.cfg.n_epochs, test_epoch_metric_tensor[0], test_epoch_metric_tensor[1], test_epoch_metric_tensor[2]))
            self.epoch += 1

    def eval_kl(self):
        self.marnet.eval()
        if self.cfg.eval_reverse_kl:
            samples = self.ising_samples + 1.0
        else:
            rand_order_gen = gen_order(self.cfg.batch_size, self.cfg.L, self.device_id, gen_order=self.cfg.gen_order)
            with torch.no_grad():
                samples = self.marnet.sample(rand_order_gen, self.cfg.batch_size)
        with torch.no_grad():
            samples_to_plot = samples[:100,]
            for i in range(10):
                samples_logp_marg, _ = self.marnet.eval_ll(samples)
                samples_logp = self.marnet.est_logp(samples, 1, self.cfg.gen_order)
            samples_logf_true = self.marnet.score_model(samples - 1.0) # convert back to [0:K-1] first
        if self.cfg.eval_reverse_kl:
            kl_div = - samples_logp.mean()
        else:
            kl_div = (samples_logp - samples_logf_true).mean()
        if self.cfg.plot_samples and self.epoch % self.cfg.plot_every == 0:
            plt.figure(self.fig1.number)
            sns.kdeplot(samples_logf_true.cpu().numpy(), fill=True)
            self.fig1.savefig(os.path.join(self.cfg.log_dir, 'samples_epoch{}.png'.format(self.epoch)))
            plt.figure(self.fig2.number)
            data_scores = self.score_model.get_scores(samples - 1.0)
            sns.kdeplot(data_scores.cpu().numpy(), fill=True)
            self.fig2.savefig(os.path.join(self.cfg.log_dir, 'data_scores_epoch{}.png'.format(self.epoch)))
            if self.cfg.dataset == 'ising':
                file_name = os.path.join(self.cfg.log_dir, 'samples_vis_epoch{}.png'.format(self.epoch))
                torchvision.utils.save_image(
                    samples_to_plot.float().reshape(samples_to_plot.shape[0], 1, self.cfg.ising_model.dim, self.cfg.ising_model.dim), file_name, normalize=True, nrow=int(samples_to_plot.shape[0] ** .5))
            with open("{}/model_samples.pkl".format(self.cfg.log_dir), 'wb') as f:
                pickle.dump(samples.cpu(), f)
            with open("{}/model_samples_scores.pkl".format(self.cfg.log_dir), 'wb') as f:
                pickle.dump(data_scores.cpu(), f)
        
        self.writer.add_scalar('Loss/p_mean', samples_logp.mean().item(), self.epoch)
        self.writer.add_scalar('Loss/p_std', samples_logp.std().item(), self.epoch)
        self.writer.add_scalar('Loss/f_t_mean', samples_logf_true.mean().item(), self.epoch)
        self.writer.add_scalar('Loss/f_t_std', samples_logf_true.std().item(), self.epoch)
        self.writer.add_scalar('Loss/test_KL_div', kl_div, self.epoch)
        if self.cfg.eval_reverse_kl:
            logging.info("test nll: %.2f, p_mean:%.2f, p_std: %.2f, f_t_mean: %.2f, f_t_std: %.2f" % (
                kl_div, samples_logp.mean(), samples_logp.std(), samples_logf_true.mean(), samples_logf_true.std()))
        else:
            logging.info("test KL_div %.2f, p_mean:%.2f, p_std: %.2f, f_t_mean: %.2f, f_t_std: %.2f" % (
                kl_div, samples_logp.mean(), samples_logp.std(), samples_logf_true.mean(), samples_logf_true.std()))
        return kl_div

    def test(self):
        self.marnet.eval()
        dataloader = self.test_loader
        mode = 'test'

        epoch_metrics = {
            'mb_loss': 0,
            'mb_loss_begin': 0,
            'mb_loss_total': 0,
            'mb_diff': 0,
            'mb_diff_var': 0,
            'count': 0,
        }
        
        pbar = tqdm(dataloader)
        pbar.set_description("Testing calculating likelihood")
        it = 0
        for x, y in pbar:
            x = x.cuda(device=self.device_id, non_blocking=True)
            x = x.squeeze(dim=1)
            y = self.score_model(x-1.0)
            with torch.no_grad():
                loss, mb_loss, mb_loss_begin, mb_loss_total, logf_t, logp_x = self.marnet(x, y)
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix({
                    "mb_loss": f"{mb_loss:.2f}", "mb_loss_begin": f"{mb_loss_begin:.2f}",\
                    "mb_loss_total": f"{mb_loss_total:.2f}", "loss": f"{loss:.2f}",\
                    "f_t_mean": f"{logf_t.mean().item():.2f}", "f_t_std": f"{logf_t.std().item():.2f}"   
                })
            it += 1
            if it==self.cfg.eval.num_batches:
                break

            count = x.shape[0]
            epoch_metrics['mb_loss'] += mb_loss.item() * count
            epoch_metrics['mb_loss_begin'] += mb_loss_begin.item() * count
            epoch_metrics['mb_loss_total'] += mb_loss_total.item() * count
            epoch_metrics['mb_diff'] += torch.abs(logp_x - logf_t).mean().item() * count
            epoch_metrics['mb_diff_var'] += torch.var(logp_x - logf_t).item() * count
            epoch_metrics['count'] += count

        with torch.no_grad():
            metric_tensor = torch.tensor( [epoch_metrics['mb_loss'], epoch_metrics['mb_loss_begin'],\
                epoch_metrics['mb_loss_total'], epoch_metrics['mb_diff'], epoch_metrics['mb_diff_var'],\
                epoch_metrics['count'] ] )
            if self.distributed:
                torch.distributed.reduce(metric_tensor, dst=0)

            if self.master_node:
                for i in range(metric_tensor.shape[0] - 1):
                    metric_tensor[i] /= metric_tensor[-1]
                logging.info("%s count, %u mb_loss: %.4f, mb_loss_begin: %.4f, mb_loss_total: %.4f" % (
                    mode, metric_tensor[-1], metric_tensor[0], metric_tensor[1], metric_tensor[2]))

        return metric_tensor
    
    def generate(self):
        self.marnet.eval()
        rand_order_gen = gen_order(self.cfg.gen_num_samples, self.cfg.L, self.device_id, gen_order=self.cfg.gen_order)
        with torch.no_grad():
            samples = self.marnet.sample(rand_order_gen, self.cfg.gen_num_samples)
            samples = samples.cuda(device=self.device_id, non_blocking=True)
            with torch.no_grad():
                logp_x, _ = self.marnet.net(samples) # (B)
                samples_logp = self.marnet.est_logp(samples, 1, self.cfg.gen_order)
            samples_logf_true = self.marnet.score_model(samples - 1.0)
            kl_div = (samples_logp - samples_logf_true).mean()
        logging.info("test KL div: %.2f, f_t_mean:%.2f, f_t_std: %.2f, logp_est_mean: %.2f, logp_est_std: %.2f" % (
            kl_div, samples_logf_true.mean(), samples_logf_true.std(), samples_logp.mean(), samples_logp.std()))
        logging.info("logp_x_mean: %.2f, logp_x_std: %.2f" % (logp_x.mean().item(), logp_x.std().item()))
        data_scores = self.score_model.get_scores(samples - 1.0)
        save_path = os.path.join(self.cfg.log_dir, 'samples.png')
        self.score_model.plot_scores(data_scores.cpu().numpy(), save_path)
        with open("{}/model_samples.pkl".format(self.cfg.log_dir), 'wb') as f:
            pickle.dump(samples.cpu(), f)
        with open("{}/model_samples_scores.pkl".format(self.cfg.log_dir), 'wb') as f:
            pickle.dump(data_scores.cpu(), f)
        if self.cfg.dataset =='ising' and self.cfg.eval_reverse_kl:
            samples = self.ising_samples + 1.0 # convert from [0:K-1] to [1:K] for NN model
            with torch.no_grad():
                samples_logp = self.marnet.est_logp(samples, 1, self.cfg.gen_order)
                samples_scores = self.score_model.get_scores(samples - 1.0)             
                samples_logf_true = self.marnet.score_model(samples - 1.0) # convert back to [0:K-1] for score model
            save_path = os.path.join(self.cfg.log_dir, 'ising_samples_scores.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(samples_scores.cpu(), f)
            nll = - samples_logp.mean()
            logging.info("test nll: %.2f, f_t_mean:%.2f, f_t_std: %.2f" % (nll, samples_logf_true.mean(), samples_logf_true.std()))
        samples = (samples-1.0).int().cpu().numpy().tolist()
        # record down the selfies and their corresponding smiles as well
        x_gen_selfies = multiple_indices_to_string(samples, self.cfg.alphabet)
        with open(os.path.join(self.cfg.log_dir, 'generated_samples_selfies.txt'), 'w') as f:
            for j in range(len(x_gen_selfies)):
                f.write(x_gen_selfies[j] + '\n')
        x_gen_smiles = list(map(sf.decoder, x_gen_selfies))
        with open(os.path.join(self.cfg.log_dir, 'generated_samples_smiles.txt'), 'w') as f:
            for j in range(len(x_gen_smiles)):
                f.write(x_gen_smiles[j] + '\n')

    def cond_gen_mols(self):
        # the model generates SELFIES strings
        self.marnet.eval()
        num_samples = self.cfg.gen_num_samples
        iters = num_samples // self.cfg.generate_batch_size
        # open a file to write generated samples
        os.makedirs(self.cfg.gen_dir, exist_ok=True)
        pbar = tqdm(range(iters))
        for i in pbar:
            with torch.no_grad():
                # conditional generation from a substring
                # convert from smiles to indices
                string_selfies = sf.encoder(self.cfg.string_example)
                print("----Given base example molecule string------:\n" + string_selfies)
                string_smiles = sf.decoder(string_selfies)
                x_cond = string_to_int(string_selfies, self.cfg.string_type, self.cfg.L, self.cfg.alphabet)
                x_cond = torch.tensor(x_cond, dtype=torch.long, device=self.device_id).unsqueeze(0) + 1 # add 1 to include unknown as 0                       
                # mask the unknown bits at specified locations
                x_cond[:, self.cfg.start:self.cfg.end] = BIT_UNKNOWN_VAL
                # set the unknown bits to ? in string_selfies
                string_smiles_cond = indices_to_string((x_cond[0]-1).cpu().numpy().tolist(), self.cfg.alphabet)
                print("----Generate from given molecule string substructure------:\n" + string_smiles_cond)
                rand_order_gen = gen_order(
                    self.cfg.generate_batch_size, self.cfg.end-self.cfg.start, self.device_id, gen_order=self.cfg.gen_order
                )
                x_gen = self.marnet_module.cond_sample(x_cond, rand_order_gen, self.cfg.generate_batch_size)
                
                logp_mam, log_z = self.marnet_module.eval_ll(x_gen)
                logp = self.marnet_module.est_logp(x_gen, self.cfg.eval.mc_ll, self.cfg.gen_order) # (B,)
                samples_logf_true = self.marnet.score_model(x_gen - 1.0)
                kl_div = (logp - samples_logf_true).mean()
                logging.info("logp_mam: %.4f logp mean: %.4f std: %.4f" % ((logp_mam - log_z).mean(), logp.mean(), logp.std()))
                logging.info("test KL div: %.4f, f_t_mean:%.4f, f_t_std: %.4f" % (
                    kl_div, samples_logf_true.mean(), samples_logf_true.std()))
            x_gen = (x_gen-1.0).int().cpu().numpy().tolist()
            x_gen_selfies = multiple_indices_to_string(x_gen, self.cfg.alphabet)
            with open(os.path.join(self.cfg.log_dir, 'generated_samples_selfies.txt'), 'w') as f:
                f.write(string_smiles_cond + '\n') # record down the conditional substring
                for j in range(len(x_gen_selfies)):
                    f.write(x_gen_selfies[j] + '\n')
            x_gen_smiles = list(map(sf.decoder, x_gen_selfies))
            # record down the corresponding smiles as well
            with open(os.path.join(self.cfg.log_dir, 'generated_samples_smiles.txt'), 'w') as f:
                for j in range(len(x_gen_smiles)):
                    f.write(x_gen_smiles[j] + '\n')
            pbar.set_postfix({"generated samples": f"{(i+1)*self.cfg.generate_batch_size}"})