import os
import logging
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import pickle
import selfies as sf

from models.nn import MLPResSingle
from models.arm_eb import ARMModel
from utils.ising_utils import prepare_ising_data, LatticeIsingModel
from utils.data_utils import load_dataset, load_ising_gt_samples
from utils.mar_utils_mol import gen_order
from utils.mol_utils import multiple_indices_to_string
from utils.eval_mol import MolEvalModel

class ARMRunner(object):
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
            self.train_loader, self.val_loader, self.test_loader = prepare_ising_data(self.score_model, cfg, distributed=self.distributed)
            if cfg.eval_reverse_kl:
                # load ground truth samples generated from MCMC for evalution
                # note: ground truthe samples are not avaialable during energy-based training
                ising_samples = load_ising_gt_samples(cfg, self.score_model)
                self.ising_samples = ising_samples.to(self.device_id)
        elif self.cfg.dataset == 'molecule':
            self.train_loader, self.val_loader, self.test_loader = load_dataset(cfg, distributed=self.distributed)
            self.score_model = MolEvalModel(cfg.alphabet, cfg.string_type, cfg.metric_name, cfg.target_value, self.cfg.tau)
        else:
            raise ValueError("Unknown dataset {}".format(cfg.dataset))
        self.epoch = 0

        if cfg.arch == 'mlp':
            self.nn = MLPResSingle(cfg.nn.hidden_dim, cfg.K, cfg.L, cfg.nn.n_layers, cfg.nn.res)
        else:
            raise ValueError("Unknown model {}".format(cfg.arch))
        self.nn.to(self.device_id)
        logging.info(self.nn)

        init_samples = next(iter(self.train_loader))[0].float()
        self.marnet = ARMModel(self.nn, self.score_model, init_samples, cfg)
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

        self.save_every = 200
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
                'loss': 0,
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
                loss, logf_t, logp_x = self.marnet(x, y)
                loss.backward()

                count = x.shape[0]
                epoch_metrics['loss'] += loss * count
                epoch_metrics['count'] += count

                bsz += x.shape[0]
                accum += x.shape[0]

                if bsz >= 32 // self.cfg.world_size:
                    if self.clip_grad > 0:
                        total_norm = torch.nn.utils.clip_grad_norm_(self.marnet.parameters(), self.clip_grad)
                        if total_norm > 1e4:
                            print("grad_norm is {}".format(total_norm))
                    self.optimizer.step()
                    # self.scheduler.step()
                    self.optimizer.zero_grad()
                    bsz = 0

                if accum >= 5120 // self.cfg.world_size:
                    if self.master_node:
                        logging.info("Iter %u out of %u, loss: %.2f, logp: %.2f, logf_t: %.2f"
                            % (it, len(dataloader), loss, logp_x.mean().item(), logf_t.mean().item()))
                        self.writer.add_scalar('Obj/obj', loss, it + 1)
                        self.writer.add_scalar('Obj/logZ', self.marnet_module.LogZ, it + 1)
                        self.writer.add_scalar('Obj/f_t_mean', logf_t.mean(), it + 1)
                        self.writer.add_scalar('Obj/f_t_std', logf_t.std(), it + 1)
                        accum = 0
                        accumll = 0.0

                pbar.set_postfix({"loss": f"{loss.item():.2e}", "logZ": f"{self.marnet_module.LogZ.item():.2f}",\
                    "f_t_mean": f"{logf_t.mean().item():.2f}", "f_t_std": f"{logf_t.std().item():.2f}"})
                it += 1

            if self.epoch % self.eval_every == 0:
                with torch.no_grad():
                    metric_tensor = torch.tensor([  epoch_metrics['loss'], epoch_metrics['count'] ] )
                    if self.distributed:
                        torch.distributed.reduce(metric_tensor, dst=0)

                if self.master_node:
                    kl_div_est = self.eval_kl()
                    if best_kl_div is None:
                        best_kl_div = kl_div_est
                    if self.cfg.save_model:
                        if kl_div_est <= best_kl_div:
                            best_kl_div = kl_div_est
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
                    metric_tensor[0] /= metric_tensor[-1]
                    self.writer.add_scalar('Loss/train_loss', metric_tensor[0], self.epoch)
                    self.writer.add_scalar('Loss/test_loss', test_epoch_metric_tensor[0], self.epoch)
                    self.writer.add_scalar('Loss/test_mb_diff', test_epoch_metric_tensor[1], self.epoch)
                    self.writer.add_scalar('Loss/test_mb_diff_var', test_epoch_metric_tensor[2], self.epoch)
                    logging.info("Epoch %u out of %u, test mb_loss: %.2f" % (
                        self.epoch, self.cfg.n_epochs, test_epoch_metric_tensor[0]))
            
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
        
        self.writer.add_scalar('Loss/f_t_mean', samples_logf_true.mean(), self.epoch)
        self.writer.add_scalar('Loss/f_t_std', samples_logf_true.std(), self.epoch)
        self.writer.add_scalar('Loss/test_KL_div', kl_div, self.epoch)
        logging.info("test KL div: %.2f, f_t_mean: %.2f, f_t_std: %.2f" % (
            kl_div, samples_logf_true.mean(), samples_logf_true.std()))
        return kl_div

    def test(self):
        self.marnet.eval()
        dataloader = self.test_loader
        mode = 'test'

        epoch_metrics = {
            'loss': 0,
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
                loss, logf_t, logp_x= self.marnet(x, y, training=False)
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix({"loss": f"{loss:.2f}",\
                    "f_t_mean": f"{logf_t.mean().item():.2f}", "f_t_std": f"{logf_t.std().item():.2f}",\
                })
            it += 1
            if it==self.cfg.eval.num_batches:
                break

            count = x.shape[0]
            epoch_metrics['loss'] += loss.item() * count
            epoch_metrics['mb_diff'] += torch.abs(logp_x + self.marnet_module.LogZ - logf_t).mean().item() * count
            epoch_metrics['mb_diff_var'] += torch.var(logp_x + self.marnet_module.LogZ - logf_t).item() * count
            epoch_metrics['count'] += count

        with torch.no_grad():
            metric_tensor = torch.tensor( [epoch_metrics['loss'], epoch_metrics['mb_diff'],\
                epoch_metrics['mb_diff_var'], epoch_metrics['count'] ] )
            if self.distributed:
                torch.distributed.reduce(metric_tensor, dst=0)

            if self.master_node:
                for i in range(metric_tensor.shape[0] - 1):
                    metric_tensor[i] /= metric_tensor[-1]
                logging.info("%s count, %u loss: %.4f, logp: %.2f, logf_t: %.2f," % (
                    mode, metric_tensor[-1], metric_tensor[0], logp_x.mean().item(), logf_t.mean().item()))

        return metric_tensor

    def generate(self):
        self.marnet.eval()
        rand_order_gen = gen_order(self.cfg.gen_num_samples, self.cfg.L, self.device_id, gen_order=self.cfg.gen_order)
        with torch.no_grad():
            samples = self.marnet.sample(rand_order_gen, self.cfg.gen_num_samples)
            samples = samples.cuda(device=self.device_id, non_blocking=True)
            with torch.no_grad():
                samples_logp = self.marnet.est_logp(samples, 1, self.cfg.gen_order)
            samples_logf_true = self.marnet.score_model(samples - 1.0)
            kl_div = (samples_logp - samples_logf_true).mean()
        logging.info("test KL div: %.2f, f_t_mean:%.2f, f_t_std: %.2f, logp_est_mean: %.2f, logp_est_std: %.2f" % (
            kl_div, samples_logf_true.mean(), samples_logf_true.std(), samples_logp.mean(), samples_logp.std()))
        data_scores = self.score_model.get_scores(samples - 1.0)
        save_path = os.path.join(self.cfg.log_dir, 'samples.png')
        self.score_model.plot_scores(data_scores.cpu().numpy(), save_path)
        with open("{}/model_samples.pkl".format(self.cfg.log_dir), 'wb') as f:
            pickle.dump(samples.cpu(), f)
        with open("{}/model_samples_scores.pkl".format(self.cfg.log_dir), 'wb') as f:
            pickle.dump(data_scores.cpu(), f)
        if self.cfg.eval_reverse_kl:
            samples = self.ising_samples + 1.0
            with torch.no_grad():
                samples_logp = self.marnet.est_logp(samples, 1, self.cfg.gen_order)
                samples_logf_true = self.marnet.score_model(samples - 1.0) # convert back to [0:K-1] first
            nll = - samples_logp.mean()
            logging.info("test nll: %.2f, f_t_mean:%.2f, f_t_std: %.2f" % (nll, samples_logf_true.mean(), samples_logf_true.std()))
        samples = (samples-1.0).int().cpu().numpy().tolist()
        # record down the selfies and their corresponding smiles as well
        if self.cfg.string_type == 'SELFIES':
            x_gen_selfies = multiple_indices_to_string(samples, self.cfg.alphabet)
            with open(os.path.join(self.cfg.log_dir, 'generated_samples_selfies.txt'), 'w') as f:
                for j in range(len(x_gen_selfies)):
                    f.write(x_gen_selfies[j] + '\n')
            x_gen_smiles = list(map(sf.decoder, x_gen_selfies))
        with open(os.path.join(self.cfg.log_dir, 'generated_samples_smiles.txt'), 'w') as f:
            for j in range(len(x_gen_smiles)):
                f.write(x_gen_smiles[j] + '\n')
