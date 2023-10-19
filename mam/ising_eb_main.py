import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from eb_runner import Runner
from eb_arm_runner import ARMRunner

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

@hydra.main(config_path="config", config_name="config_ising_eb")
def main(cfg: DictConfig):
    # get data path with absolute path
    cfg.data_dir = os.path.abspath(os.path.join(hydra.utils.get_original_cwd(), cfg.data_dir))
    cfg.L = cfg.ising_model.dim ** 2
    # relative to hydra path
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    logging.info(os.getcwd())
    logging.info(OmegaConf.to_yaml(cfg))

    if cfg.model == 'ARM':
        runner = ARMRunner(cfg)
    elif cfg.model == 'MAM':
        runner = Runner(cfg)
    else:
        raise NotImplementedError
    if cfg.mode == 'train':
        runner.train()
    elif cfg.mode == 'generate':
        runner.generate()
    elif cfg.mode == 'evaluate':
        runner.eval_kl()
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()
    
