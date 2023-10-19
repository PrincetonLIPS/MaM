import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from text_runner import Runner

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

@hydra.main(config_path="config", config_name="config_text")
def main(cfg: DictConfig):
    # absolute path
    cfg.data_dir = os.path.abspath(os.path.join(hydra.utils.get_original_cwd(), cfg.data_dir))
    # relative to hydra path
    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    logging.info(os.getcwd())
    logging.info(OmegaConf.to_yaml(cfg))

    runner = Runner(cfg)

    if cfg.mode == 'train':
        runner.train()
    elif cfg.mode == 'eval_mam_quality_test':
        runner.eval_mam_quality_test()
    elif cfg.mode == 'eval_test_ll':
        runner.eval_test_ll()
    elif cfg.mode == 'eval_mam_quality':
        runner.eval_mam_quality()
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()
    
