from typing import *
import torch 
import torch.backends.cudnn as cudnn

import os
import random
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig,OmegaConf
import numpy as np
import copy
import wandb
import warnings

from utils import helper
from trainer import Trainer

warnings.filterwarnings("ignore", category=UserWarning, module="torch")


def set_seeds(seed:Union[float, int]=None):
    if seed is not None:
        seed = int(seed)
        # setup cuda
        cudnn.enabled = True
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    #torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True # set to false if deterministic
    torch.set_printoptions(precision=10)

@hydra.main(config_path="configs", config_name="config",version_base=None)
def main(cfg: DictConfig):
    
    cudnn.benchmark = True
    torch.set_float32_matmul_precision(cfg.trainer.tf_matmul_precision)
    
    if cfg.trainer.distributed:
        #torch.cuda.set_device(cfg.trainer.local_rank)
        torch.distributed.init_process_group(backend='nccl')#, init_method='env://')
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    else:
        rank = 0
        world_size = 1

    #get overrides and merge.
    hydra_cfg = HydraConfig.get()
    hydra_info = {}    
    if 'id' in hydra_cfg.job:
        print("multi-run: ",hydra_cfg.job.id,hydra_cfg.mode)
        print("ouptut_dir: ",hydra_cfg.runtime.output_dir)
        hydra_info['hydra_id'] = hydra_cfg.job.id
    else:
        print("single-run: ",hydra_cfg.mode)
        print("ouptut_dir: ",hydra_cfg.runtime.output_dir)

    hydra_info['hydra_mode'] = str(hydra_cfg.mode)
    hydra_info['output_dir'] = hydra_cfg.runtime.output_dir

    
    if cfg.trainer.resume or cfg.trainer.eval_only:
        print('Resuming from checkpoint.')
        save_dir = cfg.logging.checkpoint.logdir
        if os.path.exists(save_dir+'/exp_config.yaml'):
            exp_cfg = OmegaConf.load(save_dir+'/exp_config.yaml')
            print("Loaded existing config from: ",save_dir+'/exp_config.yaml') 
            
            OmegaConf.resolve(exp_cfg)
            cfg = copy.deepcopy(exp_cfg)

            overrides = HydraConfig.get().overrides
            overrides = [str(item) for item in overrides.task]
            temp_cfg = OmegaConf.from_dotlist(overrides)
            
            #get new overrides and merge, for training args.
            cfg.trainer = OmegaConf.merge(exp_cfg.trainer,temp_cfg.trainer)
            cfg.logging = OmegaConf.merge(exp_cfg.logging,temp_cfg.logging)

    helper.make_dir(cfg.logging.checkpoint.logdir)    

    if cfg.common.seed == -1:
        cfg.common.seed = random.randint(0,10000)

    if cfg.trainer.num_iters_first is None:
        cfg.trainer.num_iters_first = cfg.trainer.num_iters

    print("Seed: ",cfg.common.seed)
    set_seeds(cfg.common.seed)

    if cfg.trainer.eval_only:
        cfg.logging.wandb.enabled=False

    #log model config. USed to infer trained models. 
    trainer = Trainer(cfg,local_rank=rank,world_size=world_size)

    if cfg.trainer.eval_only:
        trainer.infer()
        return
    trainer.train()


    if cfg.logging.wandb.enabled:
        wandb.finish()




if __name__ == '__main__':
    main()

