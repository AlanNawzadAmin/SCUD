import sys
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
import pytorch_lightning as pl

from evodiff.utils import Tokenizer

from d3pm_sc.ct_sched_cond import ScheduleCondition
from d3pm_sc.ct_sched_cond_sparse_k import ScheduleConditionSparseK
from d3pm_sc.masking_diffusion import MaskingDiffusion
from d3pm_sc.sedd import SEDD
from d3pm_sc.d3pm_classic import D3PMClassic
from d3pm_sc.discrete_sc import DiscreteScheduleCondition

from nets import get_model_setup
from data import get_dataloaders
from ema import EMA

import getpass

import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

@hydra.main(version_base=None, config_path="configs", config_name="basic")
def train(cfg: DictConfig) -> None:
    @rank_zero_only
    def init_wandb():
        wandb_key = "9e61d229e6b9dbfef3e2199c7e093a75bfe53135" if 'nvg' \
            in getpass.getuser() else "6a47f093d2a55e4f4e85b33767423f2db66355b8"
        wandb.login()
        wandb.init()
    init_wandb()
    ##### Load data
    pl.seed_everything(cfg.model.seed, workers=True)
    print("Getting dataloaders.")
    train_dataloader, test_dataloader = get_dataloaders(cfg)
    tokenizer = train_dataloader.tokenizer if hasattr(train_dataloader, "tokenizer") else None

    ##### Setup x0_model
    print("Setting up model.")
    x0_model_class, nn_params = get_model_setup(cfg, tokenizer) 
    
    print(cfg)
    
    ##### Pick model
    model_name_dict = {"ScheduleCondition":ScheduleCondition,
                       "ScheduleConditionSparseK":ScheduleConditionSparseK,
                       "MaskingDiffusion":MaskingDiffusion,
                       "SEDD": SEDD,
                       "DiscreteScheduleCondition":DiscreteScheduleCondition,
                       "D3PMClassic":D3PMClassic}
    using_lang = cfg.model.model == "ScheduleConditionSparseK"
    if not cfg.model.restart:
        model = model_name_dict[cfg.model.model](
            x0_model_class,
            nn_params,
            num_classes=len(tokenizer) if tokenizer else cfg.data.N,
            hybrid_loss_coeff=cfg.model.hybrid_loss_coeff,
            gamma=cfg.model.gamma,
            forward_kwargs=OmegaConf.to_container(cfg.model.forward_kwargs, resolve=True),
            schedule_type=cfg.model.schedule_type,
            logistic_pars=cfg.model.logistic_pars,
            fix_x_t_bias=cfg.model.fix_x_t_bias,
            n_T=cfg.model.n_T,
            t_max=cfg.model.t_max,
            seed=cfg.model.seed,
            sedd_param=cfg.model.sedd_param,
            eff_num_classes=cfg.model.eff_num_classes,
            input_logits=cfg.model.input_logits,
            tokenizer=tokenizer if cfg.data.data != 'uniref50' else Tokenizer(),
            **OmegaConf.to_container(cfg.train, resolve=True),
        )
        ckpt_path = None
    else:
        ckpt_path = f'checkpoints/{cfg.model.restart}'
        ckpt_path = max(glob.glob(os.path.join(ckpt_path, '*.ckpt')), key=os.path.getmtime)
        model = model_name_dict[cfg.model.model].load_from_checkpoint(ckpt_path)

    ##### Load data
    model.pre_configure_model(train_dataloader)

    ##### Train
    # wandb.init()
    wandb_logger = WandbLogger(project="debugging")
    lightning_model = model
    torch.set_float32_matmul_precision('high')
    @rank_zero_only
    def update_wandb_config():
        wandb.config.update(lightning_model.hparams)
    update_wandb_config()

    if cfg.data.data == 'uniref50':
        val_check_interval = 2 * (210000//cfg.train.batch_size)
    else:
        val_check_interval = 1.0
    print("broadcast buffer:", not using_lang)
    trainer = Trainer(
        max_epochs=cfg.train.n_epoch, 
        accelerator='auto', 
        devices=torch.cuda.device_count(), 
        logger=wandb_logger, 
        strategy=DDPStrategy(broadcast_buffers=not using_lang),
        callbacks=([EMA(0.9999)] * cfg.train.ema
                   +[ModelCheckpoint(dirpath=f'checkpoints/{wandb_logger.experiment.name}',
                                   save_on_train_epoch_end=False)]),
        val_check_interval=val_check_interval,
        accumulate_grad_batches=cfg.train.accumulate,
    )
    trainer.fit(lightning_model, train_dataloader, test_dataloader, ckpt_path=ckpt_path)
    wandb.finish()

if __name__ == "__main__":
    train()
