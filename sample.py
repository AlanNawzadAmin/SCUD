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

@hydra.main(version_base=None, config_path="configs", config_name="sample_cifar")
def train(cfg: DictConfig) -> None:
    ##### Load data
    pl.seed_everything(cfg.model.seed, workers=True)
    print("Getting dataloaders.")
    train_dataloader, test_dataloader = get_dataloaders(cfg)
    tokenizer = train_dataloader.tokenizer if hasattr(train_dataloader, "tokenizer") else None

    print(cfg)
    
    ##### Pick model
    model_name_dict = {"ScheduleCondition":ScheduleCondition,
                       "ScheduleConditionSparseK":ScheduleConditionSparseK,
                       "MaskingDiffusion":MaskingDiffusion,
                       "SEDD": SEDD,
                       "DiscreteScheduleCondition":DiscreteScheduleCondition,
                       "D3PMClassic":D3PMClassic}

    if '/' in cfg.model.restart:
        ckpt_path = cfg.model.restart
    else:
        ckpt_path = f'checkpoints/{cfg.model.restart}'
        ckpt_path = max(glob.glob(os.path.join(ckpt_path, '*.ckpt')), key=os.path.getmtime)
    if cfg.model.ema:
        checkpoint = torch.load(ckpt_path)
        ema_state = checkpoint['optimizer_states'][0]['ema']
        model = model_name_dict[cfg.model.model].load_from_checkpoint(ckpt_path)
        for param, ema_param in zip(model.parameters(), ema_state):
            param.data.copy_(ema_param)
    
    ##### Load data
    model.pre_configure_model(train_dataloader)
    model = model.cuda().eval()
    # print(cfg.model.eps)
    model.eps = cfg.model.eps

    ##### get example x
    batch = next(iter(train_dataloader))
    if isinstance(batch, tuple): #image datasets
        sample_x, cond = batch
        sample_x = sample_x.cuda()
        attn_mask = None
        if cond is not None:
            if cond.dim() == sample_x.dim(): #protein datasets
                attn_mask = cond.cuda()
                cond = None
            else:
                cond = cond.cuda()
    elif isinstance(batch, dict): #text datasets
        sample_x, attn_mask = batch['input_ids'].cuda(), batch['attention_mask'].cuda()
    cond = batch['cond'].cuda() if 'cond' in batch else None

    ##### load old samples
    if cfg.sample.restart:
        all_images = []
    else:
        try:
            all_images = np.load(cfg.sample.save_path)
        except FileNotFoundError:
            all_images = []
    
    ##### Sample
    torch.set_float32_matmul_precision('high')
    batch_size = cfg.sample.batch_size
    n_steps = int(np.ceil(cfg.sample.n_samples / batch_size))
    with torch.no_grad():
        for k in range(n_steps):
            p = model.get_stationary()
            samples = torch.multinomial(p, num_samples=batch_size*sample_x.shape[1:].numel(), replacement=True)
            init_noise = samples.reshape((batch_size,)+sample_x.shape[1:]).to(sample_x.device)
            final_samples = model.sample_sequence(
                init_noise, cond, attn_mask, stride=3, n_T=cfg.sample.gen_trans_step,
                temperature=cfg.sample.temperature, n_corrector_steps=cfg.sample.n_corrector_steps,
            )[-1].detach().cpu().numpy()
            if len(all_images) > 0:
                all_images = np.concatenate([all_images, final_samples], axis=0)
            else:
                all_images = final_samples
            np.save(cfg.sample.save_path, all_images)
    

if __name__ == "__main__":
    train()
