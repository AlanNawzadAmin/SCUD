import io
import tempfile
import numpy as np
import pytorch_lightning as pl
import wandb
import torch
import torch.optim as optim
from PIL import Image
from torchvision.utils import make_grid
import torch.nn.functional as F
from tqdm import tqdm
from .inception_score import inception_score
from pytorch_lightning.utilities import rank_zero_only

def get_gif(sample_x, sample_a, model, gen_trans_step, batch_size):
    # save images
    cond = None#torch.arange(0, batch_size).to(sample_x.device) % 10
    p = model.get_stationary()
    samples = torch.multinomial(p, num_samples=batch_size*sample_x.shape[1:].numel(), replacement=True)
    init_noise = samples.reshape((batch_size,)+sample_x.shape[1:]).to(sample_x.device)
    if sample_a is not None:
        attn_mask = sample_a.repeat(batch_size, *[1]*(sample_a.dim()-1))
    else:
        attn_mask = None
    images = model.sample_sequence(
        init_noise, cond, attn_mask, stride=3, n_T=gen_trans_step,
    )
    if images is not None:
        # image sequences to gif
        gif = []
        for image in images:
            x_as_image = make_grid(image.float() / (model.num_classes - 1), nrow=2)
            img = x_as_image.permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype(np.uint8)
            gif.append(Image.fromarray(img))

        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as temp_file:
            gif[0].save(
                temp_file.name,
                format='GIF',
                save_all=True,
                append_images=gif[1:],
                duration=100,
                loop=0,
            )
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file_img:
            last_img = gif[-1]
            last_img.save(temp_file_img)
        return temp_file.name, temp_file_img.name
    else: 
        return None, None

def get_text(sample_x, sample_a, model, gen_trans_step, batch_size, tokenizer):
    # save images
    cond = None
    p = model.get_stationary()
    samples = torch.multinomial(p, num_samples=batch_size*sample_x.shape[1:].numel(), replacement=True)
    init_noise = samples.reshape((batch_size,)+sample_x.shape[1:]).to(sample_x.device)
    if sample_a is not None:
        attn_mask = sample_a.repeat(batch_size, *[1]*(sample_a.dim()-1))
    else:
        attn_mask = None
    tokens = model.sample_sequence(
        init_noise, cond, attn_mask, stride=3, n_T=gen_trans_step,
    )
    if tokens is not None:
        last_token = tokens[-1]
        stride_tokens = tokens[::(gen_trans_step // 3)//10 + 1]
        if sample_a is not None:
            if hasattr(tokenizer, 'pad_id'):
                pad_id = tokenizer.pad_id
            elif hasattr(tokenizer, 'pad_token_id'):
                pad_id = tokenizer.pad_token_id
            last_token[attn_mask == 0.] = pad_id
            for t in stride_tokens:
                t[attn_mask == 0.] = pad_id
        if hasattr(tokenizer, 'decode'):
            dt = lambda tok: [tokenizer.decode(t) for t in tok]
        elif hasattr(tokenizer, 'untokenize'):
            dt = lambda tok: [tokenizer.untokenize(t)[:int(a.sum())]
                              for t, a in zip(tok,attn_mask)]
        return dt(last_token), [dt(t) for t in stride_tokens]
    else:
        return None

class DiffusionTrainer(pl.LightningModule):
    def __init__(self, lr=1e-3, gen_trans_step=1000, n_gen_images=4, grad_clip_val=1, weight_decay=0, seed=0, n_stat_samples=2e6, tokenizer=None, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['tokenizer'])
        self.lr = lr
        self.grad_clip_val = grad_clip_val
        self.weight_decay = weight_decay
        # logging
        self.sample_x = None
        self.validation_step_outputs = []
        self.gen_trans_step = gen_trans_step
        self.n_gen_images = n_gen_images
        self.n_stat_samples = n_stat_samples
        self.tokenizer = tokenizer
        # self.to(torch.float32)

    def forward(self, x):
        return NotImplementedError

    def get_kl_t1(self, x):
        return NotImplementedError

    def pre_configure_model(self, dataloader):
        pass

    def calc_p0(self, dataloader):
        # get stationary dist
        p0 = torch.ones(self.num_classes)
        pbar = tqdm(total=self.n_stat_samples)
        for i, batch in tqdm(enumerate(dataloader)):
            if p0.sum() > self.n_stat_samples:  
                break
            if isinstance(batch, tuple): #image datasets
                x, _ = batch
            elif isinstance(batch, dict): #text datasets
                x = batch['input_ids']
            new =  F.one_hot(x.long(), num_classes=self.num_classes).to(torch.float32).view((-1, self.num_classes)).sum(0)
            p0 = p0 + new
            pbar.update(new.sum().item())
        pbar.close()
        p0 = p0 / p0.sum()
        self.p0 = p0
        # self.register_buffer("p0", p0)

    def training_step(self, batch, batch_idx):
        if isinstance(batch, tuple): #image datasets
            x, cond = batch
            attn_mask = None
            if cond is not None:
                if cond.dim() == x.dim(): #protein datasets
                    attn_mask = cond
                    cond = None
        elif isinstance(batch, dict): #text datasets
            x, attn_mask = batch['input_ids'], batch['attention_mask']
            cond = batch['cond'] if 'cond' in batch else None
        # x, cond = x.to(device), cond.to(device)
        loss, info = self(x, cond, attn_mask)
        if self.sample_x is None:
            self.sample_x = x[:1]
            self.sample_a = None if attn_mask is None else attn_mask[:1]
                
        self.log('train_loss', info['vb_loss'], sync_dist=True)
        self.log('train_ce_loss', info['ce_loss'], sync_dist=True)
        # with torch.no_grad():
        #     param_norm = sum([torch.norm(p) for p in self.parameters()])
        # self.log('param_norm', param_norm, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):        
        if isinstance(batch, tuple): #image datasets
            x, cond = batch
            attn_mask = None
            if cond is not None:
                if cond.dim() == x.dim(): #protein datasets
                    attn_mask = cond
                    cond = None
        elif isinstance(batch, dict): #text datasets
            x, attn_mask = batch['input_ids'], batch['attention_mask']
            cond = batch['cond'] if 'cond' in batch else None

        # x, cond = x.to(device), cond.to(device)
        loss, info = self(x, cond, attn_mask)
        print(info['vb_loss'], self.get_kl_t1(x).detach().item())
        self.log('val_l01', info['vb_loss'], on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_l1', self.get_kl_t1(x).detach().item(), on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_ce_loss', info['ce_loss'], on_step=False, on_epoch=True, sync_dist=True)
        loss_dict = {
            "val_ce_loss": info['ce_loss'],
            "val_l01": info['vb_loss'],
            "val_l1": self.get_kl_t1(x).detach().item()
        }
        return loss_dict

    @rank_zero_only
    def on_validation_epoch_end(self,):
        # generate image
        if self.sample_x is not None:
            with torch.no_grad():
                if self.tokenizer is None:
                    gif_fname, img_fname = get_gif(self.sample_x, self.sample_a, self, self.gen_trans_step, self.n_gen_images)
                    if gif_fname is not None:
                        if isinstance(self.logger, pl.loggers.WandbLogger):
                            wandb.log({"sample_gif": wandb.Image(gif_fname)})
                            wandb.log({"sample_gif_last": wandb.Image(img_fname)})
                else:
                    print("getting text")
                    last_text, gen_text = get_text(self.sample_x, self.sample_a, self, self.gen_trans_step, self.n_gen_images, self.tokenizer)
                    if last_text is not None:
                        if isinstance(self.logger, pl.loggers.WandbLogger):
                            joined_text = "\n\n".join(last_text)
                            wandb.log({"sample_text": wandb.Table(columns=["text"], data=[[joined_text]])})
                            joined_text_gen = ["\n\n".join(t) for t in gen_text]
                            wandb.log({"sample_text_process": wandb.Table(columns=["text"], data=[[jt] for jt in joined_text_gen])})

    # def on_fit_start(self):
    #     if isinstance(self.logger, pl.loggers.WandbLogger):
    #         wandb.config.update(self.hparams)

    def on_before_optimizer_step(self, optimizer):
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip_val)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer,
            "gradient_clip_val": self.grad_clip_val,
            "weight_decay": self.weight_decay,
            "gradient_clip_algorithm": "norm"
        }
