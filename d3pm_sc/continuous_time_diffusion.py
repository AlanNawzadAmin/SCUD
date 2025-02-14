import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from .trainer import DiffusionTrainer
import pytorch_lightning as pl

from .schedule_sample import sample_n_transitions, sample_full_transitions
from .schedule_sample import sample_n_transitions_cont
from d3pm_sc.mutual_info_schedule import get_a_b_func_mi

def get_betas(schedule_type):
    if schedule_type in ['cos', 'linear']:
        def get_funcs(L, p0, model='SEDD', scale=1, type_=None):
            if schedule_type == 'cos':
                alpha = lambda t: 1-torch.cos((1 - t) * torch.pi / 2)
                alpha_prime = lambda t: -torch.sin((1 - t) * torch.pi / 2) * torch.pi / 2
            if schedule_type == 'linear':
                alpha = lambda t: 1-t
                alpha_prime = lambda t: -1
            beta = lambda t: - scale * alpha_prime(t) / alpha(t)
            log_alpha = lambda t: scale * torch.log(alpha(t))
            return log_alpha, beta
    elif schedule_type in ['mutual_information']:
        return get_a_b_func_mi
    return get_funcs

class ContinuousTimeDiffusion(DiffusionTrainer): #schedule conditioning is True!
    def __init__(
        self,
        x0_model_class,
        nn_params,
        num_classes: int = 10,
        schedule_type="cos",
        hybrid_loss_coeff=0.001,
        logistic_pars=False,
        t_max=0.999,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=['x0_model_class'])
        self.hparams.update(x0_model_class=x0_model_class.__name__)
        self.x0_model = x0_model_class(**nn_params)
        self.hybrid_loss_coeff = hybrid_loss_coeff
        self.eps = 1e-9
        self.num_classes = num_classes
        self.t_max = t_max
        self.logistic_pars = logistic_pars

        # Precalculate betas
        self.get_beta_func = get_betas(schedule_type)

    def get_stationary(self):
        raise NotImplementedError

    def base_predict(self, x_t, t, cond, S=None):
        return self.x0_model(x_t, t, cond, S).to(torch.float32)

    def model_predict(self, x_t, t, cond, S=None):
        pred = self.base_predict(x_t, t, cond, S)
        if not self.logistic_pars:
            return pred
        else:
            loc = pred[..., 0].unsqueeze(-1)
            log_scale = pred[..., 1].unsqueeze(-1)
            inv_scale = torch.exp(- (log_scale - 2.))
            bin_width = 2. / (self.num_classes - 1.)
            bin_centers = torch.linspace(-1., 1., self.num_classes).to(pred.device)
            bin_centers = bin_centers - loc
            log_cdf_min = torch.nn.LogSigmoid()(
                inv_scale * (bin_centers - 0.5 * bin_width))
            log_cdf_max = torch.nn.LogSigmoid()(
                inv_scale * (bin_centers + 0.5 * bin_width))
            logits = log_cdf_max + torch.log1p(-torch.exp(log_cdf_min-log_cdf_max)+self.eps)
            return logits

    def q_posterior_logits(self, x_0, x_t, t, S=None):
        raise NotImplementedError

    def x_t_sample(self, x_0, t, noise, S=None):
        raise NotImplementedError

    def sample_point(self, x, attn_mask=None, rand_shape=None):   
        t = torch.rand(x.shape[0], device=x.device) * self.t_max
        S = sample_n_transitions_cont(self.log_alpha, x[0].flatten().shape[0], t)
        S = S.swapaxes(0, 1).reshape(*x.shape).long()
        x_t = self.x_t_sample(
            x, t, torch.rand((*x.shape, rand_shape if rand_shape is not None else self.num_classes), device=x.device), S
        )
        # if attn_mask is not None:
        #     x_t = torch.where(attn_mask==1, x_t, x)
        #     S = torch.where(attn_mask==1, S, 0 * S)
        return t, S, x_t

    def load_state_dict(self, state_dict, strict=False):
        # Call the parent class's load_state_dict method
        missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict=False)

        # Load the additional state dict variables
        for key in ['p0_inds', 'p0_rank', 'K', 'L', 'K_coo', 'K_csc', 'K_T', 'L_T', 'stat', 'stationary']:
            if key in state_dict:
                setattr(self, key, state_dict[key])
                if key in unexpected_keys:
                    unexpected_keys.remove(key)
            # elif strict:
            #     missing_keys.append(key)

        if strict:
            error_msgs = []
            if len(unexpected_keys) > 0:
                error_msgs.append('unexpected key(s) in state_dict: {}. '.format(', '.join('"{}"'.format(k) for k in unexpected_keys)))
            if len(missing_keys) > 0:
                error_msgs.append('missing key(s) in state_dict: {}. '.format(', '.join('"{}"'.format(k) for k in missing_keys)))

            if len(error_msgs) > 0:
                raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                                    self.__class__.__name__, "\n\t".join(error_msgs)))

        return missing_keys, unexpected_keys

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None, **kwargs):
        print("Loading checkpoint ...")
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        hparams = checkpoint['hyper_parameters']
        
        # Get the x0_model_class
        from d3pm_sc.unet import UNet, KingmaUNet, SimpleUNet, GigaUNet
        from d3pm_sc.dit_vision import DiT_Llama
        from d3pm_sc.dit_text import DIT
        from d3pm_sc.protein_convnet import ByteNetLMTime
        x0_model_class = {
            "SimpleUNet":SimpleUNet,
            "KingmaUNet":KingmaUNet,
            "UNet":UNet,
            "GigaUNet":GigaUNet,
            "DiT_Llama":DiT_Llama,
            "DIT": DIT,
            "ByteNetLMTime": ByteNetLMTime
        }[hparams['x0_model_class']]
        hparams['x0_model_class'] = x0_model_class

        # Create model
        print("Setting up class ...")
        model = cls(**hparams)
        print("Loading params ...")
        model.load_state_dict(checkpoint['state_dict'])
        return model
