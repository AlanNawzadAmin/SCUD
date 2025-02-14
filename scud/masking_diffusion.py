import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .utils import kls, convert_to_distribution
from .schedule_sample import sample_n_transitions_cont
from d3pm_sc.ct_sched_cond import ScheduleCondition


class MaskingDiffusion(ScheduleCondition): #schedule conditioning is True!
    def __init__(
        self,
        x0_model_class,
        nn_params,
        num_classes: int = 10,
        schedule_type="cos",
        hybrid_loss_coeff=0.01,
        fix_x_t_bias=False,
        logistic_pars=False,
        input_logits=False,
        **kwargs,
    ):
        forward_kwargs={"type":"uniform"}
        gamma = 1 / num_classes
        if 'gamma' in kwargs:
            del kwargs['gamma']
        if 'forward_kwargs' in kwargs:
            del kwargs['forward_kwargs']
        super().__init__(x0_model_class, nn_params, num_classes, forward_kwargs, schedule_type, gamma, hybrid_loss_coeff,
                         fix_x_t_bias, logistic_pars, input_logits, **kwargs)
        self.use_bad_model_predict = ~ logistic_pars
        # with this choice, x_t_sample is uniform and 
        # q_posterior_logits returns uniform if S>1 and x_0 pred if S==1
        # The only differences is the predictions and marginalizing over S>1 in the weight
        # so we always assume S==1.
        # in principle we could also speed up sampling by ignoring S>1

    def base_predict(self, x_t, t, cond, S):
        masked_pos = S > 0
        masked_x_t = torch.where(masked_pos, self.num_classes, x_t)
        if cond.sum() > 0 and torch.all(cond <= 1):
            attn_mask = cond # don't mask already masked pos
            masked_x_t = torch.where(attn_mask==1, masked_x_t, x_t) 
        return self.x0_model(masked_x_t, t, cond, S=S)[..., :-1]
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor = None, attn_mask: torch.Tensor = None) -> torch.Tensor:
        t, S, x_t = self.sample_point(x, attn_mask)
        S = (S>0).long()
        # predict x_0 and prev(x_t)
        predicted_x0_logits = self.model_predict(x_t, t, cond if cond is not None else attn_mask, S).float()
        true_q_posterior_logits = self.q_posterior_logits(x, x_t, t, S)
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x_t, t, S)

        # get kls and loss
        kl = kls(true_q_posterior_logits, pred_q_posterior_logits) # shape x
        if attn_mask is not None:
            kl = kl * attn_mask
        alpha_t = torch.exp(self.log_alpha(t))
        weight = self.beta(t) * alpha_t / (1 - alpha_t) #mult by p(S=1|t)
        weight = (S.swapaxes(0, -1) * weight).swapaxes(0, -1)
        vb_loss = (kl * weight).mean() * self.t_max
        if attn_mask is not None:
            vb_loss = vb_loss / attn_mask.mean()

        # Also calculate cross entropy loss
        predicted_x0_logits = predicted_x0_logits.flatten(start_dim=0, end_dim=-2)
        x = x.flatten(start_dim=0, end_dim=-1)
        ce_loss = torch.nn.CrossEntropyLoss(reduction='none')(predicted_x0_logits, x)
        if attn_mask is not None:
            ce_loss = (ce_loss * attn_mask.flatten()).sum() / attn_mask.sum()
        else:
            ce_loss = ce_loss.mean()

        return self.hybrid_loss_coeff * ce_loss + vb_loss, {
            "vb_loss": vb_loss.detach().item(),
            "ce_loss": ce_loss.detach().item(),
        }

    def sample_sequence(self, x, cond=None, attn_mask=None, n_T=200, stride=10):
        t = self.t_max * torch.ones(x.shape[0], device=x.device)
        t = t * 0 + 1e-6
        S = (1. + 0. * x).long() # this is the only line changed
        steps = 0
        images = []
        n_steps = torch.tensor([S[b].sum() for b in range(len(S))]).max().item()
        pbar = tqdm(total=n_steps, unit="iteration",
                    position=0, leave=True)
        trans_step = max([n_steps // n_T, 1])
        while S.sum() > 0:
            # predict what comes next
            x_next = self.p_sample(
                x, t, cond, attn_mask, torch.rand((*x.shape, self.num_classes), device=x.device), S, temperature=1
            )
            for b in range(len(x)):
                trans_indices = torch.argwhere(S[b] > 0)
                trans_indices = trans_indices[torch.randperm(len(trans_indices))]
                if len(trans_indices) > 0:
                    # randomly transiiton
                    for idx in trans_indices[:trans_step]:
                        idx_tuple = (b,) + tuple(idx)
                        x[idx_tuple] = x_next[idx_tuple]
                        S[idx_tuple] -= 1
            pbar.update(trans_step)
            steps += 1
            if steps % stride == 0:
                images.append(torch.clone(x))
        pbar.close()
        # if last step is not divisible by stride, we add the last image.
        if steps % stride != 0:
            images.append(x)

        return images