import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math
import time

from .utils import kls, convert_to_probs, get_inf_gen, sample_index_S
from .schedule_sample import sample_n_transitions_cont
from .continuous_time_diffusion import ContinuousTimeDiffusion

class ScheduleCondition(ContinuousTimeDiffusion): #schedule conditioning is True!
    def __init__(
        self,
        x0_model_class,
        nn_params,
        num_classes: int = 10,
        forward_kwargs={"type":"uniform"},
        schedule_type="cos",
        gamma=0,
        hybrid_loss_coeff=0.01,
        fix_x_t_bias=False,
        logistic_pars=False,
        input_logits=False,
        **kwargs
    ):
        # Precalculate betas, define model_predict, p_sample
        super().__init__(x0_model_class, nn_params, num_classes, schedule_type, hybrid_loss_coeff, logistic_pars, **kwargs)
        self.save_hyperparameters(ignore=['x0_model_class'])
        self.fix_x_t_bias = fix_x_t_bias
        self.input_logits = input_logits
        assert gamma >= 0 and gamma < 1 # full schedule and classical resp.

        # Precalculate Ls
        L = get_inf_gen(forward_kwargs, num_classes)
        # Get Ks
        rate = - (L.diagonal().min()) / (1-gamma) # L^* in sec 6.6 of the notes
        K = L / rate + torch.eye(num_classes)
        self.rate = rate
        eigenvalues, eigenvectors = torch.linalg.eig(K.double())
        eigenvalues[torch.real(eigenvalues) > 1-self.eps] = 1
        eigenvectors_inv = torch.linalg.inv(eigenvectors)
        self.register_buffer("eigenvalues", eigenvalues)
        self.register_buffer("eigenvectors", eigenvectors)
        self.register_buffer("eigenvectors_inv", eigenvectors_inv)
        
        # Precalculate K_powers
        num_powers = 5000
        assert (num_classes <= 512 and forward_kwargs['type'] != "bert_embed")
        K_powers = torch.stack([torch.linalg.matrix_power(K, i) for i in range(5000)])
        self.register_buffer("K", K)
        self.register_buffer("K_powers", K_powers)

    def pre_configure_model(self, dataloader):
        self.calc_p0(dataloader)
        self.log_alpha, self.beta, *_ = self.get_beta_func(self.K.cpu(), self.p0.cpu(), type_='schedule_condition', scale=self.rate.cpu())
    
    def get_stationary(self):
        evals, evecs = torch.linalg.eig(self.K.T)
        norms_sq = torch.real(evals * evals.conj())
        assert torch.isclose(evals[torch.argmax(norms_sq)], torch.tensor(1, dtype=torch.complex64))
        stationary = evecs[:, torch.argmax(norms_sq)]
        assert torch.allclose(torch.imag(stationary), torch.tensor(0, dtype=self.K.dtype))
        stationary = torch.real(stationary)
        stationary = stationary * torch.sign(stationary)
        assert torch.all(stationary >= 0)
        return stationary / stationary.sum()

    def get_trans_mats_mvp(self, Smk, v):
        """ v is a ...c matrix where K is cd, and Smk is ... """
        dv = v.to(dtype=self.eigenvectors.dtype).reshape(-1, v.shape[-1])
        diag = self.eigenvalues ** F.relu(Smk.flatten()[..., None])
        dv = dv @ self.eigenvectors
        dv = dv * diag
        dv = dv @ self.eigenvectors_inv
        return F.relu(dv.double()).to(torch.float32).reshape(v.shape)
    
    def get_kl_t1(self, x):
        # sample S
        t = self.t_max * torch.ones(x.shape[0], device=x.device)
        S = sample_n_transitions_cont(self.log_alpha, x[0].flatten().shape[0], t)
        S = S.swapaxes(0, 1).reshape(*x.shape).long()
        softmaxed = convert_to_probs(x, self.num_classes)  # bs, ..., num_classes
        trans = self.get_trans_mats_mvp(S, softmaxed)
        x_1 = torch.log(trans+self.eps)
        kl = kls(x_1, torch.log(self.get_stationary() + self.eps))
        return kl.mean()

    def x_t_sample(self, x_0, t, noise, S):
        # forward process, x_0 is the clean input.
        probs = self.K_powers[S, x_0, :]
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = 1/(-torch.log(noise))
        x_t = torch.argmax(probs * gumbel_noise, dim=-1)
        return x_t

    def q_posterior_logits(self, x_0, x_t, t, S, k=1, log=True):
        """ probs for x_{t-k}|x_t, x_0 """
        fact1 = self.K_powers.swapaxes(1, 2)[k, x_t, :] # x_t | x_{t-1}
        softmaxed = convert_to_probs(x_0, self.num_classes) # bs, ..., num_classes
        fact2 = self.get_trans_mats_mvp(S - k, softmaxed) # x_{t-1} | x_{0}
        assert torch.all(fact1 >=0) and torch.all(fact2 >=0)
        if log:
            return torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)
        else:
            return fact1 * fact2

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None, attn_mask=None,) -> torch.Tensor:
        t, S, x_t = self.sample_point(x, attn_mask)
        # predict x_0 and prev(x_t)
        predicted_x0_logits = self.model_predict(x_t, t, cond if cond is not None else attn_mask, S).to(torch.float32)
        true_q_posterior_logits = self.q_posterior_logits(x, x_t, t, S)
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x_t, t, S)
        # get kls and loss
        kl = kls(true_q_posterior_logits, pred_q_posterior_logits) # shape x
        if attn_mask is not None:
            kl = kl * attn_mask
        weight = - self.beta(t) / self.log_alpha(t)
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

    
    def p_sample(self, x, t, cond, attn_mask, noise, S=None, k=1, temperature=1):
        # predict prev(x_t) or x_{t-1}
        predicted_x0_logits = self.model_predict(x, t, cond if cond is not None else attn_mask, S) / temperature
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x, t, S, k=k, log=False)
        # sample
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = 1 / (-torch.log(noise))
        sample = torch.argmax(
            pred_q_posterior_logits * gumbel_noise, dim=-1
        )
        return sample

    def corrector_sample(self, x, t, cond, attn_mask, noise, S=None, k=1, temperature=1):
        # predict prev(x_t) or x_{t-1}
        predicted_x0_logits = self.model_predict(x, t, cond if cond is not None else attn_mask, S) / temperature
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x, t, S, k=k, log=False)
        # K'_x,.K_.,. denoises and then renoises immediately
        sample_logits = torch.einsum('...i,...ij->...j',pred_q_posterior_logits, self.K_powers[k])
        # sample
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = 1 / (-torch.log(noise))
        sample = torch.argmax(sample_logits * gumbel_noise, dim=-1)
        return sample
    
    def sample_sequence(self, x, cond=None, attn_mask=None, n_T=200, stride=10,
                        n_corrector_steps=10, temperature=1, use_tau=False):
        t = self.t_max * torch.ones(x.shape[0], device=x.device)
        S = sample_n_transitions_cont(self.log_alpha, x[0].flatten().shape[0], t)
        t = t * 0
        S = S.swapaxes(0, 1).reshape(*x.shape).long()
        steps = 0
        images = []
        n_steps = torch.tensor([S[b].sum() for b in range(len(S))]).max().item()
        pbar = tqdm(total=n_steps, unit="iteration",
                    position=0, leave=True)
        trans_step = max([n_steps // n_T, 1]) * (n_corrector_steps + 1)
        total_steps = math.ceil(n_steps/trans_step)
        trans_corrector_k = max([trans_step // np.prod(x[0].shape), 1])
        print("Corrector_k:", trans_corrector_k)
        # get order of denoising
        ks = torch.zeros([len(x), total_steps, len(S[0].flatten())], device=S.device, dtype=torch.long)
        for b in range(len(x)):
            # count how many in each bin
            if use_tau:
                ts = torch.linspace(0, self.t_max, total_steps+1, device=S.device).to(torch.float32)
                weights = - self.log_alpha(ts)
                diffs = weights[1:] - weights[:-1]
                n_steps = torch.bincount(torch.multinomial(diffs, num_samples=S[b].sum(), replacement=True),
                                         None, n_T)
                n_steps = torch.cumsum(n_steps, -1)
                n_steps = torch.cat([torch.zeros_like(n_steps[[0]]), n_steps], axis=-1).long()
                assert n_steps[-1] == S[b].sum()
            else:
                n_steps = trans_step * torch.ones(total_steps, device=S.device).to(torch.float32)
                n_steps = torch.cumsum(n_steps, -1)
                n_steps = torch.cat([torch.zeros_like(n_steps[[0]]), n_steps], axis=-1).long()
                assert n_steps[-1] >= S[b].sum()
            
            indices = torch.argwhere(S[b].flatten() > 0)[:, 0]
            values = S[b].flatten()[indices]
            repeated_indices = torch.repeat_interleave(indices, values.long(), dim=0)
            repeated_indices = repeated_indices[torch.randperm(repeated_indices.size(0))]
            uniq = [torch.unique(repeated_indices[n_steps[step]:n_steps[1+step]],
                    return_counts=True) for step in range(total_steps)]
            for (u, c), i in zip(uniq, range(len(uniq))):
                if len(u) > 0:
                    ks[b][i][u] += c
            ones = torch.ones_like(S)
        assert torch.all(ks.sum(1) == S.reshape(len(x), -1))
        while S.sum() > 0:
            k = ks[:, steps, :].reshape(S.shape)
            S_temp = S - k
            assert torch.all(S_temp >= 0)

            # predict what comes next
            x = self.p_sample(
                x, t, cond, attn_mask,
                torch.rand((*x.shape, self.num_classes), device=x.device),
                S, k=k, temperature=temperature,
            )
            assert torch.all(S_temp <= S)
            S = S_temp
            for l in range(n_corrector_steps):
                x = self.corrector_sample(x, t, cond, attn_mask,
                    torch.rand((*x.shape, self.num_classes), device=x.device),
                    S, k=torch.minimum(S, torch.tensor(trans_corrector_k)), temperature=temperature,
                )
            pbar.update(trans_step)
            steps += 1
            if steps % stride == 0:
                images.append(torch.clone(x))
        pbar.close()
        # if last step is not divisible by stride, we add the last image.
        if steps % stride != 0:
            images.append(x)

        return images
