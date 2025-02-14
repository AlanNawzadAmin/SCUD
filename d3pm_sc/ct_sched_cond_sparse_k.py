import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from tqdm import tqdm
import time

from .utils import kls, get_sort_S, get_counts_S_flat, sample_index_S, sparse_zeros_like
from .schedule_sample import sample_n_transitions_cont
from .continuous_time_diffusion import ContinuousTimeDiffusion

from .language_model_inf_gen import get_L_and_K


def convert_to_norm_distribution(x_0, num_classes, eps):
    # returns log probs of x_0 as a distribution
    if x_0.dtype == torch.int64 or x_0.dtype == torch.int32:
        softmax = torch.nn.functional.one_hot(x_0, num_classes).float()
    else:
        softmax = torch.softmax(x_0, dim=-1)
    return softmax
            
class ScheduleConditionSparseK(ContinuousTimeDiffusion): #schedule conditioning is True!
    def __init__(
        self,
        x0_model_class,
        nn_params,
        num_classes: int = 10,
        forward_kwargs={"type":"uniform"},
        schedule_type="cos",
        gamma=0,
        hybrid_loss_coeff=0.01,
        sedd_param=True,
        eff_num_classes=1000000,
        **kwargs
    ):
        # Precalculate betas, define model_predict, p_sample
        super().__init__(x0_model_class, nn_params, num_classes, schedule_type, hybrid_loss_coeff, **kwargs)
        self.save_hyperparameters(ignore=['x0_model_class'])
        self.cache_fact2 = True
        self.sedd_param = sedd_param
        self.eff_num_classes = min([eff_num_classes, num_classes])
        print([eff_num_classes, num_classes])
        self.forward_kwargs = forward_kwargs
        self.gamma = gamma
        assert gamma >= 0 and gamma < 1 # full schedule and classical resp.
        # not gonna implement fix_x_t_bias, input_logits, logistic_pars

    def calc_stationary(self):
        K_T_gpu = self.K_T.cuda()
        stat_gpu = self.stat.cuda()
        stationary = torch.ones(K_T_gpu.shape[0]).float().cuda()
        stationary = stationary / stationary.sum()
        pbar = tqdm(range(100000))
        for i in pbar:
            stationary_new = self.K_T_operator(K_T_gpu, stationary, stat_gpu)
            err = torch.sqrt(((stationary_new - stationary) ** 2).sum())
            if torch.allclose(err, torch.zeros_like(err)):
                break
            stationary = stationary_new
            if i%1000 == 0:
                pbar.set_description(f"Getting stationary. err:{err.item()}")
        stationary = stationary / torch.sqrt((stationary ** 2).sum())
        self.register_buffer("stationary", stationary)

    def pre_configure_model(self, dataloader):
        if not hasattr(self, 'p0'):
            self.calc_p0(dataloader)
        if not hasattr(self, 'p0_inds'):
            p0_inds = torch.flip(torch.argsort(self.p0), (0,))
            p0_rank = torch.argsort(p0_inds)
            self.register_buffer("p0_inds", p0_inds)
            self.register_buffer("p0_rank", p0_rank)
        else:
            p0_inds = self.p0_inds
            p0_rank = self.p0_rank
        if self.eff_num_classes < self.num_classes:
            self.freq_order = True
            mat_inds = p0_inds[:self.eff_num_classes].cpu()
        else:
            self.freq_order = False
            self.eff_num_classes = self.num_classes
            mat_inds = None
        
        # Precalculate K_powers of KNN process
        L, K, L_cpu, K_cpu, rate = get_L_and_K(self.forward_kwargs, self.gamma, mat_inds)
        if not hasattr(self, 'K'):    
            self.rate = max([rate, 1 if self.freq_order else 1e-6])
            self.register_buffer("K", K)
            self.register_buffer("L", L)
            K_coo = K.to_sparse_coo()
            K_csc = K.to_sparse_csc()
            K_T = K_coo.transpose(0, 1).coalesce().to_sparse_csr()
            L_coo = L.to_sparse_coo()
            L_T = L_coo.transpose(0, 1).coalesce().to_sparse_csr()
            self.register_buffer("K_coo", K_coo)
            self.register_buffer("K_csc", K_csc)
            self.register_buffer("K_T", K_T)
            self.register_buffer("L_T", L_T)
        else:
            self.rate = max([rate, 1 if self.freq_order else 1e-6])
            K = self.K
            L = self.L
            K_coo = self.K_coo
            K_csc = self.K_csc
            K_T = self.K_T
            L_T = self.L_T
            
        print("Setting up K and L operators...")
        # Add in uniform process
        self.unif_to_stat = self.forward_kwargs['unif_to_stat']
        if not hasattr(self, 'stat'):
            stat = (self.p0[self.p0_inds][:self.eff_num_classes] / self.p0[self.p0_inds][:self.eff_num_classes].sum() if self.unif_to_stat
                    else torch.ones(self.eff_num_classes) / self.eff_num_classes)
            stat = stat / stat.sum()
            self.register_buffer("stat", stat)
        else:
            stat = self.stat
        self.unif_rate = self.forward_kwargs['uniform_rate'] * (1 if self.freq_order else torch.tensor(1-self.stat).max())
        self.rate = self.rate + self.unif_rate
        self.up = self.unif_rate / self.rate
        if self.up > 0.999:
            self.up = 1
            K = sparse_zeros_like(self.K)
            K_coo = sparse_zeros_like(self.K_coo)
            K_csc = sparse_zeros_like(self.K_csc)
            K_T = sparse_zeros_like(self.K_T)
        print("Uniform rate is:", self.up)
        # @torch.jit.script
        def K_T_operator_first(K_T, x_0_probs, stat, eff_num_classes:int, up:float): # assumes x_0_probs are normalized
            x_0_probs_top = x_0_probs[:eff_num_classes]
            other_part = x_0_probs[eff_num_classes:].sum(0)
            stat_broadcast = stat.view((eff_num_classes,) + (1,) * (x_0_probs_top.dim()-1))
            return (1-up) * (K_T @ x_0_probs_top) + (up + other_part*(1-up)) * stat_broadcast
        # @torch.jit.script
        def K_T_operator_next(K_T, x_0_probs, stat, eff_num_classes:int, up:float): # assumes x_0_probs are normalized
            stat_broadcast = stat.view((eff_num_classes,) + (1,) * (x_0_probs.dim()-1))
            return (1-up) * (K_T @ x_0_probs) + up * stat_broadcast
        def K_csc_operator_next(K_csc, x_0_probs, stat, eff_num_classes:int, up:float): # assumes x_0_probs are normalized
            return (1-up) * (x_0_probs @ K_csc) + up * stat
        def K_T_operator(K_T, x_0_probs, stat):
            if len(x_0_probs) > self.eff_num_classes:
                return K_T_operator_first(K_T, x_0_probs, stat, self.eff_num_classes, self.up)
            else:
                return K_T_operator_next(K_T, x_0_probs, stat, self.eff_num_classes, self.up)
        # @torch.jit.script
        def K_operator_vec_jit(K, x_t_probs, stat, up):        
            """ Ignores rare tokens in input and output! """
            return (1-up) * (K @ x_t_probs) + up * stat @ x_t_probs
        def K_operator_vec(K, x_t_probs, stat):
            return K_operator_vec_jit(K, x_t_probs, stat, self.up)
        def K_operator(x_t_index):
            """ Gives wrong answer when x_t has rare tokens! """
            clamp_index = torch.clamp(x_t_index.flatten(), max=self.eff_num_classes-1)
            struct = self.K_coo.index_select(1, clamp_index).to_dense()
            struct = struct.T.reshape(*x_t_index.shape, self.eff_num_classes)
            unif_part = self.stat[torch.clamp(x_t_index, max=self.eff_num_classes-1)].unsqueeze(-1)   
            struct = (1 - self.up) * struct + self.up * unif_part
            return torch.cat([struct, unif_part.expand(*unif_part.shape[:-1], self.num_classes - self.eff_num_classes)], dim=-1)
        # @torch.jit.script
        def K_T_power_loop(liks, K_T, K_csc, start_indices, stat,
                         eff_num_classes:int, freq_order:bool, up:float):
            # S=1
            active = start_indices[0]
            liks[:eff_num_classes, :active] = K_T_operator_first(
                    K_T, liks[:, :active], stat, eff_num_classes, up)
            if freq_order:
                liks[eff_num_classes:, :active] = 0
            if len(start_indices) > 1:
                submat = liks[:eff_num_classes, :]#.T.contiguous()
                # S>1
                for i in range(1, len(start_indices)):
                    active = start_indices[i]
                    submat[:, :active] = K_T_operator_next(K_T, submat[:, :active].contiguous(), stat,
                                                         eff_num_classes, up)
                # liks[:eff_num_classes, :] = submat.T
            return liks
        def K_power_loop(x_grad, K, start_indices, stat,
                         eff_num_classes:int, freq_order:bool, up:float):
            submat = x_grad[:eff_num_classes, :]
            # S>1
            for i in range(1, len(start_indices))[::-1]:
                active = start_indices[i]
                submat[:, :active] = K_operator_vec(K, submat[:, :active].contiguous(), stat)
            # S=1
            active = start_indices[0]
            if freq_order:
                x_grad[eff_num_classes:, :active] = (1-self.up) * stat @ submat[:, :active]
            submat[:, :active] = K_operator_vec(K, submat[:, :active], stat)
            return x_grad
        def K_power_loop_naive(x_grad, K, S, stat,
                         eff_num_classes:int, freq_order:bool, up:float):
            submat = x_grad[:eff_num_classes, :]
            # S>1
            for i in range(1, S.max())[::-1]:
                active = S >= (i+1)
                submat[:, active] = K_operator_vec(K, submat[:, active].contiguous(), stat)
            # S=1
            active = S >= 1
            if freq_order:
                x_grad[eff_num_classes:, active] = stat @ submat[:, active]
            submat[:, active] = K_operator_vec(K, submat[:, active], stat)
            return x_grad

        self.K_T_operator = K_T_operator
        self.K_operator_vec = K_operator_vec
        self.K_operator = K_operator
        self.K_T_power_loop = K_T_power_loop
        self.K_power_loop = K_power_loop
        self.K_power_loop_naive = K_power_loop_naive
        self.second_eval = self.up
        if not hasattr(self, 'stationary'):
            self.calc_stationary()

        print("Getting Mutual info schedule")
        mis = []
        p0_gpu = self.p0.cuda()[self.p0_inds]
        p0_gpu = p0_gpu / p0_gpu.sum()
        stat_cuda = self.stat.cuda()
        mat = torch.eye(self.num_classes).float().cuda()
        K_T_gpu = self.K_T.cuda()
        ent_p0 = -torch.xlogy(p0_gpu, p0_gpu).sum()
        pbar = tqdm(np.arange(int(10/self.second_eval)))
        for i in pbar:
            p = p0_gpu[:, None] * mat.T
            p = torch.where(p < 0, 0, p)
            p_sum = p.sum(0)
            mi = (torch.xlogy(p, p).sum() - torch.xlogy(p_sum, p_sum).sum()) / ent_p0 + 1
            mis.append(mi)
            if mi < 1e-5:
                break
            pbar.set_description(f"MI:{mi.item()}")
            mat = self.K_T_operator(K_T_gpu, mat, stat_cuda)
        self.precompute_mis = mis
        self.log_alpha, self.beta, *_ = self.get_beta_func(None, self.p0, type_='schedule_condition', scale=self.rate, precompute_mis=mis, second_eval=self.second_eval)
        del mat
        import gc
        torch.cuda.empty_cache()
        gc.collect()

    def K_T_power_mult(self, S, x_0):
        class K_T_power_mult_class(Function):
            @staticmethod
            def forward(ctx, S, x_0):
                cumulative_counts = get_counts_S_flat(S.flatten())
                start_indices = cumulative_counts[:-1].flip(0)
                ctx.save_for_backward(start_indices)
                liks = x_0.reshape(-1, x_0.shape[-1]).T.clone()
                out = self.K_T_power_loop(
                    liks, self.K_T, self.K_csc, start_indices, self.stat,
                    self.eff_num_classes, self.freq_order, self.up)
                return out.T.reshape(x_0.shape)
        
            @staticmethod
            def backward(ctx, grad_output):
                start_indices, = ctx.saved_tensors
                x_grad = grad_output.reshape(-1, grad_output.shape[-1]).T
                x_grad = self.K_power_loop(x_grad, self.K, start_indices, self.stat,
                                      self.eff_num_classes, self.freq_order, self.up)
                return None, x_grad.T.reshape(grad_output.shape)#.to(torch.bfloat16)
        return K_T_power_mult_class.apply(S, x_0)
    
    def get_stationary(self):
        return self.stationary / self.stationary.sum()

    def get_kl_t1(self, x):
        if self.freq_order:
            x = self.p0_rank[x]
        # sample S
        t = self.t_max * torch.ones(x.shape[0], device=x.device)
        S = sample_n_transitions_cont(self.log_alpha, x[0].flatten().shape[0], t)
        S = S.swapaxes(0, 1).reshape(*x.shape).long()
        softmaxed = convert_to_norm_distribution(x, self.num_classes, self.eps)
        x_1 = self.K_T_power_mult(S, softmaxed)
        kl = kls(torch.log(x_1[..., :self.eff_num_classes] + self.eps), torch.log(self.get_stationary() + self.eps), self.eps)
        not_included = torch.logical_and(S==0, x>=self.eff_num_classes).float()
        return (kl * (1-not_included)).mean() - torch.log(torch.tensor(self.eps)) * not_included.mean()

    def x_t_sample(self, x_0, t, noise, S):
        S_sort, sort, unsort = get_sort_S(S)
        softmaxed = convert_to_norm_distribution(x_0.flatten()[sort], self.num_classes, self.eps)
        if self.cache_fact2:
            self.cache_sm1_power = self.K_T_power_mult(S_sort-1, softmaxed.float())
            x_t_probs = self.K_T_power_mult((S>0).long(), self.cache_sm1_power)
        else:
            x_t_probs = self.K_T_power_mult(S_sort, softmaxed)
        # print("normalization check:", probs[..., [-1]].min(), probs[..., [-1]].max())
        inv_gumbel_noise = (1e-10 - (torch.rand_like(x_t_probs) + 1e-10).log())
        x_t_sample_sort = torch.argmax(x_t_probs/inv_gumbel_noise, dim=-1)
        return x_t_sample_sort

    def q_posterior_logits(self, x_0_sort, x_t_sort, t, S_sort,
                           use_cached_fact2=False, log_fact1=None, k_sort=1):
        if use_cached_fact2:
            fact2 = self.cache_sm1_power
            # print("cache check:", ((fact2 - self.K_T_power_mult(S_sort-1, softmaxed.float()))**2).sum())
        else:
            softmaxed = convert_to_norm_distribution(x_0_sort, self.num_classes, self.eps)
            if torch.all((S_sort-k_sort) == 0):
                fact2 = softmaxed
            else:
                fact2 = self.K_T_power_mult(S_sort-k_sort, softmaxed.float())
            # print(fact2[..., 2].max())
            # print(x_t_sort[fact2[..., 2].argmax()])
        if log_fact1 is None:
            if (isinstance(k_sort, (int, float)) and k_sort == 1):
                log_fact1 = torch.log(self.K_operator(x_t_sort) + self.eps)
            else:
                softmax_t = convert_to_norm_distribution(x_t_sort, self.num_classes, self.eps)
                x_grad = softmax_t.reshape(-1, softmax_t.shape[-1]).T
                K_k_x_t = self.K_power_loop_naive(x_grad, self.K, k_sort, self.stat,
                                         self.eff_num_classes, self.freq_order, self.up)
                log_fact1 = torch.log(K_k_x_t.T.reshape(softmax_t.shape) + self.eps)
                # print(torch.exp(log_fact1[..., 2]).max())
                # print(x_t_sort[torch.exp(log_fact1[..., 2]).argmax()])
        out = torch.log(fact2 + self.eps) + log_fact1
        return out

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None, *args) -> torch.Tensor:
        if self.freq_order:
            x = self.p0_rank[x]
        t, S, x_t_sort = self.sample_point(x, 1)
        S_sort, sort, unsort = get_sort_S(S)
        assert torch.all(sort[unsort] == torch.arange(len(sort), device=sort.device))
        x_sort = x.flatten()[sort]
        x_t = x_t_sort[unsort].reshape(x.shape)
        log_fact1 = torch.log(self.K_operator(x_t_sort) + self.eps)
        # predict x_0 and prev(x_t)
        predicted_x0_logits = self.model_predict(x_t, t, cond, S)
        predicted_x0_logits_sort = predicted_x0_logits.reshape(-1, self.num_classes)[sort]
        true_q_posterior_logits = self.q_posterior_logits(x_sort, x_t_sort, t, S_sort, use_cached_fact2=self.cache_fact2, log_fact1=log_fact1)
        if self.sedd_param:
            pred_q_posterior_logits = predicted_x0_logits
        else:
            pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits_sort, x_t_sort, t, S_sort, log_fact1=log_fact1)
        
        # get kls and loss
        weight = - self.beta(t) / self.log_alpha(t)
        weight = (S.swapaxes(0, -1) * weight).swapaxes(0, -1)
        kl = kls(true_q_posterior_logits, pred_q_posterior_logits, self.eps) # shape x
        vb_loss = (kl * weight.flatten()[sort]).mean() * self.t_max

        # Also calculate cross entropy loss
        ce_loss = torch.nn.CrossEntropyLoss()(predicted_x0_logits_sort, x_sort)

        # print(vb_loss)
        # print(ce_loss)
        return vb_loss, {
            "vb_loss": vb_loss.detach().item(),
            "ce_loss": ce_loss.detach().item(),
        }

    def p_sample(self, x, t, cond, attn_mask, noise, S=None, k=1):
            # predict prev(x_t) or x_{t-1}
            predicted_x0_logits = self.model_predict(x, t, cond if cond is not None else attn_mask, S)
            Smk_sort, sort, unsort = get_sort_S(S-k)
            x_sort = x.flatten()[sort]
            S_sort = S.flatten()[sort]
            k_sort = k.flatten()[sort]
            predicted_x0_logits_sort = predicted_x0_logits.reshape(-1, self.num_classes)[sort]
            pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits_sort, x_sort, t, S_sort, k_sort=k_sort)[unsort].reshape(*x.shape, self.num_classes)
            # sample
            noise = torch.clip(noise, self.eps, 1.0)
            gumbel_noise = -torch.log(-torch.log(noise))
            sample = torch.argmax(
                pred_q_posterior_logits + gumbel_noise, dim=-1
            )
            return sample
        
    def sample_sequence(self, x, cond=None, attn_mask=None, n_T=200, stride=10):
        # identical to sched cond
        t = self.t_max * torch.ones(x.shape[0], device=x.device)
        S = sample_n_transitions_cont(self.log_alpha, x[0].flatten().shape[0], t)
        t = t * 0
        S = S.swapaxes(0, 1).reshape(*x.shape).long()
        steps = 0
        images = []
        n_steps = torch.tensor([S[b].sum() for b in range(len(S))]).max().item()
        if n_steps > 1e6:
            return None
        pbar = tqdm(total=n_steps, unit="iteration",
                    position=0, leave=True)
        trans_step = n_steps // n_T
        while S.sum() > 0:
            k = torch.zeros_like(S)
            S_temp = S.clone()
            for b in range(len(x)):
                for i in range(trans_step):
                    if S_temp[b].sum()>0:
                        ind = sample_index_S(S_temp[b] / S_temp[b].sum())
                        k[b][ind] += 1
                        S_temp[b][ind] -= 1

            # predict what comes next
            x = self.p_sample(
                x, t, cond, attn_mask, torch.rand((*x.shape, self.num_classes), device=x.device), S, k=k,
            )
            assert torch.all(S_temp <= S)
            S = S_temp
            pbar.update(trans_step)
            steps += 1
            if steps % stride == 0:
                images.append(torch.clone(x))
        pbar.close()
        # if last step is not divisible by stride, we add the last image.
        if steps % stride != 0:
            images.append(x)

        if self.freq_order:
            images = [self.p0_inds[im.long()] for im in images]
        return images


