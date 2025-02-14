import torch
import math
import numpy as np
from tqdm import tqdm
from d3pm_sc.root_finder import root_finder, newton_root_finder
import hashlib
import os

def hash_matrix(matrix):
    # Convert the matrix to a byte string
    byte_string = matrix.cpu().numpy().tobytes()
    
    # Create a hash object (using SHA-256 in this example)
    hash_object = hashlib.sha256()
    
    # Update the hash object with the byte string
    hash_object.update(byte_string)
    
    # Get the hexadecimal representation of the hash
    return hash_object.hexdigest()

def try_load(func, fname, path='/scratch/aa11803/d3pm/save_alphas/'):
    if fname in os.listdir(path):
        print("Loading alphas. Note: I hope p0 is similar to before!")
        val = np.load(path+fname)
        val = torch.tensor(val)
    else:
        val = func()#- log_alpha_naive(base_ts)
        np.save(path+fname, val.cpu().numpy())
    return val

def get_a_b_func_cont(L, p0, **kwargs):
    N = len(p0)
    ent_p0 = -torch.xlogy(p0, p0).sum()
    evals, V = torch.linalg.eig(L.double())
    evals[torch.real(evals) > -1e-6] = 0
    second_eval = torch.real(evals)
    second_eval = -second_eval.sort().values[-2]
    V_inv = torch.linalg.inv(V)
    def mi(t, t_shift=1):
        """ t_shift is 1-t """
        evals_skew = torch.exp(t[:, None, None] * evals[None, None, :])
        too_big = torch.real(evals_skew * evals_skew.conj()) > 1
        evals_skew = torch.where(too_big, 1, evals_skew)
        mat = torch.where((t > 1e-5)[:, None, None],
                          torch.real((V[None,:, :] * evals_skew) @ V_inv), 
                          torch.eye(len(L)) + t[:, None, None] * L
                          + 0.5 * (t**2)[:, None, None] * (L @ L)) # stable for small t
        p = p0[None, :, None] * mat
        p = torch.where(p < 1e-12, 0, p)
        mi_m1 = (torch.xlogy(p, p).sum(-1) - torch.xlogy(p.sum(-2), p.sum(-2))).sum(-1) / ent_p0
        return (mi_m1 + t_shift)

    base_ts = torch.linspace(0., 0.9991, 1000000) 
    def log_alpha_naive(ts, bs=1000):
        batches = [ts.double()[i*bs:(i+1)*bs] for i in range(
            math.ceil(len(ts)/bs))]
        guess_ts = torch.tensor([batch[len(batch) // 2] for batch in batches])
        guesses = newton_root_finder(mi, 1/second_eval, guess_ts, print_=False)
        out = [newton_root_finder(mi, guess, batch, print_=False, max_iter=20)
               for batch, guess in tqdm(list(zip(batches, guesses)))]
        out = torch.concat(out)
        # out = root_finder(mi, 0, 20/second_eval, ts)
        return -torch.where(out>1e-8, out, 1e-8)
    hash_mat = hash_matrix(L)
    if N > 100:
        try_load(lambda : L, f'{hash_mat}_mat.npy')
        base_alphas = try_load(lambda: -log_alpha_naive(base_ts), f'{hash_mat}.npy')
    else:
        base_alphas = -log_alpha_naive(base_ts)
    def log_alpha(ts):
        closest_index = torch.searchsorted(base_ts, ts.to('cpu'))
        best_guess_l = base_alphas[closest_index]
        # best_guess_u = base_alphas[closest_index]
        # out = newton_root_finder(mi, best_guess_l, ts.to('cpu'))
        out = best_guess_l
        # out = root_finder(mi, best_guess_l, best_guess_u, ts)
        return -torch.where(out>1e-8, out, 1e-8).to(ts.device).to(ts.dtype)
    
    def naive_beta(ts, bs=10000):
        batches = [ts.double()[i*bs:(i+1)*bs] for i in range(
            math.ceil(len(ts)/bs))]
        grad = []
        for batch in batches:
            out_tensor = -log_alpha(batch.to('cpu'))
            grad.append(-1 / torch.func.grad(lambda o: mi(o).sum())(out_tensor))
        return torch.cat(grad).to(ts.device).to(ts.dtype)
    if N > 100:
        base_betas = try_load(lambda: naive_beta(base_ts), f'{hash_mat}_beta.npy')
    else:
        base_betas = naive_beta(base_ts)
    def beta(ts):
        closest_index = torch.searchsorted(base_ts, ts.to('cpu'))
        best_guess_l = base_betas[closest_index]
        out = best_guess_l
        return out.to(ts.device).to(ts.dtype)
    return log_alpha, beta, mi


def get_a_b_func_sc(K, p0, precompute_mis=None, second_eval=None, **kwargs):
    if second_eval is None:
        L_ish = K.double() - torch.eye(len(K), dtype=torch.float64)
        evals, V = torch.linalg.eig(L_ish)
        evals[torch.real(evals) > -1e-6] = 0
        second_eval = torch.real(evals)
        second_eval = -second_eval.sort().values[-2]
    max_n = int(40 / second_eval)
    ent_p0 = -torch.xlogy(p0, p0).sum()
    if precompute_mis is None:
        V_inv = torch.linalg.inv(V)
        def mi_p(n):
            if n > 0:
                mat = torch.real((V * ((1+evals[None, :]) ** n)) @ V_inv)
            else:
                mat = torch.eye(len(K), dtype=K.dtype)
            p = p0[:, None] * mat
            p = torch.where(p < 0, 0, p)
            return ((torch.xlogy(p, p).sum(-1) - torch.xlogy(p.sum(-2), p.sum(-2))).sum(-1) / ent_p0 + 1)
        precompute_mis = [mi_p(n) for n in range(max_n)]
    precompute_mis = torch.tensor(precompute_mis).double()
    precompute_mis = torch.maximum(precompute_mis, torch.zeros(1))
    max_n = (precompute_mis > 1e-7).sum()
    precompute_mis = torch.cummin(precompute_mis[:max_n], 0)[0]
    range_ = torch.arange(max_n, dtype=p0.dtype)
    precompute_log_factorial = torch.lgamma(1+range_)
    def mi(t, t_shift=1.):
        log_probs = (range_ * torch.log(t[:, None]+1e-6) - t[:, None]
                     - precompute_log_factorial)
        return torch.exp(log_probs) @ precompute_mis - 1. + t_shift

    base_ts = torch.linspace(0., 0.9991, 1000000)
    def log_alpha_naive(ts):
        out = newton_root_finder(mi, max_n/20, ts.double())
        return -torch.where(out>1e-6, out, 1e-6).float()
    hash_mat = hash_matrix(K)
    if len(K) > 100:
        try_load(lambda : K, f'{hash_mat}_mat.npy')
        base_alphas = try_load(lambda: -log_alpha_naive(base_ts), f'scud_{hash_mat}.npy')
    else:
        base_alphas = -log_alpha_naive(base_ts)
    base_alphas = torch.cummax(base_alphas, 0)[0] # fix any errors
    def log_alpha(ts):
        closest_index = torch.searchsorted(base_ts, ts.to('cpu'))
        best_guess_l = base_alphas[closest_index]
        out = best_guess_l
        return -torch.where(out>1e-6, out, 1e-6).to(ts.device).to(ts.dtype)
    
    def naive_beta(ts, bs=10000):
        out_tensor = -log_alpha(ts.to('cpu').double())
        batches = [ts[i*bs:(i+1)*bs].double() for i in range(
                   math.ceil(len(ts)/bs))]
        grad = []
        for batch in batches:
            out_tensor = -log_alpha(batch.to('cpu'))
            grad.append(-1 / torch.func.grad(lambda o: mi(o).sum())(out_tensor))
        return torch.cat(grad).to(ts.device).to(ts.dtype)
    if len(K) > 100:
        base_betas = try_load(lambda: naive_beta(base_ts), f'scud_{hash_mat}_beta.npy')
    else:
        base_betas = naive_beta(base_ts)
    def beta(ts):
        closest_index = torch.searchsorted(base_ts, ts.to('cpu'))
        best_guess_l = base_betas[closest_index]
        out = best_guess_l
        return out.to(ts.device).to(ts.dtype)
    return log_alpha, beta, mi, precompute_mis

def get_a_b_func_mi(mat, p0, type_, **kwargs):
    if type_ == 'schedule_condition':
        return get_a_b_func_sc(mat, p0, **kwargs)
    elif type_ == 'SEDD':
        return get_a_b_func_cont(mat, p0, **kwargs)
