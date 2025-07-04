import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from transformers import BertModel
import faiss
import scipy
from scipy.sparse import coo_matrix

def _at(a, t, x):
    # t is 1-d, x is integer value of 0 to num_classes - 1
    bs = t.shape[0]
    t = t.reshape((bs, *[1] * (x.dim() - 1)))
    return a[t, x, :]

def kls(dist1, dist2, eps=None): # KL of dists on last dim
    out = F.kl_div(torch.log_softmax(dist2, dim=-1),
                   torch.log_softmax(dist1, dim=-1),
                  log_target=True, reduction='none').sum(-1)
    return out

def convert_to_distribution(x_0, num_classes, eps):
    # returns log probs of x_0 as a distribution
    if x_0.dtype == torch.int64 or x_0.dtype == torch.int32:
        x_0_logits = torch.log(
            torch.nn.functional.one_hot(x_0, num_classes) + eps
        )
    else:
        x_0_logits = x_0.clone()
    return x_0_logits

def convert_to_probs(x_0, num_classes):
    # returns probs of x_0 as a distribution. input is either indices or logits
    if x_0.dtype == torch.int64 or x_0.dtype == torch.int32:
        x_0_probs = torch.nn.functional.one_hot(x_0, num_classes)
    else:
        x_0_probs = torch.softmax(x_0.clone(), dim=-1)
    return x_0_probs

def get_inf_gen(forward_kwargs, num_classes):
    if forward_kwargs['type'] == "uniform":
        L = torch.ones(num_classes, num_classes) / (num_classes-1)
        L.diagonal().fill_(-1)
    elif forward_kwargs['type'] == "gaussian":
        bandwidth = forward_kwargs['bandwidth']
        range_ = torch.arange(num_classes)
        diff_mat = (range_[:, None] - range_[None, :]) ** 2
        L = torch.exp(- diff_mat / (2 * (bandwidth * num_classes) ** 2))
        L = L / (L.sum(-1).max() - 1)
        L.diagonal().fill_(0)
        L[range_, range_] = -L.sum(-1)
    elif forward_kwargs['type'] == "blosum":
        from evodiff.utils import Tokenizer
        tokenizer = Tokenizer()
        # from https://web.expasy.org/protscale/pscale/A.A.Swiss-Prot.html
        aa_freq = np.array([8.25, 5.53, 4.06, 5.45, 1.37, 3.93, 6.75,
                            7.07, 2.27, 5.96, 9.66, 5.84, 2.42, 3.86,
                            4.70, 6.56, 5.34, 1.08, 2.92, 6.87] + 11*[0]) / 100 
        blosum_alphabet = np.array(list('ARNDCQEGHILKMFPSTWYVBZXJOU-'))
        tok_alphabet = np.array(tokenizer.alphabet)
        with open('data/blosum62-special-MSA.mat') as f:
            load_matrix = np.array([line.split()[1:] for line in f if line[0] in blosum_alphabet], dtype=int)
        map_ = blosum_alphabet[:, None] == tok_alphabet[None, :]
        blosum_matrix = np.zeros((len(tok_alphabet), len(tok_alphabet)))
        for i, ind_i in enumerate(np.argmax(map_, axis=1)):
            for j, ind_j in enumerate(np.argmax(map_, axis=1)):
                blosum_matrix[ind_i, ind_j] = load_matrix[i, j]
        # X_ij = BLOSUM_ij * p(aa_j) = p(aa_j | aa_i)
        cond_liks = (2. ** (blosum_matrix/2)) * aa_freq[None, :] 
        cond_liks = cond_liks ** forward_kwargs['beta']
        cond_liks = cond_liks / cond_liks.sum(-1)[:, None]
        L = cond_liks - np.eye(len(cond_liks))
        # break up
        l, V = np.linalg.eig(cond_liks[:20, :20])
        V_inv = np.linalg.inv(V)

        # alpha
        alpha = forward_kwargs['alpha']
        if alpha > 0:
            evals = (l**alpha - 1)[None, :] / alpha
        else:
            evals = np.log(l)
        L[:20, :20] = (V * evals) @ V_inv
        L[20:] *= - np.diagonal(L).min()
        L[L<0] = 0
        L = torch.tensor(L).float()
        range_ = torch.arange(num_classes)
        L[range_, range_] = -L.sum(-1)
    if ("make_sym" in forward_kwargs.keys() and forward_kwargs['make_sym']):
        L = (L + L.T) / 2
        range_ = torch.arange(num_classes)
        L.diagonal().fill_(0)
        L[range_, range_] = -L.sum(-1)
    if (("normalize" in forward_kwargs.keys() and forward_kwargs['normalize'])
        or ("normalized" in forward_kwargs.keys() and forward_kwargs['normalized'])):
        L = L / (- L.diagonal()[:, None])
        range_ = torch.arange(num_classes)
        L.diagonal().fill_(0)
        L[range_, range_] = -L.sum(-1)
    return L

def get_sort_S(S):
    S_flat, sort = torch.sort(S.flatten(), descending=True)
    S_sort = S_flat.reshape(S.shape)
    unsort = torch.zeros_like(sort)
    unsort[sort] = torch.arange(len(S_flat), device=S_flat.device)
    return S_sort, sort, unsort
    
def get_counts_S_flat(S_flat):
    unique, counts = torch.unique(torch.clamp(S_flat, min=0), return_counts=True)
    full_counts = torch.zeros(unique.max()+1, device=unique.device, dtype=torch.long)
    full_counts[unique] = counts
    return full_counts.flip(0).cumsum(0)

def _pad(tokenized, value, dim=2):
    """
    Utility function that pads batches to the same length.

    tokenized: list of tokenized sequences
    value: pad index
    """
    batch_size = len(tokenized)
    max_len = max(len(t) for t in tokenized)
    if dim == 3: # dim = 3 (one hot)
        categories = tokenized[0].shape[-1]
        output = torch.zeros((batch_size, max_len, categories)) + value
        for row, t in enumerate(tokenized):
            output[row, :len(t), :] = t
    elif dim == 2: # dim = 2 (tokenized)
        output = torch.zeros((batch_size, max_len)) + value
        for row, t in enumerate(tokenized):
            output[row, :len(t)] = t
    else:
        print("padding not supported for dim > 3")
    return output

def sample_index_S(S):
    # Flatten the array
    S_flat = S.flatten()
    
    # Ensure all values are non-negative
    if torch.any(S_flat < 0):
        raise ValueError("All entries in S must be non-negative for probability sampling")
    
    # Sample an index
    sampled_flat_index = torch.multinomial(S_flat, num_samples=1)
    
    # Convert the flat index back to multidimensional index
    sampled_index = np.unravel_index(sampled_flat_index.item(), S.shape)
    
    return sampled_index

def log1p(x):
    result = torch.log1p(x)
    mask = x < -0.7
    if mask.any():
        x_neg = x[mask]
        neg_x = -x_neg
        inv_xp1 = 1.0 / (neg_x - 1.0)
        result[mask] = torch.log1p(inv_xp1) + torch.log(neg_x)
    return result

