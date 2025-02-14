import numpy as np
import torch
from transformers import BertModel, BertTokenizer, GPT2TokenizerFast, GPT2LMHeadModel
import faiss
import scipy

english_alphabet = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]

def get_knn(k, embeds, mask_pairs):
    range_ = np.arange(len(embeds))
    indices = np.empty([len(embeds), k])
    similarities = np.empty([len(embeds), k])
    for mask, search_mask in mask_pairs:
        if mask.sum()>0:
            index = faiss.IndexFlatIP(embeds.shape[1]) 
            index.add(embeds[search_mask])
            similarities_temp, indices_temp = index.search(embeds[mask], k+1)
            similarities[mask] = similarities_temp[:, 1:]
            indices[mask] = range_[search_mask][indices_temp[:, 1:]]
    return indices, similarities

def get_L_and_K(forward_kwargs, gamma, inds=None):
    if forward_kwargs['type'] == "bert_embed":
        if forward_kwargs['tokenizer'] == 'bert-base-uncased':
            embeds = BertModel.from_pretrained("bert-base-uncased").embeddings.word_embeddings.weight
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif forward_kwargs['tokenizer'] == 'gpt2':
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            embeds = GPT2LMHeadModel.from_pretrained('gpt2').lm_head.weight
            embeds = torch.concat([embeds, torch.randn(1, embeds.shape[1], dtype=embeds.dtype, device=embeds.device)])            

        vocab = np.array([tokenizer.decode(i) for i in torch.arange(len(tokenizer))])
        embeds = embeds.detach().cpu().numpy()
        if inds is not None:
            vocab = vocab[inds]
            embeds = embeds[inds]
        
        norms = np.linalg.norm(embeds, axis=1, keepdims=True)
        embeds_normalized = embeds / norms
        
        print("Constructing nearest neighbor matrix...")
        k = forward_kwargs['knn']
        strong_masking = forward_kwargs['strong_masking']
        is_unused = ((strong_masking) * np.array([t.startswith("[") and t.endswith("]") for t in vocab])).astype(bool)
        is_suffix = ((strong_masking) * np.array([t.startswith("##") for t in vocab])).astype(bool)
        not_english = ((strong_masking) * (~np.array([all([x in english_alphabet for x in t]) for t in vocab]))).astype(bool)
        is_number = (np.array([any([x in np.arange(10).astype(str) for x in t]) for t in vocab])).astype(bool)
        masks = [is_unused, is_suffix, not_english, is_number]
        for mask in masks:
            if mask.sum() < k + 1:
                mask = (0 * mask).astype(bool)
        is_normal = ~np.any(masks, axis=0)
        masking = ([(is_number, is_number+is_normal), (is_normal, is_normal)] if not strong_masking
            else ([(m, m) for m in masks] + [(is_normal, is_normal)]))
        indices, similarities = get_knn(k, embeds_normalized,
            masking)

        print("Constructing sparse matrix...")
        row_indices = np.repeat(np.arange(embeds.shape[0]), k) 
        col_indices = indices.flatten()
        dot_products = similarities.flatten()
        # rates = distances.sum(-1)
        # assert (dot_products > 0).all()
        not_diag = row_indices != col_indices
        row_indices, col_indices, dot_products = row_indices[not_diag], col_indices[not_diag], dot_products[not_diag]
        # assert (row_indices != col_indices).all()
        if forward_kwargs['make_sym']:
            row_indices, col_indices = np.r_[row_indices, col_indices], np.r_[col_indices, row_indices]
            dot_products = np.r_[dot_products, dot_products]
        dot_products = np.exp((dot_products - 1) / forward_kwargs['bandwidth'])

        sparse_matrix = scipy.sparse.coo_array((dot_products, (row_indices, col_indices)),
                                  shape=(embeds.shape[0], embeds.shape[0]))
        sparse_matrix.sum_duplicates()
        L_off_diag = sparse_matrix.tocsr()
        L_diag = - L_off_diag.sum(-1)
        if forward_kwargs['normalize']:
            L_off_diag = scipy.sparse.coo_array(L_off_diag / (-L_diag)[:, None])
            L_diag = -1 + 0 * L_diag
        rate = - (L_diag.min()) / (1-gamma) 
        L_cpu = L_off_diag / rate + scipy.sparse.diags(L_diag / rate)
        K_cpu = L_off_diag / rate + scipy.sparse.diags(L_diag / rate + 1)
        
        L_cpu = L_cpu.tocoo()
        K_cpu = K_cpu.tocoo()
        K = torch.sparse_coo_tensor((K_cpu.row, K_cpu.col), K_cpu.data,
                                    size=(embeds.shape[0], embeds.shape[0])).float()
        K = K.to_sparse_csr()
        L = torch.sparse_coo_tensor((L_cpu.row, L_cpu.col), L_cpu.data,
                                    size=(embeds.shape[0], embeds.shape[0])).float()
        L = L.to_sparse_csr()
        print("Done!")
        return L, K, L_cpu, K_cpu, rate
    if forward_kwargs['type'] == "uniform":
        sparse_matrix = scipy.sparse.coo_array(([0], ([0], [0])),
                                  shape=(embeds.shape[0], embeds.shape[0]))
        L_cpu = sparse_matrix.tocsr()
        K_cpu = L_cpu
        L_cpu = L_cpu.tocoo()
        K_cpu = K_cpu.tocoo()
        K = torch.sparse_coo_tensor((K_cpu.row, K_cpu.col), K_cpu.data,
                                    size=(embeds.shape[0], embeds.shape[0])).float()
        K = K.to_sparse_csr()
        L = torch.sparse_coo_tensor((L_cpu.row, L_cpu.col), L_cpu.data,
                                    size=(embeds.shape[0], embeds.shape[0])).float()
        L = L.to_sparse_csr()
        return L, K, L_cpu, K_cpu, 0
