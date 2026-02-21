import torch
import torch.nn as nn

def sparse_gpt_layer(weight, inputs, sparsity, damp=0.01):
    out_f, in_f = weight.shape
    X = inputs
    H = X.T @ X
    damp_val = damp * torch.mean(torch.diag(H))
    H = H + damp_val * torch.eye(in_f, device=H.device)
    H_inv = torch.linalg.inv(H)
    diag_H_inv = torch.diag(H_inv)

    W_new = weight.clone()
    num_prune = int(sparsity * in_f)
    if num_prune == 0:
        return W_new

    for i in range(out_f):
        w = W_new[i, :]
        importance = (w ** 2) / (diag_H_inv ** 2 + 1e-8)
        sorted_idx = torch.argsort(importance)
        prune_idx = sorted_idx[:num_prune]

        old_vals = w[prune_idx].clone()
        w[prune_idx] = 0

        correction = torch.zeros(in_f, device=w.device)
        for idx_in_list, j in enumerate(prune_idx):
            factor = old_vals[idx_in_list] / H_inv[j, j]
            correction -= factor * H_inv[:, j]

        remaining_mask = torch.ones(in_f, dtype=torch.bool, device=w.device)
        remaining_mask[prune_idx] = False
        w[remaining_mask] += correction[remaining_mask]

        W_new[i, :] = w
    return W_new