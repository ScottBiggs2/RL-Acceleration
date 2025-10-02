import torch

def compute_cav(pos_acts, neg_acts):
    # pos_acts, neg_acts: [n_samples, hidden_dim]
    pos_mean = pos_acts.mean(dim=0)
    neg_mean = neg_acts.mean(dim=0)
    cav = pos_mean - neg_mean
    cav = cav / cav.norm()  # normalize
    return cav
