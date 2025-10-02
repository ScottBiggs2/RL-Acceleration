import torch
import numpy as np

def rank_neurons(acts, cav):
    # acts: [n_samples, hidden_dim]
    projections = torch.matmul(acts, cav)
    scores = projections.abs().mean(dim=0)  # neuron importance
    ranked = torch.argsort(scores, descending=True)
    return ranked, scores
