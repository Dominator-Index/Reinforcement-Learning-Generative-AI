import torch
import numpy as np

import ot
from torchcfm.utils import *
from torchdyn.core import NeuralODE

def w2_dist(X, Y, numItermax=1_000_000):
    # Computes W_2 distance. Note ot.emd2 computes squared w2 distance.

    n = X.shape[0]
    a, b = np.ones((n,)) / n, np.ones((n,)) / n

    M = ot.dist(X, Y)
    W = ot.emd2(a, b, M, numItermax=numItermax)

    return np.sqrt(W)


def eval_model(model, source_data, target_data):
    # Evaluates the W2 distance between samples of a FM model and testing samples
    node = NeuralODE(
        torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
    )
    with torch.no_grad():
        traj = node.trajectory(
            source_data,
            t_span=torch.linspace(0, 1, 100, device=source_data.device),
        )

        samples = traj[-1].cpu().numpy()
        target_data = target_data.cpu().numpy()

    w2 = w2_dist(target_data, samples)
    return w2

class StandardScaler:
    
    def __init__(self, mean=None, std=None, eps=1e-7):
        self.mean = mean
        self.std = std
        self.eps = eps
        
    def fit(self, data):
        # data: [batch, ...]
        self.mean = torch.mean(data, axis=0)
        self.std = torch.std(data, axis=0)
        
    def transform(self, data):
        return (data - self.mean) / (self.std + self.eps)
    
    def inverse_transform(self, data):
        return (self.std + self.eps) * data + self.mean