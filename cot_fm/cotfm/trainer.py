import sys
sys.path.append('../')
sys.path.append('./')

import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
from sklearn.model_selection import ParameterGrid
from argparse import ArgumentParser
import itertools

import ot
from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.utils import *
from torchdiffeq import odeint
from torchdyn.core import NeuralODE

from models.mlp import TriangularMLP
from util import eval_model, StandardScaler
from data.synthetic import SyntheticData
from data.lotka_volterra import LotkaVolterraData
from cot_flow_matching import BatchCOTConditionalFlowMatcher, FullCOTConditionalFlowMatcher



def train_cotfm(hp, dataloader_source, dataloader_target, source_te, target_te, u_dim, y_dim,
                cotmode='batch', condition_coordinates=[1], device='cuda'):
    print(f'Hyperparameters: {hp}')
    print(f'Device: {device}')

    model_cot = TriangularMLP(condition_coords=condition_coordinates, u_dim=u_dim, y_dim=y_dim, 
                              w=hp['width'], layers=hp['layers'], time_varying=True)
    model_cot = model_cot.to(device)
        
    optimizer = torch.optim.Adam(model_cot.parameters(), lr=hp['lr'])

    if cotmode == 'batch':
        FM = BatchCOTConditionalFlowMatcher(condition_coordinates=condition_coordinates,
                                            eps=hp['eps'],
                                            sigma=hp['sigma'])
        
        # Wrap dataloaders in a cycle to allow for infinite looping
        dataloader_source = itertools.cycle(dataloader_source)
        dataloader_target = itertools.cycle(dataloader_target)

    elif cotmode == 'full':
        print(f'Fitting COT plan...')
        source_data = dataloader_source.dataset.tensors[0]
        target_data = dataloader_target.dataset.tensors[0]
        FM = FullCOTConditionalFlowMatcher(source_data, target_data,
                                    condition_coordinates=condition_coordinates,
                                    eps=hp['eps'],
                                    sigma=hp['sigma'],
                                    numitermax=1_000_000)
        
    u_coords = [k for k in range(u_dim + y_dim) if k not in condition_coordinates]

    print('Starting training...')
    start = time.time()
    best_w2 = np.inf
    best_w2_k = 0
    for k in range(100_000):
        optimizer.zero_grad()

        if cotmode == 'batch':
            x0 = next(dataloader_source)[0]
            x1 = next(dataloader_target)[0]
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
        elif cotmode == 'full':
            t, xt, ut = FM.sample_location_and_conditional_flow(hp['batch_size'], return_x0=False)

        t = t.to(device)
        xt = xt.to(device)
        ut = ut.to(device)

        vt = model_cot(torch.cat([xt, t[:, None]], dim=-1))
        loss = torch.mean((vt[:, u_coords] - ut[:, u_coords]) ** 2)

        loss.backward()
        optimizer.step()

        if (k + 1) % 5000 == 0:
            end = time.time()
            w2 = eval_model(model_cot, source_te.to(device), target_te)
            if w2 < best_w2:
                best_w2 = w2
                best_w2_k = k+1
                torch.save(model_cot, f"{savedir}/cot_fm_{k+1}_{hp['eps']}_{hp['sigma']}_{hp['batch_size']}_{hp['width']}_{hp['lr']}.pt")
            print(f"{k+1}: loss {loss.item():0.3f} w2 {w2:0.6f} time {(end - start):0.2f} | best w2 {best_w2:0.6f} @ {best_w2_k}")

            start = end

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--savedir', type=str)
    parser.add_argument('--dataset', type=str, choices=['lv', '2moons', 'circles', 'checkerboard', 'swissroll'])
    parser.add_argument('--cotmode', type=str, choices=['full', 'batch'])
    args = parser.parse_args()

    savedir = args.savedir
    device = torch.device('cuda')

    os.makedirs(savedir, exist_ok=True)

    print(f'\nDataset: {args.dataset}')
    print(f'COT Mode: {args.cotmode}')

    print('Loading data...')
    if args.dataset == 'lv':
        dataset = LotkaVolterraData() # Note: in (y, u) format
        scaler = StandardScaler()

        n_tr = 10_000
        n_te = 5_000
        target_data = dataset.sample_target(n_tr, log=True, scaler=None)  
        scaler.fit(target_data)
        target_data = scaler.transform(target_data)
        source_data = dataset.sample_source(n_tr, log=True, scaler=scaler)
        target_te = dataset.sample_target(n_te, log=True, scaler=scaler)
        source_te = dataset.sample_source(n_te, log=True, scaler=scaler)

        condition_coords = list(range(0, 22))
        u_dim = 4
        y_dim = 22
    else:
        dataset = SyntheticData(args.dataset)  # Note: in (u, y) format

        n_tr = 20_000 
        n_te = 5_000
        target_data = dataset.sample_target(n_tr)
        source_data = dataset.sample_source(n_tr, noise='gaussian')
        target_te = dataset.sample_target(n_te)
        source_te = dataset.sample_source(n_te, noise='gaussian')

        condition_coords = [1]
        u_dim = 1
        y_dim = 1

    hp_grid = {
        'eps': [1e-6, 1e-4, 1e-2, 1e-1],
        'sigma': [0.001, 0.01, 0.1, 0.5],
        'batch_size': [256, 512, 1024],
        'width': [256, 512, 1024, 2048],
        'lr': [1e-4, 3e-4, 7e-4, 1e-3],
        'layers': [4, 6, 8]  
        }
    grid = ParameterGrid(hp_grid)
    grid = list(grid)
    np.random.shuffle(grid)
    grid = grid[:100]

    for k, hp in enumerate(grid):
        print(f'\nRun: {k+1}/100')

        dataloader_source = DataLoader(TensorDataset(source_data), batch_size=hp['batch_size'])
        dataloader_target = DataLoader(TensorDataset(target_data), batch_size=hp['batch_size'])
        
        train_cotfm(hp, dataloader_source, dataloader_target, 
                    source_te, target_te, u_dim, y_dim,
                    cotmode=args.cotmode, device=device,
                    condition_coordinates=condition_coords)
