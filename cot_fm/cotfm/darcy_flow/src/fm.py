import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import ot
from ot import bregman, emd

from scipy.optimize import linear_sum_assignment

from tqdm import tqdm
from functools import partial
import lightning as L
import matplotlib.pyplot as plt

from typing import Tuple
from jaxtyping import Int, Float
from torch import Tensor

from src.si import *
from src.util.ot import ot_matching, compute_wd_diff

def COT_empirical_resample(y0, u0, y1, u1, eps=1e-5, method='emd', reg=1e-1):
    C_y = ot.dist(y0.view(-1, 1), y1.view(-1, 1))
    C_u = ot.dist(u0.view(-1, 1), u1.view(-1, 1))
    C = eps*C_u + C_y
    ind = ot_matching(C, method=method, reg=reg)
    y1 = y1[ind]
    u1 = u1[ind]
    return y1, u1

def OT_empirical_resample(u0, u1, y1, method='emd', reg=1e-1):
    C = ot.dist(u0.view(-1, 1), u1.view(-1, 1))
    ind = ot_matching(C, method=method, reg=reg)
    y1 = y1[ind]
    u1 = u1[ind]
    return y1, u1

def trainer(model, 
            optimizer,
            train_loader, 
            val_loader = None, 
            scheduler=None,
            max_epochs=1000, 
            method='emd',
            eps=1e-8,
            sinkhorn_reg=1e-1,
            device='cuda',
            eval_freq=10,
            minibatch_cot=False,
            minibatch_ot=False,
            wandb_run=None,
            checkpoint_every=100,
            ):

    def training_step(batch):
        sample0, sample1 = batch
        loss = 0.

        y0, u0 = sample0
        y0, u0 = y0.to(device), u0.to(device)
        y1, u1 = sample1
        y1, u1 = y1.to(device), u1.to(device)

        b= y0.shape[0]
        if minibatch_cot == True:
            y1, u1 = COT_empirical_resample(y0, u0, y1, u1, eps=eps, method=method, reg=sinkhorn_reg)
        elif minibatch_ot == True:
            y1, u1 = OT_empirical_resample(u0, u1, y1, method=method, reg=sinkhorn_reg)
        t = torch.rand(b, device=y0.device)

        ut, z = model.simulate(t, u0, u1)
        # vt0 = model(t, ut, y0)
        # Only consider translation towards the target
        vt1 = model(t, ut, y1)
        bt = model.b(t, u0, u1, z)
        
        # loss = model.loss(bt, (vt0, vt1))
        loss = model.loss(bt, vt1)
        # Record loss
        if wandb_run is not None:
            wandb_run.log({
                'loss': loss.item(),
            })
        return loss

    loss_hist = []
    if val_loader is not None:
        val_loss_hist = []
    for epoch in tqdm(range(max_epochs)):
        for batch in train_loader:
            optimizer.zero_grad()
            loss = training_step(batch)
            loss_hist.append(loss.item())
            loss.backward()
            optimizer.step()
        if val_loader is not None and epoch % eval_freq == 0:
            val_loss = 0
            for batch in val_loader:
                val_loss += training_step(batch).item()
            val_loss_hist.append(val_loss/ len(val_loader[0]))
        if scheduler is not None:
            scheduler.step()
        if epoch % checkpoint_every == 0:
            torch.save(model.model.state_dict(), f'../trained_models/darcy/cot_ffm/si_{wandb_run.id}_{epoch}.pth')    
        
    out = {}
    out['loss_hist'] = loss_hist
    if val_loader is not None:
        out['val_loss_hist'] = val_loss_hist
    return out