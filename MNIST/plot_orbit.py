import math
from typing import List, Tuple

import os
import importer

import torch
from torch import Tensor
# from torchvision import functional as F
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib import cm as pltcm, colors
from matplotlib.colors import ListedColormap
import numpy as np

from src.datamodules import MnistDM, get_normalize_layer
from deq.standard.models.core import DEQLayer
from deq.standard.lib import solvers
from src.models import LitModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import yaml
from omegaconf import DictConfig

# just plot the dataset first
# cm = plt.cm.RdBu
# cm = pltcm.ScalarMappable(colors.Normalize(vmin=0, vmax=45), cmap=plt.get_cmap('Set1'))
cm = pltcm.ScalarMappable(colors.Normalize(vmin=0, vmax=9), cmap=plt.cm.RdBu)


# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")

def load_data():
    datamodule = MnistDM(data_dir='data', train_batch_size=200)
    datamodule.setup() 
    return datamodule

def load_model(ckpt):
    # configs = '''
    #     init_std: 0.001
    #     arch:
    #         h_features:
    #         - 64
    #         - 32
    #         f_solver: anderson
    #         f_thres: 30
    #         f_eps: 0.001
    #         b_solver: anderson
    #         b_thres: 30
    #         b_eps: 0.001
    #         stop_mode: rel
    #         num_layers: 5
    #         depth: 0
    #         core: deq
    #     lr: 0.001
    #     lr_schedule:
    #     - 120
    #     - 280
    #     lr_decay: step
    #     lr_factor: 0.2
    #     optimizer_name: adam
    #     compute_jac_loss: false
    #     spectral_radius_mode: false
    #     noise_sd: 0.0
    # '''
    # configs = yaml.safe_load(configs)
    # configs = DictConfig(configs)
    # model = LitModel(**configs)
    model = LitModel.load_from_checkpoint(ckpt)
    return model

def collect_state_deq(model, batch_x, batch_y, solver, solver_args):
    model.eval()
    with torch.no_grad():
    # Collect trajectory
        orbits = []
        def state2rep(state):
            state = model.model.out_trans[0](state)
            pooled_state = model.model.out_trans[1](state)
            flatten = model.model.out_trans[2](pooled_state)
            return flatten

        normalize_input = get_normalize_layer()
        batch_x = normalize_input(batch_x)
        phix = model.model.in_trans(batch_x)
        state = torch.zeros_like(phix)
        func = lambda z: model.model.core.f(z, phix)
        solver_run = solver(func, state, solver_args)
        states = solver_run.gen()

        for state in states:
            orbits.append(state2rep(state))
    return orbits

def collect_state_recur(model, batch_x, batch_y, iters=None):
    model.eval()
    with torch.no_grad():
    # Collect trajectory
        orbits = []
        def state2rep(state):
            pooled_state = model.model.out_trans[0](state)
            flatten = model.model.out_trans[1](pooled_state)
            return flatten

        norm_x = get_normalize_layer()(batch_x)
        phix = model.model.in_trans(norm_x)

        states = model.model.core.forward_generator(phix, iters=math.floor(model.model.core.iters*1.5))
        for state in states:
            orbits.append(state2rep(state))
    return orbits

def plot_orbit(orbits, batch, ax):
    batch_x, batch_y = batch
    batch_sz = batch_x.shape[0]

    len_orbits = len(orbits)
        
    # Dim Reduce
    orbits = torch.cat(orbits, dim=0).cpu().numpy()
    print(orbits.shape)
    pca = PCA(n_components=2)
    trans = pca.fit_transform(StandardScaler().fit_transform(orbits))
    print(trans.shape)
    trans = trans.reshape((len_orbits, batch_sz, -1)).transpose((1, 0, 2))
    print(trans.shape)
    
    # Plot
    for sample, c in zip(trans, batch_y):
        ax.plot(sample[:, 0], sample[:, 1], c=cm.to_rgba(c))

def main(ckpt):
    print('Plotting orbit for checkpoint:', ckpt)
    save_dir = f'{os.path.dirname(ckpt)}/orbit'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    datamodule = load_data()
    batch_x, batch_y = next(iter(datamodule.train_dataloader()))
    batch_sz = batch_x.shape[0]

    model = load_model(ckpt)
    
    if 'recur' in ckpt:
        orbits = collect_state_recur(model, batch_x, batch_y)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        solver = solvers.ForwardRun
        solver_args = solvers.SolverArgs(threshold=60, stop_mode='rel', eps=1e-3)
        orbits = collect_state_deq(model, batch_x, batch_y,  solver, solver_args, )
        plot_orbit(orbits, (batch_x, batch_y), axes[0])
        axes[0].set_title('Forward Run')

        solver = solvers.AndersonRun
        solver_args = solvers.AndersonArgs(threshold=60, stop_mode='rel', eps=1e-3)
        orbits = collect_state_deq(model, batch_x, batch_y,  solver, solver_args, )
        plot_orbit(orbits, (batch_x, batch_y), axes[1])
        axes[1].set_title('Anderson Run')

        fig.savefig(os.path.join(save_dir, 'v02'))
        plt.close('all')

def cal_variance(X):
    mean = np.mean(X, axis=0)
    ds = np.sqrt(np.sum((X - mean)**2, axis=1))
    return np.var(ds)

def perturb(ckpt):
    print('Plotting orbit for checkpoint:', ckpt)
    save_dir = f'{os.path.dirname(ckpt)}/orbit'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    datamodule = load_data()
    x, y = datamodule.data_train[np.random.randint(0, len(datamodule.data_train))]
    eps = torch.normal(0, 0.1, (30, x.shape[0], x.shape[1], x.shape[2]))
    batch_x = x + eps
    batch_sz = batch_x.shape[0]
    batch_y = torch.ones(batch_sz)*y

    model = load_model(ckpt)
    
    if 'recur' in ckpt:
        orbits = collect_state_recur(model, batch_x, batch_y)
        iters = model.model.core.iters
    else:
        orbits = collect_state_deq(model, batch_x, batch_y)
        iters=45

    len_orbits = len(orbits)
        
    # Dim Reduce
    orbits = torch.cat(orbits, dim=0).cpu().numpy()
    print(orbits.shape)
    pca = PCA(n_components=2)
    trans = pca.fit_transform(StandardScaler().fit_transform(orbits))
    print(trans.shape)
    trans = trans.reshape((len_orbits, batch_sz, -1))
    print(trans.shape)
    
    # Cal varience
    with open(os.path.join(save_dir, 'var'), 'w') as f: 
        variance = []
        print(f'Step - Var', file=f)
        for step_idx, b in enumerate(trans):
            variance.append(cal_variance(b))
            print(f'{step_idx} - {variance[-1]}', file=f)
    
    # Plot
    trans = trans.transpose((1, 0, 2))
    print(trans.shape)
    for i, (sample, c) in enumerate(zip(trans, batch_y)):
        plt.plot(sample[:, 0], sample[:, 1], c=cm.to_rgba(c))
        plt.plot(sample[:iters + 1, 0], sample[:iters + 1, 1], c=cm.to_rgba(i))
    
        # plt.plot(sample[iters + 1:, 0], sample[iters + 1:, 1], c=cm.to_rgba(c + 1))

    plt.savefig(os.path.join(save_dir, 'perturb_over'))
    print(tuple())
    plt.close('all')

if __name__ == "__main__":
    from glob import glob
    # ckpts=['logs/asd.last']
    # ckpts = glob("logs/experiments/fractal_mondeq2_*/**/checkpoints/last.ckpt", recursive=True)
    ckpts = glob("logs/experiments/fp[16*/**/checkpoints/last.ckpt", recursive=True)
    # ckpts = glob("logs/experiments/recur*_0.0_*/**/checkpoints/last.ckpt", recursive=True)
    for ckpt in ckpts:
        # main(ckpt)
        main(ckpt)

