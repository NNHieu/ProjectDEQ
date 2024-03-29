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
cm = pltcm.ScalarMappable(colors.Normalize(vmin=0, vmax=9), cmap=plt.get_cmap('Set1'))
# cm = pltcm.ScalarMappable(colors.Normalize(vmin=0, vmax=9), cmap=plt.cm.RdBu)


# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")

def load_data():
    datamodule = MnistDM(data_dir='data', train_batch_size=400)
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
    '''
    return: Tensor - num_step x bs x features
    '''
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
    orbits = torch.stack(orbits, dim=0)
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


def stats_variance(orbits: torch.Tensor):
    '''
    input: # num_step x bs x features
    return: Tensor - num_step
    '''
    means = torch.mean(orbits, dim=1)
    diff = orbits - means[:, None, :] # num_step x bs x features
    dist = torch.norm(diff, dim=-1) # num_step x bs
    var = torch.var(dist, dim=-1)
    return var # num_step

def reduce_dim(orbits: torch.Tensor, n_components: int):
    '''
    ----------------
    PCA
    input: Tensor - num_step x bs x features
    return: orbits with shape # num_step x bs x n_components
    '''
    len_orbits, batch_sz, nfeatures = orbits.shape
    # Dim Reduce
    orbits = orbits.view(-1, nfeatures)
    pca = PCA(n_components=n_components)
    trans = pca.fit_transform(StandardScaler().fit_transform(orbits))
    trans = trans.reshape((len_orbits, batch_sz, -1)) # num_step x bs x features
    return trans

def plot_orbit(orbits, batch, ax):
    batch_x, batch_y = batch
    batch_sz = batch_x.shape[0]
    trans = trans.transpose((1, 0, 2))
    print(trans.shape)
    
    # Plot
    for sample, c in zip(trans, batch_y):
        ax.plot(sample[:, 0], sample[:, 1], c=cm.to_rgba(c))

def plot_hidden_z(batch_z, c, ax):
    ax.scatter(batch_z[:, 0], batch_z[:, 1], batch_z[:, 2], c=c)


def main(ckpt):
    print('Plotting orbit for checkpoint:', ckpt)
    save_dir = f'{os.path.dirname(ckpt)}/orbit'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    datamodule = load_data()
    batch_x, batch_y = next(iter(datamodule.train_dataloader()))
    print(batch_x.shape, batch_y.shape)
    batch_sz = batch_x.shape[0]

    model = load_model(ckpt)
    
    if 'recur' in ckpt:
        orbits = collect_state_recur(model, batch_x, batch_y)
    else:
        plot_steps = [5, 15, 30, 45, 60]
        nrows, ncols = 2, len(plot_steps)
        fig = plt.figure(figsize=(7*ncols, 7*nrows))
        ax_idx = 1

        solver = solvers.ForwardRun
        solver_args = solvers.SolverArgs(threshold=60, stop_mode='rel', eps=1e-3)
        orbits = collect_state_deq(model, batch_x, batch_y,  solver, solver_args) # num_step x bs x features
        # step_variences = stats_variance(orbits)
        print(orbits.shape)
        orbits = reduce_dim(orbits, 3)
        print(orbits.shape)

        for step_idx in plot_steps:
            ax = fig.add_subplot(nrows, ncols, ax_idx, projection='3d')
            plot_hidden_z(orbits[step_idx - 1], cm.to_rgba(batch_y), ax)
            ax.set_title(f'FI step={step_idx}')
            ax_idx += 1

        # plot_orbit(orbits, (batch_x, batch_y), axes[0])

        solver = solvers.AndersonRun
        solver_args = solvers.AndersonArgs(threshold=60, stop_mode='rel', eps=1e-3)
        orbits = collect_state_deq(model, batch_x, batch_y,  solver, solver_args, )
        orbits = reduce_dim(orbits, 3)
        for step_idx in plot_steps:
            ax = fig.add_subplot(nrows, ncols, ax_idx, projection='3d')
            plot_hidden_z(orbits[step_idx - 1], cm.to_rgba(batch_y), ax)
            ax.set_title(f'AA step={step_idx}')
            ax_idx += 1

        fig.savefig(os.path.join(save_dir, 'v02_1'))
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

