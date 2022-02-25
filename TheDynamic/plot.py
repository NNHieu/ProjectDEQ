import hydra
import os
from pathlib import Path
import importer

import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib import cm as pltcm, colors

import numpy as np

from src.datamodules import GMDataModule
import deq
from deq.standard.lib import solvers
from src.models import LitModel
from src.utils.animation import AnimatedDynamic, plot_decision_bound, plot_hidden_representation

# just plot the dataset first
cm = pltcm.ScalarMappable(colors.Normalize(vmin=0, vmax=1), cmap=plt.cm.RdBu)
# cm_bright = ListedColormap(["#FF0000", "#0000FF"])
# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")

def create_mesh(xy_range, z=None, h = 0.02):
    '''
        h: Step size in the mess
    '''
    x_min, x_max, y_min, y_max = xy_range
    mesh = [np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)]
    if z is not None:
        mesh.append(np.array([z]))
    mesh = np.meshgrid(*mesh)
    batch_mesh = torch.Tensor(np.stack([a.ravel() for a in mesh], axis=-1))
    return mesh, batch_mesh


def load_dm():
    datamodule = GMDataModule(dim=2, rho=0.2, data_dir='data')
    datamodule.setup()
    X, target = datamodule.data_train.tensors
    return X, target

def load_model(ckpt):
    model = LitModel.load_from_checkpoint(ckpt)
    return model


    

def viz_dynamic(ckpt):
    X, target = load_dm()
    model = load_model(ckpt)
    model = model.to(device)

    anim_dir = f'{os.path.dirname(ckpt)}/dynamics'
    if not os.path.exists(anim_dir):
        os.makedirs(anim_dir)
    
    # model.model.core.f_thres = 100
    for solver_name in ['anderson', 'forward']:
        if solver_name == 'anderson':
            solver = solvers.AndersonRun
            solver_args = solvers.AndersonArgs(threshold=100, stop_mode=model.model.core.stop_mode, eps=model.model.core.f_eps, m=6, beta=1.0)
            post_fix = f'm={solver_args.m}_beta={solver_args.beta}'
        elif solver_name == 'forward':
            solver = solvers.ForwardRun
            solver_args = solvers.SolverArgs(threshold=100, stop_mode=model.model.core.stop_mode, eps=model.model.core.f_eps)
            post_fix = ''
        else:
            raise ValueError
        
    anim = AnimatedDynamic(X, target, model, solver, solver_args)
    anim.ani.save(os.path.join(anim_dir, f'{solver_name}_{post_fix}.gif'), fps=1.0)

def collect_state_deq(model, batch_x, solver, solver_args, sfunc):
    model.eval()
    with torch.no_grad():
    # Collect trajectory
        def state2score(state):
            logits = model.model.out_trans(state)
            scores = torch.softmax(logits, dim=-1)[:, 1]
            return scores
        batch_x = batch_x.to(device)
        phix = model.model.in_trans(batch_x)
        state = torch.zeros_like(phix)
        func = lambda z: model.model.core.f(z, phix)
        solver_run = solver(func, state, solver_args)
        states = solver_run.gen()

        for state in states:
            scores = state2score(state)
            sfunc(state.cpu(), scores.cpu(), solver_run)

def plot_evol_z(ckpt):
    ckpt = Path(ckpt)
    save_dir = Path(ckpt.parent.joinpath('evol_z'))
    save_dir.mkdir(exist_ok=True)
    
    X, target = load_dm()
    model = load_model(ckpt)
    model = model.to(device)
    xy_range = (X[:,0].min(), X[:,0].max(), X[:,1].min(), X[:,1].max())
    mesh, batch_x = create_mesh(xy_range, h=0.01)

    row=2
    col=5
    figure = plt.figure(figsize=(col*6, row*6))

    plot_states = {}
    def plot_func(state, scores, solver_run):
        if solver_run.k in plot_states.keys():
            ax = plot_states[solver_run.k]
            ax.scatter(state[:,0], 
                        state[:,1], 
                        state[:,2],
                        c=cm.to_rgba(scores), cmap=cm)

    for i, step_idx in enumerate([1, 10, 30, 60, 100]):
        plot_states[step_idx] = figure.add_subplot(row, col, i + 1, projection='3d')
        plot_states[step_idx].set_title(f'FI Step {step_idx}')
    solver = solvers.ForwardRun
    solver_args = solvers.SolverArgs(threshold=101, stop_mode='rel', eps=-1)
    collect_state_deq(model, batch_x, solver, solver_args, plot_func)

    plot_states = {}
    for i, step_idx in enumerate([10, 30, 60, 100]):
        plot_states[step_idx] = figure.add_subplot(row, col, i + col + 2, projection='3d')
        plot_states[step_idx].set_title(f'AA Step {step_idx}')
    solver = solvers.AndersonRun
    solver_args = solvers.AndersonArgs(threshold=101, stop_mode='rel', eps=-1)
    collect_state_deq(model, batch_x, solver, solver_args, plot_func)


    plt.subplots_adjust(left=0.025, bottom=0.025, right=0.95, top=0.975)
    cax = plt.axes([0.975, 0.1, 0.01, 0.8])
    plt.colorbar(cm,cax=cax)

    figure.savefig(os.path.join(save_dir, 'v01'))
    plt.close('all')

if __name__ == "__main__":
    from glob import glob

    ckpts=['logs/experiments/fractal_fp2_4_std=0.01_tanh_12345_forward_iteration/checkpoints/last.ckpt']
    # ckpts = glob("logs/experiments/fractal_mondeq2_*/**/checkpoints/last.ckpt", recursive=True)
    for ckpt in ckpts:
        print(ckpt)
        plot_evol_z(ckpt)