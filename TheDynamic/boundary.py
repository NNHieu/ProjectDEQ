import dotenv
import hydra
import os
import importer

import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from src.datamodules import GMDataModule
import deq
from deq.standard.lib import solvers
from src.models import LitModel
from src.utils.animation import AnimatedDynamic

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)



# just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(["#FF0000", "#0000FF"])
# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")

def create_matrix(pred_func, z=None, h = 0.02, xy_range=None):
    '''
        h: Step size in the mess
    '''
    x_min, x_max, y_min, y_max = xy_range
    mesh = [np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)]
    if z is not None:
        mesh.append(np.array([z]))
    mesh = np.meshgrid(*mesh)
    X = torch.Tensor(np.stack([a.ravel() for a in mesh], axis=-1))
    Z, diff, nstep = pred_func(X)
    Z = Z.reshape(mesh[0].shape)
    return mesh, Z, diff, nstep

def decision_bound(X, target, pred_func, ax, h = 0.02, xy_range=None):
    if xy_range is None:
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xy_range = (x_min, x_max, y_min, y_max)
    mesh, Z, diff, nstep = create_matrix(pred_func, h=h, xy_range=xy_range)

    i = 1
    ax.contourf(mesh[0], mesh[1], Z, cmap=cm, alpha=0.8)

    # Plot the training points
    ax.scatter(
        X[:, 0], X[:, 1], c=target, cmap=cm_bright, edgecolors="k"
    )
    ax.text(0.05, 0.05, 'diff={:.4f},nstep={}'.format(diff, nstep), transform = ax.transAxes, va='center')
    ax.set_xlim(mesh[0].min(), mesh[0].max())
    ax.set_ylim(mesh[1].min(), mesh[1].max())
    ax.set_xticks(())
    ax.set_yticks(())

def pred_no_core(model, X):
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        phi_X = model.model.in_trans(X)
        logits = model.model.out_trans(phi_X)
        Z = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return Z, 0, 0

def basic_pred(model, X):
    model.eval()
    solver = model.model.core.mon
    with torch.no_grad():
        X = X.to(device)
        logits, _, _ = model(X)
        Z = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        diff = solver.stats.fwd_err.avg
        nstep = solver.stats.fwd_iters.avg
    return Z, diff, nstep

def pred(model, solver, X, thres, stop_mode, eps, latest=False):
    model.eval()
    # target = torch.ones(X.shape[0])
    with torch.no_grad():
        X = X.to(device)
        phi_X = model.model.in_trans(X)
        state = torch.zeros_like(phi_X)
        func = lambda z: model.model.core.f(z, phi_X)
        result = solver(func, state, threshold=thres, stop_mode=stop_mode, name="forward", eps=eps)
        if not latest:
            state = result['result']
        else:
            state = result['latest']
        diff = result['lowest']
        nstep = result['nstep']
        logits = model.model.out_trans(state)
        Z = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return Z, diff, nstep

def pred_mon(model, solver, X, thres, stop_mode, eps):
    model.eval()
    # target = torch.ones(X.shape[0])
    orig_max_iter = solver.max_iter
    orig_tol = solver.tol
    solver.max_iter = thres
    solver.tol = eps
    with torch.no_grad():
        X = X.to(device)
        phi_X = model.model.in_trans(X)
        state = torch.zeros_like(phi_X)
        func = lambda z: model.model.core.f(z, phi_X)
        state = solver(state.view(state.shape[0], -1))[-1]
        diff = solver.stats.fwd_err.avg
        nstep = solver.stats.fwd_iters.avg
        logits = model.model.out_trans(state)
        Z = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    solver.max_iter = orig_max_iter
    solver.tol = orig_tol
    return Z, diff, nstep

def pred_batch(model, solver, X, thres, stop_mode, eps, bs=50, latest=False):
    model.eval()
    # target = torch.ones(X.shape[0])
    Z = []
    ds = TensorDataset(X)
    dl = DataLoader(ds, batch_size=bs, shuffle=False)
    diff = 0
    nstep = 0
    with torch.no_grad():
        for x in dl:
            phi_X = model.model.in_trans(x[0].to(device))
            state = torch.zeros_like(phi_X)
            func = lambda z: model.model.core.f(z, phi_X)
            result = solver(func, state, threshold=thres, stop_mode=stop_mode, name="forward", eps=eps)
            if not latest:
                state = result['result']
            else:
                state = result['latest']
            diff += result['lowest']
            nstep += result['nstep']
            logits = model.model.out_trans(state)
            Z.append(torch.softmax(logits, dim=1)[:, 1].cpu())
    diff /= len(Z)
    nstep /= len(Z)
    Z = torch.cat(Z, dim=0).numpy()
    return Z, diff, nstep

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


def main(ckpt):
    X, target = load_dm()
    model = load_model(ckpt)
    model = model.to(device)
    

    row=1
    col=6
    figure = plt.figure(figsize=(col*6, row*6))
    decision_bound(X, target, lambda X: pred_no_core(model, X), plt.subplot(row, col, 1))
    for i, iters in enumerate([1, 10, 30, 50, 70]):
        ax = plt.subplot(row, col, i + 2)
        ax.set_title(f'MONPeacemanRachford MaxIter {iters}')
        # decision_bound(X, target,
        #                lambda X: pred(model, deq.lib.solvers.forward_iteration, X, iter, 'rel', 1e-5, latest=True),
        #                ax)
        solver = model.model.core.mon
        orig_max_iter = solver.max_iter
        orig_tol = solver.tol
        solver.max_iter = iters
        solver.tol = 1e-5
        decision_bound(X, target,
                       lambda X: basic_pred(model, X),
                       ax)
        solver.stats.reset()
        solver.max_iter = orig_max_iter
        solver.tol = orig_tol
    ax_row = col + 1
    # for i, iter in enumerate([10, 30, 50, 70]):
    #     ax = plt.subplot(row, col, i + 1 + ax_row)
    #     ax.set_title(f'Anderson MaxIter {iter}')
    #     decision_bound(X, target, lambda X: pred(model, deq.lib.solvers.anderson, X, iter, 'rel', 1e-5),
    #                     ax)
    # ax_row += col
    # for i, eps in enumerate([1e-1, 1e-2, 1e-3, 1e-4, 1e-5]):
    #     ax = plt.subplot(row, col, i + ax_row + 1)
    #     ax.set_title(f'Anderson Tol {eps}')
    #     decision_bound(X, target, lambda X: pred(model, deq.lib.solvers.anderson, X, 500, 'rel', eps),
    #                     ax)

    plt.tight_layout()
    figure.savefig(os.path.join(os.path.dirname(ckpt), 'decision_boundary_latestf'))
    plt.close('all')

if __name__ == "__main__":
    from glob import glob

    # ckpts=['logs/experiments/fractal_fp2_3_std=0.1_tanh_12345_forward_iteration/multiruns/2022-01-14/02-30-44/1/checkpoints/last.ckpt']
    ckpts = glob("logs/experiments/fractal_mondeq2_*/**/checkpoints/last.ckpt", recursive=True)
    for ckpt in ckpts:
        print(ckpt)
        main(ckpt)
