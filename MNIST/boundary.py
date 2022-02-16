import dotenv
import hydra
import os
import importer
from typing import List, Tuple

import torch
from torch import Tensor
# from torchvision import functional as F
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib import cm as pltcm, colors
from matplotlib.colors import ListedColormap
import numpy as np

from src.datamodules import MnistDM
from deq.standard.models.core import DEQLayer
from deq.standard.lib import solvers
from src.models import LitModel
# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)
# just plot the dataset first
# cm = plt.cm.RdBu
cm = plt.get_cmap('Set1')
cm_bright = ListedColormap(["#FF0000", "#0000FF"])
# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")

def make_grid(based_image: torch.Tensor, pixels, h: float = 0.02):
    # vmin, vxmax = based_image.min().item(), based_image.max().item()
    vmin, vmax = 0,1.
    mesh = [np.arange(vmin, vmax, h), np.arange(vmin, vmax, h)]
    mesh = np.meshgrid(*mesh)
    X = torch.Tensor(np.stack([a.ravel() for a in mesh], axis=-1)).sub_(0.1307).div(0.3081)
    batch = based_image.repeat(X.shape[0], 1, 1, 1)
    batch[:, 0, pixels[0, 0], pixels[0, 1]] = X[:, 0]
    batch[:, 0, pixels[1, 0], pixels[1, 1]] = X[:, 1]
    return mesh, batch

def create_matrix(batch, pred_func, mesh_shape):
    Z, diff, nstep = pred_func(batch)
    idx = np.argmax(Z, axis=-1)
    score = Z[:, 5]
    score = score.reshape(mesh_shape)
    idx = idx.reshape(mesh_shape)
    return idx, score, diff, nstep

def decision_boundary(mesh, batch, model, solver, solver_args: solvers.SolverArgs, plot_states):
    # x_min, x_max, y_min, y_max = xy_range
    def scores_from_state(state):
        logits = model.model.out_trans(state)
        scores = torch.softmax(logits, dim=-1).cpu().numpy()
        return logits, scores
    model.eval()
    with torch.no_grad():
        phi_X = model.model.in_trans(batch)
        state = torch.zeros_like(phi_X)
        func = lambda z: model.model.core.f(z, phi_X)
        solver_run = solver(func, phi_X, solver_args)
        states = solver_run.gen()
        for i, state in enumerate(states):
            logits, score = scores_from_state(state)
            rel_diff = solver_run.trace_dict['rel'][-1]
            abs_diff = solver_run.trace_dict['abs'][-1]
            if solver_run.k in plot_states.keys():
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                preds = preds.reshape(mesh[0].shape)
                # rel_diff, abs_diff
                ax = plot_states[solver_run.k]
                ax.imshow(preds, cmap=cm, vmin=0, vmax=9)
                ax.text(0.05, 0.05, 'diff={:.4f},nstep={}'.format(rel_diff, solver_run.k), transform = ax.transAxes, va='center')
                # ax.set_xlim(mesh[0].min(), mesh[0].max())
                # ax.set_ylim(mesh[1].min(), mesh[1].max())
                ax.set_xticks(())
                ax.set_yticks(())
    # ax.contourf(mesh[0], mesh[1], score, cmap=cm, alpha=0.8)
    # idx = idx / 10
    # ax.contourf(mesh[0], mesh[1], idx, cmap=cm, alpha=0.8)

def load_data():
    datamodule = MnistDM(data_dir='data')
    datamodule.setup() 
    return datamodule

def load_model(ckpt):
    model = LitModel.load_from_checkpoint(ckpt)
    return model


def main(ckpt, pixel1, pixel2):
    datamodule = MnistDM(data_dir='data')
    datamodule.setup()
    based_image, target = datamodule.data_train[0]

    model = LitModel.load_from_checkpoint(ckpt)
    model.to(device)

    h = 0.02
    pixels = np.array([pixel1, pixel2])
    original_values = np.array([based_image[0, pixels[i, 0], pixels[i, 1]] for i in range(pixels.shape[0])])
    original_values = original_values * 0.3081 + 0.1307
    mesh, batch = make_grid(based_image, pixels,h=h)

    row, col = 3, 6
    figure = plt.figure(figsize=(col*6, row*6))

    # Plot based image
    based_image_ax = plt.subplot(row, col, 1)
    based_image_ax.imshow(based_image[0], cmap='gray')
    based_image_ax.scatter(pixels[:, 1], pixels[:, 0], color='red', s=40)

    # decision_boundary(mesh, batch, lambda X: pred(model, X), plt.subplot(row, col, 2))
    plot_states = {}
    for i, iter in enumerate([1, 9, 19, 39, 59, 69]):
        plot_states[iter] = plt.subplot(row, col, col + i + 1)
        plot_states[iter].scatter(original_values[0]/h, original_values[1]/h, color='red', s=40)
    decision_boundary(mesh, batch, model, solvers.ForwardRun, solvers.SolverArgs(threshold=80, stop_mode='rel', eps=1e-3), plot_states)

    plot_states = {}
    for i, iter in enumerate([9, 19, 39, 59, 69, 79]):
        plot_states[iter] = plt.subplot(row, col, 2*col + i + 1)
        plot_states[iter].scatter(original_values[0]/h, original_values[1]/h, color='red', s=40)
    decision_boundary(mesh, batch, model, solvers.AndersonRun, solvers.AndersonArgs(threshold=80, stop_mode='rel', eps=1e-3, m=6), plot_states)

    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(pltcm.ScalarMappable(colors.Normalize(vmin=0, vmax=9), cmap=cm),cax=cax)
    plt.tight_layout()
    figure.savefig(os.path.join(os.path.dirname(ckpt), f'decision_boundary_latestf_{str(pixels[0])}_{str(pixels[1])}_test'))
    plt.close('all')

if __name__ == "__main__":
    from glob import glob
    # ckpts=['logs/experiments/fractal_fp2_3_std=0.1_tanh_12345_forward_iteration/multiruns/2022-01-14/02-30-44/1/checkpoints/last.ckpt']
    # ckpts = glob("logs/experiments/fractal_mondeq2_*/**/checkpoints/last.ckpt", recursive=True)
    ckpts = glob("logs/experiments/fp_*/**/checkpoints/last.ckpt", recursive=True)
    for ckpt in ckpts:
        pixels = []
        while len(pixels) < 20:
            pixel = list(np.random.randint(0, high=28, size=2))
            if pixel not in pixels: pixels.append(pixel)
        for i in range(20):
            for j in range(i):
                main(ckpt, pixels[i], pixels[j])

