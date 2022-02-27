from typing import List

import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from deq.standard.lib import solvers



def collect_state(batch, model, solver, solver_args: solvers.SolverArgs, plot_func):
    # x_min, x_max, y_min, y_max = xy_range
    def scores_from_state(state):
        logits = model.model.out_trans(state)
        return logits
    model.eval()
    with torch.no_grad():
        phi_X = model.model.in_trans(batch)
        logits = scores_from_state(phi_X)
        
        plot_func(-1, {'logits': logits}, None)
        func = lambda z: model.core.f(z, phi_X)
        state = torch.zeros_like(phi_X)
        solver_run = solver(func, phi_X, solver_args)
        states = solver_run.gen(check_tol=False)
        prev_state = state
        for i, state in enumerate(states):
            logits = scores_from_state(state)
            plot_func(solver_run.k, {'logits': logits, 'state': state, 'prev_state':prev_state}, solver_run)
            prev_state = state
