import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.animation as animation

from deq.standard.lib import solvers

# just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(["#FF0000", "#0000FF"])

def make_grid(h: float = 0.02, xy_range=(0, 1, 0, 1)):
    xmin, xmax, ymin, ymax = xy_range
    mesh = [np.arange(xmin, xmax, h), np.arange(ymin, ymax, h)]
    mesh = np.meshgrid(*mesh)
    X = torch.Tensor(np.stack([a.ravel() for a in mesh], axis=-1))
    return mesh, X

def plot_decision_bound(score, mesh, diff, nstep, ax):
    xmin, xmax, ymin, ymax = mesh[0].min(), mesh[0].max(), mesh[1].min(), mesh[1].max()
    boundary = ax.contourf(mesh[0], mesh[1], score, cmap=cm, alpha=0.8)
    ax.text(0.05, 0.05, 'diff={:.4f},nstep={}'.format(diff, nstep), transform = ax.transAxes, va='center')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(())
    ax.set_yticks(())
    return boundary

def plot_hidden_representation(x, y, z, c, ax):
    scat = ax.scatter(x, y, z, c=c)
    # self.ttl = self.ax.text(0.5, 1.05, '', transform = self.ax.transAxes, va='center')
    # ax_title = ax.set_title('State space')
    return scat


def state_generator(X, model, solver, solver_args):
    def scores_from_state(state):
        logits = model.model.out_trans(state)
        scores = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        return logits, scores
    norm_X = torch.norm(X, dim=1)
    yield np.c_[X[:, 0], X[:, 1]], norm_X
    model.eval()
    with torch.no_grad():
        phi_X = model.model.in_trans(X)
        yield phi_X[:, :3].detach().cpu().numpy(), norm_X

        state = torch.zeros_like(phi_X)
        logits, scores = scores_from_state(state)
        yield state[:, :3].detach().cpu().numpy(), scores, torch.argmax(logits, dim=-1), 0, 0, norm_X

        func = lambda z: model.model.core.f(z, phi_X)
        solver_run = solver(func, phi_X, solver_args)
        states = solver_run.gen()
        prev_state = state
        for i, state in enumerate(states):
            diff = solver_run.trace_dict[solver_run.args.stop_mode][-1]
            logits, score = scores_from_state(state)
            norm_gx = torch.norm(state - prev_state, dim=1)
            prev_state = state

            yield state[:, :3].detach().cpu().numpy(), score, torch.argmax(logits, dim=-1), i + 1, diff, norm_gx


class AnimatedDynamic(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, X, target, model, solver: solvers.SolverRun, solver_args: solvers.SolverArgs, xy_range=None):
        self.solver=solver
        self.solver_args = solver_args
        self.model = model
        # Setup the figure and axes...
        # self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig = plt.figure(figsize=(40, 20))
        row, col = 1, 4
        self.x_ax = self.fig.add_subplot(row, col, 1)
        self.phix_ax = self.fig.add_subplot(row, col, 2, projection='3d')
        self.ax = self.fig.add_subplot(row, col, 3, projection='3d')
        self.ax_boundary = self.fig.add_subplot(row, col, 4)

        # Then setup FuncAnimation.
        self.xy_range=xy_range
        if self.xy_range is None:
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            self.xy_range = (x_min, x_max, y_min, y_max)
        self.target = target
        self.mesh, self.batch = make_grid(xy_range = self.xy_range, h=0.02)
        self.stream = state_generator(self.batch, model, solver, solver_args)
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=1,
                                          init_func=self.setup_plot, blit=False)
        self._plane = None


    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        # Plot Input
        points, c = next(self.stream)
        x, y = points.T
        self.input_plot = self.x_ax.scatter(x,y,c=c)

        # Plot phi x
        phix_points, c = next(self.stream)
        x, y, z = phix_points.T
        self.phix_ax.scatter(x, y, z, c=c)

        # Plot decision surface
        w = (self.model.model.out_trans.weight[1] - self.model.model.out_trans.weight[0]).numpy()
        b = (self.model.model.out_trans.bias[1] - self.model.model.out_trans.bias[0]).numpy()
        xx, yy = self.mesh
        zz = (-b - xx*w[0] - yy*w[1]) / w[2]
        self.ax.plot_surface(xx, yy, zz, alpha=0.5)

        self.scat  = plot_hidden_representation(x, y, z, c, self.ax)
        self.ax_title = self.ax.set_title('State space')
        self.ax.axis(self.xy_range)
        self.ax.set_zlim3d(-1.5,1.5)

        score = np.zeros_like(self.mesh[0])
        self.boundary = plot_decision_bound(score, self.mesh, 0, 0, self.ax_boundary)
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.input_plot, self.scat, self.boundary

    def update(self, i):
        """Update the scatter plot."""
        points, score, c, iter_id, diff, norm_gx = next(self.stream)
        score = score.reshape(self.mesh[0].shape)
        self.input_plot.set_array(norm_gx)

        x, y, z = points.T
        self.ax_title.set_text(f'State space, iter {iter_id}, diff {diff}')
        self.scat._offsets3d = (x, y, z)
        self.scat.set_array(c)
        for c in self.boundary.collections:
            c.remove()  # removes only the contours, leaves the rest intact
        self.boundary = plt.contourf(self.mesh[0], self.mesh[1], score, cmap=cm, alpha=0.8)
        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.input_plot, self.scat, self.boundary