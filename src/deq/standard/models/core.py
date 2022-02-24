import logging
import time

import torch
from icecream import ic
from torch import autograd, nn

from ..lib.jacobian import jac_loss_estimate, power_method
from ..lib.solvers import anderson, broyden, forward_iteration, AndersonRun, ForwardRun
from deq.shared.stats import SolverStats

logger = logging.getLogger(__name__)

class DEQLayer(nn.Module):
    def __init__(self, f, conf):
        super(DEQLayer, self).__init__()
        self.f = f
        self.parse_cfg(conf)
        self.save_result = False
        self.stats = SolverStats()

    def parse_cfg(self, cfg):
        """
        Parse a configuration file
        """
        self.num_layers = cfg["num_layers"]
        # DEQ related
        self.f_solver = eval(cfg["f_solver"])
        self.b_solver = eval(cfg.get("b_solver", "None"))
        if self.b_solver is None:
            self.b_solver = self.f_solver
        self.f_thres = cfg["f_thres"]
        self.b_thres = cfg["b_thres"]
        self.stop_mode = cfg["stop_mode"]
        self.f_eps = cfg.get("f_eps", 1e-3)
        self.b_eps = cfg.get("f_eps", 1e-3)

    def forward(
        self,
        x,
        deq_mode=True,
        compute_jac_loss=False,
        spectral_radius_mode=False,
        backward: bool = True,
        **kwargs
    ):
        start = time.time()
        # ----------------Setting up-------------------------------
        bsz = x.shape[0]

        f_thres = kwargs.get("f_thres", self.f_thres)
        b_thres = kwargs.get("b_thres", self.b_thres)
        f_eps = kwargs.get("f_eps", self.f_eps)
        b_eps = kwargs.get("b_eps", self.b_eps)

        func = lambda z: self.f(z, x)
        jac_loss = torch.tensor(0.0).to(x)
        sradius = torch.zeros(bsz, 1).to(x)
        # deq_mode = (train_step < 0) or (train_step >= self.pretrain_steps)

        z1 = torch.zeros_like(x)
        # ----------------Computation part-------------------------------
        if not deq_mode:
            for layer_ind in range(self.num_layers):
                z1 = func(z1)
            new_z1 = z1
            if self.training and compute_jac_loss:
                    z2 = z1.clone().detach().requires_grad_()
                    new_z2 = func(z2)
                    jac_loss = jac_loss_estimate(new_z2, z2)
        else:
            with torch.no_grad():
                result = self.f_solver(
                    func, z1, threshold=f_thres, stop_mode=self.stop_mode, name="forward", eps=f_eps
                )
                z1 = result["result"]
                self.stats.fwd_iters.update(result['nstep'])
                self.stats.fwd_err.update(result['lowest'])
            new_z1 = z1

            if self.training and backward:
                new_z1 = func(z1.requires_grad_())
                if compute_jac_loss: jac_loss = jac_loss_estimate(new_z1, z1)

                def backward_hook(grad):
                    start = time.time()
                    if self.hook is not None:
                        self.hook.remove()
                        torch.cuda.synchronize()
                    result = self.b_solver(
                        lambda y: autograd.grad(new_z1, z1, y, retain_graph=True)[0] + grad,
                        torch.zeros_like(grad),
                        threshold=b_thres,
                        stop_mode=self.stop_mode,
                        name="backward",
                        eps=b_eps
                    )
                    r = result["result"]
                    self.stats.bkwd_iters.update(result['nstep'])
                    self.stats.bkwd_err.update(result['lowest'])
                    self.stats.bkwd_time.update(time.time() - start)
                    return r

                self.hook = new_z1.register_hook(backward_hook)
            elif spectral_radius_mode:
                with torch.enable_grad():
                    new_z1 = func(z1.requires_grad_())
                _, sradius = power_method(new_z1, z1, n_iters=150)
        self.stats.fwd_time.update(time.time() - start)
        return new_z1, jac_loss.view(1, -1), sradius.view(-1, 1)


class DEQLayer2(nn.Module):
    def __init__(self, f, conf):
        super(DEQLayer2, self).__init__()
        self.f = f
        self.parse_cfg(conf)
        self.save_result = False
        self.stats = SolverStats()

    def parse_cfg(self, cfg):
        """
        Parse a configuration file
        """
        self.num_layers = cfg["num_layers"]
        # DEQ related
        self.f_solver = eval(cfg["f_solver"])
        self.b_solver = eval(cfg.get("b_solver", "None"))
        if self.b_solver is None:
            self.b_solver = self.f_solver
        self.f_thres = cfg["f_thres"]
        self.b_thres = cfg["b_thres"]
        self.stop_mode = cfg["stop_mode"]
        self.f_eps = cfg.get("f_eps", 1e-3)
        self.b_eps = cfg.get("f_eps", 1e-3)

    def forward(
        self,
        x,
        deq_mode=True,
        compute_jac_loss=False,
        spectral_radius_mode=False,
        backward: bool = True,
        **kwargs
    ):
        start = time.time()
        # ----------------Setting up-------------------------------
        bsz = x.shape[0]

        f_thres = kwargs.get("f_thres", self.f_thres)
        b_thres = kwargs.get("b_thres", self.b_thres)
        f_eps = kwargs.get("f_eps", self.f_eps)
        b_eps = kwargs.get("b_eps", self.b_eps)

        func = lambda z: self.f(z, x)
        jac_loss = torch.tensor(0.0).to(x)
        sradius = torch.zeros(bsz, 1).to(x)
        # deq_mode = (train_step < 0) or (train_step >= self.pretrain_steps)

        z1 = torch.zeros_like(x)
        # ----------------Computation part-------------------------------
        if not deq_mode:
            for layer_ind in range(self.num_layers):
                z1 = func(z1)
            new_z1 = z1
            if self.training and compute_jac_loss:
                    z2 = z1.clone().detach().requires_grad_()
                    new_z2 = func(z2)
                    jac_loss = jac_loss_estimate(new_z2, z2)
        else:
            with torch.no_grad():
                result = self.f_solver(
                    func, z1, threshold=f_thres, stop_mode=self.stop_mode, name="forward", eps=f_eps
                )
                z1 = result["result"]
                self.stats.fwd_iters.update(result['nstep'])
                self.stats.fwd_err.update(result['lowest'])
            new_z1 = z1

            if self.training and backward:
                new_z1 = func(z1.requires_grad_())
                if compute_jac_loss: jac_loss = jac_loss_estimate(new_z1, z1)

                def backward_hook(grad):
                    start = time.time()
                    if self.hook is not None:
                        self.hook.remove()
                        torch.cuda.synchronize()
                    result = self.b_solver(
                        lambda y: autograd.grad(new_z1, z1, y, retain_graph=True)[0] + grad,
                        torch.zeros_like(grad),
                        threshold=b_thres,
                        stop_mode=self.stop_mode,
                        name="backward",
                        eps=b_eps
                    )
                    r = result["result"]
                    self.stats.bkwd_iters.update(result['nstep'])
                    self.stats.bkwd_err.update(result['lowest'])
                    self.stats.bkwd_time.update(time.time() - start)
                    return r

                self.hook = new_z1.register_hook(backward_hook)
            elif spectral_radius_mode:
                with torch.enable_grad():
                    new_z1 = func(z1.requires_grad_())
                _, sradius = power_method(new_z1, z1, n_iters=150)
        self.stats.fwd_time.update(time.time() - start)
        return new_z1, jac_loss.view(1, -1), sradius.view(-1, 1)

class RecurLayer(nn.Module):
    def __init__(self, block, iters):
        super(RecurLayer, self).__init__()
        self.f = block
        self.iters=iters
        self.stats = SolverStats()

    def forward(self, x, iters=None, **kwargs):
        if iters is None:
            iters = self.iters
        # self.rel = []
        out = torch.zeros_like(x)
        for i in range(iters):
            out = self.f(out, x)
        return out, None, None
    
    def forward_generator(self, x, iters=None, **kwargs):
        if iters is None:
            iters = self.iters
        # self.rel = []
        out = torch.zeros_like(x)
        yield out
        for i in range(iters):
            out = self.f(out, x)
            yield out
        return out, None, None
