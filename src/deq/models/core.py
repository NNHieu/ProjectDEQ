import logging
import os
import sys

import torch
from icecream import ic
from torch import autograd, nn

from ..lib.jacobian import jac_loss_estimate, power_method
from ..lib.solvers import anderson, broyden

logger = logging.getLogger(__name__)


class DEQLayer(nn.Module):
    def __init__(self, f, conf):
        super(DEQLayer, self).__init__()
        self.f = f
        self.parse_cfg(conf)
        self.save_result = False

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

    def forward(
        self,
        x,
        deq_mode=True,
        compute_jac_loss=False,
        spectral_radius_mode=False,
        writer=None,
        **kwargs
    ):
        # ----------------Setting up-------------------------------
        bsz = x.shape[0]

        f_thres = kwargs.get("f_thres", self.f_thres)
        b_thres = kwargs.get("b_thres", self.b_thres)

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

            if self.training:
                if compute_jac_loss:
                    z2 = z1.clone().detach().requires_grad_()
                    new_z2 = func(z2)
                    jac_loss = jac_loss_estimate(new_z2, z2)
        else:
            with torch.no_grad():
                result = self.f_solver(
                    func, z1, threshold=f_thres, stop_mode=self.stop_mode, name="forward"
                )
                z1 = result["result"]
                if self.save_result:
                    del result["result"]
                    self.f_result = result
            new_z1 = z1

            if (not self.training) and spectral_radius_mode:
                with torch.enable_grad():
                    new_z1 = func(z1.requires_grad_())
                _, sradius = power_method(new_z1, z1, n_iters=150)

            if self.training:
                new_z1 = func(z1.requires_grad_())
                if compute_jac_loss:
                    jac_loss = jac_loss_estimate(new_z1, z1)

                def backward_hook(grad):
                    # ic()
                    if self.hook is not None:
                        self.hook.remove()
                        torch.cuda.synchronize()
                    # ic()
                    result = self.b_solver(
                        lambda y: autograd.grad(new_z1, z1, y, retain_graph=True)[0] + grad,
                        torch.zeros_like(grad),
                        threshold=b_thres,
                        stop_mode=self.stop_mode,
                        name="backward",
                    )
                    r = result["result"]
                    if self.save_result:
                        del result["result"]
                        self.b_result = result
                    return r

                self.hook = new_z1.register_hook(backward_hook)
                # new_z1.register_hook(backward_hook)

        return new_z1, jac_loss.view(1, -1), sradius.view(-1, 1)

    # def forward(
    #     self,
    #     x,
    #     deq_mode=True,
    #     compute_jac_loss=False,
    #     spectral_radius_mode=False,
    #     writer=None,
    #     **kwargs
    # ):
    #     # ----------------Setting up-------------------------------
    #     bsz = x.shape[0]
    #     f_thres = kwargs.get("f_thres", self.f_thres)
    #     b_thres = kwargs.get("b_thres", self.b_thres)
    #     func = lambda z: self.f(z, x)
    #     # deq_mode = (train_step < 0) or (train_step >= self.pretrain_steps)

    #     # ----------------Computation part-------------------------------
    #     if not deq_mode:
    #         z1 = torch.zeros_like(x)
    #         for layer_ind in range(self.num_layers):
    #             z1 = func(z1)
    #         z = z1
    #     else:
    #         # with torch.no_grad():
    #         #     result = self.f_solver(func, torch.zeros_like(x), threshold=f_thres, stop_mode=self.stop_mode, name="forward")
    #         #     z1 = result['result']
    #         # new_z1 = z1

    #         # if self.training:
    #         #     new_z1 = func(z1.requires_grad_())
    #         #     def backward_hook(grad):
    #         #         result = self.b_solver(lambda y: autograd.grad(new_z1, z1, y, retain_graph=True)[0] + grad, torch.zeros_like(grad),
    #         #                                 threshold=b_thres, stop_mode=self.stop_mode, name="backward")
    #         #         return result['result']
    #         #     new_z1.register_hook(backward_hook)

    #         with torch.no_grad():
    #             result = self.f_solver(
    #                 lambda z: self.f(z, x),
    #                 torch.zeros_like(x),
    #                 threshold=f_thres,
    #                 stop_mode=self.stop_mode,
    #                 name="forward",
    #             )
    #             z = result["result"]
    #             self.f_nstep = result["nstep"]
    #             self.f_lowest = result["lowest"]
    #         z = self.f(z, x)
    #         if self.training:
    #             # set up Jacobian vector product (without additional forward calls)
    #             z0 = z.clone().detach().requires_grad_()
    #             f0 = self.f(z0, x)

    #             def backward_hook(grad):
    #                 result = self.b_solver(
    #                     lambda y: autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
    #                     grad,
    #                     threshold=b_thres,
    #                     stop_mode=self.stop_mode,
    #                     name="backward",
    #                 )
    #                 self.b_nstep = result["nstep"]
    #                 self.b_lowest = result["lowest"]
    #                 return result["result"]

    #             z.register_hook(backward_hook)
    #     return z


class RecurLayer(nn.Module):
    def __init__(self, block, iters):
        super(RecurLayer, self).__init__()
        self.recur_block = block

    def forward(self, x, iters, proj_out=None):
        # self.rel = []
        out = x
        for i in range(iters):
            out = self.recur_block(out)
            # self.rel.append((new_out - out).norm().item()/ (1e-8 + new_out.norm().item()))
            # out = new_out
            if proj_out is not None:
                proj_out(i, out)
        return out
