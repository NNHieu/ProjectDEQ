from ctypes import Union
import os
import time
import torch
import numpy as np
import h5py
import tqdm
from torch import Tensor

from . import projection as proj, scheduler
from .landscape import Direction, Surface
from .utils import SamplerStats, format_num
# from utils import mpi4pytorch as mpi
mpi = None

class Sampler:
    def __init__(self, model, surface: Surface, layer_names: Union, device: torch.DeviceObjType, comm=None, rank: int=-1, logger=None) -> None:
        self.model = model
        self.surface = surface
        self.device = device
        self.rank = rank
        self.layer_names = layer_names
        self.comm = comm
        self.logger = logger
        self.stats = SamplerStats()

    def prepair(self):
        # if rank == 0: self.surface.open('r+')
        self.surface.dirs.to_tensor()
        self.layers = [self.surface.layers[name] for name in self.layer_names]
        self.layers_flat = [layer.ravel() for layer in self.layers]
        # model.to(self.device)

    def reduce(self):
        '''
        Send updated plot data to the master node
        '''
        if self.rank < 0: return 0
        syc_start = time.time()
        for layer in self.layers_flat:
            # dist.reduce(layer, 0, op=dist.ReduceOp.MAX)
            if mpi is not None:
                mpi.reduce_max(self.comm, layer)
        syc_time = time.time() - syc_start
        return syc_time


    def write(self):
        # Only the master node writes to the file - this avoids write conflicts
        if self.rank <= 0:
            for name, layer in zip(self.layer_names, self.layers):
                self.surface.h5_file['layers'][name][:] = layer
            self.surface.flush()

    def run(self, evaluate, job_schedule):
        """
            Calculate the loss values and accuracies of modified models.
        """
        inds, coords, inds_nums = job_schedule
        self.stats.start(len(inds))

        max_inds_nums = max(inds_nums)
        # dirs_tensor = (proj.tensorlist_to_tensor(directions[0]), proj.tensorlist_to_tensor(directions[1]))
        # self.logger.info('Computing %d values for rank %d'% (len(inds), self.rank))
        self.model.eval()
        with torch.no_grad():
            model = self.model
            weights = [torch.clone(p) for p in model.parameters()]
            # Loop over all uncalculated loss values
            for count, ind in enumerate(inds):
                # Get the coordinates of the loss value being calculated
                coord = coords[count]
                Direction.set_weights(model, weights, self.surface.dirs.tensors, coord)
                # Record the time to compute the loss value
                loss_start = time.time()
                values = evaluate(model)
                loss_compute_time = time.time() - loss_start
                # Record the result in the local array
                for i, val in enumerate(values):
                    self.layers_flat[i][ind] = val

                syc_time = self.reduce()
                self.write()

                self.stats.update(loss_compute_time, syc_time, coord, tuple([format_num(v) for v in values]))
            # This is only needed to make MPI run smoothly. If this process has less work than
            # the rank0 process, then we need to keep calling reduce so the rank0 process doesn't block
            for i in range(max_inds_nums - len(inds)):
                self.reduce()

        self.logger.info(f'Rank {self.rank} done! {self.stats.report()}')

# def main():
#     model = None
#     surface = Surface.load(path)
#     loss_key = ('loss', 'acc')
#     inds, coords, inds_nums = scheduler.get_job_indices(*surface.get_unplotted_indices('loss'), 0, 1)
#     surface.open('r+')
#     sampler = Sampler(model, surface, loss_key,'gpu:0', 0)
#     sampler.prepair()
#     sampler.run(evaluation, inds, coords, inds_nums)
#     surface.close()
