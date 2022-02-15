import time
from tqdm import tqdm
import os
import h5py
from .landscape import Dir2D, Surface

def name_surface_file(rect, res, dir_file):
    # use args.dir_file as the perfix
    surf_file = dir_file
    xmin, ymin, xmax, ymax = rect
    xnum, ynum = res
    # resolution
    surf_file += '_[%s,%s,%d]' % (str(xmin), str(xmax), int(xnum))
    surf_file += 'x[%s,%s,%d]' % (str(ymin), str(ymax), int(ynum))

    return surf_file + ".h5"

def create_surfile(model, layers, dir_file, surf_file, rect, resolution, logger):
    if not os.path.exists(dir_file):
        logger.info('Create dir file at {}'.format(dir_file))
        dir2d = Dir2D(model=model)
        try:
            with h5py.File(dir_file, 'w') as f:
                dir2d.save(f)
        except Exception as e:
            os.remove(dir_file)
            raise e

    if not os.path.exists(surf_file):
        logger.info('Create surface file at {}'.format(surf_file))
        surface = Surface(dir_file, rect, resolution, surf_file, {})
        surface.add_layer(*layers)
        surface.save()

    return surf_file

class Meter(object):
    """Computes and stores the min, max, avg, and current values"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = -float("inf")
        self.min = float("inf")

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = max(self.max, val)
        self.min = min(self.min, val)

def format_num(n):
    f = '{0:.3g}'.format(n)
    f = f.replace('+0', '+')
    f = f.replace('-0', '-')
    n = str(n)
    return f if len(f) < len(n) else n

class SamplerStats(object):
    def __init__(self):
        self.loss_compute_time = Meter()
        self.syc_time = Meter()
        self.start_time = 0

    def reset(self):
        self.loss_compute_time.reset()
        self.syc_time.reset()
        self.start_time = 0

    def start(self, total):
        self.reset()
        self.start_time = time.time()
        self.pbar = tqdm(total=total, desc="Sampling Surface Value")

    def update(self, loss_compute_timestep, syc_timestep, coord, values):
        self.loss_compute_time.update(loss_compute_timestep)
        self.syc_time.update(syc_timestep)
        self.pbar.set_postfix({'coord': str(coord),
                               'values': values,
                               'ttime': format_num(loss_compute_timestep),
                               'tsync': format_num(syc_timestep)})
        self.pbar.update(1)

        # self.logger.info('Evaluating rank %d  %d/%d  (%.1f%%)  coord=%s \t%s \ttime=%.2f \tsync=%.2f' % (
        #         self.rank, count, len(inds), 100.0 * count/len(inds), str(coord), log_values, loss_compute_time, syc_time))

    def report(self):
        # Rank %d done!
        self.pbar.close()
        return 'Total time: {:.2f} Sync: {:.2f} \
                (Avg loss-computation time: {:.2f}\tAvg sync time: {:.2f})' \
                .format(time.time() - self.start_time, self.syc_time.sum,
                        self.loss_compute_time.avg, self.syc_time.avg)