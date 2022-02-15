
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

class SolverStats(object):
    def __init__(self):
        self.fwd_iters = Meter()
        self.bkwd_iters = Meter()
        self.fwd_err = Meter()
        self.bkwd_err = Meter()
        self.fwd_time = Meter()
        self.bkwd_time = Meter()

    def reset(self):
        self.fwd_iters.reset()
        self.fwd_time.reset()
        self.bkwd_iters.reset()
        self.bkwd_time.reset()
        self.fwd_err.reset()
        self.bkwd_err.reset()

    def report(self):
        print('Fwd iters: {:.2f}\tFwd Time: {:.4f}\tBkwd Iters: {:.2f}\tBkwd Time: {:.4f}\n'.format(
                self.fwd_iters.avg, self.fwd_time.avg,
                self.bkwd_iters.avg, self.bkwd_time.avg))