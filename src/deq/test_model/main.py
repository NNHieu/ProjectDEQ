import sys

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# CIFAR10 data loader
from torchvision import datasets, transforms

sys.path.append("..")
from lib.solvers import anderson
from models.core import DEQLayer
from omegaconf import OmegaConf


class ResNetLayer(nn.Module):
    def __init__(self, n_channels, n_inner_channels, kernel_size=3, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(
            n_channels, n_inner_channels, kernel_size, padding=kernel_size // 2, bias=False
        )
        self.conv2 = nn.Conv2d(
            n_inner_channels, n_channels, kernel_size, padding=kernel_size // 2, bias=False
        )
        self.norm1 = nn.GroupNorm(num_groups, n_inner_channels)
        self.norm2 = nn.GroupNorm(num_groups, n_channels)
        self.norm3 = nn.GroupNorm(num_groups, n_channels)
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

    def forward(self, z, x):
        y = self.norm1(F.relu(self.conv1(z)))
        return self.norm3(F.relu(z + self.norm2(x + self.conv2(y))))


class DEQFixedPoint(nn.Module):
    def __init__(self, f, solver, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs

    def forward(self, x):
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            result = self.solver(lambda z: self.f(z, x), torch.zeros_like(x), **self.kwargs)
            z = result["result"]
        z = self.f(z, x)

        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0, x)

        def backward_hook(grad):
            result = self.solver(
                lambda y: autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                grad,
                **self.kwargs
            )
            return result["result"]

        z.register_hook(backward_hook)
        return z


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cfg = OmegaConf.create(
    {
        "train": {
            "pretrain_step": 10,
        },
        "model": {
            "f_solver": "anderson",
            "b_solver": "anderson",
            "f_thres": 25,
            "b_thres": 20,
            "stop_mode": "rel",
            "num_layers": 1,
        },
    }
)

torch.manual_seed(0)
chan = 48
f = ResNetLayer(chan, 64, kernel_size=3)
model = nn.Sequential(
    nn.Conv2d(3, chan, kernel_size=3, bias=True, padding=1),
    nn.BatchNorm2d(chan),
    DEQLayer(f, cfg),
    #   DEQFixedPoint(f, anderson, tol=1e-2, max_iter=25, m=5),
    nn.BatchNorm2d(chan),
    nn.AvgPool2d(8, 8),
    nn.Flatten(),
    nn.Linear(chan * 4 * 4, 10),
).to(device)


cifar10_train = datasets.CIFAR10(
    "../datasets", train=True, download=True, transform=transforms.ToTensor()
)
cifar10_test = datasets.CIFAR10(
    "../datasets", train=False, download=True, transform=transforms.ToTensor()
)
train_loader = DataLoader(cifar10_train, batch_size=100, shuffle=True, num_workers=8)
test_loader = DataLoader(cifar10_test, batch_size=100, shuffle=False, num_workers=8)

# standard training or evaluation loop
def epoch(loader, model, opt=None, lr_scheduler=None):
    total_loss, total_err = 0.0, 0.0
    model.eval() if opt is None else model.train()
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp, y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_scheduler.step()

        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]

    return total_err / len(loader.dataset), total_loss / len(loader.dataset)


opt = optim.Adam(model.parameters(), lr=1e-3)
print("# Parmeters: ", sum(a.numel() for a in model.parameters()))

max_epochs = 50
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, max_epochs * len(train_loader), eta_min=1e-6)

for i in range(50):
    print(epoch(train_loader, model, opt, scheduler))
    print(epoch(test_loader, model))
