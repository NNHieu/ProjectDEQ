import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta=1.0):
    """ Anderson acceleration for fixed point iteration. """
    bsz = x0.shape[0]
    x0_flatten = x0.view(bsz, -1)
    X = torch.zeros(bsz, m, x0_flatten.shape[1], dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, x0_flatten.shape[1], dtype=x0.dtype, device=x0.device)
    X[:, 0], F[:, 0] = x0_flatten, f(x0).view(bsz, -1)
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].view_as(x0)).view(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1 : n + 1, 1 : n + 1] = (
            torch.bmm(G, G.transpose(1, 2))
            + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[None]
        )
        alpha = torch.solve(y[:, : n + 1], H[:, : n + 1, : n + 1])[0][:, 1 : n + 1, 0]  # (bsz x n)

        X[:, k % m] = (
            beta * (alpha[:, None] @ F[:, :n])[:, 0]
            + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        )
        F[:, k % m] = f(X[:, k % m].view_as(x0)).view(bsz, -1)
        res.append((F[:, k % m] - X[:, k % m]).norm().item() / (1e-5 + F[:, k % m].norm().item()))
        if res[-1] < tol:
            break
    return X[:, k % m].view_as(x0), res
