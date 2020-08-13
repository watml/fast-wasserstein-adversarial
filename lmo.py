import torch
import torch.nn.functional as F

from utils import tensor_norm
from utils import bisection_search


def entr_support_func(G, X, cost, inf, eps, gamma, dual_max_iter, grad_tol, int_tol):
    """
    Compute entropic regularized support function for a transportation;
        max_pi <pi, G> + gamma * entropy(pi)
        s.t. pi 1 = x, pi >= 0, <C, pi> <= eps
    Equivalently, compute min_pi <pi, -G> - gamma * entropy(pi)
    Args:
        G: constant in the objective function. If G is gradient, calling this subroutine in Frank-Wolfe will maximize the objective
        X: constant for the marginal constraint
        cost : constant for transportation constraint
        eps: constant for transportation constraint
        gamma: constant for entropic regularization
    """
    batch_size, c, h, w = X.size()
    img_size = h * w

    def recover(lam):
        G_lambda_C = -G + lam.view(-1, 1, 1, 1) * cost
        optimal_pi = X.view(batch_size, c, img_size, 1) * F.softmin((G_lambda_C / gamma), dim=-1)

        return optimal_pi

    """
    def dual(lam):
        G_lambda_C = -G + lam.view(-1, 1, 1, 1) * cost
        return - lam * eps + gamma * (
            torch.sum(X.view(batch_size, c, img_size) * (
                torch.log(X.view(batch_size, c, img_size) + 1e-10) - torch.logsumexp(-G_lambda_C / gamma, dim=3)
            ), dim=(1, 2))
        )
    """

    left = X.new_zeros(batch_size)
    # right = 2 * tensor_norm(G, p='inf') + gamma * 1 / eps * (X.view(batch_size, c, img_size, 1) * cost * (cost < inf)).sum(dim=(1, 2, 3))

    """The following is a sharper bound"""
    right = 2 * tensor_norm(G, p='inf') + gamma * torch.log(1 / eps * (X.view(batch_size, c, img_size, 1) * cost * (cost < inf)).sum(dim=(1, 2, 3)))
    right = right.clamp(min=0.)

    def grad_fn(lam):
        tilde_pi = recover(lam)
        return (tilde_pi * cost).sum(dim=(1, 2, 3)) - eps

    lam_star, num_iter = bisection_search(grad_fn,
                                          left,
                                          right,
                                          max_iter=dual_max_iter,
                                          grad_tol=grad_tol,
                                          int_tol=int_tol,
                                          verbose=False)

    pi_star = recover(lam_star)

    return pi_star, num_iter
