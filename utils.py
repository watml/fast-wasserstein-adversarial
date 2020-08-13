
import random

import numpy as np

import torch


def str2bool(x):
    return bool(int(x))


def str_or_none(x):
    return None if x == "None" else str(x)


def int_or_none(x):
    return None if x == "None" else int(x)


def float_or_none(x):
    return None if x == "None" else float(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def test(net, loader, device, attacker, num_batch, save_img_loc=None):
    correct = 0
    total = 0

    if save_img_loc is not None:
        save_cln_img_list = []
        save_adv_img_list = []
        save_labels_list = []
        save_predictions_list = []

    for batch_idx, (cln_data, target) in enumerate(loader):
        cln_data, target = cln_data.to(device), target.to(device)

        adv_data = attacker.perturb(cln_data, target)

        if adv_data is None:
            assert 0

        with torch.no_grad():
            output = net(adv_data)

        prediction = output.max(dim=1)[1]

        correct += (prediction == target).sum().item()
        total += target.size(0)

        print("****************************************************************")
        print("batch idx: {:4d} num_batch: {:4d} acc: {:.3f}% ({:5d} / {:5d})".format(batch_idx + 1,
                                                                                      len(loader),
                                                                                      100. * correct / total,
                                                                                      correct,
                                                                                      total))
        print("****************************************************************")

        if save_img_loc is not None:
            save_cln_img_list.append(cln_data.clone().detach().cpu().numpy())
            save_adv_img_list.append(adv_data.clone().detach().cpu().numpy())
            save_labels_list.append(target.clone().detach().cpu().numpy())
            save_predictions_list.append(prediction.clone().detach().cpu().numpy())

        if num_batch is not None and batch_idx + 1 >= num_batch:
            break

        if attacker.__class__.__name__ == "Sinkhorn" and attacker.overflow is True:
            break

    if save_img_loc is not None:
        save_cln_img_array = np.concatenate(save_cln_img_list, axis=0)
        save_adv_img_array = np.concatenate(save_adv_img_list, axis=0)
        save_labels_array = np.concatenate(save_labels_list, axis=0)
        save_predictions_array = np.concatenate(save_predictions_list, axis=0)
        torch.save((save_cln_img_array, save_labels_array, save_adv_img_array, save_predictions_array), save_img_loc)

    return 100.0 * correct / total


def _violation_nonnegativity(pi):
    diff = pi.clamp(max=0.).abs().sum(dim=(1, 2, 3)).max().item()
    return diff


def _check_nonnegativity(pi, tol, verbose=False):
    """pi: tensor of size (batch_size, c, img_size, img_size)"""
    # diff = pi.clamp(max=0.).abs().sum(dim=(1, 2, 3)).max().item()
    diff = _violation_nonnegativity(pi)

    if verbose:
        print("check nonnegativity: {:.9f}".format(diff))

    assert diff < tol


def _violation_marginal_constraint(pi, X):
    batch_size, c, h, w = X.size()
    img_size = h * w

    diff = (pi.sum(dim=-1) - X.view(batch_size, c, img_size)).abs().sum(dim=(1, 2)).max().item()

    return diff


def _check_marginal_constraint(pi, X, tol, verbose=False):
    """
    pi: dense tensor of size (batch_size, c, img_size, img_size)
                          or (batch_size, c, img_size, kernel_size^2)
    X: tensor of size (batch_size, c, h, w)
    """
    diff = _violation_marginal_constraint(pi, X)

    if verbose:
        print("check marginal constraint: {:.9f}".format(diff))

    assert diff < tol


def _violation_transport_cost(pi, cost, eps):
    diff = (cost * pi).sum(dim=(1, 2, 3)).max().item()
    return diff


def _check_transport_cost(pi, cost, eps, tol, verbose=False):
    """
    pi: dense tensor of size (batch_size, c, img_size, img_size)
                          or (batch_size, c, img_size, kernel_size^2)
    cost: tensor of size (img_size, img_size)
                      or (img_size, kernel_size^2)
    """
    diff = _violation_transport_cost(pi, cost, eps)

    if verbose:
        print("check transportation cost: {:.9f}".format(diff))

    assert diff < eps + tol


def check_hypercube(adv_example, tol=None, verbose=True):
    if verbose:
        print("----------------------------------------------")
        print("num of pixels that exceed exceed one {:d}  ".format((adv_example > 1.).sum(dim=(1, 2, 3)).max().item()))
        print("maximum pixel value                  {:f}".format(adv_example.max().item()))
        print("total pixel value that exceed one    {:f}".format((adv_example - 1.).clamp(min=0.).sum(dim=(1, 2, 3)).max().item()))
        print("% of pixel value that exceed one     {:f}%".format(
            100 * ((adv_example - 1.).clamp(min=0.).sum(dim=(1, 2, 3)) / adv_example.sum(dim=(1, 2, 3))).max().item()))
        print("----------------------------------------------")

    if tol is not None:
        assert(((adv_example - 1.).clamp(min=0.).sum(dim=(1, 2, 3)) / adv_example.sum(dim=(1, 2, 3))).max().item() < tol)


# def unsqueeze3(tensor):
#     assert 0
#     """Receive a tensor of size (len, ) and reshape it to (len, 1, 1, 1)"""
#     return tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


def tensor_norm(tensor, p=2):
    """
    Return the norm for a batch of samples
    Args:
        tensor: tensor of size (batch, channel, img_size, last_dim)
        p: 1, 2 or inf

        if p is inf, the size of tensor can also be (batch, channel, img_size)
    Return:
        tensor of size (batch, )
    """
    assert tensor.layout == torch.strided

    if p == 1:
        return tensor.abs().sum(dim=(1, 2, 3))
    elif p == 2:
        return torch.sqrt(torch.sum(tensor * tensor, dim=(1, 2, 3)))
    elif p == 'inf':
        return tensor.abs().view(tensor.size(0), -1).max(dim=-1)[0]
    else:
        assert 0


def bisection_search(grad_fn, a, b, max_iter, grad_tol, int_tol, verbose=False):
    assert (a < b).all()

    for i in range(max_iter):
        mid = (a + b) / 2
        grad = grad_fn(mid)
        idx = grad > 0.

        a[idx] = mid[idx]
        b[~idx] = mid[~idx]

        assert (a < b).all()

        if grad_tol is not None and (grad.abs() < grad_tol).all():
            break

        if int_tol is not None and torch.max(b - a) < int_tol:
            break

        if verbose:
            print("bisection iter {:2d}, gradient".format(i), grad_fn(mid))

    pnt = True

    if grad_tol is not None and (grad.abs() < grad_tol).all():
        pnt = False

    if int_tol is not None and torch.max(b - a) < int_tol:
        pnt = False

    if grad_tol is None and int_tol is None:
        pnt = False

    if pnt:
        print("WARNING: bisection search does not converge in {:2d} iterations".format(max_iter))

    return b, i + 1


if __name__ == "__main__":
    def obj_fn(x):
        y = x.new_zeros(x.size())
        for i in range(x.size(0)):
            y[i] = - (x[i] - i) ** 2
        return y

    def grad_fn(x):
        y = x.new_zeros(x.size())
        for i in range(x.size(0)):
            y[i] = - 2 * (x[i] - i)
        return y

    x = torch.ones(5, dtype=torch.float)
    print("----------ojective values----------")
    print(obj_fn(x))
    print("----------gradient values----------")
    print(grad_fn(x))

    print()
    print("expecting [0, 1, 2, 3, 4]")

    print("----------test bisection search-----------")
    maximizer = bisection_search(grad_fn,
                                 torch.zeros(5, dtype=torch.float),
                                 100 * torch.ones(5, dtype=torch.float),
                                 max_iter=50,
                                 grad_tol=1e-6,
                                 int_tol=1e-6,
                                 )

    print(maximizer)
