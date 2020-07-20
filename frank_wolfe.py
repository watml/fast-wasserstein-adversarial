import torch
import torch.nn as nn

from data import str2dataset
from model import str2model
from utils import *

from wasserstein_attack import WassersteinAttack
from lmo import entr_support_func
from projection import dual_capacity_constrained_projection


class FrankWolfe(WassersteinAttack):

    def __init__(self,
                 predict, loss_fn,
                 eps, kernel_size,
                 nb_iter, entrp_gamma, dual_max_iter, grad_tol, int_tol,
                 device="cuda",
                 postprocess=False,
                 verbose=True,
                 ):

        super().__init__(predict=predict, loss_fn=loss_fn,
                         eps=eps, kernel_size=kernel_size,
                         device=device,
                         postprocess=postprocess,
                         verbose=verbose,
                         )

        self.nb_iter = nb_iter
        self.entrp_gamma = entrp_gamma
        self.dual_max_iter = dual_max_iter
        self.grad_tol = grad_tol
        self.int_tol = int_tol

        self.inf = 1000000

    def perturb(self, X, y):
        batch_size, c, h, w = X.size()

        self.initialize_cost(X, inf=self.inf)
        pi = self.initialize_coupling(X).clone().detach().requires_grad_(True)
        normalization = X.sum(dim=(1, 2, 3), keepdim=True)

        for t in range(self.nb_iter):
            adv_example = self.coupling2adversarial(pi, X)
            scores = self.predict(adv_example.clamp(min=self.clip_min, max=self.clip_max))

            loss = self.loss_fn(scores, y)
            loss.backward()

            with torch.no_grad():
                self.lst_loss.append(loss.item())
                self.lst_acc.append((scores.max(dim=1)[1] == y).sum().item())

                """Add a small constant to enhance numerical stability"""
                pi.grad /= (tensor_norm(pi.grad, p='inf').view(batch_size, 1, 1, 1) + 1e-35)
                assert (pi.grad == pi.grad).all() and (pi.grad != float('inf')).all() and (pi.grad != float('-inf')).all()

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()

                optimal_pi, num_iter = entr_support_func(pi.grad,
                                                         X,
                                                         cost=self.cost,
                                                         inf=self.inf,
                                                         eps=self.eps * normalization.squeeze(),
                                                         gamma=self.entrp_gamma,
                                                         dual_max_iter=self.dual_max_iter,
                                                         grad_tol=self.grad_tol,
                                                         int_tol=self.int_tol)
                end.record()

                torch.cuda.synchronize()

                self.run_time += start.elapsed_time(end)
                self.num_iter += num_iter
                self.func_calls += 1

                if self.verbose and (t + 1) % 10 == 0:
                    print("num of iters : {:4d}, ".format(t + 1),
                          "loss : {:12.6f}, ".format(loss.item()),
                          "acc : {:5.2f}%, ".format((scores.max(dim=1)[1] == y).sum().item() / batch_size * 100),
                          "dual iter : {:2d}, ".format(num_iter),
                          "per iter time : {:7.3f}ms".format(start.elapsed_time(end) / num_iter))


                step = 2. / (t + 2)
                pi += step * (optimal_pi - pi)
                pi.grad.zero_()

                self.check_nonnegativity(pi / normalization, tol=1e-6, verbose=False)
                self.check_marginal_constraint(pi / normalization, X / normalization, tol=1e-5, verbose=False)
                self.check_transport_cost(pi / normalization, tol=1e-3, verbose=False)

        with torch.no_grad():
            adv_example = self.coupling2adversarial(pi, X)
            check_hypercube(adv_example, verbose=self.verbose)
            self.check_nonnegativity(pi / normalization, tol=1e-4, verbose=self.verbose)
            self.check_marginal_constraint(pi / normalization, X / normalization, tol=1e-4, verbose=self.verbose)
            self.check_transport_cost(pi / normalization, tol=self.eps * 1e-3, verbose=self.verbose)

            if self.postprocess is True:
                if self.verbose:
                    print("==========> post-processing projection........")

                pi = dual_capacity_constrained_projection(pi,
                                                          X,
                                                          self.cost,
                                                          eps=self.eps * normalization.squeeze(),
                                                          transpose_idx=self.forward_idx,
                                                          detranspose_idx=self.backward_idx,
                                                          coupling2adversarial=self.coupling2adversarial)

                adv_example = self.coupling2adversarial(pi, X)
                check_hypercube(adv_example, tol=self.eps * 1e-1, verbose=self.verbose)
                self.check_nonnegativity(pi / normalization, tol=1e-6, verbose=self.verbose)
                self.check_marginal_constraint(pi / normalization, X / normalization, tol=1e-6, verbose=self.verbose)
                self.check_transport_cost(pi / normalization, tol=self.eps * 1e-3, verbose=self.verbose)

        """Do not clip the adversarial examples to preserve pixel mass"""
        return adv_example


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--checkpoint', type=str, default='mnist_vanilla')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_batch', type=int_or_none, default=5)

    parser.add_argument('--eps', type=float, default=0.5, help='the perturbation size')
    parser.add_argument('--kernel_size', type=int_or_none, default=5)

    parser.add_argument('--nb_iter', type=int, default=20)
    parser.add_argument('--entrp_gamma', type=float, default=1e-6)
    parser.add_argument('--dual_max_iter', type=int, default=15)
    parser.add_argument('--grad_tol', type=float, default=1e-5)
    parser.add_argument('--int_tol', type=float, default=1e-5)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--postprocess', type=str2bool, default=False)

    parser.add_argument('--save_img_loc', type=str_or_none, default=None)
    parser.add_argument('--save_info_loc', type=str_or_none, default=None)

    args = parser.parse_args()

    print(args)

    device = "cuda"

    set_seed(args.seed)

    testset, normalize, unnormalize = str2dataset(args.dataset)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    net = str2model(args.checkpoint, dataset=args.dataset, pretrained=True).eval().to(device)

    for param in net.parameters():
        param.requires_grad = False

    frank_wolfe = FrankWolfe(predict=lambda x: net(normalize(x)),
                             loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                             eps=args.eps,
                             kernel_size=args.kernel_size,
                             nb_iter=args.nb_iter,
                             entrp_gamma=args.entrp_gamma,
                             dual_max_iter=args.dual_max_iter,
                             grad_tol=args.grad_tol,
                             int_tol=args.int_tol,
                             device=device,
                             postprocess=args.postprocess,
                             verbose=True)

    acc = test(lambda x: net(normalize(x)),
               testloader,
               device=device,
               attacker=frank_wolfe,
               num_batch=args.num_batch,
               save_img_loc=args.save_img_loc)

    frank_wolfe.print_info(acc)

    if args.save_info_loc is not None:
        frank_wolfe.save_info(acc, args.save_info_loc)
