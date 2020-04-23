import torch
import torch.nn as nn

import torch.backends.cudnn as cudnn

from model import str2model, load_model

from data import str2dataset

from utils import test, get_wasserstein_attack_parser, tensor_norm, set_seed
from utils import check_hypercube

from wasserstein_attack import WassersteinAttack
from conjugate import entr_support_func


class FrankWolfe(WassersteinAttack):

    def __init__(self,
                 predict, loss_fn,
                 eps, kernel_size,
                 nb_iter,
                 entrp_gamma, dual_max_iter,
                 grad_tol,
                 int_tol,
                 device="cuda",
                 clip_min=0., clip_max=1.,
                 clipping=False,
                 postprocess=False,
                 verbose=True,
                 ):

        super().__init__(predict=predict, loss_fn=loss_fn,
                         eps=eps, kernel_size=kernel_size,
                         nb_iter=nb_iter,
                         device=device,
                         clip_min=clip_min, clip_max=clip_max,
                         clipping=clipping,
                         postprocess=postprocess,
                         verbose=verbose,
                         )

        self.entrp_gamma = entrp_gamma
        self.dual_max_iter = dual_max_iter

        self.grad_tol = grad_tol
        self.int_tol = int_tol

        self.inf = 1000000

    def perturb(self, X, y):
        batch_size, c, h, w = X.size()

        self.cost = self.initialize_cost(X, inf=self.inf)
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

                gradient = pi.grad
                """
                # print("grad norm", tensor_norm(pi.grad, p='inf').min())
                add a small constant to enhance numerical stability
                """
                gradient /= (tensor_norm(pi.grad, p='inf').view(batch_size, 1, 1, 1) + 1e-8)

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()

                optimal_pi, num_iter = entr_support_func(gradient,
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

                from projection import dual_capacity_constrained_projection
                pi = dual_capacity_constrained_projection(pi,
                                                          X,
                                                          self.cost,
                                                          eps=self.eps * normalization.squeeze(),
                                                          transpose_idx=self.forward_idx,
                                                          detranspose_idx=self.backward_idx,
                                                          coupling2adversarial=self.coupling2adversarial)

                adv_example = self.coupling2adversarial(pi, X)
                check_hypercube(adv_example, tol=self.eps * 5e-2, verbose=self.verbose)
                self.check_nonnegativity(pi / normalization, tol=1e-6, verbose=self.verbose)
                self.check_marginal_constraint(pi / normalization, X / normalization, tol=1e-6, verbose=self.verbose)
                self.check_transport_cost(pi / normalization, tol=self.eps * 1e-3, verbose=self.verbose)

        if self.clipping:
            return adv_example.clamp(min=self.clip_min, max=self.clip_max)
        else:
            return adv_example


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(parents=[get_wasserstein_attack_parser()])

    parser.add_argument('--entrp_gamma', type=float, default=1e-6)
    parser.add_argument('--dual_max_iter', type=int, default=15)
    parser.add_argument('--grad_tol', type=float, default=1e-5)
    parser.add_argument('--int_tol', type=float, default=1e-5)

    args = parser.parse_args()

    print(args)

    device = "cuda"

    set_seed(args.seed)

    if args.benchmark:
        cudnn.benchmark = True

    testset, normalize, unnormalize = str2dataset(args.dataset)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    net = str2model(args.checkpoint, dataset=args.dataset, pretrained=True).eval().to(device)

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
                             clip_min=0., clip_max=1.,
                             clipping=True,
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
