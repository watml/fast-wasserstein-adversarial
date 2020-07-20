import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from projected_sinkhorn import conjugate_sinkhorn, projected_sinkhorn
from projected_sinkhorn import wasserstein_cost
from projected_sinkhorn import attack

from wasserstein_attack import WassersteinAttack

from utils import test
from utils import get_wasserstein_attack_parser
from utils import set_seed
from utils import tensor_norm

from data import str2dataset

from model import str2model, load_model


class Sinkhorn(WassersteinAttack):

    def __init__(self,
                 predict, loss_fn,
                 eps, kernel_size,
                 lr, nb_iter,
                 lam, sinkhorn_max_iter,
                 stop_abs, stop_rel,
                 device="cuda",
                 clip_min=0., clip_max=1.,
                 clipping=False,
                 postprocess=False,
                 verbose=True
                 ):
        super().__init__(predict, loss_fn,
                         eps, kernel_size,
                         nb_iter,
                         device,
                         clip_min, clip_max,
                         clipping=clipping,
                         postprocess=postprocess,
                         verbose=verbose,
                         )
        self.lr = lr
        self.lam = lam
        self.sinkhorn_max_iter = sinkhorn_max_iter
        self.stop_abs = stop_abs
        self.stop_rel = stop_rel

        self.adaptive = False

        self.run_time = 0.0
        self.num_iter = 0
        self.func_calls = 0

    def perturb(self, X, y):
        if self.adaptive:
            assert 0
            # return attack(X=X, y=y,
            #               net=self.predict,
            #               epsilon=0.1,
            #               epsilon_iters=5,
            #               epsilon_factor=1.4,
            #               p=2,
            #               kernel_size=self.kernel_size,
            #               maxiters=50,
            #               alpha=self.lr,
            #               xmin=self.clip_min,
            #               xmax=self.clip_max,
            #               regularization=self.lam,
            #               sinkhorn_maxiters=self.sinkhorn_max_iter,
            #               ball='wasserstein', norm='linfinity',
            #               verbose=5)[0]
        else:
            return self.perturb_fix(X, y)

    def perturb_fix(self, X, y):
        batch_size = X.size(0)
        epsilon = X.new_ones(batch_size) * self.eps
        C = wasserstein_cost(X, p=1, kernel_size=self.kernel_size)
        normalization = X.sum(dim=(1, 2, 3), keepdim=True)

        X_ = X.clone().detach().requires_grad_(True)

        for t in range(self.nb_iter):
            # scores = self.predict(X_)
            adv_example = X_.clamp(min=self.clip_min, max=self.clip_max)
            scores = self.predict(adv_example)
            loss = self.loss_fn(scores, y)
            loss.backward()

            with torch.no_grad():
                self.lst_loss.append(loss.item())
                self.lst_acc.append((scores.max(dim=1)[1] == y).sum().item())

                X_ += self.lr * torch.sign(X_.grad)

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()

                X_, num_iter = projected_sinkhorn(X.clone().detach() / normalization,
                                                  X_.clone().detach() / normalization,
                                                  C,
                                                  epsilon,
                                                  self.lam,
                                                  verbose=False,
                                                  maxiters=self.sinkhorn_max_iter,
                                                  termination=(self.stop_abs, self.stop_rel),
                                                  return_iters=True)

                if X_ is None and num_iter is None:
                    self.overflow = True
                    return X

                if num_iter >= self.sinkhorn_max_iter:
                    self.converge = False

                X_ *= normalization

                end.record()

                torch.cuda.synchronize()

                self.run_time += start.elapsed_time(end)
                self.num_iter += num_iter
                self.func_calls += 1

                if self.verbose and (t + 1) % 10 == 0:
                    print("num of iters : {:4d}, ".format(t + 1),
                          "loss : {:12.6f}, ".format(loss.item()),
                          "acc : {:5.2f}%, ".format((scores.max(dim=1)[1] == y).sum().item() / batch_size * 100),
                          "dual iter : {:3d}, ".format(num_iter),
                          "per iter time : {:7.3f}ms".format(start.elapsed_time(end) / num_iter))

                # X_ = torch.clamp(X_, min=self.clip_min, max=self.clip_max)

            X_ = X_.clone().detach().requires_grad_(True)

        from utils import check_hypercube
        check_hypercube(X_, verbose=self.verbose)

        if self.clipping:
            return X_.clamp(min=self.clip_min, max=self.clip_max)
        else:
            return X_


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(parents=[get_wasserstein_attack_parser()])

    parser.add_argument('--lr', type=float, default=0.1, help='gradient step size')
    parser.add_argument('--lam', type=float, default=1000, help='entropic regularization constanst')
    parser.add_argument('--sinkhorn_max_iter', type=int, default=400)
    parser.add_argument('--stop_abs', type=float, default=1e-4)
    parser.add_argument('--stop_rel', type=float, default=1e-4)

    args = parser.parse_args()

    print(args)

    device = "cuda"

    set_seed(args.seed)

    # print(torch.backends.cudnn.version())

    cudnn.enabled = True

    if args.benchmark:
        cudnn.benchmark = True

    testset, normalize, unnormalize = str2dataset(args.dataset)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    net = str2model(args.checkpoint, dataset=args.dataset, pretrained=True).eval().to(device)

    for param in net.parameters():
        param.requires_grad = False

    sinkhorn = Sinkhorn(predict=lambda x: net(normalize(x)),
                        loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                        eps=args.eps,
                        kernel_size=args.kernel_size,
                        lr=args.lr,
                        nb_iter=args.nb_iter,
                        lam=args.lam,
                        sinkhorn_max_iter=args.sinkhorn_max_iter,
                        stop_abs=args.stop_abs,
                        stop_rel=args.stop_rel,
                        device=device,
                        clip_min=0.0, clip_max=1.0,
                        clipping=False,
                        postprocess=False,
                        verbose=True)

    acc = test(lambda x: net(normalize(x)),
               testloader,
               device=device,
               attacker=sinkhorn,
               num_batch=args.num_batch,
               save_img_loc=args.save_img_loc)

    sinkhorn.print_info(acc)
    # print(sinkhorn.lst_acc)

    if args.save_info_loc is not None:
        sinkhorn.save_info(acc, args.save_info_loc)
