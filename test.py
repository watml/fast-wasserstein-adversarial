import torch
import torch.nn as nn

from advertorch.attacks import Attack

from data import str2dataset
from model import str2model, load_model

from utils import test, get_test_parser, set_seed
from utils import str_or_none

from advertorch.attacks import PGDAttack

from advertorch.attacks import SparseL1DescentAttack
from advertorch.attacks import L2PGDAttack
from advertorch.attacks import LinfPGDAttack



class Clean(Attack):

    def __init__(self):
        self.clip_min = 0.0
        self.clip_max = 1.0

    def perturb(self, X, y):
        return X


class _L1PGDAttack(PGDAttack):

    def __init__(self,
                 predict, loss_fn=None, eps=10., nb_iter=40,
                 eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
                 targeted=False):

        super().__init__(predict=predict, loss_fn=loss_fn, eps=eps, nb_iter=nb_iter,
                         eps_iter=eps_iter, rand_init=rand_init, clip_min=clip_min,
                         clip_max=clip_max, targeted=targeted)

    def perturb(self, x, y):
        batch, c, h, w = x.size()

        delta = x.new_zeros(x.size()).requires_grad_(True)

        for t in range(self.nb_iter):
            scores = self.predict(x + delta)
            loss = self.loss_fn(scores, y)
            loss.backward()

            with torch.no_grad():
                delta += self.eps_iter * delta.grad / delta.grad.abs().sum(dim=(1, 2, 3), keepdim=True)

                from projection import simplex_projection
                tmp = simplex_projection(delta.abs().view(batch, 1, 1, -1),
                                         delta.new_ones(batch, 1, 1, 1) * self.eps)

                delta = tmp.view(x.size()) * delta.sign()

                delta = (x + delta).clamp(self.clip_min, self.clip_max) - x

            delta = delta.clone().detach().requires_grad_(True)

        print(delta.abs().sum(dim=(1, 2, 3)).max().item())

        return (x + delta).clamp(self.clip_min, self.clip_max)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(parents=[get_test_parser()])

    parser.add_argument('--eps', type=float, default=0.5, help='the perturbation size')
    parser.add_argument('--norm', type=str_or_none, default=None)

    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--nb_iter', type=int, default=20, help='number of gradient iterations')

    args = parser.parse_args()

    print(args)

    device = "cuda"

    set_seed(args.seed)

    testset, normalize, unnormalize = str2dataset(args.dataset)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    net = str2model(args.checkpoint, dataset=args.dataset, pretrained=True).eval().to(device)
    # net = str2model(args.checkpoint, pretrained=False).eval().to(device)

    if args.norm is None:
        adversary = Clean()
    elif args.norm == "1":
        adversary = _L1PGDAttack(predict=lambda x: net(normalize(x)), loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=args.eps,
                                 nb_iter=args.nb_iter, eps_iter=args.lr, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
    elif args.norm == "2":
        adversary = L2PGDAttack(predict=lambda x: net(normalize(x)), loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=args.eps,
                                nb_iter=args.nb_iter, eps_iter=args.lr, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
    elif args.norm == "inf":
        adversary = LinfPGDAttack(predict=lambda x: net(normalize(x)), loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=args.eps,
                                  nb_iter=args.nb_iter, eps_iter=args.lr, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)

    acc = test(lambda x: net(normalize(x)),
               testloader,
               device=device,
               attacker=adversary,
               num_batch=args.num_batch,
               save_img_loc=args.save_img_loc)

    print(acc)
