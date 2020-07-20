import torch
import torch.nn as nn

from data import str2dataset
from model import str2model

from utils import *

from advertorch.attacks import Attack

from advertorch.attacks import L2PGDAttack
from advertorch.attacks import LinfPGDAttack


class Clean(Attack):

    def __init__(self):
        self.clip_min = 0.0
        self.clip_max = 1.0

    def perturb(self, X, y):
        return X


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--checkpoint', type=str, default='mnist_vanilla')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_batch', type=int_or_none, default=5)

    parser.add_argument('--norm', type=str_or_none, default=None, help='None | 2 | inf (none for clean accuracy)')
    parser.add_argument('--eps', type=float, default=0.5, help='the perturbation size')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--nb_iter', type=int, default=20, help='number of gradient iterations')

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--save_img_loc', type=str_or_none, default=None)

    args = parser.parse_args()

    print(args)

    device = "cuda"

    set_seed(args.seed)

    testset, normalize, unnormalize = str2dataset(args.dataset)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    net = str2model(args.checkpoint, dataset=args.dataset, pretrained=True).eval().to(device)

    if args.norm is None:
        adversary = Clean()

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
