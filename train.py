import torch
import torch.nn as nn
import torch.optim as optim

from utils import set_seed
from data import str2dataset
from model import str2model

from frank_wolfe import FrankWolfe


def train(model, loader, device, lr, epoch, attacker):
    model.train()

    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0005)

    for i in range(epoch):
        correct = 0
        total = 0
        total_loss = 0

        for batch_idx, (cln_data, target) in enumerate(loader):
            cln_data, target = cln_data.to(device), target.to(device)

            # model.eval()
            for param in model.parameters():
                param.requires_grad = False

            adv_data = attacker.perturb(cln_data, target)

            # model.train()
            for param in model.parameters():
                param.requires_grad = True

            optimizer.zero_grad()

            scores = model(adv_data)
            loss = loss_fn(scores, target)
            loss.backward()

            optimizer.step()

            correct += scores.max(dim=1)[1].eq(target).sum().item()
            total += scores.size(0)
            total_loss += loss.item()

        print("epoch: {:4d} loss: {:10.6f} acc: {:.3f}% ({:5d} / {:5d})"
              .format(i + 1, total_loss, 100. * correct / total, correct, total))


if __name__ == "__main__":
    import argparse

    # parser = argparse.ArgumentParser(parents=[get_wasserstein_attack_parser()])
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--parallel', type=int, default=1)


    parser.add_argument('--attack', type=str)
    parser.add_argument('--eps', type=float)
    parser.add_argument('--nb_iter', type=int, default=40, help='number of attack iterations')


    parser.add_argument('--resume', type=int, default=0)
    parser.add_argument('--save_model_loc', type=str, default=None)

    args = parser.parse_args()

    print(args)

    device = "cuda"

    set_seed(0)

    trainset, normalize, unnormalize = str2dataset(args.dataset, train=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    net = str2model(path=args.save_model_loc, dataset=args.dataset, pretrained=args.resume).eval().to(device)

    if args.parallel:
        print("visible GPUs: {:d}".format(torch.cuda.device_count()))
        net = nn.DataParallel(net)

    if args.attack == "frank":
        attacker = FrankWolfe(predict=lambda x: net(normalize(x)),
                              loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                              eps=args.eps,
                              kernel_size=5,
                              nb_iter=args.nb_iter,
                              entrp_gamma=1e-3,
                              dual_max_iter=30,
                              grad_tol=1e-4,
                              int_tol=1e-4,
                              device=device,
                              clip_min=0., clip_max=1.,
                              clipping=False,
                              postprocess=False,
                              verbose=False)
    else:
        assert 0

    train(net, trainloader, device, args.lr, args.epoch, attacker)

    ckpt = {'model_state_dict': net.state_dict()}

    if args.save_model_loc is None:
        torch.save(ckpt, "./checkpoints/{}_adv_training_attack-{}_eps-{}.pth".format(args.dataset, args.attack, args.eps))
    else:
        torch.save(ckpt, args.save_model_loc)
