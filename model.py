import torch
import torch.nn as nn

import torchvision

from resnet import ResNet18


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


# class DebugNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.w = nn.Parameter(torch.tensor([[1, -1], [-1, 1]], dtype=torch.float))

#     def forward(self, x):
#         return x.view(-1, 1 * 1 * 1 * 2).mm(self.w)


def str2model(path, dataset=None, pretrained=True):
    if dataset == "mnist":
        net = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(32 * 7 * 7, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

        if pretrained:
            load_model(net, path)

    elif dataset == "cifar":
        net = ResNet18()

        if pretrained:
            load_model(net, path)

    elif dataset == "imagenet":
        assert pretrained
        net = torchvision.models.resnet50(pretrained=True)

    else:
        assert 0

    return net


def load_model(net, path):
    if not path.endswith(".pth") and not path.endswith(".pkl"):
        path = "./checkpoints/" + path + ".pth"

        ckpt = torch.load(path)

        if "net" in ckpt.keys():
            for key in ckpt["net"].keys():
                assert "module" in key

            ckpt["net"] = dict((key[7:], value) for key, value in ckpt["net"].items())

            net.load_state_dict(ckpt["net"])

        elif "state_dict" in ckpt:
            # assert 0
            net.load_state_dict(ckpt["state_dict"][0])

        else:
            net.load_state_dict(ckpt['model_state_dict'])

    else:
        ckpt = torch.load(path)
        net.load_state_dict(ckpt['model_state_dict'])
