import torch
import os
import re
import shutil
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.utils import extract_archive, verify_str_arg
from torchvision.datasets.folder import ImageFolder


class ImageNetLoader(ImageFolder):

    GROUND_TRUTH_FILE = 'ILSVRC2012_validation_ground_truth.txt'
    MAPPING_FILE = 'ILSVRC2012_mapping.txt'
    LABELS_FILE = 'labels.txt'
    IMAGE_TAR_FILE = 'ILSVRC2012_img_val.tar'

    def __init__(self, root='./data/ImageNet', split='val', **kwargs):
        root = self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", ("train", "val"))

        if not os.path.exists(self.split_folder):
            os.mkdir(self.split_folder)

        wnid_to_classes = self.load_label_file()
        mapped_wnid_to_idx, mapped_idx_to_wnid = self.load_mapping()
        targets = self.load_ground_truth()
        target_wnids = [mapped_idx_to_wnid[idx] for idx in targets]
        self.wnids = list(wnid_to_classes.keys())
        self.classes = list(wnid_to_classes.values())

        alphbetical_wnid_to_idx = {wnid: i for i, wnid in enumerate(sorted(self.wnids))}
        imgs = self.parse_image_tar(target_wnids, alphbetical_wnid_to_idx)

        super(ImageNetLoader, self).__init__(self.split_folder, **kwargs)

        self.classes = [cls for wnid, clss in wnid_to_classes.items() for cls in clss]
        self.wnid_to_idx = alphbetical_wnid_to_idx

        self.class_to_idx = {cls: idx
                             for wnid, idx in alphbetical_wnid_to_idx.items() if wnid in wnid_to_classes
                             for cls in wnid_to_classes[wnid]}

        self.samples = imgs
        self.targets = targets
        self.imgs = imgs


    def parse_image_tar(self, wnids, wnid_to_idx, split='val'):
        imgs = []

        root = os.path.join(self.root, split)
        extract_archive(os.path.join(self.root, ImageNetLoader.IMAGE_TAR_FILE), root)
        img_files = sorted([
            os.path.join(root, image) for image in os.listdir(root)
            if bool(re.match(r"ILSVRC[0-9]*_[a-zA-Z]*_[0-9]*.JPEG", image))
        ])

        for wnid in set(wnids):
            if not os.path.exists(os.path.join(root, wnid)):
                os.mkdir(os.path.join(root, wnid))

        for wnid, img_file in zip(wnids, img_files):
            # shutil.move(img_file, os.path.join(root, wnid, os.path.basename(img_file)))
            imgs.append(
                (os.path.join(root, wnid, os.path.basename(img_file)), wnid_to_idx[wnid])
            )

        return imgs


    def load_mapping(self):
        file = open(os.path.join(self.root, ImageNetLoader.MAPPING_FILE), 'r')

        wnid_to_idx = {}
        idx_to_wnid = {}
        for line in file.readlines():
            idx, wnid = list(map(str.rstrip, line.split(' ')))
            idx = int(idx)
            wnid_to_idx[wnid] = idx
            idx_to_wnid[idx] = wnid

        return wnid_to_idx, idx_to_wnid

    def load_ground_truth(self):
        file = open(os.path.join(self.root, ImageNetLoader.GROUND_TRUTH_FILE), 'r')

        ground_truth = []
        for line in file.readlines():
            ground_truth.append(int(line))

        return ground_truth

    def load_label_file(self):
        file = open(os.path.join(self.root, ImageNetLoader.LABELS_FILE), 'r')

        wnid_to_classes = {}
        for line in file.readlines():
            wnid, class_tuple = line.split('   ')
            wnid_to_classes[wnid] = list(map(str.rstrip, class_tuple.split(', ')))

        return wnid_to_classes

    @property
    def split_folder(self):
        return os.path.join(self.root, self.split)


def str2dataset(name, device="cuda", train=False):
    if name == "MNIST" or name == "mnist":
        mnist = torchvision.datasets.MNIST(root="./data", train=train, download=False, transform=transforms.ToTensor())

        return (mnist, lambda x: x, lambda x: x)

    elif name == "CIFAR" or name == "cifar":
        if train:
            transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            ])
        else:
            transform = transforms.ToTensor()

        cifar = torchvision.datasets.CIFAR10(root="./data", train=train, download=False, transform=transform)

        mu = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float, device=device).unsqueeze(-1).unsqueeze(-1)
        std = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float, device=device).unsqueeze(-1).unsqueeze(-1)

        normalize = lambda x: (x - mu) / std
        unnormalize = lambda x: x * std + mu

        return (cifar, normalize, unnormalize)

    elif name == "ImageNet" or name == "imagenet":
        imagenet = ImageNetLoader(root="./data/ImageNet", split="val",
                                  transform=transforms.Compose(
                                      [transforms.CenterCrop(size=224),
                                       transforms.ToTensor()])
                                  )

        mu = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float, device=device).unsqueeze(-1).unsqueeze(-1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float, device=device).unsqueeze(-1).unsqueeze(-1)

        normalize = lambda x: (x - mu) / std
        unnormalize = lambda x: x * std + mu

        return (imagenet, normalize, unnormalize)

    elif name == "Toy" or name == "toy":
        toyset = torch.utils.data.TensorDataset(torch.tensor([[0, 1], [1, 0]], dtype=torch.float).view(2, 1, 1, 2),
                                                torch.tensor([1, 0], dtype=torch.long))
        return (toyset, lambda x: x, lambda x: x)

    else:
        raise Exception('data set not supported')
