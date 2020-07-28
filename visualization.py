import numpy as np

import torch
from constants import class_mapping

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)

# from skimage.exposure import equalize_hist

import argparse

mnist_classes = tuple([str(i) for i in range(10)])

cifar_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

imagenet_class2idx = class_mapping.imagenet_class2idx
imagenet_classes = {val: key for key, val in imagenet_class2idx.items()}

dataset = None


def display_label(ax, label, loc, fontsize=15):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if loc == 'top':
        ax.set_title(label, fontsize=fontsize)
    elif loc == 'left':
        ax.set_ylabel(label, fontsize=fontsize)
    elif loc == 'bottom':
        ax.set_xlabel(label, fontsize=fontsize)
    elif loc == 'right':
        ax.yaxis.set_label_position('right')
        ax.set_ylabel(label, fontsize=fontsize)


def display_image(ax, img, label, loc, fontsize=15):
    display_label(ax, label, loc, fontsize)

    if len(img.shape) == 2:
        ax.imshow(img, cmap='gray')
    else:
        ax.imshow(img)


def display_perturbation(ax, delta, propress=lambda x: x):
    display_label(ax, None, None)

    xmin = delta.min()
    xmax = delta.max()

    delta = (delta - xmin) / ((xmax - xmin) + 1e-30)
    # delta = (delta - xmin) / (xmax - xmin)

    delta = propress(delta)

    if len(delta.shape) == 2:
        ax.imshow(delta, cmap='gray')
    else:
        ax.imshow(delta)


def np2img(array):
    if array.shape[0] == 1:
        return array[0]
    elif array.shape[0] == 3:
        return np.transpose(array, (1, 2, 0))


def display(cln_imgs, labels, adv_imgs, predictions, indices):
    num_imgs = len(indices)

    fig, axes = plt.subplots(figsize=(3.33 * 3, num_imgs * 3), nrows=num_imgs, ncols=3)
    for i, index in enumerate(indices):
        display_image(axes[i, 0],
                      np2img(cln_imgs[index]),
                      classes[labels[index]],
                      loc='left',
                      fontsize=20)

        display_perturbation(axes[i, 1],
                             np2img(adv_imgs[index] - cln_imgs[index])
                             )

        display_image(axes[i, 2],
                      np2img(adv_imgs[index]),
                      classes[predictions[index]],
                      loc='right',
                      fontsize=20)

    plt.subplots_adjust(hspace=0.0, wspace=0.0)
    # plt.subplots_adjust(left=0.1, right=0.9, top=1.0, bottom=0.0)
    plt.subplots_adjust(left=0.05, right=0.95, top=1.0, bottom=0.0)


def plot_linf_vs_wasserstein():
    pattern_linf = "./adversarial_examples/linf_{:4.2f}.pt"
    pattern_l2 = "./adversarial_examples/l2_{:4.2f}.pt"
    pattern_wass = "./adversarial_examples/wasserstein_{:4.2f}.pt"

    lst_eps_linf = [0.05, 0.10, 0.20, 0.40]
    lst_eps_l2 = [0.50, 1.00, 2.00, 4.00]
    lst_eps_wass = [0.05, 0.10, 0.20, 0.40]

    index = 4

    num_imgs = len(lst_eps_linf)

    fig, axes = plt.subplots(figsize=(num_imgs, (3 + 0.3 * 2) / 0.9), nrows=3, ncols=num_imgs)

    def plot(lst_eps, pattern, row_idx):
        for i in range(len(lst_eps)):
            eps = lst_eps[i]

            cln_imgs, labels, adv_imgs, predictions = torch.load(pattern.format(eps))
            adv_imgs = np.clip(adv_imgs, a_min=0., a_max=1.)


            display_image(axes[row_idx, i],
                          np2img(adv_imgs[index]),
                          # r"$\epsilon={:4.2f}$, {:d}".format(eps, int(classes[predictions[index]])),
                          r"$\epsilon={:4.2f}$".format(eps),
                          loc='top',
                          fontsize=14)

            # axes[row_idx, i].set_xlabel(r"$\hat{{y}}={:d}$".format(int(classes[predictions[index]])), fontsize=14))

        return cln_imgs, labels

    plot(lst_eps_linf, pattern_linf, 0)
    plot(lst_eps_l2, pattern_l2, 1)
    cln_imgs, labels = plot(lst_eps_wass, pattern_wass, 2)

    plt.subplots_adjust(hspace=0.3, wspace=0.)
    plt.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.0)

    # display_image(plt.subplot(),
    #               np2img(cln_imgs[index]),
    #               r"clean".format(int(classes[labels[index]])),
    #               loc='top',
    #               fontsize=14)

    fig = plt.figure("clean_img_mnist", figsize=(1, 1))
    ax = fig.add_subplot(111)
    display_image(ax, np2img(cln_imgs[index]), label=None, loc=None)

    fig.subplots_adjust(hspace=0.0, wspace=0.0)
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)


def plot_imagenet_comparison():
    lst_files = ['./adversarial_examples/projected_gradient_imagenet_resnet50.pt',
                 './adversarial_examples/frank_wolfe_imagenet_resnet50_small.pt',
                 './adversarial_examples/frank_wolfe_imagenet_resnet50_large.pt',
                 './adversarial_examples/sinkhorn_imagenet_resnet50.pt',
                 ]
    index = 2

    cln_imgs, labels, adv_imgs, predictions = torch.load(lst_files[0])

    fig, ax = plt.subplots(figsize=(5 / 0.8, 5))
    display_image(ax, np2img(cln_imgs[index]), classes[labels[index]], loc='left', fontsize=75)

    fig.subplots_adjust(hspace=0.0, wspace=0.0)
    fig.subplots_adjust(left=0.2, right=1.0, top=1.0, bottom=0.0)

    def plot_perturbation(lst_imgs, title, propress=lambda x: x):
        num_imgs = len(lst_imgs)
        for i in range(num_imgs):
            fig = plt.figure("{}-{:d}".format(title, i), figsize=(5, 5))
            ax = fig.add_subplot(111)
            display_perturbation(ax, lst_imgs[i], propress)

            fig.subplots_adjust(hspace=0.0, wspace=0.0)
            fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

    def plot_adv_imgs(lst_imgs, lst_perturbations, title):
        num_imgs = len(lst_imgs)
        for i in range(num_imgs):
            fig = plt.figure("{}-{:d}".format(title, i), figsize=(5 / 0.8, 5))
            ax = fig.add_subplot(111)
            display_image(ax,
                          lst_imgs[i],
                          classes[lst_predictions[i]],
                          loc='right',
                          fontsize=75)

            fig.subplots_adjust(hspace=0.0, wspace=0.0)
            fig.subplots_adjust(left=0.0, right=0.8, top=1.0, bottom=0.0)

    lst_perturbations = []
    lst_adv_imgs = []
    lst_predictions = []

    for file in lst_files:
        cln_imgs, labels, adv_imgs, predictions = torch.load(file)
        adv_imgs = np.clip(adv_imgs, a_min=0., a_max=1.)

        lst_perturbations.append(np2img(adv_imgs[index] - cln_imgs[index]))
        lst_adv_imgs.append(np2img(adv_imgs[index]))
        lst_predictions.append(predictions[index])

    # fig_per = plot(lst_perturbations, None, True, figsize=(1, len(lst_files)))
    # fig_per.subplots_adjust(hspace=0.0, wspace=0.0)
    # fig_per.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    plot_perturbation(lst_perturbations, "perturbation")

    # fig_adv = plot(lst_adv_imgs, lst_predictions, False, figsize=(1 / 0.8, len(lst_files)))
    # fig_adv.subplots_adjust(hspace=0.0, wspace=0.0)
    # fig_adv.subplots_adjust(left=0.0, right=0.8, top=1.0, bottom=0.0)
    plot_adv_imgs(lst_adv_imgs, lst_predictions, "adv_imgs")

    # fig_per_zoom = plot(lst_perturbations, None, True, figsize=(1, 4), nrows=4, ncols=1, propress=lambda x: x[50:150, 80:180, :])
    # fig_per_zoom.subplots_adjust(hspace=0.0, wspace=0.0)
    # fig_per_zoom.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    plot_perturbation(lst_perturbations, "perturbation_zoom", propress=lambda x: x[50:150, 80:180, :])


def plot_mnist_cifar_comparision():
    lst_pattern = ['projected_gradient_{}_vanilla.pt',
                   'frank_wolfe_{}_vanilla_small.pt',
                   'frank_wolfe_{}_vanilla_large.pt',
                   'sinkhorn_{}_vanilla.pt',
                   ]

    lst_dataset = ["mnist", "cifar"]

    for dataset in lst_dataset:
        num_methods = len(lst_pattern)
        num_imgs = 4

        classes = eval(dataset + '_classes')

        # fig = plt.figure(figsize=(2 * num_methods + 1, (num_imgs + 0.31 * (num_imgs - 1)) / 0.98))
        fig = plt.figure(figsize=(2 * num_methods + 1, (num_imgs + 0.31 * (num_imgs - 1)) / 0.94))
        gs = fig.add_gridspec(nrows=num_imgs, ncols=2 * num_methods + 1)

        def set_method_name(ax, name):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlabel(name, fontsize=10)

        for i in range(len(lst_pattern)):
            cln_imgs, labels, adv_imgs, predictions = torch.load("./adversarial_examples/" + lst_pattern[i].format(dataset))
            adv_imgs = np.clip(adv_imgs, a_min=0., a_max=1.)

            for j, index in enumerate(range(num_imgs)):
                ax = fig.add_subplot(gs[j, 2 * i + 1])
                display_perturbation(ax, np2img(adv_imgs[index] - cln_imgs[index]))

                ax = fig.add_subplot(gs[j, 2 * i + 2])
                display_image(ax, np2img(adv_imgs[index]), label=classes[predictions[index]], loc='top', fontsize=20)

                ax = fig.add_subplot(gs[j, 0])
                display_image(ax, np2img(cln_imgs[index]), label=classes[labels[index]], loc='top', fontsize=20)

                if index == num_imgs - 1:
                    ax = fig.add_subplot(gs[j, 0])

        fig.subplots_adjust(hspace=0.31, wspace=0.0)
        # fig.subplots_adjust(left=0.0, right=1.0, top=0.98, bottom=0.0)
        fig.subplots_adjust(left=0.0, right=1.0, top=0.94, bottom=0.0)

    plt.show()


if __name__ == "__main__":
    # classes = eval('mnist' + '_classes')
    # plot_linf_vs_wasserstein()
    # plt.show()

    # classes = eval('imagenet' + '_classes')
    # plot_imagenet_comparison()
    # plt.show()

    # plot_mnist_cifar_comparision()
    # plt.show()

    parser = argparse.ArgumentParser()

    parser.add_argument('--file', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--n', type=int, default=2)
    parser.add_argument('--selection', type=str, default='first')

    args = parser.parse_args()

    dataset = args.dataset

    cln_imgs, labels, adv_imgs, predictions = torch.load(args.file)

    if args.selection == 'first':
        indices = np.arange(args.n)
    else:
        indices = np.random.choice(cln_imgs.shape[0], args.n)

    classes = eval(args.dataset + '_classes')

    display(np.clip(cln_imgs, a_min=0., a_max=1.), labels,
            np.clip(adv_imgs, a_min=0., a_max=1.), predictions,
            indices)

    plt.show()
