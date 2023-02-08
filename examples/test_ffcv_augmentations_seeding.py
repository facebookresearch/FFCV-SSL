# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Example on how to use seeding with FFCV-SSL.
"""

import torch as ch
from torchvision.utils import make_grid
import ffcv
from tqdm import tqdm
import matplotlib.pyplot as plt
from ffcv.fields.basics import IntDecoder
import numpy as np
import os
from ffcv.loader import Loader, OrderOption

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256

def create_validation_plot(da_seed, loader_seed):
    # We specify two da_seed for both transforms
    def create_loader(da_seed, loader_seed):
        transforms = [
            ffcv.transforms.RandomResizedCrop(
                output_size=(224, 224), scale=(0.99, 1), seed=da_seed
            ),
            ffcv.transforms.RandomTranslate(128, seed=da_seed),
            ffcv.transforms.ToTensor(),
            ffcv.transforms.ToDevice(ch.device('cuda:0'), non_blocking=True),
            ffcv.transforms.ToTorchImage(),
            ffcv.transforms.NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
        ]

        label_pipeline = [
            IntDecoder(),
            ffcv.transforms.ToTensor(),
        ]

        loader = Loader(
            path=os.getenv('FFCV_DATAT_PATH'),
            num_workers=10,
            batch_size=3,
            pipelines={
                "image": transforms,
                "label": label_pipeline,
            },
            order=OrderOption.RANDOM,
            seed=loader_seed,
        )
        return loader

    images = []
    for i in range(3):
        loader = create_loader(da_seed, loader_seed)
        images.append([])
        for (x, _) in tqdm(loader, total=3):
            images[-1].append(
                make_grid(
                    x,
                    normalize=True,
                    scale_each=True,
                    nrows=4,
                )
            )
            if len(images[-1]) == 3:
                break
        images[-1] = ch.cat(images[-1], 2)
    images = ch.cat(images, 1)
    images -= images.min()
    images /= images.max()
    images *= 255
    images = images.cpu().numpy().transpose(1, 2, 0).astype("int")

    fig, axs = plt.subplots(1, 1, figsize=(9, 3))
    axs.imshow(images, extent=[0, 1, 0, 1], interpolation="nearest", aspect="auto")

    axs.plot([0.33, 0.33], [0, 1], c="red", linewidth=3)
    axs.plot([0.67, 0.67], [0, 1], c="red", linewidth=3)
    axs.plot([0, 1], [0.33, 0.33], c="green", linewidth=3)
    axs.plot([0, 1], [0.66, 0.66], c="green", linewidth=3)

    axs.set_xticks([])
    axs.set_yticks([])
    fig.text(
        0.02,
        0.5,
        f"load-seed:{loader_seed}  DA-seed:{da_seed}",
        rotation=90,
        color="blue",
        fontweight="bold",
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=12,
    )
    for i, pos in enumerate([0.18, 0.45, 0.75]):
        fig.text(
            0.05,
            pos,
            f"run {3-i}",
            rotation=90,
            color="green",
            fontweight="bold",
            horizontalalignment="center",
            verticalalignment="center",
        )
    for i, pos in enumerate([0.22, 0.53, 0.84]):
        fig.text(
            pos,
            0.93,
            f"batch {i+1}",
            color="red",
            fontweight="bold",
            horizontalalignment="center",
        )
    plt.subplots_adjust(0.06, 0.0, 1, 0.9)
    plt.savefig(f"visual_images_loader{loader_seed}_DA{da_seed}.png", dpi=80)
    plt.close()


if __name__ == "__main__":
    create_validation_plot(None, None)
    create_validation_plot(None, 0)
    create_validation_plot(None, 1)
    create_validation_plot(0, None)
    create_validation_plot(1, None)
    create_validation_plot(0, 0)
