# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
Example on how to return multiple views of a given images with FFCV-SSL.
"""

import torch as ch
import numpy as np
from ffcv.loader import Loader, OrderOption
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
import ffcv
from tqdm import tqdm
from ffcv.fields.basics import IntDecoder
import os

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256

def create_loader():
    image_pipeline1 = [
        ffcv.transforms.RandomResizedCrop((224, 224), ratio=(0.4, 1)),
        ffcv.transforms.RandomGrayscale(seed=0),
        ffcv.transforms.ToTensor(),
        ffcv.transforms.ToDevice(ch.device('cuda:0'), non_blocking=True),
        ffcv.transforms.ToTorchImage(),
        ffcv.transforms.NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
    ]
    image_pipeline2 = [
        ffcv.transforms.RandomResizedCrop((224, 224), ratio=(0.4, 1)),
        ffcv.transforms.RandomGrayscale(seed=0),
        ffcv.transforms.ToTensor(),
        ffcv.transforms.ToDevice(ch.device('cuda:0'), non_blocking=True),
        ffcv.transforms.ToTorchImage(),
        ffcv.transforms.NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
    ]
    image_pipeline3 = [
        ffcv.transforms.RandomResizedCrop((224, 224), ratio=(0.4, 1)),
        ffcv.transforms.RandomHorizontalFlip(flip_prob=0.5),
        ffcv.transforms.RandomSolarization(),
        ffcv.transforms.RandomColorJitter(),
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
        os.getenv('FFCV_DATAT_PATH'),
        num_workers=10,
        batch_size=3,
        pipelines={
            "image": image_pipeline1,
            "image2": image_pipeline2,
            "image3": image_pipeline3,
            "label": label_pipeline,
        },
        # We need this custom mapper to map the additional pipeline to
        # the label used in the dataset (image in this case)
        custom_field_mapper={"image2": "image", "image3": "image"},
        order=OrderOption.RANDOM,
    )
    return loader


if __name__ == "__main__":
    loader = create_loader()
    images = [[], [], []]
    for (x1, y, x2, x3) in tqdm(loader, total=3):
        for i, x in enumerate([x1, x2, x3]):
            images[i].append(
                make_grid(
                    x,
                    normalize=True,
                    scale_each=True,
                    nrows=4,
                )
            )
        if len(images[-1]) == 3:
            break

    for i in range(3):

        images[i] = ch.cat(images[i], 2)
        images[i] -= images[i].min()
        images[i] /= images[i].max()
        images[i] *= 255
        images[i] = images[i].cpu().numpy().transpose(1, 2, 0).astype("int")

    images = np.concatenate(images, 0)

    fig, axs = plt.subplots(1, 1, figsize=(9, 3))
    axs.imshow(images, extent=[0, 1, 0, 1], interpolation="nearest", aspect="auto")

    axs.plot([0.33, 0.33], [0, 1], c="red", linewidth=3)
    axs.plot([0.67, 0.67], [0, 1], c="red", linewidth=3)
    axs.plot([0, 1], [0.67, 0.67], c="green", linewidth=3)
    axs.plot([0, 1], [0.33, 0.33], c="green", linewidth=3)

    axs.set_xticks([])
    axs.set_yticks([])
    for i, pos in enumerate([0.15, 0.45, 0.75]):
        fig.text(
            0.02,
            pos,
            f"view {3-i}",
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
    plt.subplots_adjust(0.03, 0.0, 1, 0.9)
    plt.savefig("visual_images_ssl.png", dpi=80)
    plt.close()
