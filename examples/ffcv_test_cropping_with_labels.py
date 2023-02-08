# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
This file is an example about how one can use FFCV-SSL to retrieve the
parameters of a given data augmentations such as Random Cropping.
"""
import torch as ch
import numpy as np
import ffcv
from ffcv.loader import Loader, OrderOption
from torchvision.utils import make_grid, save_image, draw_bounding_boxes
import numpy as np
import matplotlib.pyplot as plt
from ffcv.fields.basics import IntDecoder
from ffcv.transforms import ToTensor, ToDevice, Squeeze, ToTorchImage, NormalizeImage
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import os

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256

def create_loader():
    image_pipeline = [
        ffcv.transforms.RandomResizedCrop(
            output_size=(224, 224), ratio=(0.4, 1.0), seed=5, scale=(0.08, 1.0)
        ),
        ToTensor(),
        ToDevice(ch.device('cuda:0'), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
        ]

    image_pipeline2 = [
        ffcv.transforms.RandomResizedCrop(
            output_size=(224, 224), ratio=(0.4, 1.0), seed=10, second_seed=5, scale=(0.08, 1.0)
        ),
        ToTensor(),
        ToDevice(ch.device('cuda:0'), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
        ]

    label_pipeline = [
        IntDecoder(),
        ToTensor(),
    ]

    image_pipeline_pad = [
        ffcv.transforms.PadRGBImageDecoder(),
        ToTensor(),
        ToDevice(ch.device('cuda:0'), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
    ]

    label_crops_pipeline1 = [
        ffcv.transforms.LabelRandomResizedCrop(
            output_size=(224, 224),
            ratio=(0.4, 1.0),
            seed=5,
            scale=(0.08, 1.0),
        ),
        ToTensor(),
        Squeeze(),
        ToDevice(ch.device("cpu"), non_blocking=True)
    ]

    label_crops_pipeline2 = [
        ffcv.transforms.LabelRandomResizedCrop(
            output_size=(224, 224),
            ratio=(0.4, 1.0),
            seed=10,
            second_seed=5,
            scale=(0.08, 1.0),
        ),
        ToTensor(),
        Squeeze(),
        ToDevice(ch.device("cpu"), non_blocking=True)
    ]


    loader = Loader(
        os.getenv('FFCV_DATAT_PATH'),
        num_workers=4,
        batch_size=4,
        pipelines={
            "image": image_pipeline,
            "label": label_pipeline,
            "image2": image_pipeline2,
            "crop_image": label_crops_pipeline1,
            "crop_image2": label_crops_pipeline2,
            "full_image": image_pipeline_pad
        },
        order=OrderOption.RANDOM,
        custom_field_mapper={"image2": "image", "crop_image": "image", "crop_image2": "image", "full_image": "image"}
    )
    return loader


if __name__ == "__main__":
    loader = create_loader()
    images = [[], [], []]
    for loaders in tqdm(loader, total=3):
        labels_crops = ch.cat([loaders[3], loaders[4]], dim=1).half()
        images_big = ch.cat((loaders[0], loaders[2]), dim=0)
        save_image(images_big, "./images_cropped.jpeg", normalize=True)
        list_images = []
        for k in range(4):
            bbox = labels_crops[k]
            y1, x1, y2, x2 = bbox[2:6].int()
            new_bbox = ch.tensor((x1, y1, x2, y2), dtype=ch.int)
            new_bbox[2] += new_bbox[0]
            new_bbox[3] += new_bbox[1]
            y1, x1, y2, x2 = bbox[8:12].int()
            new_bbox2 = ch.tensor((x1, y1, x2, y2), dtype=ch.int)
            new_bbox2[2] += new_bbox2[0]
            new_bbox2[3] += new_bbox2[1]
            new_bbox = ch.cat((new_bbox.unsqueeze(0), new_bbox2.unsqueeze(0)), dim=0)
            grid_image = (make_grid(loaders[-1][k].unsqueeze(0), normalize=True)*255).byte()
            new_image = draw_bounding_boxes(grid_image, new_bbox)
            list_images.append(new_image)
        new_image = ch.cat(list_images, dim=2)
        import torchvision
        img = torchvision.transforms.ToPILImage()(new_image)
        img.save("./images_with_bouding_boxes.jpeg")
        exit(0)
