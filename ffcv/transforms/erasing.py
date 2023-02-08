"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""

from typing import Callable, Optional, Tuple
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from ffcv.pipeline.compiler import Compiler
from dataclasses import replace
import numpy as np
import random
from ffcv.utils import set_seed
from .resized_crop import get_random_crop


class RandomErasing(Operation):
    """Applies random erasing data augmentation with given probability and scale/ratio

    Operates on raw arrays (not tensors).

    @inproceedings{zhong2020random,
    title={Random Erasing Data Augmentation},
    author={Zhong, Zhun and Zheng, Liang and Kang, Guoliang and Li, Shaozi and Yang, Yi},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
    year={2020}
    }
    Parameters
    ----------
    erase_prob : float
        The probability to apply the masking
    scale: tuple
        The lower bound and upper bound of the scale
    ratio: tuple
        The lower bound and upper bound of the ratio
    mean: tuple
        The values to fill into the erased area
    seed: (optional) int
        The seed for sampling
    """

    def __init__(
        self,
        erase_prob: float = 0.5,
        scale=(0.08, 0.4),
        ratio=(0.3, 3),
        mean=[128, 128, 128],
        seed: int = None,
    ):
        super().__init__()
        self.erase_prob = erase_prob
        self.scale = scale
        self.ratio = ratio
        self.mean = mean
        self.seed = seed

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        erase_prob = self.erase_prob

        scale = self.scale
        ratio = self.ratio
        seed = self.seed
        mean = np.array(self.mean)
        if isinstance(scale, tuple):
            scale = np.array(scale)
        if isinstance(ratio, tuple):
            ratio = np.array(ratio)
        if seed is None:

            def erase(images, _):
                height, width = images.shape[-3:-1]

                for dst_ix in my_range(len(images)):
                    if random.uniform(0, 1) > erase_prob:
                        continue
                    i, j, h, w = get_random_crop(
                        height,
                        width,
                        scale,
                        ratio,
                        np.random.rand(5),
                        np.random.rand(5),
                        np.random.rand(5),
                        np.random.rand(5),
                    )
                    for channel in range(images.shape[-1]):
                        images[dst_ix, i : i + h, j : j + w, channel] = mean[channel]
                return images

            erase.is_parallel = True
            return erase

        def erase(images, _, counter):
            height, width = images.shape[-3:-1]
            random.seed(seed + counter)
            N = len(images)
            r = np.zeros((N, 4, 5))
            values = np.zeros(N)
            for i in range(N):
                values[i] = random.uniform(0, 1)
                for j in range(4):
                    for k in range(5):
                        r[i, j, k] = random.uniform(0, 1)

            for dst_ix in my_range(N):
                if values[i] > erase_prob:
                    continue
                i, j, h, w = get_random_crop(
                    height,
                    width,
                    scale,
                    ratio,
                    r[dst_ix, 0],
                    r[dst_ix, 1],
                    r[dst_ix, 2],
                    r[dst_ix, 3],
                )
                for channel in range(images.shape[-1]):
                    images[dst_ix, i : i + h, j : j + w, channel] = mean[channel]
            return images

        erase.with_counter = True
        erase.is_parallel = True
        return erase

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        return (replace(previous_state, jit_mode=True), None)
