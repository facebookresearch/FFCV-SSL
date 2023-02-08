"""
File: translate.py
Project: torchstrap
-----
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""


import numpy as np
from typing import Callable, Optional, Tuple
from dataclasses import replace
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from ffcv.pipeline.compiler import Compiler
import random


class RandomTranslate(Operation):
    """Translate each image randomly in vertical and horizontal directions
    up to specified number of pixels.
    Parameters
    ----------
    padding : int
        Max number of pixels to translate in any direction.
    fill : tuple
        An RGB color ((0, 0, 0) by default) to fill the area outside the shifted image.
    """

    def __init__(
        self, padding: int, fill: Tuple[int, int, int] = (128, 128, 128), seed=None
    ):
        super().__init__()
        self.padding = padding
        self.seed = seed
        self.fill = np.array(fill)
        self.counter = 0

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        pad = self.padding
        fill = self.fill

        if self.seed is None:

            def translate(images, _):
                N, H, W, _ = images.shape

                for i in my_range(N):
                    h = np.random.randint(low=-pad, high=pad + 1)
                    w = np.random.randint(low=-pad, high=pad + 1)

                    h_start = max(-h, 0)
                    w_start = max(-w, 0)
                    h_start_ = max(h, 0)
                    w_start_ = max(w, 0)
                    H_ = H - np.abs(h)
                    W_ = W - np.abs(w)

                    images[i, h_start : h_start + H_, w_start : w_start + W_] = images[
                        i, h_start_ : h_start_ + H_, w_start_ : w_start_ + W_
                    ]
                    images[i, :h_start] = fill
                    images[i, h_start + H_ :] = fill
                    images[i, :, :w_start] = fill
                    images[i, :, w_start + W_ :] = fill

                return images
            translate.is_parallel = True
            return translate

        seed = self.seed

        def translate(images, _, counter):
            np.random.seed(counter + seed)
            N, H, W, _ = images.shape

            hw = np.zeros((N, 2))
            for i in range(N):
                hw[i, 0] = np.random.randint(low=-pad, high=pad + 1)
                hw[i, 1] = np.random.randint(low=-pad, high=pad + 1)

            for i in my_range(N):

                h, w = hw[i]
                h_start = max(-h, 0)
                w_start = max(-w, 0)
                h_start_ = max(h, 0)
                w_start_ = max(w, 0)
                H_ = H - np.abs(h)
                W_ = W - np.abs(w)
                images[i, h_start : h_start + H_, w_start : w_start + W_] = images[
                    i, h_start_ : h_start_ + H_, w_start_ : w_start_ + W_
                ]
                images[i, :h_start] = fill
                images[i, h_start + H_ :] = fill
                images[i, :, :w_start] = fill
                images[i, :, w_start + W_ :] = fill

            return images

        translate.is_parallel = True
        translate.with_counter = True
        return translate

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        return (
            replace(previous_state, jit_mode=True),
            None,
        )


class LabelTranslate(Operation):
    """ColorJitter info added to the labels. Should be initialized in exactly the same way as
    :cla:`transforms.Translate`.
    """

    def __init__(
        self, padding: int, fill: Tuple[int, int, int] = (0, 0, 0), seed: int = None
    ):
        super().__init__()
        self.padding = padding
        self.fill = np.array(fill)

        self.seed = np.random.RandomState(seed).randint(0, 2**32 - 1)

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        pad = self.padding
        seed = self.seed
        shift = 2 * pad + 1

        def translate(labels, temp_array, indices):
            rep = ""
            for i in indices:
                rep += str(i)
            local_seed = (hash(rep) + seed) % 2**31

            temp_array[:, :-2] = labels
            for i in my_range(labels.shape[0]):
                np.random.seed(local_seed + i)
                value = np.random.randint(low=0, high=shift * shift)
                temp_array[i, -2] = value // shift
                temp_array[i, -1] = value % shift
            return temp_array

        translate.is_parallel = True
        translate.with_indices = True

        return translate

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        previous_shape = previous_state.shape
        new_shape = (previous_shape[0] + 2,)
        return (
            replace(previous_state, shape=new_shape, dtype=np.float32),
            AllocationQuery(new_shape, dtype=np.float32),
        )
