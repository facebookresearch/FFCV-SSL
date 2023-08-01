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
import numpy as np
import random

class Rotate(Operation):
    """Rotate the image randomly with a given probability
    Parameters
    ----------
        solarization_prob (float): probability of the image being solarized. Default value is 0.5
        angle (float): define the bounds of the uniform angle distribution
        threshold (float): all pixels equal or above this value are inverted.
    """

    def __init__(self, rotate_prob: float = 0.5, angle: float = 0.2, seed: int = None):
        super().__init__()
        self.rot_prob = rotate_prob
        self.angle = angle
        self.seed = seed

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        rot_prob = self.rot_prob
        angle = self.angle
        seed = self.seed

        def rotate(images, dst, counter):
            if seed is not None:
                random.seed(seed + counter)
                values = np.zeros(images.shape[0])
                angles = np.zeros(images.shape[0])
                for i in range(images.shape[0]):
                    values[i] = random.uniform(0, 1)
                    angles[i] = random.uniform(-angle, angle)
            else:
                values = np.random.rand(images.shape[0])
                angles = np.random.rand(images.shape[0]) * 2 * angle - angle
            N, H, W, _ = images.shape
            x = np.repeat(np.arange(H) - H // 2, W)
            y = ((np.arange(W) - W // 2) * np.ones((H, 1))).reshape(-1)
            coords = np.stack((x, y), 1)
            for i in my_range(N):
                if values[i] < rot_prob:
                    M = [
                        [np.cos(angles[i]), -np.sin(angles[i])],
                        [np.sin(angles[i]), np.cos(angles[i])],
                    ]
                    M = np.array(M).T
                    src_coords = (coords @ M).astype(np.int32) + np.array(
                        [H // 2, W // 2]
                    )

                    for j in range(H * W):
                        if 0 <= src_coords[j, 0] < H and 0 <= src_coords[j, 1] < W:
                            dst[
                                i,
                                np.int32(coords[j, 0]) + H // 2,
                                np.int32(coords[j, 1]) + W // 2,
                                :,
                            ] = images[i, src_coords[j, 0], src_coords[j, 1], :]
                        else:
                            dst[i,
                                np.int32(coords[j, 0]) + H // 2,
                                np.int32(coords[j, 1]) + W // 2,
                                :,
                                ] = 0.
                else:
                    dst[i] = images[i]
            return dst

        rotate.is_parallel = True
        rotate.with_counter = True
        return rotate

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        assert previous_state.jit_mode
        return (
            previous_state,
            AllocationQuery(previous_state.shape, dtype=np.float32),
        )
