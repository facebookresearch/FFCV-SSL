"""
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
import numba as nb
import numbers
import math
import random
from numba import njit


@njit(parallel=False, fastmath=True, inline="always")
def apply_cj(
    im,
    apply_bri,
    bri_ratio,
    apply_cont,
    cont_ratio,
    apply_sat,
    sat_ratio,
    apply_hue,
    hue_factor,
):

    gray = (
        np.float32(0.2989) * im[..., 0]
        + np.float32(0.5870) * im[..., 1]
        + np.float32(0.1140) * im[..., 2]
    )
    one = np.float32(1)
    # Brightness
    if apply_bri:
        im = im * bri_ratio

    # Contrast
    if apply_cont:
        im = cont_ratio * im + (one - cont_ratio) * np.float32(gray.mean())

    # Saturation
    if apply_sat:
        im[..., 0] = sat_ratio * im[..., 0] + (one - sat_ratio) * gray
        im[..., 1] = sat_ratio * im[..., 1] + (one - sat_ratio) * gray
        im[..., 2] = sat_ratio * im[..., 2] + (one - sat_ratio) * gray

    # Hue
    if apply_hue:
        hue_factor_radians = hue_factor * 2.0 * np.pi
        cosA = np.cos(hue_factor_radians)
        sinA = np.sin(hue_factor_radians)
        v1, v2, v3 = 1.0 / 3.0, np.sqrt(1.0 / 3.0), (1.0 - cosA)
        hue_matrix = [
            [
                cosA + v3 / 3.0,
                v1 * v3 - v2 * sinA,
                v1 * v3 + v2 * sinA,
            ],
            [
                v1 * v3 + v2 * sinA,
                cosA + v1 * v3,
                v1 * v3 - v2 * sinA,
            ],
            [
                v1 * v3 - v2 * sinA,
                v1 * v3 + v2 * sinA,
                cosA + v1 * v3,
            ],
        ]
        hue_matrix = np.array(hue_matrix, dtype=np.float64).T
        for row in nb.prange(im.shape[0]):
            im[row] = im[row] @ hue_matrix
    return np.clip(im, 0, 255).astype(np.uint8)


class RandomColorJitter(Operation):
    """Add ColorJitter with probability jitter_prob.
    Operates on raw arrays (not tensors).

    see https://github.com/pytorch/vision/blob/28557e0cfe9113a5285330542264f03e4ba74535/torchvision/transforms/functional_tensor.py#L165
     and https://sanje2v.wordpress.com/2021/01/11/accelerating-data-transforms/
    Parameters
    ----------
    jitter_prob : float, The probability with which to apply ColorJitter.
    brightness (float or tuple of float (min, max)): How much to jitter brightness.
        brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
        or the given [min, max]. Should be non negative numbers.
    contrast (float or tuple of float (min, max)): How much to jitter contrast.
        contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
        or the given [min, max]. Should be non negative numbers.
    saturation (float or tuple of float (min, max)): How much to jitter saturation.
        saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
        or the given [min, max]. Should be non negative numbers.
    hue (float or tuple of float (min, max)): How much to jitter hue.
        hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
        Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(
        self,
        jitter_prob=0.5,
        brightness=0.8,
        contrast=0.4,
        saturation=0.4,
        hue=0.2,
        seed=None,
    ):
        super().__init__()
        self.jitter_prob = jitter_prob

        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(hue, "hue", center=0, bound=(-0.5, 0.5))
        self.seed = seed
        assert self.jitter_prob >= 0 and self.jitter_prob <= 1

    def _check_input(
        self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True
    ):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    f"If {name} is a single number, it must be non negative."
                )
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f"{name} values should be between {bound}")
        else:
            raise TypeError(
                f"{name} should be a single number or a list/tuple with length 2."
            )

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            setattr(self, f"apply_{name}", False)
        else:
            setattr(self, f"apply_{name}", True)
        return tuple(value)

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()

        jitter_prob = self.jitter_prob

        apply_bri = self.apply_brightness
        bri = self.brightness

        apply_cont = self.apply_contrast
        cont = self.contrast

        apply_sat = self.apply_saturation
        sat = self.saturation

        apply_hue = self.apply_hue
        hue = self.hue

        seed = self.seed
        if seed is None:

            def color_jitter(images, _):
                for i in my_range(images.shape[0]):
                    if np.random.rand() > jitter_prob:
                        continue

                    images[i] = apply_cj(
                        images[i].astype("float64"),
                        apply_bri,
                        np.random.uniform(bri[0], bri[1]),
                        apply_cont,
                        np.random.uniform(cont[0], cont[1]),
                        apply_sat,
                        np.random.uniform(sat[0], sat[1]),
                        apply_hue,
                        np.random.uniform(hue[0], hue[1]),
                    )
                return images

            color_jitter.is_parallel = True
            return color_jitter

        def color_jitter(images, _, counter):

            random.seed(seed + counter)
            N = images.shape[0]
            values = np.zeros(N)
            bris = np.zeros(N)
            conts = np.zeros(N)
            sats = np.zeros(N)
            hues = np.zeros(N)
            for i in range(N):
                values[i] = np.float32(random.uniform(0, 1))
                bris[i] = np.float32(random.uniform(bri[0], bri[1]))
                conts[i] = np.float32(random.uniform(cont[0], cont[1]))
                sats[i] = np.float32(random.uniform(sat[0], sat[1]))
                hues[i] = np.float32(random.uniform(hue[0], hue[1]))
            for i in my_range(N):
                if values[i] > jitter_prob:
                    continue
                images[i] = apply_cj(
                    images[i].astype("float64"),
                    apply_bri,
                    bris[i],
                    apply_cont,
                    conts[i],
                    apply_sat,
                    sats[i],
                    apply_hue,
                    hues[i],
                )
            return images

        color_jitter.is_parallel = True
        color_jitter.with_counter = True
        return color_jitter

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        return (replace(previous_state, jit_mode=True), None)


class LabelColorJitter(Operation):
    """ColorJitter info added to the labels. Should be initialized in exactly the same way as
    :cla:`ffcv.transforms.ColorJitter`.
    """

    def __init__(
        self, jitter_prob=0.5, brightness=0, contrast=0, saturation=0, hue=0, seed=None
    ):
        super().__init__()
        self.brightness = RandomColorJitter._check_input(brightness, "brightness")
        self.contrast = RandomColorJitter._check_input(contrast, "contrast")
        self.saturation = RandomColorJitter._check_input(saturation, "saturation")
        self.hue = RandomColorJitter._check_input(
            hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False
        )
        self.jitter_prob = jitter_prob
        self.seed = np.random.RandomState(seed).randint(0, 2**32 - 1)
        assert self.jitter_prob >= 0 and self.jitter_prob <= 1

    def generate_code(self) -> Callable:
        my_range = Compiler.get_iterator()
        jitter_prob = self.jitter_prob
        apply_brightness = self.brightness is not None
        if apply_brightness:
            brightness_min, brightness_max = self.brightness
        apply_contrast = self.contrast is not None
        if apply_contrast:
            contrast_min, contrast_max = self.contrast
        apply_saturation = self.saturation is not None
        if apply_saturation:
            saturation_min, saturation_max = self.saturation
        apply_hue = self.hue is not None
        if apply_hue:
            hue_min, hue_max = self.hue
        seed = self.seed

        def mixer(labels, dst, indices):
            rep = ""
            for i in indices:
                rep += str(i)
            local_seed = (hash(rep) + seed) % 2**31
            dst[:, :-4] = labels

            for i in my_range(labels.shape[0]):
                np.random.seed(local_seed + i)
                if random.uniform(0, 1) < jitter_prob:
                    r = math.trunc(random.uniform(0, 1) * 1e16)
                    a = math.trunc(r / 1e12)
                    r = r - a * 1e12
                    b = math.trunc(r / 1e8)
                    r = r - b * 1e8
                    c = math.trunc(r / 1e4)
                    d = r - c * 1e4

                    # Brightness
                    if apply_brightness:
                        a = (a / 1e4) * (
                            brightness_max - brightness_min
                        ) + brightness_min
                        dst[i, -4] = a
                    else:
                        dst[i, -4] = 1
                    # Contrast
                    if apply_contrast:
                        b = (b / 1e4) * (contrast_max - contrast_min) + contrast_min
                        dst[i, -3] = b
                    else:
                        dst[i, -3] = 1
                    # Saturation
                    if apply_saturation:
                        c = (c / 1e4) * (
                            saturation_max - saturation_min
                        ) + saturation_min
                        dst[i, -2] = c
                    else:
                        dst[i, -2] = 1
                    # Hue
                    if apply_hue:
                        d = (d / 1e4) * (hue_max - hue_min) + hue_min
                        dst[i, -1] = d
                    else:
                        dst[i, -1] = 0
                else:
                    dst[i, -4:] = np.asarray([1, 1, 1, 0], dtype=np.float32)

            return dst

        mixer.is_parallel = True
        mixer.with_indices = True

        return mixer

    def declare_state_and_memory(
        self, previous_state: State
    ) -> Tuple[State, Optional[AllocationQuery]]:
        previous_shape = previous_state.shape
        new_shape = (previous_shape[0] + 4,)
        return (
            replace(previous_state, shape=new_shape, dtype=np.float32),
            AllocationQuery(new_shape, dtype=np.float32),
        )
