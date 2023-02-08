# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
This files present two functions to return the data sequentially.
The first one: Sequential split a list of n indices in such a way that
each data point in this list is transfer to different gpus. So if we have a list
(n1,n2,n3,n4,...,nk) and 4 gpus, n1 will be on gpu1, 
n2 on gpu2, n3 on gpu3, n4 on gpu4 and n5 to gpu1.
Thse second one: SequentialContiguous split the list in such a way that the indices are splitted
in sublist of size n/(number of gpus) and each sublist is associated to a specific gpu. In this
instance, (n1,n2,n3,4) will be associated to gpu1 while (n5,n6,n7,n8) will be associated to gpu2.
"""
from typing import Sequence, TYPE_CHECKING
import numpy as np
import torch
import math

from torch.utils.data import DistributedSampler

from .base import TraversalOrder
from typing import TypeVar, Optional, Iterator

if TYPE_CHECKING:
    from ..loader.loader import Loader
    
T_co = TypeVar('T_co', covariant=True)

# This function return the data sequentially using the
# pytorch DistributedSampler. In consequence, when using distributed
# training the indices[self.rank:self.total_size:self.num_replicas]
# are returned. 
class Sequential(TraversalOrder):
    
    def __init__(self, loader:'Loader'):
        super().__init__(loader)
        
        if self.distributed:
            self.sampler = DistributedSampler(self.indices,
                                              shuffle=False,
                                              seed=self.seed,
                                              drop_last=False)
        

    def sample_order(self, epoch: int) -> Sequence[int]:
        if not self.distributed:
            return self.indices
        
        self.sampler.set_epoch(epoch)
        
        return self.indices[np.array(list(self.sampler))]

# This function return the data sequentially using the
# DistributedSamplerProxy. In consequence, when using distributed
# training the indices[self.num_samples * self.rank : self.num_samples * self.rank + self.num_samples]
# are returned. 
class SequentialContiguous(TraversalOrder):
    
    def __init__(self, loader:'Loader'):
        super().__init__(loader)
        
        if self.distributed:
            self.sampler = DistributedSamplerProxy(self.indices,
                                              shuffle=False,
                                              seed=self.seed,
                                              drop_last=False)

    def sample_order(self, epoch: int) -> Sequence[int]:
        if not self.distributed:
            return self.indices
        
        self.sampler.set_epoch(epoch)
        self.indices = self.loader.indices
        return self.indices[np.array(list(self.sampler))]

class DistributedSamplerProxy(DistributedSampler):
    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset : offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

