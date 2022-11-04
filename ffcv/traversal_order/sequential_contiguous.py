from typing import Sequence, TYPE_CHECKING
import numpy as np

from torch.utils.data import DistributedSampler

from .base import TraversalOrder
from typing import TypeVar, Optional, Iterator

if TYPE_CHECKING:
    from ..loader.loader import Loader

T_co = TypeVar('T_co', covariant=True)

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
        # indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

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
