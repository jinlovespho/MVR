import os 
import glob
import random
import numpy as np
from typing import  Optional
from torch.utils.data import DistributedSampler, Sampler


class PhoBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, epoch=0, seed=42,):
        
        self.sampler = sampler
        self.batch_size = batch_size 
        self.rng = random.Random()

        # Set the epoch for the sampler
        self.set_epoch(epoch + seed)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)
        self.epoch = epoch
        self.rng.seed(epoch * 100)

    def __iter__(self):

        # self.sampler.update_parameters(image_num=image_num, aspect_ratio=aspect_ratio)
        sampler_iter = iter(self.sampler)
        
        batch = []

        for item in sampler_iter:
            batch.append(item)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if len(batch) > 0:
            yield batch
            

    def __len__(self):
        return len(self.sampler) // self.batch_size 


class PhoSampler(DistributedSampler):
    """
    Extends PyTorch's DistributedSampler to include dynamic aspect_ratio and image_num
    parameters, which can be passed into the dataset's __getitem__ method.
    """
    def __init__(self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ):
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last
        )

        # self.image_num = None

    def __iter__(self):
        """
        Yields a sequence of (index, image_num, aspect_ratio).
        Relies on the parent class's logic for shuffling/distributing
        the indices across replicas, then attaches extra parameters.
        """
        indices_iter = super().__iter__()

        for idx in indices_iter:
            # yield (idx, self.image_num,)
            yield idx 

        
    def update_parameters(self, image_num):
        """
        Updates dynamic parameters for each new epoch or iteration.

        Args:
            aspect_ratio: The aspect ratio to set.
            image_num: The number of images to set.
        """
        self.image_num = image_num
