import os 
import glob
import random 
from torch.utils.data import Sampler 




class PhoBatchSampler(Sampler):
    def __init__(self, sampler, batch_size):
        
        self.sampler = sampler 
        self.batch_size = batch_size 
        self.epoch = 0
        
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        random.seed(epoch)
        
    
    def __iter__(self):
        image_num = 4
        aspect_ratio = 1.0

        self.sampler.update_parameters(
            image_num=image_num,
            aspect_ratio=aspect_ratio
        )

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




class PhoSampler(Sampler):
    def __init__(self, dataset, tot_frames):
        
        self.dataset = dataset 
        self.tot_frames = tot_frames 
        self.epoch = 0
        self.image_num = 4
        self.aspect_ratio = 1.0 
        
    
    def __len__(self):
        return len(self.dataset)


    def update_parameters(self, image_num, aspect_ratio):
        self.image_num = image_num
        self.aspect_ratio = aspect_ratio


    def __iter__(self):
        
        # if we receive more than one dataset, we need to consider the sampling ratio
        num_ds = len(self.dataset.datasets)
        
        
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        for idx in indices:
            yield (idx, self.image_num, self.aspect_ratio)
        
        
        
        

