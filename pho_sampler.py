import os 
import glob
import random 
from torch.utils.data import Sampler 




class PhoBatchSampler(Sampler):
    def __init__(self, sampler, batch_size):
        
        self.sampler = sampler 
        self.batch_size = batch_size 
        
    
    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)
        
    
    def __iter__(self):
        print('batch sampler')
        # HARD CODED 
        image_num = 4
        aspect_ratio = 1.0 
        
        # update the sampler algorithm before sampling
        self.sampler.update_parameters(image_num=image_num, aspect_ratio=aspect_ratio)    
        sampler_iter = iter(self.sampler)
        
        # breakpoint()
        while True: 
            batch = []
            try:
                for _ in range(self.batch_size):
                    # sample indexes using self.sampler=PhoSampler 
                    items = next(sampler_iter) 
                    print('batchsampler: ', items)
                    batch.append(items)
            except StopIteration:
                pass 
            
            if len(batch) == 0:
                break 
            
            # batch = [items1, items2, . . . ]
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
        
        
        
        

