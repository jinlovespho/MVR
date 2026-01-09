
from omegaconf import OmegaConf
from pho_hypersim import PhoHypersim
from pho_tartanair import PhoTartanAir
from pho_concat_ds import PhoConcatDataset
from pho_sampler import PhoSampler, PhoBatchSampler
from torch.utils.data import DataLoader 


# load full cfg
full_cfg = OmegaConf.load('./pho_cfg.yaml')


# load separate datasets
ds_hypersim = PhoHypersim(full_cfg.data.train.hypersim)
ds_tartanair = PhoTartanAir(full_cfg.data.train.tartanair)

# put together datasets 
train_ds = PhoConcatDataset([ds_hypersim, ds_tartanair])

# set sampler 
train_sampler = PhoSampler(train_ds, tot_frames=48)     # controls how to extract the index 
train_batchsampler = PhoBatchSampler(sampler=train_sampler, batch_size=2)

# set loader 
train_loader = DataLoader(train_ds, batch_sampler=train_batchsampler, num_workers=0, pin_memory=True)


'''
Loader Logic:
    first, the indexes are decided from samplers -> these indexes are passed to the datasets -> datasets retrieve the idx using __getitems__
'''


# training loop
for train_step, batch in enumerate(train_loader):
    print(batch)
    breakpoint()



breakpoint()