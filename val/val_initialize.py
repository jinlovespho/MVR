import os 
import sys 
sys.path.append(os.getcwd())


def load_data(cfg):

    val_data={}
    
    # load validation datas
    if 'hypersim' in cfg.val.data_list:
        from val_load_dataset import val_load_hypersim
        val_data['hypersim'] = val_load_hypersim(cfg.val.hypersim)
        
    if 'tartanair' in cfg.val.data_list:
        from val_load_dataset import val_load_tartanair
        val_data['tartanair'] = val_load_tartanair(cfg.val.tartanair)
        
    if 'eth3d' in cfg.val.data_list:
        from val_load_dataset import val_load_eth3d
        val_data['eth3d'] = val_load_eth3d(cfg.val.eth3d)
        
    if 'co3d' in cfg.val.data_list:
        pass
    
    if 'scannetpp' in cfg.val.data_list:
        pass 

    return val_data



def load_model(cfg, device, dtype):

    val_models = {}

    # load validation models
    if 'vggt' in cfg.val.model_list: 
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        model.eval()
        val_models['vggt'] = model
        
    elif 'da3' in cfg.val.model_list:
        from depth_anything_3.api import DepthAnything3
        model = DepthAnything3.from_pretrained("depth-anything/DA3-GIANT-1.1").to(device)
        model.eval()
        val_models['da3'] = model


    return val_models 

