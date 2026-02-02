import os 
import glob 


def val_load_hypersim(data_cfg):
    
    ann_path = data_cfg.ann_path
    lq_root_path = data_cfg.lq_root_path
    hq_root_path = data_cfg.hq_root_path
    
    breakpoint()
    

    
    
def val_load_tartanair(data_cfg):
    pass    


    
    
def val_load_eth3d(data_cfg):

    ann_path = data_cfg.ann_path
    lq_root_path = data_cfg.lq_root_path
    hq_root_path = data_cfg.hq_root_path
    gt_depth_root_path = data_cfg.gt_depth_root_path
    
    
    # load data paths 
    lq_paths = sorted(glob.glob(f'{lq_root_path}/image/*/*.png'))
    hq_paths = sorted(glob.glob(f'{hq_root_path}/image/*/*.JPG'))
    depth_paths = sorted(glob.glob(f'{gt_depth_root_path}/*/*/*/*/*.JPG'))
    
    
    # safety check
    assert len(lq_paths) == len(hq_paths) == len(depth_paths)
    assert len(lq_paths) != 0 
    assert len(hq_paths) != 0
    assert len(depth_paths) != 0
    
    
    files=[]
    for lq_path, hq_path, depth_path in zip(lq_paths, hq_paths, depth_paths):

        
        # safety check 
        lq_id = lq_path.split('/')[-1].split('.')[0]
        hq_id = hq_path.split('/')[-1].split('.')[0]
        depth_id = depth_path.split('/')[-1].split('.')[0]
        assert lq_id == hq_id == depth_id
        
        
        # data info 
        scene_id = lq_path.split('/')[-2]
        img_id = lq_id 
        

        files.append({
                "lq_path": lq_path,
                'hq_path': hq_path,
                'depth_path': depth_path,
                'scene_id': scene_id,
                'img_id': img_id
            })

    return files 


    
