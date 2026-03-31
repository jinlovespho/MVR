import torch 
import numpy as np

from extrinsic2pyramid.util.camera_pose_visualizer import CameraPoseVisualizer


def as_homogeneous(ext):
    """
    Accept (..., 3,4) or (..., 4,4) extrinsics, return (...,4,4) homogeneous matrix.
    Supports torch.Tensor or np.ndarray.
    """
    if isinstance(ext, torch.Tensor):
        # If already in homogeneous form
        if ext.shape[-2:] == (4, 4):
            return ext
        elif ext.shape[-2:] == (3, 4):
            # Create a new homogeneous matrix
            ones = torch.zeros_like(ext[..., :1, :4])
            ones[..., 0, 3] = 1.0
            return torch.cat([ext, ones], dim=-2)
        else:
            raise ValueError(f"Invalid shape for torch.Tensor: {ext.shape}")

    elif isinstance(ext, np.ndarray):
        if ext.shape[-2:] == (4, 4):
            return ext
        elif ext.shape[-2:] == (3, 4):
            ones = np.zeros_like(ext[..., :1, :4])
            ones[..., 0, 3] = 1.0
            return np.concatenate([ext, ones], axis=-2)
        else:
            raise ValueError(f"Invalid shape for np.ndarray: {ext.shape}")

    else:
        raise TypeError("Input must be a torch.Tensor or np.ndarray.")





visualizer = CameraPoseVisualizer([-50, 50], [-50, 50], [0, 100])

pose = torch.rand(2,3,4)
pose_h = as_homogeneous(pose).numpy()
visualizer.extrinsic2pyramid(pose_h, 'c', 10)

breakpoint()

for pos in pose_h:
    

    visualizer.extrinsic2pyramid(pos, 'c', 10)
visualizer.save('tmp.jpg')

breakpoint()