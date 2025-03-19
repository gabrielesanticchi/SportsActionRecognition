import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from typing import Optional

class PlotUtils:
    """
    Utility class for visualization functions.
    """
    @staticmethod
    def plot_video_frames(video_tensor: torch.Tensor, save_path: Optional[str] = None) -> None:
        """
        Visualize video frames in a single line.
        
        Args:
            video_tensor: Video tensor of shape (C, T, H, W)
            save_path: Optional path to save the visualization
        """
        # Reshape tensor to (T, C, H, W) for make_grid
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        
        # Denormalize the tensor
        # mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        # std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        # video_tensor = video_tensor * std + mean
        
        # Create grid of frames
        grid = make_grid(video_tensor, nrow=video_tensor.size(0), padding=2, normalize=False)
        
        # Convert to numpy and transpose for plotting
        grid_img = grid.permute(1, 2, 0).cpu().numpy()
        
        # Clip values to valid range
        grid_img = np.clip(grid_img, 0, 1)
        
        # Create figure
        plt.figure(figsize=(20, 3))
        plt.imshow(grid_img)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.close()
    
    @staticmethod
    def plot_single_frame(video_tensor: torch.Tensor, frame_idx: int = 0, save_path: Optional[str] = None) -> None:
        """
        Visualize a single frame from the video tensor.
        
        Args:
            video_tensor: Video tensor of shape (C, T, H, W)
            frame_idx: Index of the frame to plot (default: 0)
            save_path: Optional path to save the visualization
        """
        # Select the specified frame and reshape to (C, H, W)
        frame = video_tensor[:, frame_idx, :, :]
        
        # Convert to numpy and transpose for plotting (H, W, C)
        frame_img = frame.permute(1, 2, 0).cpu().numpy()
        
        # Clip values to valid range
        frame_img = np.clip(frame_img, 0, 1)
        
        # Create figure
        plt.figure(figsize=(8, 8))
        plt.imshow(frame_img)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.show()
        plt.close()
        