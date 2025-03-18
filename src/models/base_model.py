import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
import yaml
import os


class BaseModel(nn.Module):
    """
    Base class for all models.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the base model.
        
        Args:
            config_path: Path to configuration file
        """
        super(BaseModel, self).__init__()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def save_checkpoint(self, epoch: int, optimizer: torch.optim.Optimizer, 
                      scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                      metrics: Optional[Dict[str, float]] = None) -> str:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            optimizer: Optimizer state
            scheduler: Learning rate scheduler state (optional)
            metrics: Evaluation metrics (optional)
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_dir = self.config['logging']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, checkpoint_path)
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str, 
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> Tuple[int, Dict[str, float]]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            optimizer: Optimizer to load state into (optional)
            scheduler: Learning rate scheduler to load state into (optional)
            
        Returns:
            Tuple of (epoch, metrics)
        """
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        self.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        metrics = checkpoint.get('metrics', {})
        
        return epoch, metrics