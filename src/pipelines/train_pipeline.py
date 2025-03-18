import os
import torch
import yaml
import argparse
import json
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple

from src.data.data_loader import SoccerEventDataLoader
from src.data.dataset import create_data_loaders
from src.models.video_cnn_model import VideoCNN, SimplerVideoCNN
from src.trainers.trainer import Trainer


class TrainingPipeline:
    """
    Pipeline for training soccer event classification models.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the training pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set random seed for reproducibility
        self.seed = self.config['training']['seed']
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        
        # Create output directories
        os.makedirs(self.config['logging']['log_dir'], exist_ok=True)
        os.makedirs(self.config['logging']['checkpoint_dir'], exist_ok=True)
        os.makedirs(self.config['data']['metadata_dir'], exist_ok=True)
    
    def run(self) -> Tuple[Dict[str, Any], str]:
        """
        Run the full training pipeline.
        
        Returns:
            Tuple of (training history, best model path)
        """
        # Step 1: Load and prepare data
        data_loader = SoccerEventDataLoader(self.config_path)
        event_df = data_loader.load_data()
        
        # Step 2: Split data into train/val/test sets
        train_df, val_df, test_df = data_loader.create_train_val_test_split(event_df)
        
        # Step 3: Create PyTorch data loaders
        dataloaders = create_data_loaders(train_df, val_df, test_df, self.config_path)
        
        # Step 4: Initialize model
        model_type = self.config['model']['type']
        if model_type == 'video_cnn':
            model = VideoCNN(self.config_path)
        elif model_type == 'simpler_video_cnn':
            model = SimplerVideoCNN(self.config_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Step 5: Train the model
        trainer = Trainer(model, dataloaders, self.config_path)
        history, best_model_path = trainer.train()
        
        # Step 6: Save training history
        self._save_history(history)
        
        # Step 7: Plot training curves
        self._plot_training_curves(history)
        
        return history, best_model_path
    
    def _save_history(self, history: Dict[str, Any]) -> None:
        """
        Save training history to file.
        
        Args:
            history: Dictionary of training metrics
        """
        history_path = os.path.join(self.config['logging']['log_dir'], 'training_history.json')
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_history = {}
        for key, value in history.items():
            if hasattr(value, 'tolist'):
                serializable_history[key] = value.tolist()
            else:
                serializable_history[key] = value
        
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
    
    def _plot_training_curves(self, history: Dict[str, Any]) -> None:
        """
        Plot training and validation curves.
        
        Args:
            history: Dictionary of training metrics
        """
        # Create figure directory
        figure_dir = os.path.join(self.config['logging']['log_dir'], 'figures')
        os.makedirs(figure_dir, exist_ok=True)
        
        # Plot loss curves
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(figure_dir, 'loss_curves.png'))
        plt.close()
        
        # Plot accuracy curves
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_acc'], label='Training Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(figure_dir, 'accuracy_curves.png'))
        plt.close()
        
        # Plot F1 score curve
        plt.figure(figsize=(10, 6))
        plt.plot(history['val_f1'], label='Validation F1 Score')
        plt.title('F1 Score Curve')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(figure_dir, 'f1_curve.png'))
        plt.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train soccer event classification model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    return parser.parse_args()


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()
    
    # Run training pipeline
    pipeline = TrainingPipeline(args.config)
    history, best_model_path = pipeline.run()
    
    print(f"Training completed. Best model saved at: {best_model_path}")
    print(f"Final validation metrics:")
    print(f"  Accuracy: {history['val_acc'][-1]:.4f}")
    print(f"  F1 Score: {history['val_f1'][-1]:.4f}")