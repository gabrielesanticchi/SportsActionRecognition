import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import yaml
import time
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.utils.plot_utils import PlotUtils

class Trainer:
    """
    Trainer class for soccer event classification models.
    """
    
    def __init__(self, model: nn.Module, dataloaders: Dict[str, DataLoader], 
                config_path: str = "config/config.yaml"):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            dataloaders: Dictionary of DataLoaders for 'train', 'val', and 'test'
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set up model, dataloaders, and device
        self.model = model
        self.dataloaders = dataloaders
        self.device = torch.device(f"cuda:{self.config['training']['devices'][0]}" if 
                                 self.config['training']['devices'] and torch.cuda.is_available() 
                                 else "cpu")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Set up loss function, optimizer, and scheduler
        self._setup_training()
        
        # Set up logging
        self._setup_logging()
    
    def _setup_training(self):
        """
        Set up loss function, optimizer, and learning rate scheduler.
        """
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        if self.config['training']['lr_scheduler']['use']:
            scheduler_type = self.config['training']['lr_scheduler']['type']
            
            if scheduler_type == 'reduce_on_plateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=self.config['training']['lr_scheduler']['factor'],
                    patience=self.config['training']['lr_scheduler']['patience'],
                    verbose=True
                )
            elif scheduler_type == 'step':
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=self.config['training']['lr_scheduler']['step_size'],
                    gamma=self.config['training']['lr_scheduler']['factor']
                )
            elif scheduler_type == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.config['training']['num_epochs']
                )
            else:
                self.scheduler = None
        else:
            self.scheduler = None
        
        # Early stopping
        self.early_stopping = self.config['training']['early_stopping']['use']
        self.patience = self.config['training']['early_stopping']['patience']
        self.min_delta = self.config['training']['early_stopping']['min_delta']
        self.best_val_loss = float('inf')
        self.counter = 0
    
    def _setup_logging(self):
        """
        Set up logging.
        """
        log_dir = self.config['logging']['log_dir']
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        fh = logging.FileHandler(os.path.join(log_dir, 'training.log'))
        fh.setLevel(logging.INFO)
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def train(self) -> Tuple[Dict[str, List[float]], str]:
        """
        Train the model.
        
        Returns:
            Tuple of (training history, best model path)
        """
        num_epochs = self.config['training']['num_epochs']
        checkpoint_dir = self.config['logging']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize metrics history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_f1': []
        }
        
        # Best model path
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
        
        self.logger.info(f"Starting training on {self.device}")
        self.logger.info(f"Model: {type(self.model).__name__}")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training phase
            train_loss, train_acc = self._train_epoch()
            
            # Validation phase
            val_loss, val_metrics = self._validate()
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1'])
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % self.config['logging']['save_freq'] == 0:
                metrics = {
                    'val_loss': val_loss,
                    'val_accuracy': val_metrics['accuracy'],
                    'val_f1': val_metrics['f1']
                }
                checkpoint_path = self.model.save_checkpoint(epoch, self.optimizer, self.scheduler, metrics)
                self.logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.counter = 0  # Reset early stopping counter
                
                # Save best model
                torch.save(self.model.state_dict(), best_model_path)
                self.logger.info(f"Best model saved: {best_model_path}")
            else:
                self.counter += 1  # Increment early stopping counter
            
            # Log epoch results
            epoch_time = time.time() - start_time
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Time: {epoch_time:.2f}s - "
                f"Train Loss: {train_loss:.4f} - "
                f"Train Acc: {train_acc:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"Val Acc: {val_metrics['accuracy']:.4f} - "
                f"Val F1: {val_metrics['f1']:.4f}"
            )
            
            # Early stopping
            if self.early_stopping and self.counter >= self.patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Final evaluation on test set
        self.logger.info("Training completed. Evaluating on test set...")
        self._load_best_model(best_model_path)
        test_loss, test_metrics = self._evaluate('test')
        
        self.logger.info(
            f"Test Results - "
            f"Loss: {test_loss:.4f} - "
            f"Accuracy: {test_metrics['accuracy']:.4f} - "
            f"Precision: {test_metrics['precision']:.4f} - "
            f"Recall: {test_metrics['recall']:.4f} - "
            f"F1: {test_metrics['f1']:.4f}"
        )
        
        return history, best_model_path
    
    def _train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        # Get train dataloader and log frequency
        train_loader = self.dataloaders['train']
        log_freq = self.config['logging']['log_freq']
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
        
        for i, (inputs, labels) in pbar:
            # Move inputs and labels to device
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            # PlotUtils.plot_video_frames(inputs[-1], save_path='video_frames.png') 
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels) #NOTE:  In the training code, you're using nn.CrossEntropyLoss(), which internally combines a LogSoftmax and NLLLoss.
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            if (i + 1) % log_freq == 0:
                pbar.set_postfix({
                    'loss': running_loss / (i + 1),
                    'acc': accuracy_score(all_labels, all_preds)
                })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    def _validate(self) -> Tuple[float, Dict[str, float]]:
        """
        Validate the model.
        
        Returns:
            Tuple of (average loss, metrics dictionary)
        """
        return self._evaluate('val')
    
    def _evaluate(self, split: str) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate the model on a specific data split.
        
        Args:
            split: Data split to evaluate on ('val' or 'test')
        
        Returns:
            Tuple of (average loss, metrics dictionary)
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        # Get dataloader
        dataloader = self.dataloaders[split]
        
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc=f"Evaluating on {split}"):
                # Move inputs and labels to device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Update statistics
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = running_loss / len(dataloader)
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='weighted'),
            'recall': recall_score(all_labels, all_preds, average='weighted'),
            'f1': f1_score(all_labels, all_preds, average='weighted'),
            'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist()
        }
        
        return avg_loss, metrics
    
    def _load_best_model(self, model_path: str):
        """
        Load the best model weights.
        
        Args:
            model_path: Path to the model weights
        """
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))