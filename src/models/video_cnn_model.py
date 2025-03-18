import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any, Optional, Tuple
from src.models.base_model import BaseModel


class VideoCNN(BaseModel):
    """
    3D CNN model for video classification.
    Uses a 2D CNN backbone with additional temporal layers.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the Video CNN model.
        
        Args:
            config_path: Path to configuration file
        """
        super(VideoCNN, self).__init__(config_path)
        
        # Get model parameters from config
        self.feature_extractor = self.config['model']['feature_extractor']
        self.pretrained = self.config['model']['pretrained']
        self.num_classes = self.config['model']['num_classes']
        self.sequence_length = self.config['model']['sequence_length']
        self.dropout_rate = self.config['model']['dropout']
        
        # Initialize the 2D CNN backbone
        self._init_backbone()
        
        # 3D convolutional layers for temporal modeling
        self.temporal_layers = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2 * 2 * 2, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, self.num_classes)
        )
    
    def _init_backbone(self):
        """
        Initialize the 2D CNN backbone based on configuration.
        """
        # Use ResNet-18 as the backbone (can be changed in config)
        if self.feature_extractor == 'resnet18':
            backbone = models.resnet18(pretrained=self.pretrained)
        elif self.feature_extractor == 'resnet50':
            backbone = models.resnet50(pretrained=self.pretrained)
        else:
            raise ValueError(f"Unsupported feature extractor: {self.feature_extractor}")
        
        # Remove the final fully connected layer
        modules = list(backbone.children())[:-2]
        self.backbone = nn.Sequential(*modules)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, sequence_length, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        batch_size, channels, seq_len, height, width = x.size() #
        
        # Reshape for 2D CNN: (B, C, T, H, W) -> (B*T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(batch_size * seq_len, channels, height, width)
        
        # Pass through backbone CNN
        features = self.backbone(x) # torch.Size([128, 512, 7, 7])
        
        # Reshape for 3D CNN: (B*T, C, H, W) -> (B, C, T, H, W)
        _, channels, height, width = features.size()
        features = features.view(batch_size, seq_len, channels, height, width)
        features = features.permute(0, 2, 1, 3, 4).contiguous()
        
        # Apply temporal layers
        temporal_features = self.temporal_layers(features)
        
        # Apply fully connected layers
        output = self.fc_layers(temporal_features)
        
        return output


class SimplerVideoCNN(BaseModel):
    """
    A simpler 3D CNN model for soccer event classification.
    Uses a 2D CNN for feature extraction followed by LSTM for temporal modeling.
    May be more suitable for smaller datasets.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the Simpler Video CNN model.
        
        Args:
            config_path: Path to configuration file
        """
        super(SimplerVideoCNN, self).__init__(config_path)
        
        # Get model parameters from config
        self.feature_extractor = self.config['model']['feature_extractor']
        self.pretrained = self.config['model']['pretrained']
        self.num_classes = self.config['model']['num_classes']
        self.sequence_length = self.config['model']['sequence_length']
        self.dropout_rate = self.config['model']['dropout']
        
        # Initialize the 2D CNN backbone
        self._init_backbone()
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=self.dropout_rate if self.dropout_rate > 0 else 0,
            bidirectional=True
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, self.num_classes)
        )
    
    def _init_backbone(self):
        """
        Initialize the 2D CNN backbone based on configuration.
        """
        # Use ResNet-18 as the backbone (can be changed in config)
        if self.feature_extractor == 'resnet18':
            backbone = models.resnet18(pretrained=self.pretrained)
        elif self.feature_extractor == 'resnet50':
            backbone = models.resnet50(pretrained=self.pretrained)
        else:
            raise ValueError(f"Unsupported feature extractor: {self.feature_extractor}")
        
        # Remove the final fully connected layer
        modules = list(backbone.children())[:-2]
        self.backbone = nn.Sequential(*modules)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, sequence_length, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        batch_size, channels, seq_len, height, width = x.size()
        
        # Process each frame with the CNN backbone
        frame_features = []
        for t in range(seq_len):
            # Get the t-th frame
            frame = x[:, :, t, :, :]
            
            # Pass through backbone
            features = self.backbone(frame)
            
            # Global average pooling
            features = self.global_pool(features).squeeze(-1).squeeze(-1)
            
            frame_features.append(features)
        
        # Stack frame features
        sequence = torch.stack(frame_features, dim=1)  # (batch_size, seq_len, feature_dim)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(sequence)
        
        # Use the final LSTM output
        lstm_out = lstm_out[:, -1, :]
        
        # Fully connected layers
        output = self.fc(lstm_out)
        
        return output