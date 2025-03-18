import os
import torch
import yaml
import argparse
import numpy as np
import cv2
from typing import Dict, List, Tuple, Any, Optional
from torchvision import transforms

from src.models.video_cnn_model import VideoCNN, SimplerVideoCNN
from src.data.dataset import VideoFrameTransform


class InferencePipeline:
    """
    Pipeline for making predictions on new soccer event clips.
    """
    
    def __init__(self, model_path: str, config_path: str = "config/config.yaml"):
        """
        Initialize the inference pipeline.
        
        Args:
            model_path: Path to the trained model weights
            config_path: Path to configuration file
        """
        self.model_path = model_path
        self.config_path = config_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize model
        self._init_model()
        
        # Set up video frame transform
        resize_dims = tuple(self.config['model']['video_params']['resize_dim'])
        self.transform = VideoFrameTransform(is_train=False, resize_dims=resize_dims)
        
        # Set up device
        self.device = torch.device(f"cuda:{self.config['training']['devices'][0]}" if 
                                 self.config['training']['devices'] and torch.cuda.is_available() 
                                 else "cpu")
    
    def _init_model(self):
        """
        Initialize and load the model.
        """
        # Initialize model based on config
        model_type = self.config['model']['type']
        if model_type == 'video_cnn':
            self.model = VideoCNN(self.config_path)
        elif model_type == 'simpler_video_cnn':
            self.model = SimplerVideoCNN(self.config_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Load model weights
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        
        # Set model to evaluation mode
        self.model.eval()
    
    def predict_clip(self, video_path: str) -> Dict[str, Any]:
        """
        Make a prediction for a single video clip.
        
        Args:
            video_path: Path to the video clip
            
        Returns:
            Dictionary with prediction results
        """
        # Load and preprocess video
        video_tensor = self._preprocess_video(video_path)
        
        # Move tensor to device
        video_tensor = video_tensor.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(video_tensor.unsqueeze(0))
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
        
        # Prepare results
        result = {
            'success': predicted_class == 1,  # 1 = success, 0 = failure
            'confidence': confidence,
            'class_probabilities': {
                'failure': probabilities[0].item(),
                'success': probabilities[1].item()
            }
        }
        
        return result
    
    def _preprocess_video(self, video_path: str) -> torch.Tensor:
        """
        Load and preprocess a video clip.
        
        Args:
            video_path: Path to the video clip
            
        Returns:
            Tensor of preprocessed video frames (C, T, H, W)
        """
        # Load video
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices to sample
        sequence_length = self.config['model']['sequence_length']
        indices = list(np.linspace(0, frame_count - 1, sequence_length, dtype=int))
        
        for i in indices:
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            
            if ret:
                # Apply transformations
                frame_tensor = self.transform(frame)
                frames.append(frame_tensor)
            else:
                # If frame read failed, add a zero tensor
                resize_dims = tuple(self.config['model']['video_params']['resize_dim'])
                zero_frame = torch.zeros((3, *resize_dims))
                frames.append(zero_frame)
        
        cap.release()
        
        # Stack frames into a video tensor [C, T, H, W]
        video_tensor = torch.stack(frames, dim=1)
        
        return video_tensor
    
    def predict_batch(self, video_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Make predictions for a batch of video clips.
        
        Args:
            video_paths: List of paths to video clips
            
        Returns:
            List of dictionaries with prediction results
        """
        results = []
        
        for video_path in video_paths:
            try:
                result = self.predict_clip(video_path)
                result['video_path'] = video_path
                results.append(result)
            except Exception as e:
                print(f"Error processing {video_path}: {str(e)}")
                results.append({
                    'video_path': video_path,
                    'error': str(e)
                })
        
        return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference on soccer event clips')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model weights')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to video clip or directory of clips')
    return parser.parse_args()


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()
    
    # Initialize inference pipeline
    pipeline = InferencePipeline(args.model, args.config)
    
    # Check if input is a directory or a single file
    if os.path.isdir(args.video):
        # Get all video files in directory
        video_paths = [
            os.path.join(args.video, f) for f in os.listdir(args.video)
            if f.endswith(('.mp4', '.avi', '.mov'))
        ]
        
        # Make predictions
        results = pipeline.predict_batch(video_paths)
        
        # Print results
        print(f"Processed {len(results)} videos:")
        for result in results:
            if 'error' in result:
                print(f"  {os.path.basename(result['video_path'])}: Error - {result['error']}")
            else:
                outcome = "SUCCESS" if result['success'] else "FAILURE"
                print(f"  {os.path.basename(result['video_path'])}: {outcome} (Confidence: {result['confidence']:.4f})")
    else:
        # Make prediction for single video
        result = pipeline.predict_clip(args.video)
        
        # Print result
        outcome = "SUCCESS" if result['success'] else "FAILURE"
        print(f"Prediction: {outcome}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Class probabilities:")
        print(f"  Success: {result['class_probabilities']['success']:.4f}")
        print(f"  Failure: {result['class_probabilities']['failure']:.4f}")