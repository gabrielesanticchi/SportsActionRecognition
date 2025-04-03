# SportsActionRecognition Documentation

## Setup

### AWS Credentials

The system requires AWS credentials for accessing S3 storage. Create a `.env` file in the project root:

```plaintext
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
```

## Model Architecture

The system supports three different model architectures for classifying soccer technical events:

### 1. VideoCNN

A 3D CNN model that processes spatial and temporal information using:
- ResNet backbone (either ResNet-18 or ResNet-50) for spatial feature extraction
- 3D convolutional layers for temporal modeling
- Fully connected layers for classification

```yaml
model:
  type: "video_cnn"
  feature_extractor: "resnet18"  # or "resnet50"
```

### 2. SimplerVideoCNN

A hybrid model that may be more suitable for smaller datasets:
- ResNet backbone for spatial feature extraction 
- Global average pooling to reduce spatial dimensions
- Bidirectional LSTM for temporal modeling
- Fully connected layers for classification

```yaml
model:
  type: "simpler_video_cnn"
  feature_extractor: "resnet18"  # or "resnet50"
```

### 3. SlowFastNetwork

The SlowFast architecture processes videos at two different frame rates simultaneously:
- **Slow pathway**: Processes fewer frames to capture spatial semantics (player positions, field context)
- **Fast pathway**: Processes all frames to capture fine motion details (quick movements, ball trajectory)
- **Lateral connections**: Transfer information from Fast to Slow pathway

SlowFast is particularly effective for soccer action recognition because it can simultaneously track:
1. Spatial relationships (player positions, field areas) in the Slow pathway
2. Quick movements (ball trajectory, player actions) in the Fast pathway

```yaml
model:
  type: "slowfast_network"
```

## Configuration Options

### Common Parameters

```yaml
model:
  pretrained: true  # Use pretrained backbone
  num_classes: 2    # Binary classification (success/failure)
  sequence_length: 16  # Number of frames to sample 
  dropout: 0.5
  video_params:
    frame_rate: 2  # Target frame rate
    clip_duration: 10  # Clip duration in seconds
    resize_dim: [224, 224]  # Height, width
```

### SlowFast Specific Parameters

```yaml
model:
  slowfast_params:
    alpha: 8  # Speed ratio between Fast and Slow pathway
    beta: 0.125  # Channel ratio between Fast and Slow pathway
    slow_pathway_stages: [1, 2, 3, 4]  # Number of blocks per stage in Slow pathway
    fast_pathway_stages: [1, 2, 3, 4]  # Number of blocks per stage in Fast pathway
```

## Training a Model

```bash
python main.py train --config config/config.yaml
```

To train the SlowFast network, first update your config.yaml to set the model type:

```yaml
model:
  type: "slowfast_network"
```

## Making Predictions

```bash
python main.py predict --model checkpoints/best_model.pt --video path/to/video.mp4
```

## Performance Considerations

1. **VideoCNN**: Most memory-efficient but lower accuracy
2. **SimplerVideoCNN**: Good balance of accuracy and efficiency
3. **SlowFastNetwork**: Highest accuracy but most resource-intensive 

When choosing a model architecture, consider:
- Dataset size
- Computational resources
- Real-time requirements

## Technical Implementation

The SlowFast network consists of:

1. **Dual pathways architecture**:
   - Slow pathway operates on frames with α (default=8) stride
   - Fast pathway operates on all frames
   - Channel count in Fast pathway scaled by β (default=1/8)

2. **Lateral connections**:
   - Transfer information from Fast to Slow pathway
   - Implemented using 3D convolutions
   - Located after each major stage of processing

3. **Bottleneck blocks**:
   - Residual blocks with bottleneck structure
   - Separate blocks for each pathway
   - Time-strided convolutions for efficient processing

The architecture achieves state-of-the-art performance by efficiently modeling both appearance and motion information at different temporal resolutions.