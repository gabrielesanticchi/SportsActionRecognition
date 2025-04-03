# SportsActionRecognition
A modular, professional, and ready-to-use ML pipeline for analyzing soccer technical events from video clips. This system uses computer vision to estimate the accuracy of technical events, determining whether a ball is successfuly delivered to a teammate.

## Project Overview

This pipeline analyzes 10-second video clips of soccer events (passes, crosses, long balls) and predicts whether the event was successful or not. The system:

1. Uses XML metadata to extract event information
2. Downloads clips from S3 storage
3. Processes video data using PyTorch
4. Trains a 3D CNN model to classify events
5. Provides inference capabilities for new clips

## Project Structure

```
soccer-event-analysis/
├── config/
│   └── config.yaml         # Main configuration file
├── data/
│   ├── raw/                # Downloaded clips
│   ├── processed/          # Preprocessed data
│   └── metadata/           # Extracted XML metadata
├── src/
│   ├── data/               # Data handling modules
│   ├── models/             # Model architectures
│   ├── utils/              # Utility functions
│   ├── trainers/           # Training logic
│   └── pipelines/          # Full pipelines
├── checkpoints/            # Saved model weights
├── logs/                   # Training logs and visualizations
├── main.py                 # Main entry point
└── README.md               # Documentation
```

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.9 or higher
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/soccer-event-analysis.git
   cd soccer-event-analysis
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up AWS credentials for S3 access:
   ```
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   ```

## Configuration

Modify `config/config.yaml` to customize the pipeline:

- **Data Configuration**: Specify match ID, S3 bucket, and event filters
- **Model Configuration**: Choose model type, architecture, and hyperparameters
- **Training Configuration**: Set batch size, learning rate, and training epochs
- **Logging Configuration**: Configure checkpointing and logging behavior

## Usage

### Training a Model

```bash
python main.py train --config config/config.yaml
```

This will:
1. Download XML metadata and video clips from S3
2. Create train/validation/test splits
3. Train the model with the specified configuration
4. Save checkpoints and training metrics
5. Generate visualizations of the training process

### Making Predictions

```bash
python main.py predict --model checkpoints/best_model.pt --video path/to/video.mp4
```

For batch prediction on a directory of videos:

```bash
python main.py predict --model checkpoints/best_model.pt --video path/to/video_directory
```

## Model Architecture

The system provides two model architectures:

1. **VideoCNN**: A 3D CNN model that processes spatial and temporal information using 3D convolutions
2. **SimplerVideoCNN**: A hybrid model that uses a 2D CNN for spatial features and LSTM for temporal modeling

Both models use a pretrained ResNet backbone for feature extraction.

## Data Pipeline

The system processes data in the following stages:

1. **XML Parsing**: Extract event data and labels from XML files
2. **Clip Download**: Download video clips from S3 storage
3. **Preprocessing**: Extract frames, resize, and apply augmentations
4. **Dataset Creation**: Create PyTorch datasets and dataloaders
5. **Training/Inference**: Feed data to models for training or prediction

## Extending the System

### Adding New Model Architectures

1. Create a new model class in `src/models/` that inherits from `BaseModel`
2. Implement the forward method and any model-specific methods
3. Update `config.yaml` to include the new model type
4. Add the model to the model initialization in the training and inference pipelines

### Adding New Data Sources

1. Extend the data loading functionality in `src/data/data_loader.py`
2. Implement custom parsing for the new data format
3. Update the dataset classes to handle the new data format

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project uses the XSEED XML format for soccer event data
- The video processing pipeline is built on FFmpeg
