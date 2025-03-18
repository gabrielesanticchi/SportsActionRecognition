#!/usr/bin/env python3
"""
Soccer Technical Events Analysis - Main Script

This script is the entry point for the soccer technical events analysis pipeline.
It handles command line arguments and runs the appropriate pipeline (training or inference).
"""

import os
import argparse
import yaml
from src.pipelines.train_pipeline import TrainingPipeline
from src.pipelines.inference_pipeline import InferencePipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Soccer Technical Events Analysis')
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--config', type=str, default='config/config.yaml',
                             help='Path to configuration file')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions with a trained model')
    predict_parser.add_argument('--model', type=str, required=True,
                               help='Path to trained model weights')
    predict_parser.add_argument('--config', type=str, default='config/config.yaml',
                               help='Path to configuration file')
    predict_parser.add_argument('--video', type=str, required=True,
                               help='Path to video clip or directory of clips')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Evaluate model on test set')
    test_parser.add_argument('--model', type=str, required=True,
                            help='Path to trained model weights')
    test_parser.add_argument('--config', type=str, default='config/config.yaml',
                            help='Path to configuration file')
                            
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Set up directory structure and download XML data')
    setup_parser.add_argument('--config', type=str, default='config/config.yaml',
                             help='Path to configuration file')
    setup_parser.add_argument('--download-clips', action='store_true',
                             help='Download video clips (may take a long time)')
    setup_parser.add_argument('--inspect', action='store_true',
                             help='Print data statistics after download')
    
    return parser.parse_args()


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Create necessary directories
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    os.makedirs(config['data']['raw_dir'], exist_ok=True)
    os.makedirs(config['data']['processed_dir'], exist_ok=True)
    os.makedirs(config['data']['metadata_dir'], exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
    
    # Run the appropriate pipeline
    if args.command == 'train':
        print("Starting training pipeline...")
        pipeline = TrainingPipeline(args.config)
        history, best_model_path = pipeline.run()
        print(f"Training completed. Best model saved at: {best_model_path}")
    
    elif args.command == 'predict':
        print("Starting inference pipeline...")
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
    
    elif args.command == 'test':
        print("Test functionality not implemented yet.")
        
    elif args.command == 'setup':
        print("Setting up soccer event analysis environment...")
        # Create necessary directories
        os.makedirs(config['data']['raw_dir'], exist_ok=True)
        os.makedirs(config['data']['processed_dir'], exist_ok=True)
        os.makedirs(config['data']['metadata_dir'], exist_ok=True)
        os.makedirs(config['logging']['log_dir'], exist_ok=True)
        os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
        
        print(f"Created directory structure. Container ID: {config['data']['container_id']}")
        
        # Download XML data
        from src.utils.s3_utils import S3Downloader
        from src.utils.xml_utils import XMLEventExtractor
        import pandas as pd
        
        downloader = S3Downloader(args.config)
        videomatch_content, sportscode_content = downloader.download_xml_files()
        
        if videomatch_content and sportscode_content:
            print("Successfully downloaded XML files.")
            
            # Extract and analyze data if inspect flag is set
            if args.inspect:
                # Parse player mapping from videomatch
                videomatch_extractor = XMLEventExtractor(videomatch_content)
                player_mapping = videomatch_extractor.get_player_mapping()
                print(f"Found {len(player_mapping)} players in videomatch XML.")
                
                # Extract events from sportscode
                sportscode_extractor = XMLEventExtractor(sportscode_content)
                events = sportscode_extractor.extract_events_with_filenames(player_mapping)
                
                if events:
                    # Analyze event types
                    event_types = {}
                    success_count = 0
                    failure_count = 0
                    
                    for event in events:
                        event_type = event['group']
                        if event_type not in event_types:
                            event_types[event_type] = 0
                        event_types[event_type] += 1
                        
                        if event.get('success') == 1:
                            success_count += 1
                        elif event.get('success') == 0:
                            failure_count += 1
                    
                    # Print statistics
                    print(f"\nEvent analysis:")
                    print(f"Total events: {len(events)}")
                    print(f"Success/Failure distribution: {success_count}/{failure_count}")
                    print(f"Event types:")
                    for event_type, count in event_types.items():
                        print(f"  {event_type}: {count}")
                else:
                    print("No relevant events found in sportscode XML.")
            
            # Download clips if requested
            if args.download_clips:
                print("\nDownloading video clips (this may take a while)...")
                from src.data.data_loader import SoccerEventDataLoader
                
                data_loader = SoccerEventDataLoader(args.config)
                try:
                    event_df = data_loader.load_data()
                    print(f"Downloaded {len(event_df)} video clips.")
                    print(f"Success/Failure distribution: {event_df['success'].sum()}/{len(event_df) - event_df['success'].sum()}")
                except Exception as e:
                    print(f"Error downloading clips: {str(e)}")
        else:
            print("Failed to download XML files. Check your configuration and credentials.")
    
    else:
        print("Please specify a command: train, predict, test, or setup")
        print("Run with --help for more information")


if __name__ == '__main__':
    main()