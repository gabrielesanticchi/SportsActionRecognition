import os
import boto3
import yaml
from typing import List, Optional, Tuple
import xml.etree.ElementTree as ET
from .video_compressor import VideoCompressor
import tempfile


class S3Downloader:
    """
    Utility for downloading clips and XML data from S3 for ML training.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Extract S3 configuration
        self.container_ids = self.config['data']['container_ids']
        self.output_bucket = self.config['data']['output_bucket']
        self.base_paths = [f"{self.config['data']['base_path']}/{container_id}" 
                          for container_id in self.container_ids]
        
        # Create AWS S3 client
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
        )
        
        # Create data directories
        os.makedirs(self.config['data']['raw_dir'], exist_ok=True)
        os.makedirs(self.config['data']['processed_dir'], exist_ok=True)
        os.makedirs(self.config['data']['metadata_dir'], exist_ok=True)

        # Video compressor
        self.video_compressor = VideoCompressor(self.config)

    def download_xml_files(self) -> List[Tuple[Optional[str], Optional[str]]]:
        """
        Download XML files for all specified container IDs.
        
        Returns:
            List of tuples (videomatch_content, sportscode_content) for each container
        """
        all_results = []
        
        for container_id, base_path in zip(self.container_ids, self.base_paths):
            try:
                # List objects in the container directory
                response = self.s3.list_objects_v2(
                    Bucket=self.output_bucket,
                    Prefix=base_path
                )
                
                if 'Contents' not in response:
                    continue
                
                # Find videomatch and sportscode XML files
                videomatch_path = None
                sportscode_path = None
                
                for obj in response['Contents']:
                    key = obj['Key']
                    if key.endswith('_videomatch.xml'):
                        videomatch_path = key
                    elif key.endswith('_sportscode.xml'):
                        sportscode_path = key
                
                if not videomatch_path:
                    print("Videomatch XML file not found")
                    return None, None
                
                if not sportscode_path:
                    print("Sportscode XML file not found")
                    return None, None
                
                # Download XML files
                videomatch_content = self._download_file_content(videomatch_path)
                sportscode_content = self._download_file_content(sportscode_path)
                
                # Save XML files locally
                metadata_dir = self.config['data']['metadata_dir']
                os.makedirs(metadata_dir, exist_ok=True)
                
                if videomatch_content:
                    xml_local_path = os.path.join(metadata_dir, os.path.basename(videomatch_path))
                    with open(xml_local_path, 'w') as f:
                        f.write(videomatch_content)
                
                if sportscode_content:
                    xml_local_path = os.path.join(metadata_dir, os.path.basename(sportscode_path))
                    with open(xml_local_path, 'w') as f:
                        f.write(sportscode_content)
                
                print(f"Downloaded videomatch XML ({len(videomatch_content) if videomatch_content else 0} bytes) and " 
                      f"sportscode XML ({len(sportscode_content) if sportscode_content else 0} bytes)")
                
                all_results.append((videomatch_content, sportscode_content))
                
            except Exception as e:
                all_results.append((None, None))
                
        return all_results
    
    def _download_file_content(self, key: str) -> Optional[str]:
        """
        Download and return file content as string.
        
        Args:
            key: S3 object key
            
        Returns:
            File content as string, or None if error occurs
        """
        try:
            response = self.s3.get_object(Bucket=self.output_bucket, Key=key)
            return response['Body'].read().decode('utf-8')
        except Exception as e:
            print(f"Error downloading {key}: {str(e)}")
            return None
    
    def download_clip(self, s3_path: str, local_dir: str) -> Optional[str]:
        """
        Download and compress a clip from S3 to a local directory.
        """
        try:
            # Create temporary file for initial download
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_path = temp_file.name
                
                # Download to temporary file
                self.s3.download_file(self.output_bucket, s3_path, temp_path)
                
                # Create final local path
                local_path = os.path.join(local_dir, os.path.basename(s3_path))
                
                # Compress video to final location
                compressed_path = self.video_compressor.compress_video(temp_path, local_path)
                
                # Clean up temporary file
                os.unlink(temp_path)
                
                return compressed_path
                
        except Exception as e:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
            return None
    
    def download_clips_for_events(self, events: List[dict]) -> List[dict]:
        """
        Download clips for a list of events and update event dictionaries with local paths.
        
        Args:
            events: List of event dictionaries with 'clip_filename' and 'user_id' fields
            
        Returns:
            Updated list of events with 'local_path' field added
        """
        raw_dir = self.config['data']['raw_dir']
        os.makedirs(raw_dir, exist_ok=True)
        
        updated_events = []
        
        for event in events:
            user_id = event.get('user_id', 'unknown')
            group = event['group']
            clip_filename = event['clip_filename']
            container_id = event['container_id']
            
            # Get the correct base path for this container
            base_path = f"{self.config['data']['base_path']}/{container_id}"
            
            # Check if file already exists locally
            local_path = os.path.join(raw_dir, clip_filename)
            if os.path.exists(local_path):
                event['local_path'] = local_path
                updated_events.append(event)
                print(f'... skipping download for {clip_filename}')
                continue
            
            # If file doesn't exist, download it
            s3_path = f"{base_path}/{user_id}/{group}/{clip_filename}"
            print(f'Downloading {clip_filename} from container {container_id}...')
            local_path = self.download_clip(s3_path, raw_dir)
            
            if local_path:
                event['local_path'] = local_path
                updated_events.append(event)
        
        return updated_events
    
    def s3_object_exists(self, key: str) -> bool:
        """
        Check if an object exists in S3.
        
        Args:
            key: S3 object key
            
        Returns:
            True if the object exists, False otherwise
        """
        try:
            self.s3.head_object(Bucket=self.output_bucket, Key=key)
            return True
        except Exception:
            return False
    
    def find_xml_and_video_paths(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Find XML and video file paths for the container.
        
        Returns:
            Tuple of (xml_path, video_url)
        """
        try:
            # List objects in the container directory
            response = self.s3.list_objects_v2(
                Bucket=self.output_bucket,
                Prefix=self.base_path
            )
            
            if 'Contents' not in response:
                print(f"No files found for container ID {self.container_id}")
                return None, None
            
            # Find videomatch XML file and video file
            xml_path = None
            video_path = None
            
            for obj in response['Contents']:
                key = obj['Key']
                if key.endswith('_highlights_videomatch.xml'):
                    xml_path = key
                elif key.endswith('.mp4'):
                    video_path = key
            
            if not xml_path:
                print("Videomatch XML file not found")
                return None, None
            
            if not video_path:
                print("Video file not found")
                return None, None
            
            # Construct video URL
            video_url = f"https://{self.output_bucket}.s3.amazonaws.com/{video_path}"
            
            return xml_path, video_url
        
        except Exception as e:
            print(f"Error finding XML and video paths: {str(e)}")
            return None, None