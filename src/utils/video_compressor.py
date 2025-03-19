import ffmpeg
import os
from typing import Optional

class VideoCompressor:
    """
    Utility for compressing video clips using ffmpeg.
    """
    def __init__(self, config: dict):
        """
        Initialize video compressor with configuration.
        
        Args:
            config: Dictionary containing video compression settings
        """
        self.fps = config['video']['fps']
        self.resolution = config['video']['resolution']
        self.crf = config['video']['crf']  # Compression quality (0-51, lower is better)
    
    def compress_video(self, input_path: str, output_path: str) -> Optional[str]:
        """
        Compress video with specified parameters.
        
        Args:
            input_path: Path to input video
            output_path: Path to save compressed video
            
        Returns:
            Path to compressed video if successful, None otherwise
        """
        try:
            stream = ffmpeg.input(input_path)
            stream = ffmpeg.filter(stream, 'fps', fps=self.fps)
            
            # Extract resolution values
            width, height = self.resolution.split('x')
            
            stream = ffmpeg.filter(stream, 'scale', width=width, height=height)
            
            stream = ffmpeg.output(
                stream, 
                output_path,
                vcodec='libx264',
                crf=self.crf,
                acodec='aac'
            )
            
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            
            return output_path
        except Exception:
            if os.path.exists(output_path):
                os.remove(output_path)
            return None