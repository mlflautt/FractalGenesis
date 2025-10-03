"""
Animation Renderer for Fractal Evolution

This module renders frame sequences from animation parameters and converts them to video.
Supports parallel rendering, progress tracking, and multiple output formats.
"""

import os
import subprocess
import tempfile
import logging
import concurrent.futures
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Callable
from dataclasses import dataclass
import time
import json

from renderers.mandelbulber.renderer import MandelbulberRenderer
from .animation_parameters import AnimationParameters


logger = logging.getLogger(__name__)


@dataclass
class AnimationRenderResult:
    """Result of animation rendering operation"""
    success: bool
    animation_path: Optional[Path] = None
    frame_paths: List[Path] = None
    render_time_seconds: float = 0.0
    total_frames: int = 0
    failed_frames: List[int] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.frame_paths is None:
            self.frame_paths = []
        if self.failed_frames is None:
            self.failed_frames = []


class AnimationRenderer:
    """
    Renders fractal animations using evolutionary parameters.
    
    Handles frame generation, parallel processing, and video encoding.
    """
    
    def __init__(self, 
                 mandelbulber_renderer: Optional[MandelbulberRenderer] = None,
                 output_dir: Optional[str] = None,
                 temp_dir: Optional[str] = None,
                 max_workers: int = 4):
        """
        Initialize animation renderer.
        
        Args:
            mandelbulber_renderer: Renderer instance (creates new if None)
            output_dir: Directory for output videos and frames
            temp_dir: Directory for temporary files
            max_workers: Maximum parallel rendering threads
        """
        self.mandelbulber_renderer = mandelbulber_renderer or MandelbulberRenderer()
        self.output_dir = Path(output_dir) if output_dir else Path("./animations")
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "fractal_animations"
        self.max_workers = max_workers
        
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Video encoding settings
        self.video_codecs = {
            'mp4': 'libx264',
            'webm': 'libvpx-vp9',
            'gif': None  # Special handling for GIF
        }
        
        # Progress callback
        self.progress_callback: Optional[Callable[[int, int], None]] = None
    
    def render_animation(self, 
                        animation_params: AnimationParameters,
                        output_name: str = None,
                        video_format: str = 'mp4',
                        preview_mode: bool = False) -> AnimationRenderResult:
        """
        Render complete animation from parameters.
        
        Args:
            animation_params: Animation parameters defining the sequence
            output_name: Name for output files (auto-generated if None)
            video_format: Output format ('mp4', 'webm', 'gif')
            preview_mode: If True, render lower quality for preview
            
        Returns:
            AnimationRenderResult with paths and statistics
        """
        start_time = time.time()
        
        if not output_name:
            timestamp = int(time.time())
            output_name = f"fractal_animation_{timestamp}"
        
        # Create session directory
        session_dir = self.temp_dir / output_name
        session_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Step 1: Generate frame sequence
            logger.info(f"Rendering {animation_params.total_frames} frames...")
            frame_paths = self._render_frame_sequence(animation_params, session_dir, preview_mode)
            
            if not frame_paths:
                return AnimationRenderResult(
                    success=False,
                    error_message="Failed to render any frames"
                )
            
            # Step 2: Encode video
            logger.info(f"Encoding video in {video_format} format...")
            video_path = self._encode_video(frame_paths, output_name, video_format, animation_params.fps)
            
            if not video_path:
                return AnimationRenderResult(
                    success=False,
                    frame_paths=frame_paths,
                    total_frames=len(frame_paths),
                    render_time_seconds=time.time() - start_time,
                    error_message="Video encoding failed"
                )
            
            # Step 3: Save animation metadata
            self._save_animation_metadata(animation_params, session_dir, video_path)
            
            render_time = time.time() - start_time
            logger.info(f"Animation rendered successfully in {render_time:.2f}s")
            
            return AnimationRenderResult(
                success=True,
                animation_path=video_path,
                frame_paths=frame_paths,
                render_time_seconds=render_time,
                total_frames=len(frame_paths)
            )
            
        except Exception as e:
            logger.error(f"Animation rendering failed: {e}")
            return AnimationRenderResult(
                success=False,
                error_message=str(e),
                render_time_seconds=time.time() - start_time
            )
    
    def _render_frame_sequence(self, 
                             animation_params: AnimationParameters, 
                             session_dir: Path,
                             preview_mode: bool = False) -> List[Path]:
        """
        Render individual frames of the animation.
        
        Args:
            animation_params: Animation parameters
            session_dir: Directory for this rendering session
            preview_mode: Use lower quality for preview
            
        Returns:
            List of paths to rendered frame images
        """
        total_frames = animation_params.total_frames
        frame_paths = []
        failed_frames = []
        
        # Generate frame parameters
        frame_params = []
        for frame_idx in range(total_frames):
            time_position = frame_idx / max(1, total_frames - 1)
            params = animation_params.get_parameters_at_time(time_position)
            
            # Apply preview mode settings
            if preview_mode:
                params.render.image_width = min(800, params.render.image_width)
                params.render.image_height = min(600, params.render.image_height)
                params.render.detail_level *= 0.5
                params.fractal.iterations = min(100, params.fractal.iterations)
            else:
                params.render.image_width = animation_params.render_width
                params.render.image_height = animation_params.render_height
                params.render.quality = animation_params.render_quality
            
            frame_filename = f"frame_{frame_idx:06d}.png"
            frame_params.append((frame_idx, params, frame_filename))
        
        # Render frames in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all frame rendering jobs
            future_to_frame = {
                executor.submit(self._render_single_frame, params, session_dir / filename): 
                (frame_idx, session_dir / filename)
                for frame_idx, params, filename in frame_params
            }
            
            # Collect results
            completed_frames = 0
            for future in concurrent.futures.as_completed(future_to_frame):
                frame_idx, expected_path = future_to_frame[future]
                
                try:
                    result_path = future.result()
                    if result_path and result_path.exists():
                        frame_paths.append((frame_idx, result_path))
                    else:
                        failed_frames.append(frame_idx)
                        logger.warning(f"Frame {frame_idx} failed to render")
                        
                except Exception as e:
                    failed_frames.append(frame_idx)
                    logger.error(f"Frame {frame_idx} rendering error: {e}")
                
                completed_frames += 1
                if self.progress_callback:
                    self.progress_callback(completed_frames, total_frames)
        
        # Sort frames by index and return paths
        frame_paths.sort(key=lambda x: x[0])
        return [path for _, path in frame_paths]
    
    def _render_single_frame(self, 
                           parameters: 'MandelbulberParameters',
                           output_path: Path) -> Optional[Path]:
        """
        Render a single frame.
        
        Args:
            parameters: Fractal parameters for this frame
            output_path: Where to save the frame
            
        Returns:
            Path to rendered frame or None if failed
        """
        try:
            return self.mandelbulber_renderer.render_single(
                parameters=parameters,
                output_filename=output_path.name
            )
        except Exception as e:
            logger.error(f"Single frame render failed: {e}")
            return None
    
    def _encode_video(self, 
                     frame_paths: List[Path],
                     output_name: str,
                     video_format: str,
                     fps: int) -> Optional[Path]:
        """
        Encode frame sequence to video.
        
        Args:
            frame_paths: List of frame image paths
            output_name: Base name for output file
            video_format: Video format ('mp4', 'webm', 'gif')
            fps: Frames per second
            
        Returns:
            Path to encoded video or None if failed
        """
        if not frame_paths:
            return None
        
        output_path = self.output_dir / f"{output_name}.{video_format}"
        
        try:
            if video_format == 'gif':
                return self._create_gif(frame_paths, output_path, fps)
            else:
                return self._create_video_ffmpeg(frame_paths, output_path, video_format, fps)
        except Exception as e:
            logger.error(f"Video encoding failed: {e}")
            return None
    
    def _create_video_ffmpeg(self, 
                           frame_paths: List[Path],
                           output_path: Path,
                           video_format: str,
                           fps: int) -> Optional[Path]:
        """Create video using FFmpeg"""
        if not frame_paths:
            return None
        
        # Create a temporary text file listing all frames
        frames_dir = frame_paths[0].parent
        
        # Use FFmpeg with frame pattern
        codec = self.video_codecs.get(video_format, 'libx264')
        
        cmd = [
            'ffmpeg', '-y',  # Overwrite output
            '-framerate', str(fps),
            '-pattern_type', 'glob',
            '-i', str(frames_dir / 'frame_*.png'),
            '-c:v', codec,
            '-pix_fmt', 'yuv420p',
            '-crf', '18',  # High quality
            str(output_path)
        ]
        
        logger.debug(f"Running FFmpeg: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and output_path.exists():
            logger.info(f"Video created: {output_path}")
            return output_path
        else:
            logger.error(f"FFmpeg failed: {result.stderr}")
            return None
    
    def _create_gif(self, 
                   frame_paths: List[Path],
                   output_path: Path,
                   fps: int) -> Optional[Path]:
        """Create animated GIF using FFmpeg"""
        if not frame_paths:
            return None
        
        frames_dir = frame_paths[0].parent
        delay = int(100 / fps)  # Delay in centiseconds
        
        # Create high-quality GIF with palette
        palette_path = frames_dir / "palette.png"
        
        # Generate palette
        palette_cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-pattern_type', 'glob',
            '-i', str(frames_dir / 'frame_*.png'),
            '-vf', 'palettegen=stats_mode=diff',
            str(palette_path)
        ]
        
        result = subprocess.run(palette_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.warning("Palette generation failed, creating GIF without palette")
            
        # Create GIF
        gif_cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-pattern_type', 'glob',
            '-i', str(frames_dir / 'frame_*.png')
        ]
        
        if palette_path.exists():
            gif_cmd.extend([
                '-i', str(palette_path),
                '-lavfi', 'paletteuse=dither=bayer:bayer_scale=3'
            ])
        
        gif_cmd.append(str(output_path))
        
        result = subprocess.run(gif_cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and output_path.exists():
            logger.info(f"GIF created: {output_path}")
            # Clean up palette
            if palette_path.exists():
                palette_path.unlink()
            return output_path
        else:
            logger.error(f"GIF creation failed: {result.stderr}")
            return None
    
    def _save_animation_metadata(self, 
                               animation_params: AnimationParameters,
                               session_dir: Path,
                               video_path: Path):
        """Save animation metadata for later analysis"""
        metadata = {
            'animation_parameters': animation_params.to_dict(),
            'video_path': str(video_path),
            'render_timestamp': time.time(),
            'total_frames': animation_params.total_frames,
            'fps': animation_params.fps,
            'duration_seconds': animation_params.duration_seconds
        }
        
        metadata_path = session_dir / "animation_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def render_preview(self, 
                      animation_params: AnimationParameters,
                      max_frames: int = 30) -> AnimationRenderResult:
        """
        Render a quick preview of the animation.
        
        Args:
            animation_params: Animation parameters
            max_frames: Maximum frames to render for preview
            
        Returns:
            AnimationRenderResult with preview animation
        """
        # Create preview version with limited frames
        preview_params = animation_params.copy()
        
        if animation_params.total_frames > max_frames:
            # Reduce frame count while keeping duration
            preview_params.fps = max_frames / animation_params.duration_seconds
            preview_params.fps = max(1, int(preview_params.fps))
        
        return self.render_animation(
            preview_params,
            output_name=f"preview_{int(time.time())}",
            video_format='gif',  # GIF is good for previews
            preview_mode=True
        )
    
    def batch_render_animations(self, 
                              animation_list: List[Tuple[AnimationParameters, str]],
                              video_format: str = 'mp4') -> List[AnimationRenderResult]:
        """
        Render multiple animations sequentially.
        
        Args:
            animation_list: List of (parameters, name) tuples
            video_format: Output video format
            
        Returns:
            List of render results
        """
        results = []
        
        for i, (params, name) in enumerate(animation_list):
            logger.info(f"Rendering animation {i+1}/{len(animation_list)}: {name}")
            
            result = self.render_animation(
                animation_params=params,
                output_name=name,
                video_format=video_format
            )
            
            results.append(result)
            
            if not result.success:
                logger.error(f"Failed to render animation: {name}")
        
        return results
    
    def set_progress_callback(self, callback: Callable[[int, int], None]):
        """Set callback function for progress updates"""
        self.progress_callback = callback
    
    def cleanup_temp_files(self, max_age_hours: int = 24):
        """Clean up old temporary files"""
        if not self.temp_dir.exists():
            return
        
        cutoff_time = time.time() - (max_age_hours * 3600)
        cleaned_count = 0
        
        for item in self.temp_dir.iterdir():
            if item.is_dir():
                try:
                    # Check if directory is older than cutoff
                    if item.stat().st_mtime < cutoff_time:
                        # Remove entire directory tree
                        import shutil
                        shutil.rmtree(item)
                        cleaned_count += 1
                except OSError as e:
                    logger.warning(f"Failed to clean up {item}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old animation directories")
    
    def get_render_statistics(self) -> Dict[str, Any]:
        """Get statistics about rendered animations"""
        stats = {
            'total_animations': 0,
            'total_size_mb': 0,
            'formats': {},
            'output_directory': str(self.output_dir)
        }
        
        if not self.output_dir.exists():
            return stats
        
        for file_path in self.output_dir.iterdir():
            if file_path.is_file() and file_path.suffix in ['.mp4', '.webm', '.gif']:
                stats['total_animations'] += 1
                size_mb = file_path.stat().st_size / (1024 * 1024)
                stats['total_size_mb'] += size_mb
                
                fmt = file_path.suffix[1:]  # Remove dot
                stats['formats'][fmt] = stats['formats'].get(fmt, 0) + 1
        
        return stats