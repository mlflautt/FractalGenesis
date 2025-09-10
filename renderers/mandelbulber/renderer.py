"""
Mandelbulber Renderer Class

This module provides the main interface for rendering fractals using Mandelbulber.
It handles CLI calls, file management, and batch processing for the genetic algorithm.
"""

import os
import subprocess
import tempfile
import time
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from PIL import Image
import threading
from queue import Queue, Empty

from .parameters import MandelbulberParameters


logger = logging.getLogger(__name__)


class MandelbulberRenderer:
    """
    Interface for Mandelbulber fractal rendering.
    
    Manages .fract file generation, CLI calls, and batch processing
    for efficient fractal generation in genetic algorithms.
    """
    
    def __init__(self, 
                 mandelbulber_path: str = "mandelbulber2",
                 temp_dir: Optional[str] = None,
                 output_dir: Optional[str] = None):
        """
        Initialize the Mandelbulber renderer.
        
        Args:
            mandelbulber_path: Path to mandelbulber2 executable
            temp_dir: Directory for temporary files
            output_dir: Directory for rendered images
        """
        self.mandelbulber_path = mandelbulber_path
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir())
        self.output_dir = Path(output_dir) if output_dir else Path("./renders")
        
        # Ensure directories exist
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if Mandelbulber is available
        self.is_available = self._check_mandelbulber_availability()
        
        # Thread pool for batch rendering
        self.render_queue = Queue()
        self.results_queue = Queue()
        self.worker_threads: List[threading.Thread] = []
        self.max_workers = 4
        
    def _check_mandelbulber_availability(self) -> bool:
        """
        Check if Mandelbulber is available on the system.
        
        Returns:
            bool: True if Mandelbulber is available
        """
        try:
            result = subprocess.run([self.mandelbulber_path, "--version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info(f"Mandelbulber found: {result.stdout.strip()}") 
                return True
        except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Mandelbulber not found at {self.mandelbulber_path}: {e}")
            
        return False
    
    def render_single(self, 
                     parameters: MandelbulberParameters,
                     output_filename: Optional[str] = None,
                     thumbnail_size: Optional[Tuple[int, int]] = None) -> Optional[Path]:
        """
        Render a single fractal image.
        
        Args:
            parameters: Fractal parameters to render
            output_filename: Optional custom output filename
            thumbnail_size: Optional size for thumbnail generation (width, height)
            
        Returns:
            Path to rendered image file or None if failed
        """
        if not self.is_available:
            logger.error("Mandelbulber not available for rendering")
            return None
            
        # Generate unique filename if not provided
        if not output_filename:
            timestamp = int(time.time() * 1000)
            output_filename = f"fractal_{timestamp}.png"
            
        output_path = self.output_dir / output_filename
        fract_path = self.temp_dir / f"{output_filename}.fract"
        
        try:
            # Write .fract file
            fract_content = parameters.to_fract_file()
            with open(fract_path, 'w') as f:
                f.write(fract_content)
            
            # Build mandelbulber command
            cmd = [
                self.mandelbulber_path,
                "--nogui",
                "--settings", str(fract_path),
                "--output", str(output_path),
                "--format", "png"
            ]
            
            # Add size parameters if specified
            if parameters.render.image_width and parameters.render.image_height:
                cmd.extend([
                    "--size", f"{parameters.render.image_width}x{parameters.render.image_height}"
                ])
            
            logger.debug(f"Running command: {' '.join(cmd)}")
            
            # Execute render
            result = subprocess.run(cmd, 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=300)  # 5 minute timeout
            
            if result.returncode != 0:
                logger.error(f"Mandelbulber render failed: {result.stderr}")
                return None
            
            # Verify output file exists
            if not output_path.exists():
                logger.error(f"Output file not created: {output_path}")
                return None
                
            # Generate thumbnail if requested
            if thumbnail_size:
                self._create_thumbnail(output_path, thumbnail_size)
            
            # Cleanup temp file
            if fract_path.exists():
                fract_path.unlink()
                
            logger.info(f"Successfully rendered: {output_path}")
            return output_path
            
        except subprocess.TimeoutExpired:
            logger.error(f"Render timeout for {output_filename}")
        except Exception as e:
            logger.error(f"Render error for {output_filename}: {e}")
        finally:
            # Cleanup temp file on error
            if fract_path.exists():
                fract_path.unlink()
                
        return None
    
    def _create_thumbnail(self, image_path: Path, size: Tuple[int, int]):
        """
        Create a thumbnail version of the rendered image.
        
        Args:
            image_path: Path to the original image
            size: Thumbnail size (width, height)
        """
        try:
            with Image.open(image_path) as img:
                img.thumbnail(size, Image.Resampling.LANCZOS)
                thumb_path = image_path.parent / f"thumb_{image_path.name}"
                img.save(thumb_path, "PNG")
                logger.debug(f"Created thumbnail: {thumb_path}")
        except Exception as e:
            logger.error(f"Failed to create thumbnail for {image_path}: {e}")
    
    def render_batch(self, 
                    parameter_list: List[MandelbulberParameters],
                    thumbnail_size: Tuple[int, int] = (256, 256),
                    max_concurrent: int = 4) -> List[Optional[Path]]:
        """
        Render multiple fractals concurrently.
        
        Args:
            parameter_list: List of parameter sets to render
            thumbnail_size: Size for thumbnails
            max_concurrent: Maximum concurrent renders
            
        Returns:
            List of paths to rendered images (None for failed renders)
        """
        if not self.is_available:
            logger.error("Mandelbulber not available for batch rendering")
            return [None] * len(parameter_list)
        
        results = [None] * len(parameter_list)
        
        # Start worker threads
        self._start_workers(min(max_concurrent, len(parameter_list)))
        
        try:
            # Queue all render jobs
            for i, params in enumerate(parameter_list):
                self.render_queue.put((i, params, thumbnail_size))
            
            # Wait for all results
            completed = 0
            while completed < len(parameter_list):
                try:
                    index, result_path = self.results_queue.get(timeout=600)  # 10 min total timeout
                    results[index] = result_path
                    completed += 1
                    logger.info(f"Batch progress: {completed}/{len(parameter_list)}")
                except Empty:
                    logger.error("Batch render timeout")
                    break
                    
        finally:
            # Stop workers
            self._stop_workers()
            
        return results
    
    def _start_workers(self, num_workers: int):
        """Start worker threads for batch processing."""
        self.worker_threads = []
        for i in range(num_workers):
            worker = threading.Thread(target=self._worker_thread, daemon=True)
            worker.start()
            self.worker_threads.append(worker)
    
    def _worker_thread(self):
        """Worker thread for processing render queue."""
        while True:
            try:
                item = self.render_queue.get(timeout=1)
                if item is None:  # Poison pill
                    break
                    
                index, parameters, thumbnail_size = item
                result_path = self.render_single(
                    parameters, 
                    output_filename=f"batch_{index}_{int(time.time()*1000)}.png",
                    thumbnail_size=thumbnail_size
                )
                self.results_queue.put((index, result_path))
                self.render_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Worker thread error: {e}")
    
    def _stop_workers(self):
        """Stop all worker threads."""
        # Send poison pills
        for _ in self.worker_threads:
            self.render_queue.put(None)
        
        # Wait for threads to finish
        for worker in self.worker_threads:
            worker.join(timeout=5)
        
        self.worker_threads = []
    
    def get_render_statistics(self) -> Dict[str, int]:
        """
        Get statistics about rendered images.
        
        Returns:
            Dictionary with render statistics
        """
        stats = {
            'total_renders': 0,
            'total_thumbnails': 0,
            'disk_usage_mb': 0
        }
        
        if not self.output_dir.exists():
            return stats
            
        for file_path in self.output_dir.iterdir():
            if file_path.is_file():
                if file_path.name.startswith('thumb_'):
                    stats['total_thumbnails'] += 1
                elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    stats['total_renders'] += 1
                    
                stats['disk_usage_mb'] += file_path.stat().st_size / (1024 * 1024)
        
        return stats
    
    def cleanup_old_renders(self, max_age_hours: int = 24):
        """
        Clean up old render files to save disk space.
        
        Args:
            max_age_hours: Maximum age of files to keep
        """
        if not self.output_dir.exists():
            return
            
        cutoff_time = time.time() - (max_age_hours * 3600)
        removed_count = 0
        
        for file_path in self.output_dir.iterdir():
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()
                    removed_count += 1
                except OSError as e:
                    logger.warning(f"Failed to remove {file_path}: {e}")
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old render files")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self._stop_workers()
