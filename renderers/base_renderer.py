"""
Base Renderer Interface for FractalGenesis

This module provides the abstract base class for all fractal renderers
in the FractalGenesis system.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import os
import logging


class BaseRenderer(ABC):
    """
    Abstract base class for fractal renderers.
    
    This class defines the interface that all fractal renderers 
    must implement to work with the FractalGenesis system.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize base renderer.
        
        Args:
            output_dir: Directory to store rendered images
        """
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    @abstractmethod
    def generate_random_genome(self, seed: Optional[int] = None) -> Any:
        """
        Generate a random fractal genome.
        
        Args:
            seed: Optional random seed for reproducible generation
            
        Returns:
            Genome data in renderer-specific format
        """
        pass
    
    @abstractmethod
    def render_genome(self, genome: Any, output_filename: Optional[str] = None) -> str:
        """
        Render a fractal genome to an image file.
        
        Args:
            genome: Genome data in renderer-specific format
            output_filename: Optional custom filename (without extension)
            
        Returns:
            Path to rendered image file
        """
        pass
    
    def render_batch(self, genomes: List[Any], batch_name: str = "batch") -> List[str]:
        """
        Render multiple genomes in a batch.
        
        Default implementation calls render_genome for each genome.
        Subclasses can override for optimized batch processing.
        
        Args:
            genomes: List of genome data
            batch_name: Base name for batch files
            
        Returns:
            List of paths to rendered image files
        """
        rendered_files = []
        
        for i, genome in enumerate(genomes):
            output_filename = f"{batch_name}_{i:03d}"
            try:
                rendered_file = self.render_genome(genome, output_filename)
                rendered_files.append(rendered_file)
                self.logger.info(f"Rendered {output_filename}")
            except Exception as e:
                self.logger.error(f"Failed to render {output_filename}: {e}")
                # Continue with remaining genomes
                
        return rendered_files
    
    def get_output_dir(self) -> str:
        """Get the output directory for rendered images."""
        return self.output_dir
    
    def set_output_dir(self, output_dir: str):
        """Set the output directory for rendered images."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def cleanup_output_dir(self, keep_latest: int = 0):
        """
        Clean up old files in the output directory.
        
        Args:
            keep_latest: Number of most recent files to keep (0 = keep all)
        """
        if not os.path.exists(self.output_dir):
            return
        
        if keep_latest <= 0:
            return
        
        try:
            # Get all image files sorted by modification time
            files = []
            for filename in os.listdir(self.output_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    filepath = os.path.join(self.output_dir, filename)
                    if os.path.isfile(filepath):
                        files.append((filepath, os.path.getmtime(filepath)))
            
            # Sort by modification time (newest first)
            files.sort(key=lambda x: x[1], reverse=True)
            
            # Remove old files
            for filepath, _ in files[keep_latest:]:
                try:
                    os.remove(filepath)
                    self.logger.info(f"Removed old file: {filepath}")
                except OSError as e:
                    self.logger.warning(f"Failed to remove {filepath}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to cleanup output directory: {e}")


class RendererConfig:
    """Base configuration class for renderers."""
    
    def __init__(self, **kwargs):
        """Initialize with configuration parameters."""
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def update(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {key: value for key, value in self.__dict__.items() 
                if not key.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RendererConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
