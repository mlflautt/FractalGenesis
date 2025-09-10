"""
Flam3 Renderer Integration for FractalGenesis

This module provides integration with the FLAM3 fractal flame renderer,
allowing generation of fractal flames through CLI interface.
"""

import os
import subprocess
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple
import tempfile
import uuid
from pathlib import Path
import logging

from .base_renderer import BaseRenderer


class Flam3Renderer(BaseRenderer):
    """
    Flam3 fractal flame renderer integration.
    
    This renderer uses the flam3 command-line tools to generate 
    and render fractal flames in the Electric Sheep style.
    """
    
    def __init__(self, 
                 output_dir: str = "output/flam3",
                 quality: int = 50,
                 size: int = 512,
                 threads: int = 0):
        """
        Initialize Flam3 renderer.
        
        Args:
            output_dir: Directory to store rendered images
            quality: Rendering quality (1-100, higher is better/slower)
            size: Image size in pixels (width=height)
            threads: Number of threads to use (0 = auto-detect)
        """
        super().__init__(output_dir)
        self.quality = quality
        self.size = size
        self.threads = threads
        self.logger = logging.getLogger(__name__)
        
        # Verify flam3 tools are available
        self._check_flam3_installation()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def _check_flam3_installation(self):
        """Check if flam3 tools are installed and accessible."""
        required_tools = ['flam3-genome', 'flam3-render']
        missing_tools = []
        
        for tool in required_tools:
            if subprocess.run(['which', tool], capture_output=True).returncode != 0:
                missing_tools.append(tool)
        
        if missing_tools:
            raise RuntimeError(
                f"Missing required flam3 tools: {missing_tools}. "
                "Please install flam3: sudo dnf install flam3"
            )
    
    def generate_random_genome(self, seed: Optional[int] = None) -> str:
        """
        Generate a random flam3 genome using flam3-genome tool.
        
        Args:
            seed: Random seed for reproducible generation
            
        Returns:
            XML string containing the flam3 genome
        """
        try:
            # Build flam3-genome command
            env = os.environ.copy()
            env['repeat'] = '1'  # Generate single flame
            
            if seed is not None:
                env['seed'] = str(seed)
            
            # Run flam3-genome to generate random flame
            result = subprocess.run(
                ['flam3-genome'],
                env=env,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"flam3-genome failed: {result.stderr}")
            
            return result.stdout.strip()
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("flam3-genome timed out")
        except Exception as e:
            self.logger.error(f"Failed to generate flam3 genome: {e}")
            raise
    
    def render_genome(self, genome_xml: str, output_filename: Optional[str] = None) -> str:
        """
        Render a flam3 genome to an image file.
        
        Args:
            genome_xml: XML string containing flam3 genome
            output_filename: Optional custom filename (without extension)
            
        Returns:
            Path to rendered image file
        """
        if output_filename is None:
            output_filename = f"fractal_{uuid.uuid4().hex[:8]}"
        
        try:
            # Create temporary file for genome
            with tempfile.NamedTemporaryFile(mode='w', suffix='.flam3', delete=False) as f:
                f.write(genome_xml)
                temp_genome_file = f.name
            
            try:
                # Build rendering environment
                env = os.environ.copy()
                env['prefix'] = os.path.join(self.output_dir, f"{output_filename}.")
                env['quality'] = str(self.quality)
                env['size'] = str(self.size)
                env['format'] = 'png'
                env['nthreads'] = str(self.threads)
                
                # Run flam3-render (don't change directory)
                with open(temp_genome_file, 'r') as genome_file:
                    result = subprocess.run(
                        ['flam3-render'],
                        stdin=genome_file,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=120
                    )
                
                if result.returncode != 0:
                    raise RuntimeError(f"flam3-render failed: {result.stderr}")
                
                # Find the generated image file
                expected_file = f"{output_filename}.00000.png"
                output_path = os.path.join(self.output_dir, expected_file)
                    
                if not os.path.exists(output_path):
                    raise RuntimeError(f"Expected output file not found: {output_path}")
                
                return output_path
                    
            finally:
                # Clean up temporary genome file
                os.unlink(temp_genome_file)
                
        except subprocess.TimeoutExpired:
            raise RuntimeError("flam3-render timed out")
        except Exception as e:
            self.logger.error(f"Failed to render flam3 genome: {e}")
            raise
    
    def parse_genome_xml(self, genome_xml: str) -> Dict:
        """
        Parse flam3 genome XML to extract key parameters.
        
        Args:
            genome_xml: XML string containing flam3 genome
            
        Returns:
            Dictionary containing parsed genome parameters
        """
        try:
            # Parse XML
            root = ET.fromstring(genome_xml)
            
            # Find the flame element (may be wrapped in <pick>)
            flame = root.find('.//flame')
            if flame is None:
                flame = root if root.tag == 'flame' else None
                
            if flame is None:
                raise ValueError("No flame element found in genome XML")
            
            # Extract basic flame attributes
            genome_data = {
                'version': flame.get('version', 'unknown'),
                'time': float(flame.get('time', 0)),
                'size': flame.get('size', '100 100'),
                'center': flame.get('center', '0 0'),
                'scale': float(flame.get('scale', 1)),
                'rotate': float(flame.get('rotate', 0)),
                'quality': float(flame.get('quality', 1)),
                'brightness': float(flame.get('brightness', 4)),
                'gamma': float(flame.get('gamma', 4)),
                'background': flame.get('background', '0 0 0'),
                'xforms': [],
                'colors': []
            }
            
            # Extract xform (transformation) data
            for xform in flame.findall('xform'):
                xform_data = {
                    'weight': float(xform.get('weight', 0.25)),
                    'color': float(xform.get('color', 0)),
                    'coefs': xform.get('coefs', ''),
                    'variations': {}
                }
                
                # Extract variation weights
                for attr_name, attr_value in xform.attrib.items():
                    if attr_name not in ['weight', 'color', 'color_speed', 'animate', 'coefs', 'post', 'opacity']:
                        # This is likely a variation parameter
                        try:
                            xform_data['variations'][attr_name] = float(attr_value)
                        except ValueError:
                            xform_data['variations'][attr_name] = attr_value
                
                genome_data['xforms'].append(xform_data)
            
            # Extract color palette
            for color in flame.findall('color'):
                color_data = {
                    'index': int(color.get('index', 0)),
                    'rgb': color.get('rgb', '0 0 0')
                }
                genome_data['colors'].append(color_data)
            
            return genome_data
            
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML in genome: {e}")
        except Exception as e:
            self.logger.error(f"Failed to parse genome XML: {e}")
            raise
    
    def mutate_genome(self, genome_xml: str, mutation_strength: float = 0.1) -> str:
        """
        Mutate a flam3 genome using flam3-genome mutation.
        
        Args:
            genome_xml: Original genome XML
            mutation_strength: Mutation strength (not directly used by flam3)
            
        Returns:
            Mutated genome XML string
        """
        try:
            # Save original genome to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.flam3', delete=False) as f:
                f.write(genome_xml)
                temp_genome_file = f.name
            
            try:
                # Build mutation environment
                env = os.environ.copy()
                env['mutate'] = temp_genome_file
                env['repeat'] = '1'
                
                # Run flam3-genome for mutation
                result = subprocess.run(
                    ['flam3-genome'],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode != 0:
                    raise RuntimeError(f"flam3-genome mutation failed: {result.stderr}")
                
                return result.stdout.strip()
                
            finally:
                os.unlink(temp_genome_file)
                
        except subprocess.TimeoutExpired:
            raise RuntimeError("flam3-genome mutation timed out")
        except Exception as e:
            self.logger.error(f"Failed to mutate flam3 genome: {e}")
            raise
    
    def crossover_genomes(self, genome1_xml: str, genome2_xml: str, method: str = "interpolate") -> str:
        """
        Crossover two flam3 genomes using flam3-genome.
        
        Args:
            genome1_xml: First parent genome XML
            genome2_xml: Second parent genome XML  
            method: Crossover method ("interpolate", "alternate", or "union")
            
        Returns:
            Child genome XML string
        """
        try:
            # Save parent genomes to temporary files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.flam3', delete=False) as f1:
                f1.write(genome1_xml)
                temp_file1 = f1.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.flam3', delete=False) as f2:
                f2.write(genome2_xml)
                temp_file2 = f2.name
            
            try:
                # Build crossover environment
                env = os.environ.copy()
                env['cross0'] = temp_file1
                env['cross1'] = temp_file2
                env['method'] = method
                env['repeat'] = '1'
                
                # Run flam3-genome for crossover
                result = subprocess.run(
                    ['flam3-genome'],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode != 0:
                    raise RuntimeError(f"flam3-genome crossover failed: {result.stderr}")
                
                return result.stdout.strip()
                
            finally:
                os.unlink(temp_file1)
                os.unlink(temp_file2)
                
        except subprocess.TimeoutExpired:
            raise RuntimeError("flam3-genome crossover timed out")
        except Exception as e:
            self.logger.error(f"Failed to crossover flam3 genomes: {e}")
            raise
    
    def render_batch(self, genomes: List[str], batch_name: str = "batch") -> List[str]:
        """
        Render multiple genomes in a batch.
        
        Args:
            genomes: List of genome XML strings
            batch_name: Base name for batch files
            
        Returns:
            List of paths to rendered image files
        """
        rendered_files = []
        
        for i, genome_xml in enumerate(genomes):
            output_filename = f"{batch_name}_{i:03d}"
            try:
                rendered_file = self.render_genome(genome_xml, output_filename)
                rendered_files.append(rendered_file)
                self.logger.info(f"Rendered {output_filename}")
            except Exception as e:
                self.logger.error(f"Failed to render {output_filename}: {e}")
                # Continue with remaining genomes
                
        return rendered_files
    
    def get_variation_list(self) -> List[str]:
        """
        Get list of available flam3 variations.
        
        Returns:
            List of variation names supported by flam3
        """
        # Standard flam3 variations (from help output)
        return [
            "linear", "sinusoidal", "spherical", "swirl", "horseshoe", "polar",
            "handkerchief", "heart", "disc", "spiral", "hyperbolic", "diamond",
            "ex", "julia", "bent", "waves", "fisheye", "popcorn", "exponential",
            "power", "cosine", "rings", "fan", "blob", "pdj", "fan2", "rings2",
            "eyefish", "bubble", "cylinder", "perspective", "noise", "julian",
            "juliascope", "blur", "gaussian_blur", "radial_blur", "pie", "ngon",
            "curl", "rectangles", "arch", "tangent", "square", "rays", "blade",
            "secant2", "twintrian", "cross", "disc2", "super_shape", "flower",
            "conic", "parabola", "bent2", "bipolar", "boarders", "butterfly",
            "cell", "cpow", "curve", "edisc", "elliptic", "escher", "foci",
            "lazysusan", "loonie", "pre_blur", "modulus", "oscilloscope", "polar2",
            "popcorn2", "scry", "separation", "split", "splits", "stripes",
            "wedge", "wedge_julia", "wedge_sph", "whorl", "waves2", "exp", "log",
            "sin", "cos", "tan", "sec", "csc", "cot", "sinh", "cosh", "tanh",
            "sech", "csch", "coth", "auger", "flux", "mobius"
        ]


class Flam3Config:
    """Configuration settings for Flam3 renderer."""
    
    DEFAULT_CONFIG = {
        'quality': 50,      # Render quality 1-100
        'size': 512,        # Image size in pixels  
        'threads': 0,       # Number of threads (0=auto)
        'format': 'png',    # Output format
        'supersample': 1,   # Supersampling level
        'filter': 0.5,      # Spatial filter radius
        'temporal_samples': 1000,  # Temporal samples for motion blur
        'brightness': 4,    # Brightness adjustment
        'gamma': 4,         # Gamma correction
        'vibrancy': 1,      # Color vibrancy
        'background': '0 0 0',  # Background RGB color
    }
    
    @classmethod
    def get_render_env(cls, config: Optional[Dict] = None) -> Dict[str, str]:
        """
        Get environment variables for flam3-render.
        
        Args:
            config: Optional config overrides
            
        Returns:
            Environment variables dict
        """
        final_config = cls.DEFAULT_CONFIG.copy()
        if config:
            final_config.update(config)
        
        # Convert to string environment variables
        env_vars = {}
        for key, value in final_config.items():
            env_vars[key] = str(value)
            
        return env_vars
