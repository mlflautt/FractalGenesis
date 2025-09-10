"""
Flam3 Genome Adapter for FractalGenesis

This module provides conversion between the unified FractalGenome format
and Flam3's native XML flame format, enabling genetic operations on 
fractal flames.
"""

import xml.etree.ElementTree as ET
import random
import math
from typing import Dict, List, Optional, Tuple, Any
import uuid
import logging

from shared.genome import FractalGenome, CameraGene, FractalGene, ColorGene, LightingGene


class Flam3Genome:
    """
    Adapter for Flam3 fractal flame genomes.
    
    This class converts between the unified FractalGenome format used
    by FractalGenesis and the XML flame format used by FLAM3.
    """
    
    def __init__(self, xml_data: Optional[str] = None):
        """
        Initialize Flam3Genome from XML data or create empty.
        
        Args:
            xml_data: Optional XML string containing flame data
        """
        self.logger = logging.getLogger(__name__)
        
        if xml_data:
            self.parse_from_xml(xml_data)
        else:
            # Initialize with default values
            self.flame_data = self._create_default_flame()
    
    def _create_default_flame(self) -> Dict[str, Any]:
        """Create a default flame structure."""
        return {
            'version': 'FLAM3-LNX-3.1.1',
            'time': 0,
            'size': '512 512',
            'center': '0 0',
            'scale': 50,
            'rotate': 0,
            'quality': 50,
            'passes': 1,
            'temporal_samples': 1000,
            'supersample': 1,
            'filter': 0.5,
            'filter_shape': 'gaussian',
            'background': '0 0 0',
            'brightness': 4,
            'gamma': 4,
            'highlight_power': -1,
            'vibrancy': 1,
            'estimator_radius': 9,
            'estimator_minimum': 0,
            'estimator_curve': 0.4,
            'gamma_threshold': 0.01,
            'palette_mode': 'step',
            'interpolation_type': 'log',
            'palette_interpolation': 'hsv_circular',
            'xforms': [],
            'colors': self._generate_default_palette()
        }
    
    def _generate_default_palette(self) -> List[Dict[str, Any]]:
        """Generate a default 256-color palette."""
        colors = []
        for i in range(256):
            # Simple gradient from black to white
            val = int(i * 255 / 255)
            colors.append({
                'index': i,
                'rgb': f"{val} {val} {val}"
            })
        return colors
    
    def parse_from_xml(self, xml_data: str):
        """
        Parse flame data from XML string.
        
        Args:
            xml_data: XML string containing flame data
        """
        try:
            # Parse XML and find flame element
            root = ET.fromstring(xml_data)
            flame = root.find('.//flame')
            if flame is None:
                flame = root if root.tag == 'flame' else None
                
            if flame is None:
                raise ValueError("No flame element found in XML")
            
            # Extract flame attributes
            self.flame_data = {}
            for attr_name, default_value in self._create_default_flame().items():
                if attr_name in ['xforms', 'colors']:
                    continue
                    
                attr_value = flame.get(attr_name)
                if attr_value is not None:
                    # Try to convert to appropriate type
                    if isinstance(default_value, (int, float)):
                        try:
                            self.flame_data[attr_name] = type(default_value)(attr_value)
                        except ValueError:
                            self.flame_data[attr_name] = default_value
                    else:
                        self.flame_data[attr_name] = attr_value
                else:
                    self.flame_data[attr_name] = default_value
            
            # Parse xforms (transformations)
            self.flame_data['xforms'] = []
            for xform in flame.findall('xform'):
                xform_data = self._parse_xform(xform)
                self.flame_data['xforms'].append(xform_data)
            
            # Parse colors
            self.flame_data['colors'] = []
            for color in flame.findall('color'):
                color_data = {
                    'index': int(color.get('index', 0)),
                    'rgb': color.get('rgb', '0 0 0')
                }
                self.flame_data['colors'].append(color_data)
            
            # Ensure we have a complete palette
            if len(self.flame_data['colors']) < 256:
                self._fill_palette()
                
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML: {e}")
        except Exception as e:
            self.logger.error(f"Failed to parse XML: {e}")
            raise
    
    def _parse_xform(self, xform: ET.Element) -> Dict[str, Any]:
        """Parse an xform element from XML."""
        xform_data = {
            'weight': float(xform.get('weight', 0.25)),
            'color': float(xform.get('color', 0)),
            'color_speed': float(xform.get('color_speed', 0.5)),
            'animate': xform.get('animate', '1') == '1',
            'opacity': float(xform.get('opacity', 1)),
            'coefs': xform.get('coefs', '1 0 0 1 0 0'),
            'post': xform.get('post', ''),
            'variations': {}
        }
        
        # Extract variation parameters
        standard_attrs = {'weight', 'color', 'color_speed', 'animate', 'opacity', 'coefs', 'post'}
        for attr_name, attr_value in xform.attrib.items():
            if attr_name not in standard_attrs:
                # This is a variation parameter
                try:
                    xform_data['variations'][attr_name] = float(attr_value)
                except ValueError:
                    xform_data['variations'][attr_name] = attr_value
        
        return xform_data
    
    def _fill_palette(self):
        """Fill palette to 256 colors if incomplete."""
        existing_indices = {color['index'] for color in self.flame_data['colors']}
        
        for i in range(256):
            if i not in existing_indices:
                # Interpolate or use default color
                rgb = self._interpolate_palette_color(i)
                self.flame_data['colors'].append({
                    'index': i,
                    'rgb': rgb
                })
        
        # Sort by index
        self.flame_data['colors'].sort(key=lambda x: x['index'])
    
    def _interpolate_palette_color(self, index: int) -> str:
        """Interpolate a palette color based on existing colors."""
        if not self.flame_data['colors']:
            return "128 128 128"  # Default gray
            
        # Find nearest existing colors
        existing_colors = sorted(self.flame_data['colors'], key=lambda x: x['index'])
        
        # Simple interpolation between first and last colors
        first_color = existing_colors[0]['rgb'].split()
        last_color = existing_colors[-1]['rgb'].split()
        
        if len(first_color) == 3 and len(last_color) == 3:
            try:
                t = index / 255.0
                r = int(float(first_color[0]) * (1 - t) + float(last_color[0]) * t)
                g = int(float(first_color[1]) * (1 - t) + float(last_color[1]) * t)
                b = int(float(first_color[2]) * (1 - t) + float(last_color[2]) * t)
                return f"{r} {g} {b}"
            except ValueError:
                pass
        
        return "128 128 128"  # Fallback gray
    
    def to_xml(self) -> str:
        """
        Convert flame data to XML string.
        
        Returns:
            XML string in flam3 format
        """
        # Create root element
        flame = ET.Element('flame')
        
        # Set flame attributes
        for attr_name, attr_value in self.flame_data.items():
            if attr_name not in ['xforms', 'colors']:
                flame.set(attr_name, str(attr_value))
        
        # Add xforms
        for xform_data in self.flame_data['xforms']:
            xform = ET.SubElement(flame, 'xform')
            
            # Standard attributes
            for attr in ['weight', 'color', 'color_speed', 'opacity']:
                if attr in xform_data:
                    xform.set(attr, str(xform_data[attr]))
            
            # Boolean attributes
            if xform_data.get('animate', True):
                xform.set('animate', '1')
            else:
                xform.set('animate', '0')
            
            # Coefficient strings
            if 'coefs' in xform_data:
                xform.set('coefs', xform_data['coefs'])
            if 'post' in xform_data and xform_data['post']:
                xform.set('post', xform_data['post'])
            
            # Variation parameters
            for var_name, var_value in xform_data.get('variations', {}).items():
                xform.set(var_name, str(var_value))
        
        # Add colors
        for color_data in self.flame_data['colors']:
            color = ET.SubElement(flame, 'color')
            color.set('index', str(color_data['index']))
            color.set('rgb', color_data['rgb'])
        
        # Convert to string
        return ET.tostring(flame, encoding='unicode')
    
    def to_fractal_genome(self) -> FractalGenome:
        """
        Convert Flam3Genome to unified FractalGenome format.
        
        Returns:
            FractalGenome object
        """
        # Extract camera information from flame data
        center_parts = self.flame_data['center'].split()
        camera_x = float(center_parts[0]) if len(center_parts) > 0 else 0.0
        camera_y = float(center_parts[1]) if len(center_parts) > 1 else 0.0
        
        camera_gene = CameraGene(
            position=(camera_x, camera_y, -3.0),  # Flam3 is 2D, set fixed Z
            target=(camera_x, camera_y, 0.0),
            fov=self.flame_data['scale'] / 50.0,  # Convert scale to FOV-like value
            rotation=(0.0, 0.0, self.flame_data['rotate'])
        )
        
        # Extract fractal parameters from xforms
        fractal_gene = FractalGene(
            formula_type=f"flam3_{len(self.flame_data['xforms'])}xforms",
            iterations=int(self.flame_data.get('quality', 50) * 10),
            bailout=4.0,  # Standard for flame fractals
            power=2.0,    # Standard default
            julia_mode=False,
            julia_constant=(0.0, 0.0, 0.0),
            custom_params={
                'xforms': self.flame_data['xforms'],
                'supersample': self.flame_data.get('supersample', 1),
                'filter': self.flame_data.get('filter', 0.5),
                'filter_shape': self.flame_data.get('filter_shape', 'gaussian'),
                'temporal_samples': self.flame_data.get('temporal_samples', 1000),
                'estimator_radius': self.flame_data.get('estimator_radius', 9),
                'estimator_minimum': self.flame_data.get('estimator_minimum', 0),
                'estimator_curve': self.flame_data.get('estimator_curve', 0.4)
            }
        )
        
        # Extract color information - convert from average palette colors
        color_palette = self._extract_color_palette()
        avg_color = (0.5, 0.5, 0.8) if not color_palette else color_palette[len(color_palette)//2]
        
        color_gene = ColorGene(
            base_color=avg_color,
            specular_color=(1.0, 1.0, 1.0),
            roughness=0.1,
            metallic=0.0,
            emission=(0.0, 0.0, 0.0),
            transparency=0.0
        )
        
        # Extract lighting (brightness/gamma as lighting)
        lighting_gene = LightingGene(
            main_light_direction=(-45.0, 30.0),
            main_light_intensity=self.flame_data.get('brightness', 4.0) / 4.0,
            main_light_color=(1.0, 1.0, 1.0),
            ambient_intensity=0.3,
            shadows_enabled=True
        )
        
        # Create FractalGenome and set components
        from shared.genome import RendererType
        genome = FractalGenome(RendererType.FRACTAL_FLAME)
        genome.camera = camera_gene
        genome.fractal = fractal_gene
        genome.color = color_gene
        genome.lighting = lighting_gene
        genome.genome_id = genome._generate_id()
        return genome
    
    def _extract_color_palette(self) -> List[Tuple[float, float, float]]:
        """Extract color palette as list of RGB tuples."""
        colors = []
        for color_data in self.flame_data.get('colors', []):
            rgb_str = color_data['rgb']
            try:
                rgb_parts = rgb_str.split()
                if len(rgb_parts) >= 3:
                    r = float(rgb_parts[0]) / 255.0
                    g = float(rgb_parts[1]) / 255.0
                    b = float(rgb_parts[2]) / 255.0
                    colors.append((r, g, b))
            except ValueError:
                colors.append((0.5, 0.5, 0.5))  # Default gray
        
        return colors
    
    @classmethod
    def from_fractal_genome(cls, fractal_genome: FractalGenome) -> 'Flam3Genome':
        """
        Create Flam3Genome from unified FractalGenome.
        
        Args:
            fractal_genome: FractalGenome to convert
            
        Returns:
            Flam3Genome object
        """
        flam3_genome = cls()
        
        # Convert camera gene
        camera = fractal_genome.camera
        flam3_genome.flame_data['center'] = f"{camera.position[0]} {camera.position[1]}"
        flam3_genome.flame_data['scale'] = camera.fov * 50.0  # Convert FOV back to scale
        flam3_genome.flame_data['rotate'] = camera.rotation[2]
        
        # Convert fractal gene
        fractal = fractal_genome.fractal
        flam3_genome.flame_data['quality'] = max(1, min(100, int(fractal.iterations / 1000)))
        
        # Use custom params if available (from previous Flam3 genome)
        if 'xforms' in fractal.custom_params:
            flam3_genome.flame_data['xforms'] = fractal.custom_params['xforms']
        else:
            # Generate default xforms
            flam3_genome.flame_data['xforms'] = flam3_genome._generate_default_xforms()
        
        # Convert other fractal params
        for param in ['supersample', 'filter', 'filter_shape', 'temporal_samples',
                      'estimator_radius', 'estimator_minimum', 'estimator_curve']:
            if param in fractal.custom_params:
                flam3_genome.flame_data[param] = fractal.custom_params[param]
        
        # Convert color gene
        color = fractal_genome.color
        flam3_genome.flame_data['brightness'] = 4.0  # Default flame brightness
        flam3_genome.flame_data['vibrancy'] = 1.0   # Default vibrancy
        
        # Convert base color to a simple palette
        base_colors = [color.base_color]
        flam3_genome._set_color_palette(base_colors)
        
        # Convert lighting gene
        lighting = fractal_genome.lighting
        flam3_genome.flame_data['brightness'] = lighting.main_light_intensity * 4.0
        
        # Set default lighting parameters
        flam3_genome.flame_data['gamma'] = 4.0
        flam3_genome.flame_data['highlight_power'] = -1
        flam3_genome.flame_data['gamma_threshold'] = 0.01
        
        return flam3_genome
    
    def _generate_default_xforms(self) -> List[Dict[str, Any]]:
        """Generate default xform transformations."""
        num_xforms = random.randint(2, 4)
        xforms = []
        
        variations = ['linear', 'sinusoidal', 'spherical', 'swirl', 'horseshoe',
                     'polar', 'heart', 'disc', 'spiral', 'julia', 'bent']
        
        for i in range(num_xforms):
            # Random affine transformation coefficients
            coefs = [
                random.uniform(-1, 1),  # a
                random.uniform(-1, 1),  # b
                random.uniform(-1, 1),  # c
                random.uniform(-1, 1),  # d
                random.uniform(-1, 1),  # e
                random.uniform(-1, 1)   # f
            ]
            
            # Select random variations
            selected_variations = random.sample(variations, random.randint(1, 3))
            variation_dict = {}
            
            total_weight = 0
            for var in selected_variations:
                weight = random.uniform(0.1, 1.0)
                variation_dict[var] = weight
                total_weight += weight
            
            # Normalize variations to sum to 1
            if total_weight > 0:
                for var in variation_dict:
                    variation_dict[var] /= total_weight
            
            xform = {
                'weight': 1.0 / num_xforms,
                'color': i / (num_xforms - 1) if num_xforms > 1 else 0,
                'color_speed': 0.5,
                'animate': True,
                'opacity': 1.0,
                'coefs': ' '.join(map(str, coefs)),
                'post': '',
                'variations': variation_dict
            }
            xforms.append(xform)
        
        return xforms
    
    def _set_color_palette(self, colors: List[Tuple[float, float, float]]):
        """Set color palette from RGB tuples."""
        self.flame_data['colors'] = []
        
        # Resample colors to 256 entries
        for i in range(256):
            if colors:
                # Interpolate or cycle through provided colors
                color_index = int(i * len(colors) / 256)
                color_index = min(color_index, len(colors) - 1)
                r, g, b = colors[color_index]
                
                # Convert to 0-255 range
                r_int = int(r * 255)
                g_int = int(g * 255)
                b_int = int(b * 255)
                
                self.flame_data['colors'].append({
                    'index': i,
                    'rgb': f"{r_int} {g_int} {b_int}"
                })
            else:
                # Default grayscale
                val = int(i * 255 / 255)
                self.flame_data['colors'].append({
                    'index': i,
                    'rgb': f"{val} {val} {val}"
                })
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.1):
        """
        Mutate the flame genome.
        
        Args:
            mutation_rate: Probability of mutating each parameter
            mutation_strength: Strength of mutations
        """
        # Mutate global parameters
        if random.random() < mutation_rate:
            center_parts = self.flame_data['center'].split()
            if len(center_parts) >= 2:
                x = float(center_parts[0]) + random.gauss(0, mutation_strength)
                y = float(center_parts[1]) + random.gauss(0, mutation_strength)
                self.flame_data['center'] = f"{x} {y}"
        
        if random.random() < mutation_rate:
            self.flame_data['scale'] *= (1 + random.gauss(0, mutation_strength))
            self.flame_data['scale'] = max(0.1, self.flame_data['scale'])
        
        if random.random() < mutation_rate:
            self.flame_data['rotate'] += random.gauss(0, mutation_strength * 180)
        
        # Mutate xforms
        for xform in self.flame_data['xforms']:
            # Mutate weight
            if random.random() < mutation_rate:
                xform['weight'] *= (1 + random.gauss(0, mutation_strength))
                xform['weight'] = max(0.01, xform['weight'])
            
            # Mutate color
            if random.random() < mutation_rate:
                xform['color'] += random.gauss(0, mutation_strength)
                xform['color'] = max(0, min(1, xform['color']))
            
            # Mutate coefficients
            if random.random() < mutation_rate:
                coef_parts = xform['coefs'].split()
                if len(coef_parts) >= 6:
                    new_coefs = []
                    for coef_str in coef_parts:
                        try:
                            coef = float(coef_str) + random.gauss(0, mutation_strength)
                            new_coefs.append(str(coef))
                        except ValueError:
                            new_coefs.append(coef_str)
                    xform['coefs'] = ' '.join(new_coefs)
            
            # Mutate variation weights
            for var_name in list(xform.get('variations', {}).keys()):
                if random.random() < mutation_rate:
                    old_weight = xform['variations'][var_name]
                    new_weight = old_weight * (1 + random.gauss(0, mutation_strength))
                    xform['variations'][var_name] = max(0, new_weight)
        
        # Normalize xform weights
        self._normalize_xform_weights()
    
    def _normalize_xform_weights(self):
        """Normalize xform weights to sum to 1."""
        if not self.flame_data['xforms']:
            return
            
        total_weight = sum(xform['weight'] for xform in self.flame_data['xforms'])
        if total_weight > 0:
            for xform in self.flame_data['xforms']:
                xform['weight'] /= total_weight
    
    def crossover(self, other: 'Flam3Genome', crossover_rate: float = 0.5) -> 'Flam3Genome':
        """
        Crossover with another Flam3Genome.
        
        Args:
            other: Other Flam3Genome to crossover with
            crossover_rate: Probability of taking each gene from this genome
            
        Returns:
            New Flam3Genome child
        """
        child = Flam3Genome()
        
        # Crossover global parameters
        for param in ['center', 'scale', 'rotate', 'quality', 'brightness', 'gamma']:
            if random.random() < crossover_rate:
                child.flame_data[param] = self.flame_data[param]
            else:
                child.flame_data[param] = other.flame_data[param]
        
        # Copy other parameters from parent
        for param, default in self._create_default_flame().items():
            if param not in child.flame_data and param not in ['xforms', 'colors']:
                child.flame_data[param] = self.flame_data.get(param, default)
        
        # Crossover xforms
        child.flame_data['xforms'] = []
        max_xforms = max(len(self.flame_data['xforms']), len(other.flame_data['xforms']))
        
        for i in range(max_xforms):
            parent1_xform = self.flame_data['xforms'][i] if i < len(self.flame_data['xforms']) else None
            parent2_xform = other.flame_data['xforms'][i] if i < len(other.flame_data['xforms']) else None
            
            if parent1_xform and parent2_xform:
                # Crossover between both parents
                if random.random() < crossover_rate:
                    child.flame_data['xforms'].append(parent1_xform.copy())
                else:
                    child.flame_data['xforms'].append(parent2_xform.copy())
            elif parent1_xform:
                child.flame_data['xforms'].append(parent1_xform.copy())
            elif parent2_xform:
                child.flame_data['xforms'].append(parent2_xform.copy())
        
        # Crossover colors (take from one parent or other)
        if random.random() < crossover_rate:
            child.flame_data['colors'] = self.flame_data['colors'].copy()
        else:
            child.flame_data['colors'] = other.flame_data['colors'].copy()
        
        child._normalize_xform_weights()
        return child
