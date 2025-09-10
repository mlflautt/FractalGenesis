"""
Fractal Genome Definition

This module defines the genetic representation (genome) for fractal parameters
that can work across different renderers (Mandelbulber, fractal flames, etc.).
It provides a unified interface for the genetic algorithm operations.
"""

import copy
import random
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod


class RendererType(Enum):
    """Supported fractal renderers"""
    MANDELBULBER = "mandelbulber"
    FRACTAL_FLAME = "fractal_flame"
    CUSTOM = "custom"


class GenomeComponent(ABC):
    """Abstract base class for genome components"""
    
    @abstractmethod
    def mutate(self, mutation_rate: float, mutation_strength: float) -> None:
        """Apply mutations to this component"""
        pass
    
    @abstractmethod
    def crossover(self, other: 'GenomeComponent') -> 'GenomeComponent':
        """Create offspring by crossing with another component"""
        pass
    
    @abstractmethod
    def randomize(self) -> None:
        """Generate random values for this component"""
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        pass


@dataclass
class CameraGene(GenomeComponent):
    """Camera position and orientation genes"""
    position: Tuple[float, float, float] = (0.0, 0.0, -3.0)
    target: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    up_vector: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov: float = 1.0
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    def mutate(self, mutation_rate: float, mutation_strength: float) -> None:
        if random.random() < mutation_rate:
            # Mutate position
            pos = list(self.position)
            for i in range(3):
                pos[i] += random.gauss(0, mutation_strength)
            self.position = tuple(pos)
        
        if random.random() < mutation_rate:
            # Mutate target
            target = list(self.target)
            for i in range(3):
                target[i] += random.gauss(0, mutation_strength * 0.5)
            self.target = tuple(target)
        
        if random.random() < mutation_rate:
            # Mutate FOV
            self.fov += random.gauss(0, mutation_strength * 0.1)
            self.fov = max(0.1, min(3.0, self.fov))
    
    def crossover(self, other: 'CameraGene') -> 'CameraGene':
        child = CameraGene()
        # Blend positions
        child.position = tuple((a + b) / 2 for a, b in zip(self.position, other.position))
        child.target = tuple((a + b) / 2 for a, b in zip(self.target, other.target))
        child.fov = random.choice([self.fov, other.fov])
        child.rotation = random.choice([self.rotation, other.rotation])
        return child
    
    def randomize(self) -> None:
        self.position = tuple(random.uniform(-5, 5) for _ in range(3))
        self.target = tuple(random.uniform(-1, 1) for _ in range(3))
        self.fov = random.uniform(0.5, 2.0)
        self.rotation = tuple(random.uniform(-180, 180) for _ in range(3))
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass  
class FractalGene(GenomeComponent):
    """Core fractal mathematical parameters"""
    formula_type: str = "mandelbulb"
    power: float = 8.0
    bailout: float = 10.0
    iterations: int = 250
    julia_mode: bool = False
    julia_constant: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    custom_params: Dict[str, float] = field(default_factory=dict)
    
    def mutate(self, mutation_rate: float, mutation_strength: float) -> None:
        if random.random() < mutation_rate:
            # Mutate power
            self.power += random.gauss(0, mutation_strength * 2)
            self.power = max(0.1, min(30.0, self.power))
        
        if random.random() < mutation_rate:
            # Mutate bailout
            self.bailout += random.gauss(0, mutation_strength * 5)
            self.bailout = max(2.0, min(100.0, self.bailout))
        
        if random.random() < mutation_rate * 0.3:  # Less frequent
            # Mutate formula type
            formulas = ["mandelbulb", "mandelbox", "menger_sponge", "burning_ship"]
            self.formula_type = random.choice(formulas)
        
        if random.random() < mutation_rate * 0.1:  # Even less frequent
            # Toggle julia mode
            self.julia_mode = not self.julia_mode
            if self.julia_mode:
                self.julia_constant = tuple(random.uniform(-1, 1) for _ in range(3))
    
    def crossover(self, other: 'FractalGene') -> 'FractalGene':
        child = FractalGene()
        child.formula_type = random.choice([self.formula_type, other.formula_type])
        child.power = random.choice([self.power, other.power])
        child.bailout = (self.bailout + other.bailout) / 2
        child.iterations = random.choice([self.iterations, other.iterations])
        child.julia_mode = random.choice([self.julia_mode, other.julia_mode])
        if child.julia_mode:
            child.julia_constant = random.choice([self.julia_constant, other.julia_constant])
        return child
    
    def randomize(self) -> None:
        formulas = ["mandelbulb", "mandelbox", "menger_sponge"]
        self.formula_type = random.choice(formulas)
        self.power = random.uniform(2, 16)
        self.bailout = random.uniform(5, 50)
        self.iterations = random.randint(50, 400)
        self.julia_mode = random.random() < 0.2  # 20% chance
        if self.julia_mode:
            self.julia_constant = tuple(random.uniform(-1, 1) for _ in range(3))
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ColorGene(GenomeComponent):
    """Color and material parameters"""
    base_color: Tuple[float, float, float] = (0.5, 0.5, 0.8)
    specular_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    roughness: float = 0.1
    metallic: float = 0.0
    emission: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    transparency: float = 0.0
    
    def mutate(self, mutation_rate: float, mutation_strength: float) -> None:
        if random.random() < mutation_rate:
            # Mutate base color
            base = list(self.base_color)
            for i in range(3):
                base[i] += random.gauss(0, mutation_strength)
                base[i] = max(0.0, min(1.0, base[i]))
            self.base_color = tuple(base)
        
        if random.random() < mutation_rate:
            # Mutate material properties
            self.roughness += random.gauss(0, mutation_strength * 0.1)
            self.roughness = max(0.0, min(1.0, self.roughness))
            
            self.metallic += random.gauss(0, mutation_strength * 0.2)
            self.metallic = max(0.0, min(1.0, self.metallic))
    
    def crossover(self, other: 'ColorGene') -> 'ColorGene':
        child = ColorGene()
        # Blend colors
        child.base_color = tuple((a + b) / 2 for a, b in zip(self.base_color, other.base_color))
        child.specular_color = random.choice([self.specular_color, other.specular_color])
        child.roughness = (self.roughness + other.roughness) / 2
        child.metallic = random.choice([self.metallic, other.metallic])
        return child
    
    def randomize(self) -> None:
        self.base_color = tuple(random.random() for _ in range(3))
        self.specular_color = tuple(random.random() for _ in range(3))
        self.roughness = random.random()
        self.metallic = random.random()
        self.transparency = random.random() * 0.5  # Keep mostly opaque
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LightingGene(GenomeComponent):
    """Lighting configuration genes"""
    main_light_direction: Tuple[float, float] = (-45.0, 30.0)  # alpha, beta
    main_light_intensity: float = 1.0
    main_light_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ambient_intensity: float = 0.3
    shadows_enabled: bool = True
    
    def mutate(self, mutation_rate: float, mutation_strength: float) -> None:
        if random.random() < mutation_rate:
            # Mutate light direction
            direction = list(self.main_light_direction)
            direction[0] += random.gauss(0, mutation_strength * 30)  # alpha
            direction[1] += random.gauss(0, mutation_strength * 20)  # beta
            direction[0] = max(-180, min(180, direction[0]))
            direction[1] = max(-90, min(90, direction[1]))
            self.main_light_direction = tuple(direction)
        
        if random.random() < mutation_rate:
            # Mutate intensity
            self.main_light_intensity += random.gauss(0, mutation_strength * 0.3)
            self.main_light_intensity = max(0.1, min(3.0, self.main_light_intensity))
    
    def crossover(self, other: 'LightingGene') -> 'LightingGene':
        child = LightingGene()
        child.main_light_direction = random.choice([self.main_light_direction, other.main_light_direction])
        child.main_light_intensity = (self.main_light_intensity + other.main_light_intensity) / 2
        child.main_light_color = random.choice([self.main_light_color, other.main_light_color])
        child.ambient_intensity = random.choice([self.ambient_intensity, other.ambient_intensity])
        return child
    
    def randomize(self) -> None:
        self.main_light_direction = (random.uniform(-180, 180), random.uniform(-90, 90))
        self.main_light_intensity = random.uniform(0.5, 2.5)
        self.main_light_color = tuple(random.uniform(0.7, 1.0) for _ in range(3))
        self.ambient_intensity = random.uniform(0.1, 0.6)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FractalGenome:
    """
    Complete fractal genome containing all parameter genes.
    
    This is the main genetic representation used by the evolutionary algorithm.
    It provides a unified interface that can be converted to specific renderer formats.
    """
    
    def __init__(self, renderer_type: RendererType = RendererType.MANDELBULBER):
        self.renderer_type = renderer_type
        self.camera = CameraGene()
        self.fractal = FractalGene()
        self.color = ColorGene()
        self.lighting = LightingGene()
        self.fitness: Optional[float] = None
        self.generation: int = 0
        self.parent_ids: List[str] = []
        self.genome_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate a unique ID for this genome"""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.1) -> None:
        """
        Apply mutations to all genome components.
        
        Args:
            mutation_rate: Probability of mutating each component
            mutation_strength: Relative strength of mutations (0.0 to 1.0)
        """
        self.camera.mutate(mutation_rate, mutation_strength)
        self.fractal.mutate(mutation_rate, mutation_strength)
        self.color.mutate(mutation_rate, mutation_strength)
        self.lighting.mutate(mutation_rate, mutation_strength)
        
        # Update ID after mutation
        self.genome_id = self._generate_id()
    
    def crossover(self, other: 'FractalGenome') -> 'FractalGenome':
        """
        Create offspring genome by crossing over with another genome.
        
        Args:
            other: Another genome to crossover with
            
        Returns:
            New offspring genome
        """
        child = FractalGenome(self.renderer_type)
        child.camera = self.camera.crossover(other.camera)
        child.fractal = self.fractal.crossover(other.fractal)
        child.color = self.color.crossover(other.color)
        child.lighting = self.lighting.crossover(other.lighting)
        
        # Set ancestry information
        child.generation = max(self.generation, other.generation) + 1
        child.parent_ids = [self.genome_id, other.genome_id]
        child.genome_id = child._generate_id()
        
        return child
    
    def randomize(self) -> None:
        """Generate completely random genome"""
        self.camera.randomize()
        self.fractal.randomize()
        self.color.randomize()
        self.lighting.randomize()
        self.genome_id = self._generate_id()
    
    def copy(self) -> 'FractalGenome':
        """Create a deep copy of this genome"""
        return copy.deepcopy(self)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert genome to dictionary representation"""
        return {
            'genome_id': self.genome_id,
            'renderer_type': self.renderer_type.value,
            'generation': self.generation,
            'parent_ids': self.parent_ids,
            'fitness': self.fitness,
            'camera': self.camera.to_dict(),
            'fractal': self.fractal.to_dict(),
            'color': self.color.to_dict(),
            'lighting': self.lighting.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FractalGenome':
        """Create genome from dictionary representation"""
        renderer_type = RendererType(data['renderer_type'])
        genome = cls(renderer_type)
        genome.genome_id = data['genome_id']
        genome.generation = data['generation']
        genome.parent_ids = data['parent_ids']
        genome.fitness = data['fitness']
        
        # Reconstruct components
        camera_data = data['camera']
        genome.camera = CameraGene(**camera_data)
        
        fractal_data = data['fractal']
        genome.fractal = FractalGene(**fractal_data)
        
        color_data = data['color']
        genome.color = ColorGene(**color_data)
        
        lighting_data = data['lighting']
        genome.lighting = LightingGene(**lighting_data)
        
        return genome
    
    def to_mandelbulber_parameters(self):
        """
        Convert genome to MandelbulberParameters.
        Only available if MandelbulberParameters is imported.
        """
        try:
            from renderers.mandelbulber.parameters import MandelbulberParameters
        except ImportError:
            raise ImportError("MandelbulberParameters not available")
        
        params = MandelbulberParameters()
        
        # Convert camera
        params.camera.camera_x = self.camera.position[0]
        params.camera.camera_y = self.camera.position[1]
        params.camera.camera_z = self.camera.position[2]
        params.camera.target_x = self.camera.target[0]
        params.camera.target_y = self.camera.target[1]
        params.camera.target_z = self.camera.target[2]
        params.camera.fov = self.camera.fov
        
        # Convert fractal
        params.fractal.formula_name = self.fractal.formula_type
        params.fractal.power = self.fractal.power
        params.fractal.bailout = self.fractal.bailout
        params.fractal.iterations = self.fractal.iterations
        params.fractal.julia_mode = self.fractal.julia_mode
        if self.fractal.julia_mode:
            params.fractal.julia_c_x = self.fractal.julia_constant[0]
            params.fractal.julia_c_y = self.fractal.julia_constant[1]
            params.fractal.julia_c_z = self.fractal.julia_constant[2]
        
        # Convert color/material
        params.material.surface_color_r = self.color.base_color[0]
        params.material.surface_color_g = self.color.base_color[1]
        params.material.surface_color_b = self.color.base_color[2]
        params.material.roughness = self.color.roughness
        
        # Convert lighting
        params.lighting.main_light_alpha = self.lighting.main_light_direction[0]
        params.lighting.main_light_beta = self.lighting.main_light_direction[1]
        params.lighting.main_light_intensity = self.lighting.main_light_intensity
        params.lighting.main_light_color_r = self.lighting.main_light_color[0]
        params.lighting.main_light_color_g = self.lighting.main_light_color[1]
        params.lighting.main_light_color_b = self.lighting.main_light_color[2]
        
        return params
    
    def calculate_diversity(self, other: 'FractalGenome') -> float:
        """
        Calculate genetic diversity/distance between two genomes.
        Higher values indicate more diverse genomes.
        """
        diversity = 0.0
        
        # Camera diversity
        cam_dist = np.linalg.norm(np.array(self.camera.position) - np.array(other.camera.position))
        diversity += cam_dist / 10.0  # Normalize
        
        # Fractal diversity
        diversity += abs(self.fractal.power - other.fractal.power) / 20.0
        diversity += 1.0 if self.fractal.formula_type != other.fractal.formula_type else 0.0
        
        # Color diversity
        color_dist = np.linalg.norm(np.array(self.color.base_color) - np.array(other.color.base_color))
        diversity += color_dist
        
        return diversity
