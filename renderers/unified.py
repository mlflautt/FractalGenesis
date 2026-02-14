#!/usr/bin/env python3
"""
Unified Fractal Renderer Interface
===================================

Provides a consistent API across all fractal rendering backends:
- Python 3D native renderer (no external dependencies)
- Mandelbulber (3D ray-marched fractals)
- Flam3 (2D fractal flames)

This abstraction allows easy switching between renderers and enables
future evolution/AI integration.

Usage:
    from renderers.unified import create_renderer, RendererType

    # Create a renderer
    renderer = create_renderer(RendererType.PYTHON_3D)

    # Generate parameters
    params = renderer.generate_random_parameters()

    # Render
    success = renderer.render_fractal(params, "output.png")

    # For animation
    frames = renderer.create_animation(start_params, end_params, num_frames=30)
"""

import os
import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class RendererType(Enum):
    """Available fractal renderer types."""
    PYTHON_3D = "python_3d"
    MANDELBULBER = "mandelbulber"
    FLAM3 = "flam3"
    GENESIS = "genesis"


@dataclass
class FractalParams:
    """Unified parameter representation for all renderers."""
    # Core parameters
    fractal_type: str = "mandelbulb"
    power: float = 8.0
    iterations: int = 100

    # Camera
    camera_pos: Tuple[float, float, float] = (0.0, 0.0, -3.0)
    target: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    fov: float = 45.0

    # Rendering
    width: int = 800
    height: int = 600

    # Colors
    color_palette: str = "warm"
    color_intensity: float = 1.0

    # Renderer-specific parameters (stored as raw dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "fractal_type": self.fractal_type,
            "power": self.power,
            "iterations": self.iterations,
            "camera_pos": self.camera_pos,
            "target": self.target,
            "fov": self.fov,
            "width": self.width,
            "height": self.height,
            "color_palette": self.color_palette,
            "color_intensity": self.color_intensity,
            **self.extra
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "FractalParams":
        """Create from dictionary."""
        extra = {k: v for k, v in data.items() if k not in cls.__dataclass_fields__}
        return cls(
            fractal_type=data.get("fractal_type", "mandelbulb"),
            power=data.get("power", 8.0),
            iterations=data.get("iterations", 100),
            camera_pos=tuple(data.get("camera_pos", [0.0, 0.0, -3.0])),
            target=tuple(data.get("target", [0.0, 0.0, 0.0])),
            fov=data.get("fov", 45.0),
            width=data.get("width", 800),
            height=data.get("height", 600),
            color_palette=data.get("color_palette", "warm"),
            color_intensity=data.get("color_intensity", 1.0),
            extra=extra
        )


class BaseRenderer(ABC):
    """Abstract base class for all fractal renderers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable renderer name."""
        pass

    @property
    @abstractmethod
    def renderer_type(self) -> RendererType:
        """Renderer type identifier."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if renderer is installed and usable."""
        pass

    @abstractmethod
    def generate_random_parameters(self) -> Dict[str, Any]:
        """Generate random parameters for a fractal."""
        pass

    @abstractmethod
    def render_fractal(self, params: Dict, output_path: str,
                       width: int = 512, height: int = 512) -> bool:
        """Render a single fractal."""
        pass

    @abstractmethod
    def mutate_parameters(self, params: Dict, mutation_rate: float = 0.3) -> Dict:
        """Create mutated version of parameters."""
        pass

    def render_batch(self, params_list: List[Dict], output_prefix: str = "fractal",
                     width: int = 512, height: int = 512) -> List[Optional[str]]:
        """Render multiple fractals. Default implementation."""
        results = []
        for i, p in enumerate(params_list):
            path = f"{output_prefix}_{i:03d}.png"
            success = self.render_fractal(p, path, width, height)
            results.append(path if success else None)
        return results

    def crossover(self, params1: Dict, params2: Dict,
                  crossover_rate: float = 0.5) -> Dict:
        """Combine two parameter sets. Default: random selection."""
        if random.random() < crossover_rate:
            return params1.copy()
        return params2.copy()

    def create_animation(self, start_params: Dict, end_params: Dict,
                         num_frames: int = 30,
                         output_dir: str = "animation",
                         name: str = "fractal_animation",
                         width: int = 640, height: int = 480) -> List[str]:
        """Create animation frames by interpolating parameters."""
        from pathlib import Path
        import numpy as np

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        frame_paths = []

        for i in range(num_frames):
            t = i / (num_frames - 1)

            # Linear interpolation for numeric parameters
            interp_params = {}
            for key in start_params:
                if key in end_params:
                    start_val = start_params[key]
                    end_val = end_params[key]

                    if isinstance(start_val, (int, float)) and isinstance(end_val, (int, float)):
                        interp_params[key] = start_val + (end_val - start_val) * t
                    elif isinstance(start_val, tuple) and isinstance(end_val, tuple):
                        interp_params[key] = tuple(
                            a + (b - a) * t for a, b in zip(start_val, end_val)
                        )
                    else:
                        interp_params[key] = start_val if t < 0.5 else end_val
                else:
                    interp_params[key] = start_params.get(key)

            output_path = Path(output_dir) / f"{name}_frame_{i:04d}.png"
            success = self.render_fractal(interp_params, str(output_path), width, height)

            if success:
                frame_paths.append(str(output_path))
                logger.info(f"Frame {i+1}/{num_frames}: {output_path}")

        return frame_paths

    def create_gif(self, frame_paths: List[str], output_path: str,
                   fps: int = 10, loop: int = 0) -> bool:
        """Create GIF from frames."""
        try:
            from PIL import Image
            import imageio

            images = []
            for path in frame_paths:
                if os.path.exists(path):
                    images.append(Image.open(path))

            if images:
                images[0].save(
                    output_path,
                    save_all=True,
                    append_images=images[1:],
                    duration=1000 // fps,
                    loop=loop
                )
                logger.info(f"Created GIF: {output_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to create GIF: {e}")
            return False


class Python3DRenderer(BaseRenderer):
    """Native Python 3D fractal renderer using Numba."""

    def __init__(self):
        try:
            from renderers.python_3d import FractalRenderer
            self._renderer = FractalRenderer()
            self._available = True
        except ImportError as e:
            logger.warning(f"Python 3D renderer not available: {e}")
            self._available = False

    @property
    def name(self) -> str:
        return "Python 3D (Native)"

    @property
    def renderer_type(self) -> RendererType:
        return RendererType.PYTHON_3D

    def is_available(self) -> bool:
        return self._available

    def generate_random_parameters(self) -> Dict[str, Any]:
        return {
            "fractal_type": "mandelbulb",
            "power": random.uniform(2.0, 16.0),
            "iterations": random.randint(50, 150),
            "camera_pos": (
                random.uniform(-2, 2),
                random.uniform(-2, 2),
                random.uniform(-4, -1.5)
            ),
            "color_palette": random.choice(["warm", "cool", "rainbow", "fire", "ice"]),
            "width": 512,
            "height": 512
        }

    def render_fractal(self, params: Dict, output_path: str,
                       width: int = 512, height: int = 512) -> bool:
        try:
            from renderers.python_3d import FractalParams as PyFractalParams

            fractal_params = PyFractalParams(
                fractal_type=params.get("fractal_type", "mandelbulb"),
                power=params.get("power", 8.0),
                iterations=params.get("iterations", 100),
                width=width,
                height=height,
                camera_pos=params.get("camera_pos", (0.0, 0.0, -3.0)),
                color_palette=params.get("color_palette", "warm")
            )

            output_path_path = Path(output_path)
            output_path_path.parent.mkdir(parents=True, exist_ok=True)

            image, metrics = self._renderer.render(fractal_params)

            import matplotlib.pyplot as plt
            plt.imsave(output_path_path, image)
            logger.info(f"Rendered: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Python 3D render failed: {e}")
            return False

    def mutate_parameters(self, params: Dict, mutation_rate: float = 0.3) -> Dict:
        mutated = params.copy()
        for key in mutated:
            if random.random() < mutation_rate:
                if key == "power":
                    mutated[key] = max(2.0, min(16.0, mutated[key] * random.uniform(0.7, 1.4)))
                elif key == "iterations":
                    mutated[key] = max(30, min(200, mutated[key] + random.randint(-20, 20)))
                elif key == "camera_pos":
                    mutated[key] = tuple(
                        max(-5, min(5, v + random.uniform(-0.3, 0.3)))
                        for v in mutated[key]
                    )
        return mutated


class MandelbulberRenderer(BaseRenderer):
    """Mandelbulber 3D fractal renderer."""

    def __init__(self, output_dir: str = "output/mandelbulber"):
        try:
            from renderers.mandelbulber.renderer import MandelbulberRenderer as MDR
            self._renderer = MDR(output_dir)
            self._available = True
        except Exception as e:
            logger.warning(f"Mandelbulber not available: {e}")
            self._available = False

    @property
    def name(self) -> str:
        return "Mandelbulber 3D"

    @property
    def renderer_type(self) -> RendererType:
        return RendererType.MANDELBULBER

    def is_available(self) -> bool:
        return self._available

    def generate_random_parameters(self) -> Dict[str, Any]:
        return self._renderer.generate_random_parameters()

    def render_fractal(self, params: Dict, output_path: str,
                       width: int = 512, height: int = 512) -> bool:
        return self._renderer.render_fractal(params, output_path, width, height)

    def mutate_parameters(self, params: Dict, mutation_rate: float = 0.3) -> Dict:
        return self._renderer.mutate_parameters(params, mutation_rate)


class Flam3Renderer(BaseRenderer):
    """Flam3 fractal flame renderer."""

    def __init__(self, output_dir: str = "output/flam3"):
        try:
            from renderers.flam3_renderer import Flam3Renderer as F3R
            self._renderer = F3R(output_dir)
            self._available = True
        except Exception as e:
            logger.warning(f"Flam3 not available: {e}")
            self._available = False

    @property
    def name(self) -> str:
        return "Flam3 (Fractal Flames)"

    @property
    def renderer_type(self) -> RendererType:
        return RendererType.FLAM3

    def is_available(self) -> bool:
        return self._available

    def generate_random_parameters(self) -> Dict[str, Any]:
        try:
            xml = self._renderer.generate_random_genome()
            return {"genome_xml": xml}
        except:
            return {"genome_xml": "<flame/>"}

    def render_fractal(self, params: Dict, output_path: str,
                       width: int = 512, height: int = 512) -> bool:
        try:
            xml = params.get("genome_xml", "<flame/>")
            result = self._renderer.render_genome(xml, Path(output_path).stem)
            return result is not None
        except Exception as e:
            logger.error(f"Flam3 render failed: {e}")
            return False

    def mutate_parameters(self, params: Dict, mutation_rate: float = 0.3) -> Dict:
        return params  # Flam3 handles mutation internally


def create_renderer(renderer_type: RendererType, **kwargs) -> Optional[BaseRenderer]:
    """Factory function to create a renderer instance."""
    renderers = {
        RendererType.PYTHON_3D: Python3DRenderer,
        RendererType.MANDELBULBER: lambda: MandelbulberRenderer(**kwargs),
        RendererType.FLAM3: lambda: Flam3Renderer(**kwargs),
    }

    if renderer_type in renderers:
        try:
            renderer = renderers[renderer_type]()
            if renderer.is_available():
                return renderer
            logger.warning(f"Renderer {renderer_type.value} is not available")
        except Exception as e:
            logger.error(f"Failed to create renderer {renderer_type.value}: {e}")

    return None


def get_available_renderers() -> List[Tuple[RendererType, str, bool]]:
    """Get list of all available renderers."""
    results = []
    for rt in RendererType:
        renderer = create_renderer(rt)
        available = renderer is not None and renderer.is_available()
        name = renderer.name if renderer else rt.value
        results.append((rt, name, available))
    return results


def compare_renderers(params: Dict, output_dir: str = "output/comparison",
                      width: int = 400, height: int = 400) -> Dict[str, Dict]:
    """Render the same parameters with all available renderers for comparison."""
    from pathlib import Path
    import time

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = {}

    for rt in RendererType:
        renderer = create_renderer(rt)
        if not renderer or not renderer.is_available():
            continue

        timestamp = int(time.time())
        output_path = f"{output_dir}/{rt.value}_{timestamp}.png"

        start_time = time.time()
        success = renderer.render_fractal(params, output_path, width, height)
        elapsed = time.time() - start_time

        results[rt.value] = {
            "renderer": renderer.name,
            "success": success,
            "output_path": output_path if success else None,
            "time_seconds": elapsed,
            "file_size": os.path.getsize(output_path) if success else 0
        }

    return results


import random
