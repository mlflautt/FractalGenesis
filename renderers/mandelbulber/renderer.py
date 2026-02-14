#!/usr/bin/env python3
"""
Mandelbulber Renderer - Fixed for Mandelbulber 2.34+
=====================================================

Uses the new INI-style settings format that Mandelbulber 2.34 expects.
"""

import subprocess
import random
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class MandelbulberRenderer:
    """Integration with Mandelbulber 3D fractal renderer."""

    def __init__(self, output_dir: str = "output/mandelbulber"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.settings_dir = self.output_dir / "settings"
        self.settings_dir.mkdir(parents=True, exist_ok=True)

        self.mandelbulber_cmd = self._find_mandelbulber()
        if not self.mandelbulber_cmd:
            raise RuntimeError(
                "Mandelbulber not found. Install with:\n"
                "  flatpak install com.github.buddhi1980.mandelbulber2"
            )
        logger.info(f"Using Mandelbulber: {' '.join(self.mandelbulber_cmd)}")

    def _find_mandelbulber(self) -> Optional[List[str]]:
        """Find available Mandelbulber installation."""
        try:
            result = subprocess.run(
                ["flatpak", "run", "com.github.buddhi1980.mandelbulber2", "--version"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return ["flatpak", "run", "com.github.buddhi1980.mandelbulber2"]
        except:
            pass

        try:
            result = subprocess.run(
                ["mandelbulber2", "--version"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return ["mandelbulber2"]
        except:
            pass

        return None

    def generate_random_parameters(self) -> Dict[str, Any]:
        """Generate diverse random fractal parameters."""
        camera_distance = random.uniform(1.5, 12.0)
        camera_alpha = random.uniform(0, 360)
        camera_beta = random.uniform(-60, 60)

        alpha_rad = math.radians(camera_alpha)
        beta_rad = math.radians(camera_beta)

        camera_x = camera_distance * math.cos(beta_rad) * math.cos(alpha_rad)
        camera_y = camera_distance * math.cos(beta_rad) * math.sin(alpha_rad)
        camera_z = camera_distance * math.sin(beta_rad)

        return {
            "formula": random.choice([9, 10, 11, 12, 13]),
            "power": random.uniform(2.0, 16.0),
            "detail_level": random.uniform(0.8, 2.0),
            "max_iterations": random.randint(100, 400),
            "camera_x": camera_x,
            "camera_y": camera_y,
            "camera_z": camera_z,
            "target_x": random.uniform(-0.3, 0.3),
            "target_y": random.uniform(-0.3, 0.3),
            "target_z": random.uniform(-0.3, 0.3),
            "fov": random.uniform(30, 90),
            "color_r": random.randint(80, 255),
            "color_g": random.randint(80, 255),
            "color_b": random.randint(80, 255),
            "ambient_occlusion": random.random() > 0.3,
            "glow_enabled": random.random() > 0.5,
            "glow_intensity": random.uniform(0.5, 2.0),
        }

    def _create_settings_file_v234(self, params: Dict, name: str, width: int, height: int) -> str:
        """Create Mandelbulber 2.34+ INI-style settings file."""
        cam_x = params.get("camera_x", 3.0)
        cam_y = params.get("camera_y", -6.0)
        cam_z = params.get("camera_z", 2.0)
        target_x = params.get("target_x", 0.0)
        target_y = params.get("target_y", 0.0)
        target_z = params.get("target_z", 0.0)

        cam_dist = math.sqrt(cam_x**2 + cam_y**2 + cam_z**2)
        alpha = math.degrees(math.atan2(cam_y, cam_x))
        beta = math.degrees(math.asin(cam_z / cam_dist))

        rotation_y = -alpha
        rotation_x = -beta

        content = f'''[main_parameters]
image_width={width}
image_height={height}
formula_1={params.get("formula", 9)}
detail_level={params.get("detail_level", 1.0)}
N={params.get("max_iterations", 200)}
power={params.get("power", 8.0)}
camera={cam_x} {cam_y} {cam_z}
target={target_x} {target_y} {target_z}
camera_rotation={rotation_x} {rotation_y} 0
fov={params.get("fov", 53.13)}
ambient_occlusion={1 if params.get("ambient_occlusion") else 0}
ambient_occlusion_quality=4
glow_enabled={1 if params.get("glow_enabled") else 0}
glow_intensity={params.get("glow_intensity", 1.0)}
brightness=1.0
contrast=1.0
saturation=1.0
gamma=1.0
mat1_is_defined=true
mat1_surface_color={params.get("color_r", 200)*257} {params.get("color_g", 150)*257} {params.get("color_b", 100)*257}
mat1_shading=1
mat1_specular=5.0
mat1_surface_roughness=0.1
mat1_metallic=1.0
mat1_use_colors_from_palette=0
aux_light_enabled_1=true
aux_light_position_1=3 -3 -3
aux_light_intensity_1=1.0
aux_light_colour_1=ffff ffff ffff
aux_light_visibility=1
background_color_1=0000 95a2 ffff
background_color_2=ffff ffff ffff
background_color_3=0000 2710 01f4
background_3_colors_enable=1
save_image_format=0
file_destination={name}

[fractal_parameters]
'''

        settings_path = self.settings_dir / f"{name}.fract"
        with open(settings_path, 'w') as f:
            f.write(content)

        return str(settings_path)

    def render_fractal(self, params: Dict, output_path: str, width: int = 512, height: int = 512) -> bool:
        """Render a fractal with the given parameters."""
        try:
            output_path_path = Path(output_path)
            output_path_path.parent.mkdir(parents=True, exist_ok=True)

            name = output_path_path.stem
            settings_path = self._create_settings_file_v234(params, name, width, height)

            cmd = self.mandelbulber_cmd + [
                "--nogui",
                "--format", "png",
                "--res", f"{width}x{height}",
                "--output", str(output_path_path),
                settings_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                logger.error(f"Mandelbulber failed: {result.stderr}")
                return False

            if not output_path_path.exists():
                logger.error(f"Output not created: {output_path}")
                return False

            logger.info(f"Rendered: {output_path}")
            return True

        except subprocess.TimeoutExpired:
            logger.error("Render timed out")
            return False
        except Exception as e:
            logger.error(f"Render error: {e}")
            return False

    def render_batch(self, params_list: List[Dict], output_prefix: str = "fractal",
                     width: int = 512, height: int = 512, max_concurrent: int = 2) -> List[Optional[Path]]:
        """Render multiple fractals."""
        results = []
        for i, params in enumerate(params_list):
            output_path = self.output_dir / f"{output_prefix}_{i:03d}.png"
            success = self.render_fractal(params, str(output_path), width, height)
            results.append(output_path if success else None)
        return results

    def mutate_parameters(self, params: Dict, mutation_rate: float = 0.3) -> Dict:
        """Create mutated parameters for evolution."""
        mutated = params.copy()

        for key in mutated:
            if random.random() < mutation_rate:
                if key == "formula":
                    mutated[key] = random.choice([9, 10, 11, 12, 13])
                elif key == "power":
                    mutated[key] = max(2.0, min(16.0, mutated[key] * random.uniform(0.7, 1.4)))
                elif key == "detail_level":
                    mutated[key] = max(0.5, min(2.5, mutated[key] * random.uniform(0.8, 1.2)))
                elif key.startswith("camera_"):
                    mutated[key] = mutated[key] * random.uniform(0.8, 1.2)
                elif key.startswith("color_"):
                    mutated[key] = max(50, min(255, mutated[key] + random.randint(-30, 30)))

        return mutated
