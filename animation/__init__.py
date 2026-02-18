"""
Animation Package
=================

Model: minimax-m2.5 (opencode)
Created: 2026-02-16
Version: 1.1

Advanced animation system for FractalGenesis with:
- Procedural generators (orbits, formula evolution, colors)
- Animation presets & templates
- Keyframe interpolation
- GIF/Video export

Usage:
    from animation import FractalAnimator
    from animation.video_export import create_video
    from animation.procedural_generators import OrbitGenerator
    from animation.presets import AnimationPresets
"""

from .procedural_generators import (
    OrbitGenerator,
    OrbitConfig,
    OrbitPattern,
    FormulaEvolutionGenerator,
    ColorAnimationGenerator
)

from .presets import (
    AnimationPreset,
    AnimationPresets,
    PresetApplicator,
    save_preset,
    load_preset
)

from .video_export import (
    create_video,
    create_video_from_frames,
    create_video_with_audio,
    create_gif_with_ffmpeg,
    get_video_info,
    quick_video,
    check_ffmpeg
)

# Simple animation controller - works with Python 3D renderer
class FractalAnimator:
    """Handles fractal animation creation and export."""
    
    def __init__(self, renderer=None):
        if renderer:
            self.renderer = renderer
        else:
            from renderers.python_3d import FractalRenderer, FractalParams
            self.renderer = FractalRenderer()
            self.params_class = FractalParams
    
    def interpolate_params(self, start: dict, end: dict, t: float) -> dict:
        """Linearly interpolate between two parameter sets."""
        result = {}
        for key in start:
            if key in end:
                s, e = start[key], end[key]
                if isinstance(s, (int, float)) and isinstance(e, (int, float)):
                    result[key] = s + (e - s) * t
                elif isinstance(s, tuple) and isinstance(e, tuple):
                    result[key] = tuple(a + (b - a) * t for a, b in zip(s, e))
                else:
                    result[key] = e if t > 0.5 else s
            else:
                result[key] = start[key]
        for key in end:
            if key not in result:
                result[key] = end[key]
        return result
    
    def animate(self, start_params: dict, end_params: dict, num_frames: int = 30,
                output_dir: str = "output/animation", name: str = "anim",
                width: int = 400, height: int = 400) -> list:
        """Create animation by interpolating between two parameter states."""
        from pathlib import Path
        from renderers.python_3d import FractalParams
        import matplotlib.pyplot as plt
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for p in [start_params, end_params]:
            p.setdefault("width", width)
            p.setdefault("height", height)
            p.setdefault("camera_pos", (0.0, 0.0, -3.0))
            p.setdefault("fractal_type", "mandelbulb")
        
        frames = []
        for i in range(num_frames):
            t = i / (num_frames - 1) if num_frames > 1 else 0.0
            frame_params = self.interpolate_params(start_params, end_params, t)
            frame_params["fractal_type"] = frame_params.get("fractal_type", "mandelbulb")
            
            fp = FractalParams(**frame_params)
            
            # Ensure integer dimensions
            fp.width = int(fp.width)
            fp.height = int(fp.height)
            
            output_file = output_path / f"{name}_frame_{i:04d}.png"
            
            print(f"Frame {i+1}/{num_frames}", end="\r")
            img, _ = self.renderer.render(fp)
            plt.imsave(output_file, img)
            frames.append(str(output_file))
        
        print(f"\n✓ Rendered {num_frames} frames")
        return frames
    
    def create_gif(self, frame_paths: list, output_path: str, fps: int = 10) -> bool:
        """Create GIF from frames."""
        from PIL import Image
        import os
        
        images = [Image.open(f) for f in frame_paths if os.path.exists(f)]
        if images:
            images[0].save(output_path, save_all=True, append_images=images[1:],
                          duration=1000//fps, loop=0)
            print(f"✓ Created GIF: {output_path}")
            return True
        return False
    
    def create_video(self, frame_paths: list, output_path: str, fps: int = 30) -> bool:
        """Create MP4 video from frames."""
        from .video_export import create_video_from_frames
        return create_video_from_frames(frame_paths, output_path, fps)
    
    def animate_power_morph(self, power_start: float = 4.0, power_end: float = 16.0,
                            num_frames: int = 30, output_dir: str = "output/animation",
                            fractal_type: str = "mandelbulb", width: int = 400, height: int = 400,
                            create_video: bool = False) -> list:
        """Create power morph animation with optional video export."""
        frames = self.animate(
            {"fractal_type": fractal_type, "power": power_start, "camera_pos": (0, 0, -3), 
             "color_palette": "warm", "width": width, "height": height},
            {"fractal_type": fractal_type, "power": power_end, "camera_pos": (0, 0, -3),
             "color_palette": "fire", "width": width, "height": height},
            num_frames, output_dir, f"power_{power_start}_to_{power_end}", width, height
        )
        
        if create_video:
            video_path = output_dir + "/power_morph.mp4"
            self.create_video(frames, video_path, fps=10)
        
        return frames


__all__ = [
    # Core Animation
    'FractalAnimator',
    
    # Procedural Generators
    'OrbitGenerator',
    'OrbitConfig', 
    'OrbitPattern',
    'FormulaEvolutionGenerator',
    'ColorAnimationGenerator',
    
    # Presets
    'AnimationPreset',
    'AnimationPresets',
    'PresetApplicator',
    'save_preset',
    'load_preset',
    
    # Video Export
    'create_video',
    'create_video_from_frames',
    'create_video_with_audio',
    'create_gif_with_ffmpeg',
    'get_video_info',
    'quick_video',
    'check_ffmpeg'
]

__version__ = "1.1.0"
