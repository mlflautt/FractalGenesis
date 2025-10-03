# Python 3D Fractal Renderer

High-performance Python-based 3D fractal rendering system for FractalGenesis.

## Features

✅ **Multiple Fractal Types**: Mandelbulb, 3D Julia sets, extensible architecture  
✅ **High Performance**: 0.14-0.40s per frame (800x600) with Numba JIT compilation  
✅ **Advanced Materials**: Metallic/roughness PBR-style shading  
✅ **Color Palettes**: Warm, Cool, Rainbow, Fire, Ice, Monochrome  
✅ **Animation System**: Parameter interpolation and frame-by-frame rendering  
✅ **Complete Control**: All parameters accessible via Python API  

## Usage

```python
from renderers.python_3d import FractalRenderer, FractalParams

# Create renderer
renderer = FractalRenderer()

# Set parameters
params = FractalParams(
    fractal_type="mandelbulb",
    power=8.0,
    width=800, height=600,
    camera_pos=(0.0, 0.0, -3.0),
    color_palette="warm"
)

# Render
image, metrics = renderer.render(params)
```

## Performance

- **Numba JIT**: ~0.2s per frame (800x600)
- **Animation**: ~0.5s per frame (400x300) including I/O
- **Parallel Processing**: Multi-core ray marching
- **Memory Efficient**: Optimized parameter passing

## Integration

This system is designed to replace Mandelbulber for better:
- Parameter control in evolution algorithms
- Direct Python API integration
- Animation generation capabilities
- Performance for population rendering

## File: `fractal_animator.py`

Complete implementation with all classes and functions needed for fractal rendering and animation.