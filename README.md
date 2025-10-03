# FractalGenesis

Fractal generation and evolution system using genetic algorithms. Features a high-performance Python 3D fractal renderer (0.14-0.40s per frame) with complete parameter control, replacing external dependencies like Mandelbulber.

## Features

### Python 3D Renderer
- 0.14-0.40s per frame (800x600) with Numba JIT compilation
- Multiple fractal types: Mandelbulb, 3D Julia sets, extensible architecture
- Advanced materials: Metallic/roughness PBR shading, subsurface scattering
- Animation system: Parameter interpolation and frame sequencing
- Complete Python API control

### Evolution System
- Genetic algorithms for population-based fractal evolution
- Machine learning models learn user aesthetic preferences
- Interactive selection with real-time feedback
- Automated generation using trained AI models

### Rendering Quality
- 6 color palettes: Warm, Cool, Rainbow, Fire, Ice, Monochrome
- Advanced lighting: Hard/soft shadows, ambient occlusion, multiple light sources
- Material properties: Metallic, roughness, transmittance
- Multiple coloring modes: Orbit trap, distance, normal, iteration-based

## Architecture

```
FractalGenesis/
├── renderers/
│   ├── python_3d/           # 🌟 New high-performance Python renderer
│   └── mandelbulber/         # Legacy Mandelbulber integration
├── ai/                       # Machine learning components
├── shared/                   # Common genome and utility classes
├── FractalExplorer/          # Genetic algorithm engine
├── ui/                       # User interface components
├── examples/                 # Demo and example scripts
└── data/                     # User selections and training data
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/FractalGenesis.git
cd FractalGenesis

# Install dependencies
pip install -r requirements.txt

# Verify installation
python3 get_started.py
```

### Basic Usage

```python
# Import the Python 3D renderer
from renderers.python_3d import FractalRenderer, FractalParams

# Create renderer
renderer = FractalRenderer()

# Configure fractal parameters
params = FractalParams(
    fractal_type="mandelbulb",
    power=8.0,
    width=800, height=600,
    camera_pos=(0.0, 0.0, -3.0),
    color_palette="warm"
)

# Render fractal
image, metrics = renderer.render(params)
```

### Run Demo

```bash
# Demonstrate Python 3D rendering capabilities
python3 examples/python_3d_fractal_demo.py

# Launch main GUI interface
python3 fractal_launcher.py

# Start evolution with rendering
python3 examples/flam3_evolution.py --render --generations 5
```

## Performance Comparison

| System | Speed (800x600) | Control | Integration |
|--------|----------------|---------|-------------|
| Python 3D | ~0.2s | Complete | Native |
| Mandelbulber | ~2-10s | CLI Limited | External |
| Flam3 | ~1-5s | XML Config | External |

## Animation Capabilities

Create smooth animations with parameter interpolation:

```python
# Define start and end states
start_params = FractalParams(fractal_type="mandelbulb", power=2.0)
end_params = FractalParams(fractal_type="mandelbulb", power=16.0)

# Generate animation frames
frame_paths = renderer.create_animation(
    start_params, end_params,
    num_frames=30,
    output_dir="animations/power_morph",
    name="power_evolution"
)

# Create GIF
renderer.create_gif(frame_paths, "power_evolution.gif", fps=10)
```

## AI Integration

### Train AI on Your Preferences
```bash
# Collect selection data through interactive evolution
python3 ui/visual_evolution_gui.py

# Train AI model
python3 manage_ai.py train

# Use AI for automated evolution
python3 manage_ai.py evolve "My Style" --generations 10
```

### Evolution Parameters
- **Population Size**: 6-20 individuals per generation
- **Mutation Rate**: Configurable genetic variation
- **Selection Strategy**: Tournament, roulette, or AI-guided
- **Parameter Space**: ~10^10 combinations for exploration

## Technical Details

### Python 3D Renderer Architecture

**JIT-Compiled Ray Marching**:
```python
@jit(nopython=True, parallel=True, fastmath=True)
def render_fractal_fast(image, width, height, params_array):
    # Multi-core ray marching with distance estimation
    # Advanced shading and material calculations
    # Optimized color palette generation
```

**Distance Estimation Functions**:
- `mandelbulb_de()`: Classic power-law iteration
- `julia_set_de()`: 3D Julia sets with complex constants
- Extensible for new fractal types (Mandelbox, Burning Ship, etc.)

**Performance Optimizations**:
- Numba JIT compilation for near-C speed
- Parallel pixel processing across CPU cores
- Efficient parameter packing and memory usage
- Adaptive ray stepping and early termination

## Parameter Control

### Verified Parameter Impact
Our testing shows significant visual impact (>5% difference) for:
- **Power variations**: 27.5% color difference
- **Camera positions**: 17-19% surface difference, 50-67% color difference
- **Fractal types**: 5-8% surface variation
- **Color palettes**: Noticeable visual changes

### Evolution-Friendly Parameters
```python
parameter_ranges = {
    'power': (2.0, 16.0),
    'iterations': (30, 150),
    'camera_distance': (1.5, 6.0),
    'camera_angle': (0.0, 2*π),
    'julia_c': (-0.3, 0.3) for each component,
    'color_intensity': (0.5, 2.0),
    'metallic': (0.0, 1.0),
    'roughness': (0.1, 1.0)
}
```

## Development

### Adding New Fractal Types

1. Implement distance estimation function:
```python
@jit(nopython=True, fastmath=True)
def new_fractal_de(x, y, z, power, max_iter, bailout):
    # Your fractal mathematics here
    return distance, orbit_trap, iterations
```

2. Add to dispatch function:
```python
def get_distance_and_info(x, y, z, fractal_type, ...):
    if fractal_type == NEW_TYPE_ID:
        return new_fractal_de(...)
```

3. Update parameter handling and presets

### Contributing
1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## Project Structure

### Core Components
- `renderers/python_3d/`: High-performance Python 3D fractal renderer
- `FractalExplorer/genetic_algorithm/`: Evolution engine
- `ai/`: Machine learning and preference learning
- `shared/genome.py`: Universal fractal genome representation
- `ui/`: User interface components

### Key Files
- `fractal_launcher.py`: Main GUI application
- `examples/python_3d_fractal_demo.py`: Comprehensive demonstration
- `manage_ai.py`: AI model management
- `requirements.txt`: All dependencies

## Example Outputs

- High-quality static images: 200K+ unique colors, 400K+ surface pixels
- Smooth animations: Parameter interpolation with GIF/video output  
- Evolution sequences: Multi-generation fractal development
- Quality metrics: Quantified diversity and complexity measures

## Future Enhancements

### Planned Features
1. **Additional Fractal Types**: Mandelbox, Burning Ship, Menger Sponge
2. **Advanced Lighting**: Hard/soft shadows, HDR environment maps
3. **GPU Acceleration**: CUDA/OpenCL for 10x+ speed improvements
4. **Real-time Preview**: Interactive parameter adjustment
5. **Video Export**: MP4/WebM animation output
6. **VR Integration**: Immersive fractal exploration

### Research Directions
- Neural style transfer for fractal aesthetics
- Generative adversarial networks for fractal creation
- Reinforcement learning for automated parameter optimization
- Multi-objective optimization for aesthetic and mathematical properties

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built upon research in genetic algorithms and fractal mathematics
- Inspired by Mandelbulb3D's lighting and material systems
- Uses Numba for high-performance Python computation
- Integrates ideas from the fractal art community

## Contact

- **Issues**: Create GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub discussions for questions and ideas  
- **Development**: See CONTRIBUTING.md for development guidelines

---

**FractalGenesis** - Fractal evolution through genetic algorithms and high-performance computing.
