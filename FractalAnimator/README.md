# Fractal Animation Evolution System

A comprehensive system for creating and evolving fractal animations using evolutionary algorithms and Mandelbulber integration.

## Overview

The FractalAnimator provides a complete toolkit for:
- **Creating animated fractals** with keyframe interpolation
- **Evolving animations** using genetic algorithms
- **Programmatic control** of Mandelbulber rendering
- **Quality assessment** with fitness functions for motion, aesthetics, and technical quality

## Quick Start

### 1. Test the System
```bash
# Run basic system tests
cd /home/mitchellflautt/FractalGenesis
python3 test_animation_system.py
```

### 2. Create a Simple Animation
```bash
# Create an orbital animation
python3 examples/create_simple_animation.py --template orbital_mandelbulb

# Preview a zoom animation (faster)
python3 examples/create_simple_animation.py --template zoom_into_fractal --preview-only

# List all available templates
python3 examples/create_simple_animation.py --list-templates
```

### 3. Evolve Animations
```bash
# Quick evolution (5 generations, preview quality)
python3 examples/evolve_fractal_animations.py --mode quick --generations 5

# High quality evolution with specific template
python3 examples/evolve_fractal_animations.py --mode quality --template power_evolution --generations 15

# Preview-only evolution for fast experimentation  
python3 examples/evolve_fractal_animations.py --preview-only --generations 10
```

## System Architecture

### Core Components

#### 1. Animation Parameters (`animation_parameters.py`)
- **AnimationParameters**: Main container for animation definition
- **Keyframe**: Point-in-time fractal parameter snapshot
- **CameraPath**: Defines camera movement (orbit, zoom, linear, spiral)
- **InterpolationType**: Smooth transitions between keyframes

```python
from FractalAnimator import AnimationParameters, CameraPath, InterpolationType

# Create animation with orbital camera
animation = AnimationParameters()
animation.duration_seconds = 10.0
animation.fps = 30

# Set up camera path
animation.camera_path = CameraPath(
    path_type="orbit",
    orbit_radius=5.0,
    orbit_speed=1.0
)

# Add keyframes with different fractal parameters  
animation.add_keyframe(0.0, start_params, InterpolationType.EASE_IN_OUT)
animation.add_keyframe(1.0, end_params, InterpolationType.CUBIC)
```

#### 2. Animation Renderer (`animation_renderer.py`)
- **AnimationRenderer**: Renders frame sequences and creates videos
- **Parallel processing**: Multi-threaded frame rendering
- **Multiple formats**: MP4, WebM, GIF output
- **Progress tracking**: Real-time render progress

```python
from FractalAnimator import AnimationRenderer
from renderers.mandelbulber.renderer import MandelbulberRenderer

# Set up renderer
mandelbulber_renderer = MandelbulberRenderer()
animation_renderer = AnimationRenderer(
    mandelbulber_renderer=mandelbulber_renderer,
    output_dir="./animations",
    max_workers=4
)

# Render animation
result = animation_renderer.render_animation(
    animation_params=animation,
    output_name="my_fractal_animation",
    video_format='mp4'
)
```

#### 3. Evolution Engine (`animation_evolution.py`)
- **AnimationEvolutionEngine**: Genetic algorithm for animation optimization
- **Fitness evaluation**: Multi-criteria quality assessment
- **Population management**: Selection, crossover, and mutation
- **Convergence detection**: Early stopping and target fitness

```python
from FractalAnimator import AnimationEvolutionEngine

# Create evolution engine
evolution_engine = AnimationEvolutionEngine(
    renderer=animation_renderer,
    population_size=12,
    max_generations=20
)

# Run evolution
final_population = evolution_engine.evolve_animations(
    seed_animations=seed_population,
    target_fitness=0.85
)
```

#### 4. Animation Templates (`animation_templates.py`)
Pre-configured animation types for common use cases:

- **orbital_mandelbulb**: Smooth orbital motion around fractal
- **zoom_into_fractal**: Dramatic zoom with increasing detail
- **color_morph_sequence**: Color transitions through palette
- **power_evolution_sequence**: Fractal power evolution over time
- **julia_transformation_sequence**: Mandelbulb to Julia set transition
- **lighting_showcase**: Dynamic lighting effects
- **dramatic_zoom_sequence**: High-intensity zoom with parameter changes

```python
from FractalAnimator import AnimationTemplates

# Get a template
orbital_anim = AnimationTemplates.orbital_mandelbulb(duration=15.0, revolutions=2.0)

# Create variations
variation = AnimationTemplates.create_random_variation(orbital_anim, variation_strength=0.3)

# Generate diverse population for evolution
population = AnimationTemplates.create_seed_population(12)
```

### Advanced Features

#### Fitness Evaluation
The system evaluates animations across multiple criteria:

- **Visual Continuity** (30%): Smooth parameter transitions
- **Motion Smoothness** (25%): Camera path and movement quality  
- **Aesthetic Appeal** (20%): Color harmony and composition
- **Complexity Variation** (15%): Interesting changes over time
- **Color Harmony** (10%): Pleasing color relationships

#### Genetic Operations
- **Mutation**: Random parameter modifications with controlled strength
- **Crossover**: Blend keyframes and paths from two parent animations
- **Elitism**: Preserve best individuals across generations
- **Diversity**: Maintain population variety to avoid premature convergence

#### Camera Path Types
- **Orbit**: Circular motion around focal point
- **Spiral**: Inward/outward spiral with height variation
- **Zoom**: Direct approach to focal point with scaling
- **Linear**: Straight-line camera movement
- **Custom**: User-defined waypoint path

## Configuration Options

### Evolution Modes

#### Quick Mode (Testing)
```python
evolution_config = {
    'population_size': 8,
    'elite_size': 2,
    'mutation_rate': 0.2,
    'max_generations': 10
}
```

#### Standard Mode (Balanced)
```python
evolution_config = {
    'population_size': 12,
    'elite_size': 3,
    'mutation_rate': 0.15,
    'max_generations': 20
}
```

#### Quality Mode (Best Results)
```python
evolution_config = {
    'population_size': 16,
    'elite_size': 4,
    'mutation_rate': 0.12,
    'max_generations': 30
}
```

### Render Quality Settings

#### Preview (Fast)
- Resolution: 400x300
- Detail Level: 0.5
- Iterations: 80-100
- Use for: Testing, evolution fitness evaluation

#### Standard (Balanced)  
- Resolution: 1920x1080
- Detail Level: 1.0
- Iterations: 150-300
- Use for: Final outputs, sharing

#### High Quality (Best)
- Resolution: 2560x1440+
- Detail Level: 1.5+
- Iterations: 300-500
- Use for: High-resolution exports, professional use

## Dependencies

### Required
- **Python 3.8+**
- **Mandelbulber2**: Fractal rendering engine
- **FFmpeg**: Video encoding (MP4, WebM, GIF)
- **NumPy**: Numerical computations
- **Pillow**: Image processing

### Installation
```bash
# Fedora/RHEL
sudo dnf install mandelbulber2 ffmpeg python3-numpy python3-pillow

# Ubuntu/Debian  
sudo apt install mandelbulber2 ffmpeg python3-numpy python3-pil

# Or using Flatpak
flatpak install org.mandelbulber.Mandelbulber2
```

## Testing

### Comprehensive Test Suite
```python
from FractalAnimator.testing import run_animation_tests

# Run all tests (quick mode)
test_suite = run_animation_tests(quick_mode=True)

# Run full test suite (slower, more comprehensive)
test_suite = run_animation_tests(quick_mode=False, output_dir="./test_results")
```

### Test Categories
- **Import Tests**: Module loading and dependencies
- **Parameter Tests**: Animation creation and manipulation  
- **Template Tests**: Pre-configured animation templates
- **Genetic Tests**: Mutation and crossover operations
- **Renderer Tests**: Setup and basic functionality
- **Evolution Tests**: Population management and convergence
- **Performance Tests**: Speed and efficiency benchmarks

## API Reference

### AnimationParameters
```python
class AnimationParameters:
    def __init__(self):
        self.duration_seconds: float = 10.0
        self.fps: int = 30
        self.keyframes: List[Keyframe] = []
        self.camera_path: CameraPath = CameraPath()
        
    def add_keyframe(self, time: float, parameters: MandelbulberParameters, 
                    interpolation: InterpolationType = InterpolationType.LINEAR)
    def get_parameters_at_time(self, time: float) -> MandelbulberParameters
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.1)
    def crossover(self, other: 'AnimationParameters') -> 'AnimationParameters'
```

### AnimationRenderer
```python
class AnimationRenderer:
    def __init__(self, mandelbulber_renderer: MandelbulberRenderer, 
                 output_dir: str, max_workers: int = 4)
                 
    def render_animation(self, animation_params: AnimationParameters,
                        output_name: str, video_format: str = 'mp4') -> AnimationRenderResult
    def render_preview(self, animation_params: AnimationParameters,
                      max_frames: int = 30) -> AnimationRenderResult
    def batch_render_animations(self, animation_list: List[Tuple[AnimationParameters, str]]) -> List[AnimationRenderResult]
```

### AnimationEvolutionEngine
```python
class AnimationEvolutionEngine:
    def __init__(self, renderer: AnimationRenderer, population_size: int = 20,
                 max_generations: int = 50)
                 
    def evolve_animations(self, seed_animations: List[AnimationParameters] = None,
                         target_fitness: float = 0.9) -> List[AnimationIndividual]
    def get_best_individual(self) -> Optional[AnimationIndividual]
    def save_evolution_results(self, output_path: Path)
```

## Examples and Use Cases

### 1. Scientific Visualization
Create educational animations showing mathematical concepts:
```python
# Power evolution sequence showing mathematical progression
animation = AnimationTemplates.power_evolution_sequence(duration=20.0)
animation.camera_path.orbit_speed = 0.5  # Slow orbit for observation
```

### 2. Artistic Content
Generate aesthetically pleasing animations for digital art:
```python
# Color morphing with custom palette
animation = AnimationTemplates.color_morph_sequence(duration=15.0)
# Evolve for artistic fitness
evolution_engine.fitness_evaluator.default_weights['aesthetic_appeal'] = 0.5
```

### 3. Technical Demonstrations  
Showcase rendering capabilities and parameter effects:
```python
# Lighting showcase with technical details
animation = AnimationTemplates.lighting_showcase(duration=12.0)
animation.render_width = 2560
animation.render_height = 1440
```

## Troubleshooting

### Common Issues

#### "Mandelbulber not found"
- Install Mandelbulber2 package
- Check PATH environment variable
- Verify installation: `mandelbulber2 --version`

#### "FFmpeg not found" 
- Install FFmpeg package
- Check PATH for ffmpeg executable
- Test: `ffmpeg -version`

#### Slow Evolution Performance
- Reduce population size for testing
- Use preview mode during evolution
- Enable parallel processing
- Reduce animation duration/frames

#### Memory Issues
- Lower render resolution during evolution  
- Reduce max_workers parameter
- Use preview mode for fitness evaluation
- Clean up temporary files regularly

### Performance Optimization

#### For Evolution Speed
```python
# Use preview mode
evolution_config = {
    'preview_mode': True,
    'max_preview_frames': 10,
    'population_size': 8,
    'max_workers': 2
}
```

#### For Render Quality
```python
# High quality settings
animation_params.render_width = 2560
animation_params.render_height = 1440
animation_params.render_quality = 1.5

# Increase fractal detail
for keyframe in animation_params.keyframes:
    keyframe.parameters.fractal.iterations = 400
    keyframe.parameters.render.detail_level = 1.8
```

## Contributing

The system is designed to be extensible:

### Adding New Templates
```python
@staticmethod
def my_custom_template(duration: float = 10.0) -> AnimationParameters:
    params = AnimationParameters()
    params.duration_seconds = duration
    # Configure camera path, keyframes, etc.
    return params
```

### Custom Fitness Functions
```python
def custom_fitness_evaluator(individual: AnimationIndividual) -> AnimationFitness:
    # Implement custom evaluation logic
    fitness = AnimationFitness()
    fitness.total_score = my_custom_scoring(individual.animation_params)
    return fitness
```

### New Camera Path Types  
```python
def calculate_camera_position(self, time: float) -> Tuple[float, float, float]:
    if self.path_type == "my_custom_path":
        # Implement custom camera movement
        return (x, y, z)
```

## License

This fractal animation system is part of the FractalGenesis project. See the main project README for licensing information.

---

For more examples and detailed API documentation, see the `examples/` directory and inline code documentation.