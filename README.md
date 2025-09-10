# FractalGenesis

**An AI-Powered Fractal Evolution System**

FractalGenesis is a sophisticated human-in-the-loop (HITL) system for exploring and evolving 3D fractals using genetic algorithms and machine learning. The system learns your aesthetic preferences and automatically discovers visually stunning fractals by combining user feedback with AI-guided evolution.

## 🌟 Features

- **Intuitive User Interface**: Simple 4-choice selection - just pick your favorite!
- **Genetic Algorithm Evolution**: Sophisticated evolution engine that learns from your preferences
- **Multiple Renderer Support**: Built-in support for Mandelbulber 3D fractals with extensible architecture
- **AI-Guided Discovery**: Machine learning models that learn your taste and guide exploration
- **Modular Architecture**: Clean, extensible codebase for easy customization and expansion
- **Animation Tools**: Generate smooth fractal animations and parameter interpolations
- **Comprehensive Genome System**: Advanced parameter representation supporting complex fractal configurations

## 🏗️ Architecture Overview

FractalGenesis is built with a modular architecture designed for maintainability and extensibility:

```
FractalGenesis/
├── FractalExplorer/          # Main genetic algorithm system
│   ├── genetic_algorithm/    # Evolution engine and selection strategies
│   ├── ai_judge/            # AI models for preference prediction
│   └── ui/                  # User interface components
├── FractalAnimator/         # Animation and keyframe tools
├── renderers/               # Fractal renderer interfaces
│   └── mandelbulber/       # Mandelbulber integration
├── shared/                  # Common utilities and data structures
│   ├── genome.py           # Fractal parameter genome system
│   ├── database/           # Data persistence layer
│   └── config/             # Configuration management
└── examples/                # Usage examples and tutorials
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Mandelbulber2 (for 3D fractal rendering)
- Virtual environment (recommended)

### Installation

1. **Clone the repository** (if you haven't already created it):
```bash
# You're already in the project directory
pwd  # Should show /home/mitchellflautt/FractalGenesis
```

2. **Set up Python environment**:
```bash
# Activate the virtual environment we created
source fractal_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

3. **Install Mandelbulber2**:
```bash
# On Fedora Linux:
sudo dnf install mandelbulber2

# On Ubuntu/Debian:
sudo apt install mandelbulber2

# Or download from: https://sourceforge.net/projects/mandelbulber/
```

### Basic Usage

1. **Initialize a simple exploration session**:

```python
from FractalExplorer.genetic_algorithm import EvolutionEngine, EvolutionConfig
from FractalExplorer.ui import SelectionInterface
from renderers.mandelbulber import MandelbulberRenderer, ParameterTemplates
from shared.genome import FractalGenome, RendererType

# Create evolution engine
config = EvolutionConfig(population_size=20, mutation_rate=0.15)
engine = EvolutionEngine(config, RendererType.MANDELBULBER)

# Initialize with template fractals
seed_genomes = []
templates = ParameterTemplates.get_all_templates()
for template_params in templates.values():
    genome = FractalGenome(RendererType.MANDELBULBER)
    # Convert template to genome (implementation needed)
    seed_genomes.append(genome)

engine.initialize_population(seed_genomes)

# Set up renderer
renderer = MandelbulberRenderer(output_dir=\"./renders\")

# Create user interface
ui = SelectionInterface()

# Run evolution loop
for generation in range(10):
    print(f\"Generation {generation + 1}\")
    
    # Get candidates for user selection
    candidates = engine.get_candidates_for_user_selection(4)
    
    # Render candidates
    image_paths = []
    for candidate in candidates:
        mandel_params = candidate.to_mandelbulber_parameters()
        render_path = renderer.render_single(mandel_params, thumbnail_size=(256, 256))
        image_paths.append(render_path)
    
    # Show to user and get selection
    ui.load_fractal_images(image_paths)
    selected_index = ui.show_and_get_selection()
    
    # Record selection and evolve
    engine.record_user_selection(candidates, selected_index)
    engine.evolve_generation()
    
    print(f\"User selected option {selected_index + 1}\")
```

## 🎯 Core Components

### Genetic Algorithm Engine

The heart of FractalGenesis is a sophisticated genetic algorithm that evolves fractal parameters based on your preferences:

- **Tournament Selection**: Balances fitness with diversity to avoid premature convergence
- **Intelligent Crossover**: Combines fractal parameters in meaningful ways
- **Adaptive Mutation**: Evolves mutation strategies based on population diversity
- **Elitism**: Preserves the best individuals across generations

### Fractal Genome System

Our genome representation captures the full complexity of 3D fractals:

```python
# Example genome structure
genome = FractalGenome()
genome.camera.position = (2.0, -1.0, -3.0)    # Camera position
genome.fractal.formula_type = \"mandelbulb\"     # Fractal type
genome.fractal.power = 8.0                     # Mandelbulb power
genome.color.base_color = (0.3, 0.7, 1.0)     # Surface color
genome.lighting.main_light_direction = (-45, 30)  # Lighting angle
```

### User Interface

Simple, intuitive selection interface:
- **4-Option Grid**: Shows 4 fractal thumbnails
- **One-Click Selection**: Just click your favorite
- **Skip Option**: Skip if none appeal to you
- **Progress Tracking**: See evolution statistics

### Renderer Integration

Currently supports Mandelbulber with extensible architecture:
- **Parameter Translation**: Converts genomes to renderer-specific formats
- **Batch Processing**: Efficient multi-threaded rendering
- **Quality Control**: Automatic thumbnail generation and optimization

## 🧬 Advanced Usage

### Custom Fitness Functions

Create your own fitness evaluation strategies:

```python
from FractalExplorer.genetic_algorithm.fitness import FitnessEvaluator

class CustomFitnessEvaluator(FitnessEvaluator):
    def evaluate(self, genome, context=None):
        # Your custom fitness logic here
        return fitness_score
```

### Animation Generation

Use the FractalAnimator module for smooth animations:

```python
from FractalAnimator import KeyframeAnimator

animator = KeyframeAnimator()
animator.add_keyframe(genome1, time=0.0)
animator.add_keyframe(genome2, time=5.0)
animation_frames = animator.generate_frames(fps=30)
```

### AI-Guided Evolution

Enable AI assistance to accelerate discovery:

```python
from FractalExplorer.ai_judge import VisualFeatureExtractor, PreferencePredictor

# Train AI on your preferences
feature_extractor = VisualFeatureExtractor()
predictor = PreferencePredictor()

# AI will learn to predict what you like
engine.enable_ai_guidance(feature_extractor, predictor)
```

## 🔧 Configuration

Configure the system through `shared/config/`:

```python
# Evolution settings
POPULATION_SIZE = 20
MUTATION_RATE = 0.15
CROSSOVER_RATE = 0.8
ELITE_SIZE = 4

# Rendering settings
RENDER_WIDTH = 800
RENDER_HEIGHT = 600
THUMBNAIL_SIZE = 256

# AI settings
AI_TRAINING_THRESHOLD = 50  # Minimum selections before AI training
```

## 📊 Understanding the Evolution Process

### Generation Lifecycle

1. **Selection**: Choose parents based on fitness and diversity
2. **Crossover**: Create offspring by combining parent genomes
3. **Mutation**: Introduce variations to explore new parameter spaces
4. **Evaluation**: Render candidates and collect user feedback
5. **Replacement**: Form new population with elitism and diversity maintenance

### Fitness Evaluation

The system uses multiple fitness components:
- **User Preferences**: Direct feedback from your selections
- **Diversity Bonus**: Rewards unique and novel fractals
- **AI Predictions**: Learned preferences guide exploration

### Population Dynamics

- **Diversity Monitoring**: Tracks genetic diversity to prevent stagnation
- **Cluster Analysis**: Identifies similar groups in parameter space
- **Novelty Archives**: Maintains history of unique discoveries

## 🎨 Fractal Types Supported

### Current Support
- **Mandelbulb**: Classic 3D Mandelbrot extension
- **Mandelbox**: Box-folding fractals with sharp edges
- **Menger Sponge**: Recursive cubic structures
- **Julia Sets**: 3D Julia set variations

### Planned Support
- **Fractal Flames**: 2D flame algorithm fractals
- **IFS Fractals**: Iterated Function Systems
- **Custom Formulas**: User-defined mathematical expressions

## 🤝 Contributing

We welcome contributions! Areas where help is needed:

1. **New Renderer Integrations**: Support for additional fractal software
2. **AI Improvements**: Better preference learning algorithms
3. **User Interface**: Enhanced visualization and interaction
4. **Performance**: Optimization of rendering and evolution
5. **Documentation**: Examples, tutorials, and guides

## 🐛 Troubleshooting

### Common Issues

**Mandelbulber not found:**
```bash
# Check if Mandelbulber is installed
which mandelbulber2

# If not found, install it:
sudo dnf install mandelbulber2  # Fedora
sudo apt install mandelbulber2  # Ubuntu
```

**Rendering errors:**
- Ensure Mandelbulber can run in headless mode
- Check file permissions in render output directory
- Verify .fract parameter files are valid

**UI not responding:**
- Check if tkinter is properly installed
- Try alternative UI backends (PyQt5)
- Ensure image files are accessible

**Slow evolution:**
- Reduce population size for faster iterations
- Use lower render quality for thumbnails
- Enable batch processing with multiple cores

## 📈 Performance Tips

1. **Optimize Rendering**:
   - Use lower resolution for thumbnails (256x256)
   - Reduce iteration counts for quick previews
   - Enable parallel rendering

2. **Efficient Evolution**:
   - Start with smaller populations (10-20 individuals)
   - Use higher mutation rates early, lower rates later
   - Balance diversity weight (0.2-0.3 works well)

3. **Memory Management**:
   - Clear old renders periodically
   - Limit population history storage
   - Use lazy loading for large image datasets

## 🔬 Research Applications

FractalGenesis can be used for:

- **Aesthetic Research**: Study of visual preferences and beauty
- **Procedural Art**: Automated art generation systems
- **Mathematical Exploration**: Discovery of novel fractal structures  
- **Algorithm Development**: Testing of evolutionary strategies
- **Human-AI Collaboration**: Interactive machine learning studies

## 📚 References and Inspiration

- Mandelbulber Project: https://github.com/buddhi1980/mandelbulber2
- Interactive Evolution: Dawkins' \"The Blind Watchmaker\"
- Genetic Programming: Koza's seminal work on evolutionary algorithms
- Human-in-the-Loop ML: Recent advances in interactive machine learning

## 📄 License

This project is open source. Please see LICENSE file for details.

## 🙏 Acknowledgments

- The Mandelbulber team for their excellent fractal rendering software
- The fractal art community for inspiration and techniques
- Research in evolutionary algorithms and human-computer interaction

---

**Happy fractal exploring!** 🌀

For questions, issues, or contributions, please check our GitHub issues or start a discussion.
