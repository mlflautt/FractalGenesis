# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

FractalGenesis is an AI-powered fractal evolution system that uses genetic algorithms to breed fractals based on user preferences. It combines interactive evolution with machine learning to learn aesthetic preferences and automate fractal generation.

## Known Issues

- **Flam3 evolution produces identical/black images**: Genome conversion between FractalGenome and Flam3Genome loses critical variation data
- **Evolution lacks diversity**: Mutations not propagating correctly through conversion chain
- **GUI output directory handling**: Fixed - evolution scripts now respect FRACTAL_OUTPUT_DIR environment variable
- **3D Mandelbulber integration**: Incomplete, renderer exists but evolution pipeline needs work

## Common Commands

### System Setup & Status
```bash
# Check system dependencies and requirements
python3 get_started.py

# Install core dependencies (Fedora/RHEL)
sudo dnf install flam3 python3-pandas python3-scikit-learn python3-tkinter

# Install core dependencies (Ubuntu/Debian)
sudo apt install flam3 python3-pandas python3-sklearn python3-tk

# Install Python dependencies
pip install Pillow numpy
```

### Main Application
```bash
# Launch GUI interface (primary entry point)
python3 fractal_launcher.py

# Command-line fractal evolution with rendering
python3 examples/flam3_evolution.py --render --generations 5 --population 8

# Quick test without rendering
python3 examples/flam3_evolution.py --generations 3 --population 6

# 3D fractal evolution (requires Mandelbulber)
python3 examples/basic_evolution.py --generations 5 --population 8
```

### AI Training & Management
```bash
# Generate test data for AI training
python3 generate_test_data.py

# Train AI from user selection data
python3 manage_ai.py train

# View AI management options
python3 manage_ai.py

# Check selection data statistics
python3 manage_ai.py data stats

# List trained AI selectors
python3 manage_ai.py selectors

# Test an AI selector
python3 manage_ai.py test "My Preferences 2025-09-10"

# Export/import AI selectors
python3 manage_ai.py export "AI Name"
python3 manage_ai.py import shared_ai.pkl
```

### Testing & Debugging
```bash
# Test individual components
python3 test_simple_mandelbulb.py
python3 test_mandelbulber_lighting.py
python3 test_surface_color.py

# Debug Mandelbulber integration
python3 debug_mandelbulber.py

# Interactive selection testing
python3 ui/interactive_selector.py
```

### Development & Analysis
```bash
# Run visual evolution GUI
python3 ui/visual_evolution_gui.py

# Individual test files for specific features
python3 test_exact_params.py
python3 test_formula_2.py
python3 test_material_id.py
```

## High-Level Architecture

### Core Components Architecture

**1. Genome System (`shared/genome.py`)**
- **FractalGenome**: Universal fractal representation with genetic operations
- **GenomeComponent hierarchy**: CameraGene, FractalGene, ColorGene, LightingGene
- **Cross-renderer compatibility**: Translates between different fractal engines
- **Genetic operations**: mutation, crossover, randomization with domain-specific constraints

**2. Evolution Engine (`FractalExplorer/genetic_algorithm/`)**
- **EvolutionEngine**: Main genetic algorithm coordinator
- **Population**: Manages fractal individuals and diversity metrics
- **Selection strategies**: Tournament selection, roulette selection with fitness-diversity balance
- **Elitism + diversity**: Preserves best individuals while maintaining genetic diversity
- **User preference integration**: Converts selections into fitness scores

**3. Multi-Renderer Architecture (`renderers/`)**
- **BaseRenderer**: Abstract interface for all fractal engines
- **Flam3Renderer**: 2D fractal flames via flam3 CLI tools
- **MandelbulberRenderer**: 3D fractals via Mandelbulber CLI/API
- **Pluggable design**: Easy to add new fractal engines
- **Parameter translation**: Genome-to-renderer-specific format conversion

**4. AI Preference Learning (`ai/preference_learner.py`)**
- **FractalFeatureExtractor**: Converts fractal parameters to ML feature vectors
- **UserPreferenceDataset**: Manages training data from user selections
- **AISelector**: Trained models that predict user preferences
- **Multi-algorithm support**: Random Forest, potential for neural networks
- **Export/import system**: Share trained AI models between users

**5. User Interface Layer (`ui/`, `fractal_launcher.py`)**
- **FractalLauncher**: Main GUI with evolution configuration
- **VisualEvolutionGUI**: Real-time evolution with user selection
- **InteractiveSelector**: Selection interface for AI training
- **Multi-mode support**: Automated evolution vs manual selection

### Data Flow Architecture

**Evolution Pipeline:**
1. **Initialization**: Create population from templates or random generation
2. **Rendering**: Convert genomes to fractal images via appropriate renderer
3. **Selection**: User selection or AI prediction determines fitness
4. **Evolution**: Genetic operations create next generation
5. **Iteration**: Repeat with fitness-guided selection and diversity maintenance

**AI Training Pipeline:**
1. **Data Collection**: User selections recorded as JSON with fractal parameters
2. **Feature Extraction**: Convert parameters to numerical feature vectors
3. **Model Training**: RandomForest classification on selection patterns
4. **Validation**: Cross-validation and accuracy metrics
5. **Deployment**: Trained models used for automated evolution

### Key Design Patterns

**Strategy Pattern**: Multiple selection strategies (tournament, roulette) and renderers
**Template Method**: BaseRenderer defines rendering workflow, subclasses implement specifics
**Observer Pattern**: Evolution callbacks for UI updates and progress tracking
**Factory Pattern**: Genome creation and renderer instantiation
**Command Pattern**: CLI management system with pluggable commands

### Integration Points

**External Dependencies:**
- **flam3**: CLI tools for fractal flame generation and rendering
- **Mandelbulber**: 3D fractal rendering engine (optional)
- **scikit-learn**: Machine learning for preference learning
- **tkinter**: GUI framework for user interfaces

**Data Persistence:**
- **JSON files**: User selection data and evolution history
- **Pickle files**: Trained AI models and feature extractors
- **PNG images**: Rendered fractal outputs with organized directory structure

### Extension Points

**Adding New Renderers:**
1. Inherit from BaseRenderer
2. Implement generate_random_genome() and render_genome()
3. Add renderer type to RendererType enum
4. Create parameter translation in FractalGenome

**Adding New AI Models:**
1. Extend FractalFeatureExtractor for domain-specific features
2. Create new model class implementing selection interface
3. Add training pipeline in preference_learner.py
4. Update manage_ai.py CLI for new model types

**Custom Selection Strategies:**
1. Inherit from SelectionStrategy base class
2. Implement selection logic with fitness and diversity considerations
3. Register in evolution engine configuration

## Development Notes

- Evolution uses fitness + diversity scoring to prevent convergence
- All fractal parameters are normalized and bounded for genetic operations  
- AI training requires minimum 50 user selections for effective models
- GUI launcher handles dependency checking and graceful fallbacks
- Visual evolution GUI provides real-time generation progress
- Output directory structure organized by evolution type and timestamp
- Mandelbulber integration supports both system and Flatpak installations

## File Structure Patterns

- `examples/`: Runnable evolution scripts for different fractal types
- `renderers/`: Renderer implementations with consistent interface
- `FractalExplorer/genetic_algorithm/`: Core evolution engine components  
- `ai/`: Machine learning components for preference learning
- `ui/`: User interface components and interactive tools
- `shared/`: Common genome and utility classes
- `data/user_selections/`: JSON files with selection history
- `output/`: Generated fractal images organized by session
- `selectors/`: Trained AI model files for automated selection

<citations>
<document>
    <document_type>WARP_DOCUMENTATION</document_type>
    <document_id>getting-started/quickstart-guide/coding-in-warp</document_id>
</document>
</citations>
