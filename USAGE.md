# Usage Guide

FractalGenesis provides multiple interfaces for fractal evolution: GUI, command-line, and AI training tools.

## Quick Start

**System Check:**
```bash
python3 get_started.py
```

**Basic Usage:**
1. Launch GUI: `python3 fractal_launcher.py`
2. Configure parameters and start evolution
3. View results when complete
4. Optional: Train AI on your selections

## GUI Interface

**Launch:**
```bash
python3 fractal_launcher.py
```

**Configuration:**
- Fractal Type: "Fractal Flames (Flam3)" recommended
- Generations: 3-10 evolution cycles
- Population: 6-12 fractals per generation
- Rendering: "Render images" for results, "Simulation" for testing
- Quality/Size: Higher values = better images but slower

**Process:**
1. Algorithm generates random initial population
2. Evolution runs automatically with simulated selection
3. Results saved to output directory
4. Click "View Results" when complete

## AI Training

**Generate Training Data:**
```bash
python3 generate_test_data.py
```

**Train AI Model:**
```bash
python3 manage_ai.py train
```

**Manage Models:**
```bash
# View data and models
python3 manage_ai.py data stats
python3 manage_ai.py selectors

# Test and export models
python3 manage_ai.py test "Model Name"
python3 manage_ai.py export "Model Name"
```

## Command Line

**Flam3 Evolution:**
```bash
# With rendering
python3 examples/flam3_evolution.py --render --generations 5 --population 8

# Quick test
python3 examples/flam3_evolution.py --generations 3 --population 6
```

**3D Fractals (if Mandelbulber available):**
```bash
python3 examples/basic_evolution.py --generations 5 --population 8
```

## Requirements

- Linux (Fedora/Ubuntu recommended)
- Python 3.8+
- flam3 package
- Python packages: Pillow, numpy, pandas, scikit-learn, tkinter

## Output Locations

- GUI fractals: `output/gui_fractals/`
- CLI fractals: `output/flam3_evolution/`
- Selection data: `data/user_selections/`
- AI models: `selectors/`
