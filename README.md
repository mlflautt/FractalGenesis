# FractalGenesis

FractalGenesis is an interactive fractal evolution system that uses genetic algorithms to breed fractals based on user preferences. Features AI-powered selection that learns aesthetic preferences and can automate future fractal evolution.

**Working Features:**
- GUI launcher with evolution configuration
- Fractal flame evolution via Flam3
- AI preference learning system using RandomForest
- Interactive selection interface for training
- AI selector export/import functionality
- Selection data management and analysis

## Quick Start

### Install Dependencies
```bash
# Fedora/RHEL
sudo dnf install flam3 python3-pandas python3-scikit-learn python3-tkinter

# Ubuntu/Debian  
sudo apt install flam3 python3-pandas python3-sklearn python3-tk

# Python packages
pip install Pillow numpy
```

### Launch GUI
```bash
python3 fractal_launcher.py
```

### Usage
1. Select fractal type (Flam3 recommended)
2. Configure generations (3-10) and population (6-12)
3. Choose rendering mode
4. Click "Start Evolution!"
5. View results when complete

## Key Features

**Evolution:**
- One-click start with configurable parameters
- Real-time visual progress tracking
- Simulation mode for testing vs high-quality rendering

**AI Learning:**
- Interactive selection interface for training data collection
- RandomForest-based preference learning
- Automated evolution using trained models
- Model export/import for sharing

**Data Management:**
- Selection pattern tracking and analysis
- Evolution statistics and metrics
- JSON-based data storage with cleanup tools

## 📚 **Usage Guide**

### 🔥 **Fractal Evolution (GUI Method)**

**Launch the Interface:**
```bash
python3 fractal_launcher.py
```

**Settings Guide:**
- **Fractal Type**: Choose "Fractal Flames (Flam3)" ✅
- **Generations**: 3-10 (more = better results, longer time)
- **Population**: 6-12 (more = more variety per generation)
- **Rendering Mode**: 
  - "Simulate only" → Fast testing (30 seconds)
  - "Render images" → Beautiful results (5-20 minutes)
- **Quality**: 20-80 (higher = better images, slower rendering)
- **Image Size**: 512-1024 pixels

**Results**: Generated fractals saved in `output/gui_fractals/`

### AI Training Workflow

**Generate Training Data:**
```bash
python3 generate_test_data.py
```

**Train AI Model:**
```bash
python3 manage_ai.py train
```

**Manage AI Models:**
```bash
# View data statistics
python3 manage_ai.py data stats

# List trained models
python3 manage_ai.py selectors

# Test model
python3 manage_ai.py test "Model Name"

# Export/import models
python3 manage_ai.py export "Model Name"
python3 manage_ai.py import model.pkl
```

## Architecture

**Core Components:**
- `shared/genome.py`: Universal fractal representation with genetic operations
- `FractalExplorer/genetic_algorithm/`: Evolution engine with population management
- `renderers/`: Multi-renderer architecture (Flam3, Mandelbulber)
- `ai/preference_learner.py`: RandomForest-based preference learning
- `ui/`: GUI components for evolution and selection

## Known Issues

- **Flam3 genome conversion**: Loss of variation data when converting between FractalGenome and Flam3Genome formats
- **Evolution lacks diversity**: Mutations not propagating correctly through the conversion pipeline
- **Mandelbulber integration**: 3D fractal support incomplete, needs evolution pipeline work
- **Rendering quality**: Some rendered fractals appear black or identical due to parameter translation issues

## Troubleshooting

**Flam3 not found:**
```bash
sudo dnf install flam3  # Fedora
sudo apt install flam3  # Ubuntu
```

**GUI won't start:**
```bash
python3 -c "import tkinter; print('GUI available')"
sudo dnf install python3-tkinter  # if missing
```

**No images generated:**
- Check output directory permissions
- Try simulation mode first
- Verify flam3-render is working: `which flam3-render`

