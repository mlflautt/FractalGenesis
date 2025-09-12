# 🌀 FractalGenesis - Complete User Instructions

**The AI-Powered Fractal Evolution System**

## 📋 **System Check (Run This First)**

```bash
python3 get_started.py
```

This will check your system and tell you exactly what you can do next.

---

## 🚀 **Method 1: Easy GUI (Recommended)**

### Launch the Interface
```bash
python3 fractal_launcher.py
```

### Use the GUI
1. **Choose Settings**: 
   - Fractal Type: "Fractal Flames (Flam3)" ✅
   - Generations: 5 (try 3 for faster testing)
   - Population: 8 (try 6 for faster testing)
   - Rendering: "Render actual images" for beautiful results
2. **Click "Start Evolution!"**
3. **Wait** (2-10 minutes depending on settings)
4. **Click "View Results"** to see your fractals

**Results Location**: `output/gui_fractals/`

---

## 🤖 **Method 2: AI Training Workflow**

### Step 1: Create Training Data
```bash
# Generate sample data for testing
python3 generate_test_data.py
```

### Step 2: Train Your AI
```bash
# Train AI from selection data
python3 manage_ai.py train
```

### Step 3: Manage Your AI
```bash
# Check your data
python3 manage_ai.py data stats

# List your trained AIs
python3 manage_ai.py selectors

# Test an AI
python3 manage_ai.py test "My Preferences 2025-09-10"

# Share your AI
python3 manage_ai.py export "My AI Name"

# Use someone else's AI
python3 manage_ai.py import shared_ai.pkl
```

---

## 🔧 **Method 3: Command Line (Advanced)**

### Basic Evolution
```bash
# Simple evolution with rendering
python3 examples/flam3_evolution.py --render --generations 5 --population 8

# Quick test (no rendering)
python3 examples/flam3_evolution.py --generations 3 --population 6
```

---

## 📁 **Where to Find Your Results**

| What | Location |
|------|----------|
| GUI Fractals | `output/gui_fractals/` |
| CLI Fractals | `output/flam3_evolution/` |
| AI Training Data | `data/user_selections/` |
| Trained AI Models | `selectors/` |
| Raw Models | `models/` |

---

## ⚙️ **Installation & Setup**

### Requirements Check
```bash
python3 get_started.py  # This will tell you what's missing
```

### Fedora/RHEL Installation
```bash
# System packages
sudo dnf install flam3 python3-pandas python3-scikit-learn python3-tkinter

# Python packages
pip install Pillow numpy
```

### Ubuntu/Debian Installation
```bash
# System packages
sudo apt install flam3 python3-pandas python3-sklearn python3-tk

# Python packages
pip install Pillow numpy
```

---

## 🎯 **Quick Command Reference**

| Task | Command |
|------|---------|
| **System Check** | `python3 get_started.py` |
| **Launch GUI** | `python3 fractal_launcher.py` |
| **Train AI** | `python3 manage_ai.py train` |
| **View AI Data** | `python3 manage_ai.py data stats` |
| **List AIs** | `python3 manage_ai.py selectors` |
| **Generate Test Data** | `python3 generate_test_data.py` |
| **CLI Evolution** | `python3 examples/flam3_evolution.py --render` |

---

## 🆘 **Troubleshooting**

### "Command not found" errors
```bash
# Make sure you're in the right directory
pwd  # Should show .../FractalGenesis

# Run the getting started guide
python3 get_started.py
```

### GUI won't start
```bash
# Check tkinter
python3 -c "import tkinter; print('GUI available')"

# Install if missing (Ubuntu)
sudo apt install python3-tkinter
```

### "flam3-render not found"
```bash
# Fedora
sudo dnf install flam3

# Ubuntu
sudo apt install flam3
```

### AI features don't work
```bash
# Install ML libraries
sudo dnf install python3-pandas python3-scikit-learn  # Fedora
sudo apt install python3-pandas python3-sklearn      # Ubuntu
```

### No images generated
- Check the `output/` directory permissions
- Try "Simulation mode" first to test the algorithm
- Look for error messages in the GUI status

---

## 📖 **Documentation Files**

- **`get_started.py`** - Interactive system check and setup guide
- **`README.md`** - Complete project documentation
- **`USAGE.md`** - Detailed usage instructions with examples
- **`USER_INSTRUCTIONS.md`** - This file (quick reference)

---

## 🎨 **What You'll Create**

FractalGenesis generates beautiful fractal flame images like:
- Swirling patterns with vibrant colors
- Complex geometric structures
- Organic-looking flowing forms
- Unique mathematical art

Each evolution run creates 10-50+ unique fractals that get progressively more interesting as the algorithm learns and evolves.

---

## 🔄 **Typical Workflow**

1. **First Run**: `python3 get_started.py` (check system)
2. **Basic Use**: `python3 fractal_launcher.py` (create fractals)
3. **AI Training**: `python3 generate_test_data.py` then `python3 manage_ai.py train`
4. **Advanced**: Export/import AI selectors, try CLI evolution
5. **Share**: Export your trained AI for others to use

---

**🎉 That's it! You're ready to evolve beautiful fractals with AI assistance.**

For questions or issues, run `python3 get_started.py` to check your system status.
