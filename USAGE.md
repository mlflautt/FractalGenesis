# How to Use FractalGenesis

FractalGenesis lets you evolve beautiful fractals using genetic algorithms! You can use either the **simple graphical interface** or the **command line**.

## 🖥️ **Easy Way: Graphical Interface (Recommended)**

### 1. Launch the GUI
```bash
cd /path/to/FractalGenesis
python3 fractal_launcher.py
```

### 2. Use the Interface
- **Choose Fractal Type**: Select "Fractal Flames (Flam3)" (recommended)
- **Set Evolution Parameters**: 
  - Generations: 5 (how many evolution cycles)
  - Population: 8 (how many fractals per generation)  
- **Choose Rendering**:
  - "Render actual images" - See beautiful results (slower)
  - "Simulation mode" - Test quickly without rendering
- **Set Quality/Size**: Higher = better images but slower
- **Click "Start Evolution!"**
- **Wait** for it to complete
- **Click "View Results"** to see your evolved fractals!

### 3. What Happens
The algorithm will:
1. Generate random fractals
2. Show you 4 options each generation  
3. Automatically pick the "best" one (simulated user choice)
4. Create new fractals based on the chosen one
5. Repeat for the specified number of generations
6. Save all the images for you to view

## 🔧 **Advanced Way: Command Line**

### Fractal Flames (Flam3)
```bash
# Basic run with rendering
python3 examples/flam3_evolution.py --render --generations 5 --population 8

# Quick test without rendering  
python3 examples/flam3_evolution.py --generations 3 --population 6

# High quality rendering
python3 examples/flam3_evolution.py --render --quality 80 --size 1024 --generations 3
```

### 3D Fractals (Mandelbulber - if available)
```bash
# Basic evolution (simulation mode)
python3 examples/basic_evolution.py --generations 5 --population 8
```

## 📋 **Requirements**

### For Fractal Flames:
```bash
# Install Flam3 (Fedora)
sudo dnf install flam3

# Install Python dependencies  
pip install Pillow numpy
```

### For 3D Fractals:
- Mandelbulber2 (not in Fedora repos - needs manual installation)

## 📁 **Results**

Your fractals are saved in:
- GUI mode: `output/gui_fractals/`
- Command line: `output/flam3_evolution/` or `output/basic_evolution/`

## 🎯 **Tips**

- **Start with simulation mode** to test quickly
- **Fractal Flames (Flam3)** are fully working and recommended  
- **Higher quality** = better images but much slower rendering
- **More generations** = more evolution but takes longer
- **Population of 6-8** works well for most cases
- **Results get better** over multiple generations as the algorithm learns

## 🆘 **Troubleshooting**

### "Flam3 not found"
```bash
sudo dnf install flam3
```

### "PIL not found"  
```bash  
pip install Pillow
```

### GUI won't start
Make sure you have tkinter:
```bash
python3 -c "import tkinter; print('GUI available')"
```

### No images generated
- Check the output directory exists
- Try simulation mode first to test the algorithm
- Check the status log for error messages

## 🔗 **More Information**

- GitHub: https://github.com/mlflautt/FractalGenesis
- Flam3 Documentation: http://flam3.com/
- Mandelbulber: https://mandelbulber.org/

Enjoy creating beautiful evolved fractals! 🎨✨
