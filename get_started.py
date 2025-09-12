#!/usr/bin/env python3
"""
FractalGenesis Getting Started Guide

This script helps new users check their system and get started with FractalGenesis.
"""

import sys
import subprocess
import os
from pathlib import Path


def check_command(command, description, install_hint=""):
    """Check if a command is available."""
    try:
        result = subprocess.run([command, "--help"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ {description}: Available")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    print(f"❌ {description}: Not found")
    if install_hint:
        print(f"   Install with: {install_hint}")
    return False


def check_python_module(module, description, install_hint=""):
    """Check if a Python module is available."""
    try:
        __import__(module)
        print(f"✅ {description}: Available")
        return True
    except ImportError:
        print(f"❌ {description}: Not found")
        if install_hint:
            print(f"   Install with: {install_hint}")
        return False


def main():
    """Main getting started guide."""
    print("🌀 FractalGenesis - Getting Started Guide")
    print("=" * 50)
    
    # Check system components
    print("\n📋 Checking System Requirements:")
    print("-" * 30)
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"✅ Python {python_version.major}.{python_version.minor}: Good")
    else:
        print(f"❌ Python {python_version.major}.{python_version.minor}: Too old (need 3.8+)")
        return
    
    # Check core dependencies
    flam3_ok = check_command("flam3-render", "Flam3 Renderer", 
                            "sudo dnf install flam3  # Fedora")
    
    # Check Python modules
    tkinter_ok = check_python_module("tkinter", "GUI Support (tkinter)")
    pillow_ok = check_python_module("PIL", "Image Processing (Pillow)",
                                   "pip install Pillow")
    numpy_ok = check_python_module("numpy", "NumPy", 
                                  "pip install numpy")
    
    # Check optional AI modules
    pandas_ok = check_python_module("pandas", "AI Support (pandas)",
                                   "sudo dnf install python3-pandas")
    sklearn_ok = check_python_module("sklearn", "AI Support (scikit-learn)",
                                    "sudo dnf install python3-scikit-learn")
    
    print("\n" + "=" * 50)
    
    # Determine what user can do
    basic_ready = flam3_ok and tkinter_ok and pillow_ok and numpy_ok
    ai_ready = basic_ready and pandas_ok and sklearn_ok
    
    if ai_ready:
        print("🎉 System Status: FULLY READY!")
        print("   You can use all features including AI training.")
        print("\n🚀 Next Steps:")
        print("   1. python3 fractal_launcher.py          # Launch GUI")
        print("   2. python3 generate_test_data.py        # Create AI training data")
        print("   3. python3 manage_ai.py train           # Train your AI")
        
    elif basic_ready:
        print("✅ System Status: BASIC READY")
        print("   You can use fractal evolution but not AI features.")
        print("\n🚀 Next Steps:")
        print("   1. python3 fractal_launcher.py          # Launch GUI")
        print("   2. Install AI dependencies for full features")
        
    else:
        print("⚠️  System Status: NEEDS SETUP")
        print("   Please install the missing dependencies above.")
        return
    
    # Check for existing data
    print("\n📁 Data Status:")
    print("-" * 20)
    
    # Check for selection data
    selection_dir = Path("data/user_selections")
    if selection_dir.exists():
        selection_files = list(selection_dir.glob("*.json"))
        print(f"📊 Selection Data: {len(selection_files)} sessions found")
    else:
        print("📊 Selection Data: None (use generate_test_data.py)")
    
    # Check for AI selectors
    selector_dir = Path("selectors")
    if selector_dir.exists():
        selector_files = list(selector_dir.glob("*_selector.pkl"))
        print(f"🤖 AI Selectors: {len(selector_files)} trained AIs found")
    else:
        print("🤖 AI Selectors: None (train AI first)")
    
    # Check for previous results
    output_dirs = [Path("output/gui_fractals"), Path("output/flam3_evolution")]
    total_images = 0
    for output_dir in output_dirs:
        if output_dir.exists():
            images = list(output_dir.glob("*.png")) + list(output_dir.glob("*.jpg"))
            total_images += len(images)
    
    if total_images > 0:
        print(f"🎨 Generated Fractals: {total_images} images found in output/")
    else:
        print("🎨 Generated Fractals: None (run evolution first)")
    
    print("\n" + "=" * 50)
    print("📖 Full Documentation:")
    print("   README.md     - Complete guide")
    print("   USAGE.md      - Detailed usage instructions")
    print("   manage_ai.py help  - AI management commands")
    
    print("\n🎯 Quick Commands:")
    print("   python3 fractal_launcher.py  # Start GUI")
    print("   python3 manage_ai.py         # View AI commands") 
    print("   python3 get_started.py       # This guide")


if __name__ == "__main__":
    main()
