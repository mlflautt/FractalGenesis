#!/usr/bin/env python3
"""
FractalGenesis GUI Launcher

A simple graphical interface to launch fractal evolution sessions
without needing to use the command line.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import sys
import threading
import subprocess
import webbrowser
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class FractalLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("FractalGenesis - Fractal Evolution")
        self.root.geometry("600x500")
        self.root.resizable(True, True)
        
        # Configure style
        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')
        
        self.setup_ui()
        self.check_dependencies()
        
    def setup_ui(self):
        """Create the main user interface."""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights for resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="FractalGenesis", 
                               font=('TkDefaultFont', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        subtitle_label = ttk.Label(main_frame, text="Evolve Beautiful Fractals with Genetic Algorithms")
        subtitle_label.grid(row=1, column=0, columnspan=2, pady=(0, 20))
        
        # Fractal Type Selection
        type_frame = ttk.LabelFrame(main_frame, text="Fractal Type", padding="10")
        type_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        type_frame.columnconfigure(1, weight=1)
        
        self.fractal_type = tk.StringVar(value="flam3")
        ttk.Radiobutton(type_frame, text="Fractal Flames (Flam3) - Colorful 2D flames", 
                       variable=self.fractal_type, value="flam3").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Radiobutton(type_frame, text="3D Fractals (Mandelbulber) - 3D mathematical shapes", 
                       variable=self.fractal_type, value="mandelbulber").grid(row=1, column=0, sticky=tk.W, pady=2)
        
        # Evolution Settings
        settings_frame = ttk.LabelFrame(main_frame, text="Evolution Settings", padding="10")
        settings_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        settings_frame.columnconfigure(1, weight=1)
        
        # Generations
        ttk.Label(settings_frame, text="Generations:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.generations = tk.IntVar(value=5)
        generations_spin = ttk.Spinbox(settings_frame, from_=1, to=20, width=10, 
                                     textvariable=self.generations)
        generations_spin.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        ttk.Label(settings_frame, text="(How many evolution cycles)").grid(row=0, column=2, sticky=tk.W, padx=(10, 0))
        
        # Population Size
        ttk.Label(settings_frame, text="Population:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.population = tk.IntVar(value=8)
        population_spin = ttk.Spinbox(settings_frame, from_=4, to=20, width=10, 
                                    textvariable=self.population)
        population_spin.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        ttk.Label(settings_frame, text="(Number of fractals per generation)").grid(row=1, column=2, sticky=tk.W, padx=(10, 0))
        
        # Rendering Settings
        render_frame = ttk.LabelFrame(main_frame, text="Rendering Settings", padding="10")
        render_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        render_frame.columnconfigure(1, weight=1)
        
        # Render Mode
        self.render_mode = tk.StringVar(value="render")
        ttk.Radiobutton(render_frame, text="Render actual images (slower, but you can see results)", 
                       variable=self.render_mode, value="render").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Radiobutton(render_frame, text="Simulation mode (faster, for testing)", 
                       variable=self.render_mode, value="simulate").grid(row=1, column=0, sticky=tk.W, pady=2)
        
        # Quality and Size (only when rendering)
        quality_frame = ttk.Frame(render_frame)
        quality_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        quality_frame.columnconfigure(1, weight=1)
        
        ttk.Label(quality_frame, text="Quality:").grid(row=0, column=0, sticky=tk.W)
        self.quality = tk.IntVar(value=50)
        quality_scale = ttk.Scale(quality_frame, from_=10, to=100, variable=self.quality, orient=tk.HORIZONTAL)
        quality_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 10))
        self.quality_label = ttk.Label(quality_frame, text="50")
        self.quality_label.grid(row=0, column=2, sticky=tk.W)
        quality_scale.configure(command=self.update_quality_label)
        
        ttk.Label(quality_frame, text="Image Size:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.image_size = tk.StringVar(value="512")
        size_combo = ttk.Combobox(quality_frame, textvariable=self.image_size, 
                                values=["256", "512", "1024", "2048"], width=10)
        size_combo.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=(5, 0))
        size_combo.configure(state="readonly")
        
        # Output Directory
        output_frame = ttk.LabelFrame(main_frame, text="Output", padding="10")
        output_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        output_frame.columnconfigure(1, weight=1)
        
        ttk.Label(output_frame, text="Save to:").grid(row=0, column=0, sticky=tk.W)
        self.output_dir = tk.StringVar(value=str(project_root / "output" / "gui_fractals"))
        output_entry = ttk.Entry(output_frame, textvariable=self.output_dir)
        output_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 10))
        ttk.Button(output_frame, text="Browse...", command=self.browse_output).grid(row=0, column=2)
        
        # Control Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=6, column=0, columnspan=2, pady=20)
        
        # Start Evolution Button
        self.start_button = ttk.Button(button_frame, text="Start Evolution!", 
                                     command=self.start_evolution)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        # View Results Button
        self.view_button = ttk.Button(button_frame, text="View Results", 
                                    command=self.view_results, state=tk.DISABLED)
        self.view_button.pack(side=tk.LEFT, padx=5)
        
        # Help Button
        ttk.Button(button_frame, text="Help", command=self.show_help).pack(side=tk.LEFT, padx=5)
        
        # Status and Progress
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        status_frame.columnconfigure(0, weight=1)
        
        self.status_text = tk.Text(status_frame, height=8, wrap=tk.WORD)
        status_scroll = ttk.Scrollbar(status_frame, orient=tk.VERTICAL, command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=status_scroll.set)
        
        self.status_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        status_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        status_frame.rowconfigure(0, weight=1)
        
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate')
        self.progress.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def update_quality_label(self, value):
        """Update the quality label when slider moves."""
        self.quality_label.config(text=str(int(float(value))))
    
    def browse_output(self):
        """Browse for output directory."""
        directory = filedialog.askdirectory(initialdir=self.output_dir.get())
        if directory:
            self.output_dir.set(directory)
    
    def log_message(self, message):
        """Add a message to the status log."""
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()
    
    def check_dependencies(self):
        """Check if required dependencies are installed."""
        self.log_message("Checking dependencies...")
        
        # Check for flam3
        try:
            result = subprocess.run(['which', 'flam3-genome'], capture_output=True)
            if result.returncode == 0:
                self.log_message("✓ Flam3 is installed and ready")
                self.flam3_available = True
            else:
                self.log_message("⚠ Flam3 not found. Install with: sudo dnf install flam3")
                self.flam3_available = False
        except Exception:
            self.log_message("⚠ Could not check for Flam3")
            self.flam3_available = False
        
        # Check for Python dependencies
        try:
            import PIL
            self.log_message("✓ PIL/Pillow is available")
        except ImportError:
            self.log_message("⚠ PIL/Pillow not found. Install with: pip install Pillow")
    
    def start_evolution(self):
        """Start the evolution process in a separate thread."""
        # Validate settings
        if self.fractal_type.get() == "flam3" and not self.flam3_available:
            messagebox.showerror("Missing Dependency", 
                               "Flam3 is not installed. Please install it with:\n\nsudo dnf install flam3")
            return
        
        if self.fractal_type.get() == "mandelbulber":
            messagebox.showinfo("Not Yet Implemented", 
                              "Mandelbulber integration is available but not yet connected to this GUI.\n\n" +
                              "For now, please use Fractal Flames (Flam3) which is fully working!")
            return
        
        # Disable start button and show progress
        self.start_button.configure(state=tk.DISABLED)
        self.view_button.configure(state=tk.DISABLED)
        self.progress.start()
        
        # Clear status
        self.status_text.delete(1.0, tk.END)
        self.log_message(f"Starting {self.fractal_type.get()} evolution...")
        self.log_message(f"Generations: {self.generations.get()}")
        self.log_message(f"Population: {self.population.get()}")
        
        if self.render_mode.get() == "render":
            self.log_message(f"Rendering quality: {self.quality.get()}")
            self.log_message(f"Image size: {self.image_size.get()}x{self.image_size.get()}")
        else:
            self.log_message("Running in simulation mode")
        
        self.log_message(f"Output directory: {self.output_dir.get()}")
        self.log_message("\n" + "="*50)
        
        # Start evolution in thread
        evolution_thread = threading.Thread(target=self.run_evolution)
        evolution_thread.daemon = True
        evolution_thread.start()
    
    def run_evolution(self):
        """Run the evolution process."""
        try:
            # Build command
            if self.fractal_type.get() == "flam3":
                script_path = project_root / "examples" / "flam3_evolution.py"
            else:
                script_path = project_root / "examples" / "basic_evolution.py"  # fallback
            
            cmd = [sys.executable, str(script_path),
                  "--generations", str(self.generations.get()),
                  "--population", str(self.population.get())]
            
            if self.render_mode.get() == "render":
                cmd.extend(["--render", 
                           "--quality", str(self.quality.get()),
                           "--size", str(self.image_size.get())])
            
            # Set output directory environment variable
            env = os.environ.copy()
            env['FRACTAL_OUTPUT_DIR'] = self.output_dir.get()
            
            # Run the evolution script
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     universal_newlines=True, bufsize=1, env=env)
            
            # Stream output to GUI
            for line in process.stdout:
                line = line.rstrip()
                if line:
                    self.root.after(0, lambda msg=line: self.log_message(msg))
            
            process.wait()
            
            if process.returncode == 0:
                self.root.after(0, lambda: self.log_message("\n✓ Evolution completed successfully!"))
                self.root.after(0, lambda: self.view_button.configure(state=tk.NORMAL))
                self.root.after(0, lambda: messagebox.showinfo("Complete!", "Evolution finished! Click 'View Results' to see your fractals."))
            else:
                self.root.after(0, lambda: self.log_message(f"\n✗ Evolution failed with code {process.returncode}"))
                self.root.after(0, lambda: messagebox.showerror("Error", "Evolution failed. Check the status log for details."))
        
        except Exception as e:
            error_msg = f"Error running evolution: {e}"
            self.root.after(0, lambda: self.log_message(f"\n✗ {error_msg}"))
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
        
        finally:
            # Re-enable controls
            self.root.after(0, lambda: self.start_button.configure(state=tk.NORMAL))
            self.root.after(0, lambda: self.progress.stop())
    
    def view_results(self):
        """Open the output directory to view results."""
        output_path = Path(self.output_dir.get())
        if output_path.exists():
            # Try to open file manager
            try:
                if os.name == 'nt':  # Windows
                    os.startfile(output_path)
                elif os.name == 'posix':  # Linux/Mac
                    subprocess.run(['xdg-open', str(output_path)])
            except Exception:
                messagebox.showinfo("Results Location", f"Your fractals are saved in:\n{output_path}")
        else:
            messagebox.showwarning("No Results", "No output directory found. Run evolution first.")
    
    def show_help(self):
        """Show help information."""
        help_text = """FractalGenesis Help

How to use:
1. Choose your fractal type (Fractal Flames recommended)
2. Set evolution parameters (5 generations and 8 population work well)
3. Choose whether to render images or simulate
4. Click 'Start Evolution!'
5. Wait for the process to complete
6. Click 'View Results' to see your evolved fractals

Tips:
• Start with simulation mode to test quickly
• Higher quality = better images but slower rendering
• More generations = more evolution but takes longer
• The algorithm will show you 4 fractals each generation and simulate choosing the best one

For more information, visit:
https://github.com/mlflautt/FractalGenesis"""
        
        messagebox.showinfo("Help", help_text)


def main():
    """Create and run the GUI application."""
    root = tk.Tk()
    app = FractalLauncher(root)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_reqwidth()) // 2
    y = (root.winfo_screenheight() - root.winfo_reqheight()) // 2
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()


if __name__ == "__main__":
    main()
