#!/usr/bin/env python3
"""
FractalGenesis Desktop Application
==================================

GUI application for exploring and evolving fractals.
Provides easy access to 2D flame and 3D fractal evolution.

Usage:
    python3 fractalgenesis_gui.py
    
Or launch from desktop/taskbar after installation.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import subprocess
import threading
import os
import sys
from pathlib import Path
import json
from datetime import datetime
from PIL import Image, ImageTk
import webbrowser

class FractalGenesisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FractalGenesis - Fractal Evolution Explorer")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Set icon if available
        try:
            self.root.iconbitmap("assets/icon.ico")
        except:
            pass
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Color scheme
        self.bg_color = "#1e1e1e"
        self.fg_color = "#ffffff"
        self.accent_color = "#00d4ff"
        self.success_color = "#00ff88"
        self.warning_color = "#ffaa00"
        self.error_color = "#ff4444"
        
        self.root.configure(bg=self.bg_color)
        
        # Project root
        self.project_root = Path(__file__).parent
        self.output_dir = self.project_root / "output"
        self.output_dir.mkdir(exist_ok=True)
        
        # Current process
        self.current_process = None
        self.is_running = False
        
        # Build UI
        self._create_menu()
        self._create_main_layout()
        self._create_status_bar()
        
        # Log initial message
        self.log("FractalGenesis Desktop App Started")
        self.log("Ready to explore fractal parameter spaces!")
        self.check_renderers()
    
    def _create_menu(self):
        """Create application menu"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Output Folder", command=self.open_output_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Test System", command=self.run_system_test)
        tools_menu.add_command(label="Verify Diversity", command=self.verify_diversity)
        tools_menu.add_separator()
        tools_menu.add_command(label="Clear Output", command=self.clear_output)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self.open_docs)
        help_menu.add_command(label="About", command=self.show_about)
    
    def _create_main_layout(self):
        """Create main application layout"""
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        left_panel = ttk.Frame(main_container, width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        self._create_control_panel(left_panel)
        
        # Right panel - Output and Preview
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self._create_output_panel(right_panel)
    
    def _create_control_panel(self, parent):
        """Create control panel with evolution settings"""
        # Title
        title_label = ttk.Label(
            parent, 
            text="🌀 FractalGenesis", 
            font=('Helvetica', 18, 'bold')
        )
        title_label.pack(pady=(0, 20))
        
        # Fractal Type Selection
        type_frame = ttk.LabelFrame(parent, text="Fractal Type", padding=10)
        type_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.fractal_type = tk.StringVar(value="3d")
        ttk.Radiobutton(
            type_frame, 
            text="3D Fractals (Mandelbulb)", 
            variable=self.fractal_type, 
            value="3d"
        ).pack(anchor=tk.W)
        ttk.Radiobutton(
            type_frame, 
            text="2D Flames (Fractal Flames)", 
            variable=self.fractal_type, 
            value="2d"
        ).pack(anchor=tk.W)
        
        # Evolution Settings
        settings_frame = ttk.LabelFrame(parent, text="Evolution Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Generations
        ttk.Label(settings_frame, text="Generations:").grid(row=0, column=0, sticky=tk.W)
        self.generations = ttk.Spinbox(settings_frame, from_=1, to=20, width=10)
        self.generations.set(3)
        self.generations.grid(row=0, column=1, padx=5)
        
        # Population
        ttk.Label(settings_frame, text="Population:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.population = ttk.Spinbox(settings_frame, from_=4, to=20, width=10)
        self.population.set(6)
        self.population.grid(row=1, column=1, padx=5)
        
        # Mutation Rate
        ttk.Label(settings_frame, text="Mutation Rate:").grid(row=2, column=0, sticky=tk.W)
        self.mutation_rate = ttk.Spinbox(settings_frame, from_=0.1, to=0.9, increment=0.1, width=10)
        self.mutation_rate.set(0.3)
        self.mutation_rate.grid(row=2, column=1, padx=5)
        
        # Action Buttons
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=20)
        
        self.run_btn = tk.Button(
            button_frame, 
            text="▶ Start Evolution", 
            command=self.start_evolution,
            bg=self.success_color,
            fg="black",
            font=('Helvetica', 12, 'bold'),
            height=2
        )
        self.run_btn.pack(fill=tk.X, pady=(0, 5))
        
        self.stop_btn = tk.Button(
            button_frame, 
            text="⏹ Stop", 
            command=self.stop_evolution,
            bg=self.error_color,
            fg="white",
            font=('Helvetica', 10),
            state=tk.DISABLED
        )
        self.stop_btn.pack(fill=tk.X, pady=(0, 5))
        
        # Quick Actions
        quick_frame = ttk.LabelFrame(parent, text="Quick Actions", padding=10)
        quick_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(
            quick_frame, 
            text="Generate Single 3D", 
            command=lambda: self.generate_single("3d")
        ).pack(fill=tk.X, pady=2)
        
        ttk.Button(
            quick_frame, 
            text="Generate Single 2D", 
            command=lambda: self.generate_single("2d")
        ).pack(fill=tk.X, pady=2)
        
        ttk.Button(
            quick_frame, 
            text="Test System", 
            command=self.run_system_test
        ).pack(fill=tk.X, pady=2)
        
        # Renderer Status
        self.status_frame = ttk.LabelFrame(parent, text="Renderer Status", padding=10)
        self.status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.renderer_status = tk.Text(
            self.status_frame, 
            height=5, 
            width=30,
            state=tk.DISABLED,
            bg="#2e2e2e",
            fg=self.fg_color
        )
        self.renderer_status.pack(fill=tk.X)
    
    def _create_output_panel(self, parent):
        """Create output panel with log and preview"""
        # Notebook for tabs
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Log tab
        log_frame = ttk.Frame(self.notebook)
        self.notebook.add(log_frame, text="Log")
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            wrap=tk.WORD,
            font=('Consolas', 10),
            bg="#1e1e1e",
            fg=self.fg_color,
            insertbackground=self.fg_color
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Preview tab
        preview_frame = ttk.Frame(self.notebook)
        self.notebook.add(preview_frame, text="Preview")
        
        # Preview canvas with scrollbar
        self.preview_canvas = tk.Canvas(
            preview_frame,
            bg="#1e1e1e",
            highlightthickness=0
        )
        scrollbar = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.preview_canvas.yview)
        self.preview_canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.preview_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.preview_frame = ttk.Frame(self.preview_canvas)
        self.preview_canvas.create_window((0, 0), window=self.preview_frame, anchor=tk.NW)
        
        self.preview_images = []
        
        # Gallery tab
        gallery_frame = ttk.Frame(self.notebook)
        self.notebook.add(gallery_frame, text="Gallery")
        
        self.gallery_text = tk.Text(
            gallery_frame,
            wrap=tk.WORD,
            font=('Helvetica', 10),
            bg="#1e1e1e",
            fg=self.fg_color
        )
        self.gallery_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.gallery_text.insert(tk.END, "Generated fractals will appear here.\n\n")
        self.gallery_text.config(state=tk.DISABLED)
    
    def _create_status_bar(self):
        """Create status bar at bottom"""
        self.status_bar = ttk.Label(
            self.root, 
            text="Ready",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def log(self, message, tag=""):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n", tag)
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def check_renderers(self):
        """Check which renderers are available"""
        self.log("Checking available renderers...")
        
        try:
            sys.path.insert(0, str(self.project_root))
            from renderers.unified import get_available_renderers
            
            available = get_available_renderers()
            status_text = ""
            
            for rt, name, is_available in available:
                status = "✓" if is_available else "✗"
                status_text += f"{status} {name}\n"
                if is_available:
                    self.log(f"  Found: {name}")
            
            self.renderer_status.config(state=tk.NORMAL)
            self.renderer_status.delete(1.0, tk.END)
            self.renderer_status.insert(tk.END, status_text)
            self.renderer_status.config(state=tk.DISABLED)
            
        except Exception as e:
            self.log(f"Error checking renderers: {e}", "error")
    
    def start_evolution(self):
        """Start fractal evolution"""
        if self.is_running:
            return
        
        fractal_type = self.fractal_type.get()
        generations = int(self.generations.get())
        population = int(self.population.get())
        mutation = float(self.mutation_rate.get())
        
        self.log(f"\n{'='*50}")
        self.log(f"Starting {fractal_type.upper()} Fractal Evolution")
        self.log(f"Generations: {generations}, Population: {population}")
        self.log(f"Mutation Rate: {mutation}")
        self.log(f"{'='*50}\n")
        
        self.is_running = True
        self.run_btn.config(state=tk.DISABLED, text="Running...")
        self.stop_btn.config(state=tk.NORMAL)
        self.status_bar.config(text="Evolution in progress...")
        
        # Run in separate thread
        thread = threading.Thread(
            target=self._run_evolution,
            args=(fractal_type, generations, population, mutation)
        )
        thread.daemon = True
        thread.start()
    
    def _run_evolution(self, fractal_type, generations, population, mutation):
        """Run evolution in background thread"""
        try:
            if fractal_type == "3d":
                script = "fractal_evolution_3d.py"
            else:
                script = "fractal_evolution_2d.py"
            
            output_dir = self.output_dir / f"gui_{fractal_type}_{datetime.now():%Y%m%d_%H%M%S}"
            
            cmd = [
                sys.executable,
                str(self.project_root / script),
                "--generations", str(generations),
                "--population", str(population),
                "--mutation-rate", str(mutation),
                "--output", str(output_dir)
            ]
            
            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            for line in self.current_process.stdout:
                self.root.after(0, lambda l=line: self.log(l.strip()))
            
            self.current_process.wait()
            
            if self.current_process.returncode == 0:
                self.root.after(0, lambda: self.evolution_complete(output_dir))
            else:
                self.root.after(0, lambda: self.evolution_failed())
                
        except Exception as e:
            self.root.after(0, lambda: self.log(f"Error: {e}", "error"))
        finally:
            self.root.after(0, self.reset_ui)
    
    def stop_evolution(self):
        """Stop running evolution"""
        if self.current_process and self.is_running:
            self.current_process.terminate()
            self.log("\n⚠️ Evolution stopped by user")
            self.reset_ui()
    
    def reset_ui(self):
        """Reset UI after evolution completes"""
        self.is_running = False
        self.run_btn.config(state=tk.NORMAL, text="▶ Start Evolution")
        self.stop_btn.config(state=tk.DISABLED)
        self.status_bar.config(text="Ready")
        self.current_process = None
    
    def evolution_complete(self, output_dir):
        """Handle evolution completion"""
        self.log(f"\n✅ Evolution complete!")
        self.log(f"Output: {output_dir}")
        
        self.status_bar.config(text=f"Complete! Results in {output_dir}")
        
        # Load preview images
        self.load_previews(output_dir)
        
        # Update gallery
        self.update_gallery(output_dir)
        
        messagebox.showinfo(
            "Evolution Complete",
            f"Fractal evolution completed!\n\nResults saved to:\n{output_dir}"
        )
    
    def evolution_failed(self):
        """Handle evolution failure"""
        self.log("\n❌ Evolution failed!")
        self.status_bar.config(text="Evolution failed")
        messagebox.showerror("Error", "Evolution process failed. Check log for details.")
    
    def load_previews(self, output_dir):
        """Load preview images"""
        self.preview_images.clear()
        
        for widget in self.preview_frame.winfo_children():
            widget.destroy()
        
        image_files = sorted(Path(output_dir).glob("*.png"))
        
        if not image_files:
            return
        
        for i, img_path in enumerate(image_files[:12]):  # Show first 12
            try:
                img = Image.open(img_path)
                img.thumbnail((200, 200))
                photo = ImageTk.PhotoImage(img)
                
                label = tk.Label(self.preview_frame, image=photo, bg="#1e1e1e")
                label.image = photo
                label.grid(row=i//3, column=i%3, padx=5, pady=5)
                
                # Add filename tooltip
                from tkinter import Toplevel
                
                tooltip_window = None
                
                def show_tooltip(event, path=img_path):
                    nonlocal tooltip_window
                    tooltip_window = Toplevel(self.root)
                    tooltip_window.wm_overrideredirect(True)
                    tooltip_window.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
                    ttk.Label(tooltip_window, text=path.name, padding=(5, 2)).pack()
                
                def hide_tooltip(event):
                    nonlocal tooltip_window
                    if tooltip_window:
                        tooltip_window.destroy()
                        tooltip_window = None
                    
                label.bind("<Enter>", show_tooltip)
                label.bind("<Leave>", hide_tooltip)
                
            except Exception as e:
                self.log(f"Error loading preview {img_path}: {e}")
    
    def update_gallery(self, output_dir):
        """Update gallery with results"""
        self.gallery_text.config(state=tk.NORMAL)
        self.gallery_text.delete(1.0, tk.END)
        
        image_files = sorted(Path(output_dir).glob("*.png"))
        
        self.gallery_text.insert(tk.END, f"Generated {len(image_files)} fractals\n")
        self.gallery_text.insert(tk.END, f"Location: {output_dir}\n\n")
        
        for img_path in image_files:
            self.gallery_text.insert(tk.END, f"• {img_path.name}\n")
        
        self.gallery_text.config(state=tk.DISABLED)
    
    def generate_single(self, fractal_type):
        """Generate a single fractal"""
        self.log(f"\nGenerating single {fractal_type.upper()} fractal...")
        
        thread = threading.Thread(target=self._generate_single, args=(fractal_type,))
        thread.daemon = True
        thread.start()
    
    def _generate_single(self, fractal_type):
        """Generate single fractal in background"""
        try:
            sys.path.insert(0, str(self.project_root))
            from renderers.unified import create_renderer, RendererType
            
            if fractal_type == "3d":
                renderer = create_renderer(RendererType.PYTHON_3D)
            else:
                from renderers.flam3_renderer import Flam3Renderer
                renderer = Flam3Renderer(str(self.output_dir / "single"))
            
            params = renderer.generate_random_parameters()
            output_path = self.output_dir / f"single_{fractal_type}_{datetime.now():%H%M%S}.png"
            
            success = renderer.render_fractal(params, str(output_path), 512, 512)
            
            if success:
                self.root.after(0, lambda: self.log(f"✅ Generated: {output_path.name}"))
                self.root.after(0, lambda: self.load_previews(self.output_dir))
            else:
                self.root.after(0, lambda: self.log("❌ Generation failed"))
                
        except Exception as e:
            self.root.after(0, lambda: self.log(f"Error: {e}"))
    
    def run_system_test(self):
        """Run system test"""
        self.log("\n" + "="*50)
        self.log("Running System Test...")
        self.log("="*50)
        
        thread = threading.Thread(target=self._run_system_test)
        thread.daemon = True
        thread.start()
    
    def _run_system_test(self):
        """Run system test in background"""
        try:
            cmd = [sys.executable, str(self.project_root / "test_system.py"), "--quick"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            for line in result.stdout.split('\n'):
                if line.strip():
                    self.root.after(0, lambda l=line: self.log(l))
            
            if result.returncode == 0:
                self.root.after(0, lambda: self.log("\n✅ System test passed!"))
            else:
                self.root.after(0, lambda: self.log("\n❌ System test failed!"))
                
        except Exception as e:
            self.root.after(0, lambda: self.log(f"Test error: {e}"))
    
    def verify_diversity(self):
        """Run diversity verification"""
        output_dirs = [d for d in self.output_dir.iterdir() if d.is_dir()]
        
        if not output_dirs:
            messagebox.showinfo("No Data", "No evolution results to verify yet.")
            return
        
        latest_dir = max(output_dirs, key=lambda d: d.stat().st_mtime)
        
        self.log(f"\nVerifying diversity in {latest_dir}...")
        
        thread = threading.Thread(target=self._verify_diversity, args=(latest_dir,))
        thread.daemon = True
        thread.start()
    
    def _verify_diversity(self, output_dir):
        """Verify diversity in background"""
        try:
            sys.path.insert(0, str(self.project_root))
            from verification.diversity_engine import DiversityEngine
            
            image_files = list(output_dir.glob("*.png"))
            
            if len(image_files) < 2:
                self.root.after(0, lambda: self.log("Not enough images to verify"))
                return
            
            engine = DiversityEngine()
            result = engine.verify_batch_diversity([str(f) for f in image_files])
            
            report = engine.generate_report(result)
            
            for line in report.split('\n'):
                self.root.after(0, lambda l=line: self.log(l))
            
        except Exception as e:
            self.root.after(0, lambda: self.log(f"Verification error: {e}"))
    
    def open_output_folder(self):
        """Open output folder in file manager"""
        try:
            subprocess.Popen(['xdg-open', str(self.output_dir)])
        except:
            webbrowser.open(str(self.output_dir))
    
    def clear_output(self):
        """Clear output directory"""
        if messagebox.askyesno("Confirm", "Delete all generated fractals?"):
            import shutil
            for item in self.output_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            self.log("Output directory cleared")
    
    def open_docs(self):
        """Open documentation"""
        readme = self.project_root / "README.md"
        if readme.exists():
            webbrowser.open(f"file://{readme}")
    
    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo(
            "About FractalGenesis",
            "FractalGenesis - Fractal Evolution Explorer\n\n"
            "Version: 2.0\n"
            "A tool for exploring and evolving fractal parameter spaces\n\n"
            "Features:\n"
            "• 3D Mandelbulb evolution\n"
            "• 2D Fractal Flames\n"
            "• Genetic algorithm optimization\n"
            "• Diversity verification\n\n"
            "Built with Python, Tkinter, and love for fractals ❤️"
        )

def main():
    root = tk.Tk()
    app = FractalGenesisApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
