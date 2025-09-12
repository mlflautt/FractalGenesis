#!/usr/bin/env python3
"""
Visual Evolution GUI - Fixed Version

An intuitive interface showing step-by-step fractal evolution with:
- Actual fractal rendering integration
- Proper generation management
- Real candidate selection
- Evolution progression
- Data collection and storage
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import threading
import time
import json
import random
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from renderers.mandelbulber_renderer import MandelbulberRenderer
from renderers.flam3_renderer import Flam3Renderer


class GenerationProgressWidget(tk.Frame):
    """Widget to show generation progress with visual timeline."""
    
    def __init__(self, parent, max_generations=5):
        super().__init__(parent, bg='#2c3e50', relief='raised', bd=2)
        self.max_generations = max_generations
        self.current_generation = 0
        
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the progress UI."""
        title = tk.Label(self, text="Evolution Progress", 
                        font=("Arial", 12, "bold"), fg='white', bg='#2c3e50')
        title.pack(pady=5)
        
        # Progress bar frame
        progress_frame = tk.Frame(self, bg='#2c3e50')
        progress_frame.pack(fill='x', padx=10, pady=5)
        
        # Generation circles
        self.gen_circles = []
        for i in range(self.max_generations):
            circle_frame = tk.Frame(progress_frame, bg='#2c3e50')
            circle_frame.pack(side='left', fill='x', expand=True)
            
            # Circle (represented as label with special styling)
            circle = tk.Label(circle_frame, text=str(i+1), 
                            width=3, height=2, 
                            bg='#34495e', fg='white',
                            relief='raised', bd=1,
                            font=("Arial", 10, "bold"))
            circle.pack(pady=2)
            
            # Status label
            status = tk.Label(circle_frame, text="Pending", 
                            fg='#bdc3c7', bg='#2c3e50', font=("Arial", 8))
            status.pack()
            
            self.gen_circles.append((circle, status))
            
            # Connection line (except for last)
            if i < self.max_generations - 1:
                line = tk.Label(progress_frame, text="─", 
                              fg='#34495e', bg='#2c3e50')
                line.pack(side='left')
    
    def update_generation(self, generation, status="Active"):
        """Update the visual progress."""
        if generation < len(self.gen_circles):
            circle, status_label = self.gen_circles[generation]
            
            if status == "Active":
                circle.config(bg='#f39c12', relief='raised')  # Orange for active
                status_label.config(text="Rendering", fg='#f39c12')
            elif status == "Complete":
                circle.config(bg='#27ae60', relief='raised')  # Green for complete
                status_label.config(text="Done", fg='#27ae60')
            elif status == "Selection":
                circle.config(bg='#e74c3c', relief='raised')  # Red for selection
                status_label.config(text="Select", fg='#e74c3c')
        
        self.current_generation = generation


class CandidateSelectionWidget(tk.Frame):
    """Widget for visual candidate selection with thumbnails."""
    
    def __init__(self, parent, selection_callback=None):
        super().__init__(parent, bg='#34495e', relief='raised', bd=2)
        self.selection_callback = selection_callback
        self.candidates = []
        self.selected_index = -1
        
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the selection UI."""
        title = tk.Label(self, text="Select Your Favorite Fractal", 
                        font=("Arial", 14, "bold"), fg='white', bg='#34495e')
        title.pack(pady=10)
        
        # Candidates frame
        self.candidates_frame = tk.Frame(self, bg='#34495e')
        self.candidates_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Selection info
        self.selection_info = tk.Label(self, text="Generating fractals...", 
                                     fg='#bdc3c7', bg='#34495e', font=("Arial", 10))
        self.selection_info.pack(pady=5)
        
        # Buttons frame
        button_frame = tk.Frame(self, bg='#34495e')
        button_frame.pack(fill='x', padx=10, pady=10)
        
        self.confirm_btn = tk.Button(button_frame, text="Confirm Selection", 
                                   command=self.confirm_selection,
                                   state='disabled',
                                   bg='#27ae60', fg='white',
                                   font=("Arial", 12, "bold"))
        self.confirm_btn.pack(side='right', padx=5)
        
        self.skip_btn = tk.Button(button_frame, text="Skip Generation", 
                                command=self.skip_selection,
                                bg='#e67e22', fg='white',
                                font=("Arial", 10))
        self.skip_btn.pack(side='right', padx=5)
    
    def load_candidates(self, image_paths: List[str]):
        """Load candidate images for selection."""
        # Clear existing candidates
        for widget in self.candidates_frame.winfo_children():
            widget.destroy()
        
        self.candidates = []
        self.selected_index = -1
        self.confirm_btn.config(state='disabled')
        
        # Create candidate buttons
        valid_images = [path for path in image_paths if Path(path).exists()]
        
        if not valid_images:
            # Show message if no images
            no_images_label = tk.Label(self.candidates_frame, 
                                     text="No fractal images were generated.\nCheck the console for rendering errors.", 
                                     fg='#e74c3c', bg='#34495e', font=("Arial", 12))
            no_images_label.pack(expand=True)
            self.selection_info.config(text="Generation failed - no candidates to select")
            return
        
        for i, img_path in enumerate(valid_images):
            candidate_frame = tk.Frame(self.candidates_frame, bg='#34495e', 
                                     relief='raised', bd=2)
            # Use 4 columns for 8 candidates (2x4 grid)
            candidate_frame.grid(row=i//4, column=i%4, padx=5, pady=5, 
                               sticky='nsew')
            
            # Load and resize image
            try:
                img = Image.open(img_path)
                img = img.resize((250, 250), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                # Create clickable image button
                img_btn = tk.Button(candidate_frame, image=photo, 
                                  command=lambda idx=i: self.select_candidate(idx),
                                  relief='raised', bd=3)
                img_btn.image = photo  # Keep reference
                img_btn.pack(padx=5, pady=5)
                
                # Index label
                label = tk.Label(candidate_frame, text=f"Option {i+1}", 
                               fg='white', bg='#34495e', font=("Arial", 10, "bold"))
                label.pack(pady=2)
                
                self.candidates.append((candidate_frame, img_btn, label))
                
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        
        # Configure grid weights for 4 columns
        for col in range(4):
            self.candidates_frame.grid_columnconfigure(col, weight=1)
        
        self.selection_info.config(text=f"Generation complete! Choose from {len(self.candidates)} candidates:")
    
    def select_candidate(self, index):
        """Handle candidate selection."""
        # Reset all candidates to normal appearance
        for i, (frame, btn, label) in enumerate(self.candidates):
            if i == index:
                # Highlight selected
                frame.config(bg='#e74c3c', relief='sunken')
                btn.config(relief='sunken', bg='#c0392b')
                label.config(bg='#e74c3c', text=f"✓ SELECTED {i+1}")
            else:
                # Normal appearance
                frame.config(bg='#34495e', relief='raised')
                btn.config(relief='raised', bg='#f0f0f0')
                label.config(bg='#34495e', text=f"Option {i+1}")
        
        self.selected_index = index
        self.confirm_btn.config(state='normal')
        self.selection_info.config(text=f"Selected Option {index+1}. Confirm to continue evolution.")
    
    def confirm_selection(self):
        """Confirm the selection and proceed."""
        if self.selected_index >= 0 and self.selection_callback:
            self.selection_callback(self.selected_index)
    
    def skip_selection(self):
        """Skip this generation."""
        if self.selection_callback:
            self.selection_callback(-1)  # -1 indicates skip


class EvolutionEngine:
    """Manages the fractal evolution process with proper generation tracking."""
    
    def __init__(self, fractal_type="mandelbulber", output_dir=None):
        self.fractal_type = fractal_type
        # Use provided output directory or create a default one
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            base_output = Path("output/visual_evolution")
            base_output.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.output_dir = base_output / f"{fractal_type}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Evolution state
        self.current_generation = 0
        self.max_generations = 5
        self.population_size = 8  # Larger population for better diversity
        self.generation_history = []
        self.current_population = []
        
        # Initialize renderer
        if fractal_type == "mandelbulber":
            self.renderer = MandelbulberRenderer(str(self.output_dir))
        else:
            self.renderer = Flam3Renderer(str(self.output_dir))
        
        self.initialize_population()
    
    def initialize_population(self):
        """Create initial random population."""
        print(f"Initializing {self.fractal_type} population...")
        
        if self.fractal_type == "mandelbulber":
            # Generate random Mandelbulb parameters
            for i in range(self.population_size):
                params = self.renderer.generate_random_parameters()
                self.current_population.append({
                    'id': f"gen0_ind{i}",
                    'parameters': params,
                    'fitness': 0.0,
                    'generation': 0
                })
        else:
            # Generate random Flam3 parameters (placeholder - would need Flam3Renderer)
            for i in range(self.population_size):
                params = {"placeholder": True, "id": i}
                self.current_population.append({
                    'id': f"gen0_ind{i}",
                    'parameters': params,
                    'fitness': 0.0,
                    'generation': 0
                })
        
        print(f"Initialized population of {len(self.current_population)} individuals")
    
    def render_generation(self, generation):
        """Render all individuals in current generation."""
        candidate_paths = []
        
        print(f"Rendering generation {generation} ({len(self.current_population)} individuals)...")
        
        for i, individual in enumerate(self.current_population):
            if self.fractal_type == "mandelbulber":
                # Render fractal directly with parameters
                output_file = self.output_dir / f"gen{generation}_candidate{i}.png"
                
                print(f"Rendering candidate {i+1}/{len(self.current_population)}...")
                success = self.renderer.render_fractal(
                    individual['parameters'], str(output_file), 400, 400)
                
                if success and output_file.exists():
                    candidate_paths.append(str(output_file))
                    print(f"✓ Rendered: {output_file}")
                else:
                    print(f"✗ Failed to render candidate {i}")
            else:
                # Placeholder for Flam3 rendering
                placeholder_file = self.output_dir / f"gen{generation}_candidate{i}.png"
                candidate_paths.append(str(placeholder_file))
        
        print(f"Generation {generation} rendering complete: {len(candidate_paths)} images")
        return candidate_paths
    
    def evolve_population(self, selected_index):
        """Evolve population based on selection."""
        if selected_index < 0 or selected_index >= len(self.current_population):
            print("Invalid selection, generating random population")
            # Generate new random population
            self.initialize_population()
            return
        
        # Get selected individual
        selected = self.current_population[selected_index]
        selected['fitness'] = 1.0  # Mark as selected
        
        # Record selection data
        self.record_selection_data(selected_index)
        
        print(f"Selected individual {selected_index} for evolution")
        
        # Create new population with enhanced diversity
        new_population = []
        
        # Keep the selected individual (elitism)
        elite = selected.copy()
        elite['id'] = f"gen{self.current_generation + 1}_elite"
        new_population.append(elite)
        
        # Generate diverse offspring using multiple strategies
        for i in range(1, self.population_size):
            if self.fractal_type == "mandelbulber":
                if i <= self.population_size // 2:
                    # First half: mutations of selected individual
                    mutated_params = self.renderer.mutate_parameters(selected['parameters'], 0.4)
                elif i <= self.population_size * 3 // 4:
                    # Third quarter: crossover with a random previous individual
                    if len(self.current_population) > 1:
                        other = random.choice([ind for ind in self.current_population if ind != selected])
                        crossover_params = self.renderer.crossover_parameters(selected['parameters'], other['parameters'])
                        # Also mutate the crossover result
                        mutated_params = self.renderer.mutate_parameters(crossover_params, 0.2)
                    else:
                        mutated_params = self.renderer.mutate_parameters(selected['parameters'], 0.4)
                else:
                    # Last quarter: completely new random individuals for diversity
                    mutated_params = self.renderer.generate_random_parameters()
                
                new_individual = {
                    'id': f"gen{self.current_generation + 1}_ind{i}",
                    'parameters': mutated_params,
                    'fitness': 0.0,
                    'generation': self.current_generation + 1
                }
                new_population.append(new_individual)
            else:
                # Placeholder for other types
                new_individual = selected.copy()
                new_individual['id'] = f"gen{self.current_generation + 1}_ind{i}"
                new_population.append(new_individual)
        
        # Update population
        self.current_population = new_population
        self.current_generation += 1
        
        print(f"Evolved to generation {self.current_generation}")
    
    def record_selection_data(self, selected_index):
        """Record selection for AI training."""
        try:
            data_dir = Path("data/user_selections")
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract parameters for storage
            fractal_parameters = []
            for individual in self.current_population:
                fractal_parameters.append(individual['parameters'])
            
            # Create selection record
            selection_data = {
                "timestamp": time.time(),
                "generation": self.current_generation,
                "selected_index": selected_index,
                "fractal_type": self.fractal_type,
                "fractal_parameters": fractal_parameters,
                "selection_time": time.time(),  # Placeholder
                "session_id": f"visual_evolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
            
            # Save to file
            filename = f"selection_visual_gen{self.current_generation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            file_path = data_dir / filename
            
            with open(file_path, 'w') as f:
                json.dump(selection_data, f, indent=2)
            
            print(f"Recorded selection data: {file_path}")
            
        except Exception as e:
            print(f"Error recording selection data: {e}")


class VisualEvolutionGUI:
    """Main visual evolution interface with proper integration."""
    
    def __init__(self, fractal_type="mandelbulber"):
        self.fractal_type = fractal_type
        
        # Create user-friendly output directory structure
        base_output = Path("output/visual_evolution")
        base_output.mkdir(parents=True, exist_ok=True)
        
        # Create session directory with readable timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        session_name = f"{fractal_type}_{timestamp}"
        self.output_dir = base_output / session_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_generations = 5
        
        # Evolution state
        self.is_running = False
        self.selection_complete = False
        self.selected_index = -1
        self.evolution_engine = None
        
        self.create_gui()
    
    def create_gui(self):
        """Create the main GUI."""
        self.root = tk.Tk()
        self.root.title(f"FractalGenesis - Visual Evolution ({self.fractal_type.title()})")
        self.root.geometry("1200x900")
        self.root.configure(bg='#2c3e50')
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Top section - Progress and Settings
        top_frame = tk.Frame(main_frame, bg='#2c3e50')
        top_frame.pack(fill='x', pady=(0, 10))
        
        # Progress widget
        self.progress_widget = GenerationProgressWidget(top_frame, self.max_generations)
        self.progress_widget.pack(fill='x')
        
        # Middle section - Selection interface
        self.selection_widget = CandidateSelectionWidget(main_frame, self.on_selection_made)
        self.selection_widget.pack(fill='both', expand=True, pady=(0, 10))
        
        # Bottom section - Control buttons
        control_frame = tk.Frame(main_frame, bg='#2c3e50')
        control_frame.pack(fill='x')
        
        self.start_btn = tk.Button(control_frame, text="Start Visual Evolution", 
                                 command=self.start_evolution,
                                 bg='#27ae60', fg='white',
                                 font=("Arial", 14, "bold"))
        self.start_btn.pack(side='left', padx=5)
        
        self.stop_btn = tk.Button(control_frame, text="Stop", 
                                command=self.stop_evolution,
                                state='disabled',
                                bg='#e74c3c', fg='white',
                                font=("Arial", 12))
        self.stop_btn.pack(side='left', padx=5)
        
        self.results_btn = tk.Button(control_frame, text="View Results", 
                                   command=self.view_results,
                                   bg='#3498db', fg='white',
                                   font=("Arial", 12))
        self.results_btn.pack(side='right', padx=5)
        
        # Status bar
        self.status_var = tk.StringVar(value=f"Ready to start {self.fractal_type.title()} evolution")
        status_bar = tk.Label(main_frame, textvariable=self.status_var, 
                            relief='sunken', anchor='w',
                            bg='#34495e', fg='white', font=("Arial", 10))
        status_bar.pack(fill='x', pady=(10, 0))
        
        # Info label
        info_text = f"Output: {self.output_dir}"
        info_label = tk.Label(main_frame, text=info_text, 
                            fg='#bdc3c7', bg='#2c3e50', font=("Arial", 9))
        info_label.pack(fill='x')
        
        # Auto-start first generation rendering
        self.root.after(500, self.auto_start_first_generation)
    
    def auto_start_first_generation(self):
        """Automatically start rendering the first generation."""
        if not self.is_running:
            self.status_var.set("Rendering initial fractal population...")
            
            # Initialize evolution engine
            self.evolution_engine = EvolutionEngine(self.fractal_type, self.output_dir)
            
            # Start first generation rendering in background
            first_gen_thread = threading.Thread(target=self.render_first_generation)
            first_gen_thread.daemon = True
            first_gen_thread.start()
    
    def render_first_generation(self):
        """Render the first generation in background."""
        try:
            self.root.after(0, lambda: self.progress_widget.update_generation(0, "Active"))
            
            # Render first generation
            candidate_paths = self.evolution_engine.render_generation(0)
            
            if candidate_paths:
                # Load candidates for selection
                self.root.after(0, lambda: self.selection_widget.load_candidates(candidate_paths))
                self.root.after(0, lambda: self.progress_widget.update_generation(0, "Selection"))
                self.root.after(0, lambda: self.status_var.set("Select your favorite fractal to continue evolution..."))
            else:
                self.root.after(0, lambda: self.status_var.set("First generation rendering failed"))
                
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Error rendering first generation: {e}"))
            print(f"First generation error: {e}")
    
    def on_selection_made(self, selected_index):
        """Handle selection callback from selection widget."""
        self.selected_index = selected_index
        self.selection_complete = True
        
        if selected_index >= 0:
            self.status_var.set(f"Selection confirmed: Option {selected_index + 1}")
        else:
            self.status_var.set("Generation skipped")
    
    def start_evolution(self):
        """Start the visual evolution process."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Update UI
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        
        if not self.evolution_engine:
            self.status_var.set("Initializing evolution engine...")
            # Initialize evolution engine if not already done
            self.evolution_engine = EvolutionEngine(self.fractal_type, self.output_dir)
        else:
            self.status_var.set("Continuing evolution from generation 1...")
        
        # Start evolution in separate thread
        evolution_thread = threading.Thread(target=self.run_evolution)
        evolution_thread.daemon = True
        evolution_thread.start()
    
    def stop_evolution(self):
        """Stop the evolution process."""
        self.is_running = False
        self.status_var.set("Evolution stopped")
        
        # Update UI
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
    
    def run_evolution(self):
        """Main evolution loop (runs in separate thread)."""
        try:
            # Start from generation 1 if first generation is already rendered
            start_gen = 1 if self.evolution_engine and self.evolution_engine.current_generation == 0 else 0
            
            for generation in range(start_gen, self.max_generations):
                if not self.is_running:
                    break
                
                # Update progress
                self.root.after(0, lambda g=generation: 
                              self.progress_widget.update_generation(g, "Active"))
                
                self.root.after(0, lambda g=generation: 
                              self.status_var.set(f"Rendering generation {g + 1}..."))
                
                # Render fractal candidates
                candidate_paths = self.evolution_engine.render_generation(generation)
                
                if not candidate_paths:
                    self.root.after(0, lambda: 
                                  self.status_var.set("Rendering failed - stopping evolution"))
                    break
                
                # Update progress to selection phase
                self.root.after(0, lambda g=generation: 
                              self.progress_widget.update_generation(g, "Selection"))
                
                # Load candidates for selection
                self.root.after(0, lambda paths=candidate_paths: 
                              self.selection_widget.load_candidates(paths))
                
                # Wait for user selection
                self.selection_complete = False
                self.selected_index = -1
                
                self.root.after(0, lambda: 
                              self.status_var.set("Select your favorite fractal to continue..."))
                
                while not self.selection_complete and self.is_running:
                    time.sleep(0.1)
                
                if not self.is_running:
                    break
                
                # Process selection and evolve
                if self.selected_index >= 0:
                    self.evolution_engine.evolve_population(self.selected_index)
                    self.root.after(0, lambda: 
                                  self.status_var.set(f"Evolved population based on selection"))
                else:
                    self.root.after(0, lambda: 
                                  self.status_var.set(f"Generation {generation + 1} skipped"))
                
                # Mark generation complete
                self.root.after(0, lambda g=generation: 
                              self.progress_widget.update_generation(g, "Complete"))
                
                time.sleep(1)  # Brief pause between generations
            
            # Evolution complete
            self.root.after(0, self.evolution_complete)
            
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Evolution error: {e}"))
            print(f"Evolution error: {e}")
            self.root.after(0, self.stop_evolution)
    
    def evolution_complete(self):
        """Handle evolution completion."""
        self.status_var.set("Evolution complete! View results to see your evolved fractals.")
        self.stop_evolution()
        
        # Show completion message
        messagebox.showinfo("Evolution Complete", 
                          f"Fractal evolution finished!\n\n"
                          f"Results saved to: {self.output_dir}\n\n"
                          f"Click 'View Results' to see your fractals.")
    
    def view_results(self):
        """Open results directory."""
        import subprocess
        import sys
        
        try:
            if sys.platform == "linux":
                subprocess.run(["xdg-open", str(self.output_dir)])
            elif sys.platform == "darwin":
                subprocess.run(["open", str(self.output_dir)])
            elif sys.platform == "win32":
                subprocess.run(["explorer", str(self.output_dir)])
        except Exception as e:
            messagebox.showinfo("Results Location", f"Results saved to:\n{self.output_dir}")
    
    def run(self):
        """Start the GUI."""
        self.root.mainloop()


def main():
    """Demo the visual evolution GUI."""
    import sys
    
    fractal_type = "mandelbulber" if len(sys.argv) > 1 and sys.argv[1] == "3d" else "mandelbulber"
    
    print(f"Starting Visual Evolution GUI ({fractal_type})...")
    print("This GUI will render actual fractals and let you select your favorites.")
    
    gui = VisualEvolutionGUI(fractal_type)
    gui.run()


if __name__ == "__main__":
    main()
