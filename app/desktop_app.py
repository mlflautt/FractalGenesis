#!/usr/bin/env python3
"""
FractalGenesis Desktop Application
====================================

Model: minimax-m2.5 (opencode)
Created: 2026-02-16
Version: 1.0

Desktop application for:
- Generating fractal renders
- User-guided evolution
- Session management
- AI-assisted selection (future)
- Animation creation

Usage:
    python3 app/desktop_app.py
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import threading
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import json
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class FractalGenesisApp:
    """Main desktop application."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("FractalGenesis Desktop")
        self.root.geometry("1000x700")
        
        # State
        self.evolver = None
        self.session_manager = None
        self.current_population = []
        self.session = None
        
        # Setup
        self.setup_styles()
        self.create_ui()
        self.init_systems()
        
    def setup_styles(self):
        """Configure UI styling."""
        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')
        
        style.configure('Title.TLabel', font=('Helvetica', 14, 'bold'))
        style.configure('Section.TLabelframe', font=('Helvetica', 10, 'bold'))
        
    def init_systems(self):
        """Initialize backend systems."""
        try:
            from evolution import FractalEvolver, SessionManager
            self.evolver = FractalEvolver()
            self.session_manager = SessionManager()
            self.log("✓ Systems initialized")
        except Exception as e:
            self.log(f"✗ Init error: {e}")
    
    def create_ui(self):
        """Create main UI."""
        # Paned window for resizable sections
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Controls
        left_frame = ttk.Frame(paned, width=300)
        paned.add(left_frame, weight=0)
        self.create_control_panel(left_frame)
        
        # Right panel - Display
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=1)
        self.create_display_panel(right_frame)
        
        # Bottom - Status bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_control_panel(self, parent):
        """Create control panel."""
        # Title
        title = ttk.Label(parent, text="FractalGenesis", style='Title.TLabel')
        title.pack(pady=10)
        
        # Notebook for tabs
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 1: Generate
        generate_tab = ttk.Frame(notebook)
        notebook.add(generate_tab, text="Generate")
        self.create_generate_tab(generate_tab)
        
        # Tab 2: Evolution
        evolve_tab = ttk.Frame(notebook)
        notebook.add(evolve_tab, text="Evolution")
        self.create_evolution_tab(evolve_tab)
        
        # Tab 3: Sessions
        session_tab = ttk.Frame(notebook)
        notebook.add(session_tab, text="Sessions")
        self.create_session_tab(session_tab)
        
        # Tab 4: Animation
        anim_tab = ttk.Frame(notebook)
        notebook.add(anim_tab, text="Animation")
        self.create_animation_tab(anim_tab)
    
    def create_generate_tab(self, parent):
        """Generate tab."""
        frame = ttk.LabelFrame(parent, text="Quick Generate", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Fractal type
        ttk.Label(frame, text="Fractal Type:").pack(anchor=tk.W, pady=(0, 5))
        self.gen_type = ttk.Combobox(frame, values=[
            "mandelbulb", "julia", "mandelbox", "burning_ship", "kifs", "tricorn"
        ], state="readonly")
        self.gen_type.current(0)
        self.gen_type.pack(fill=tk.X, pady=(0, 10))
        
        # Power
        ttk.Label(frame, text="Power:").pack(anchor=tk.W, pady=(0, 5))
        self.gen_power = ttk.Scale(frame, from_=2, to=16, orient=tk.HORIZONTAL)
        self.gen_power.set(8)
        self.gen_power.pack(fill=tk.X, pady=(0, 10))
        
        # Generate button
        self.gen_btn = ttk.Button(frame, text="Generate", command=self.generate_single)
        self.gen_btn.pack(fill=tk.X, pady=10)
        
        # Output dir
        ttk.Label(frame, text="Output:").pack(anchor=tk.W, pady=(10, 5))
        self.output_dir = ttk.Entry(frame)
        self.output_dir.insert(0, "output/singles")
        self.output_dir.pack(fill=tk.X)
    
    def create_evolution_tab(self, parent):
        """Evolution tab."""
        frame = ttk.LabelFrame(parent, text="Evolution Controls", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Population size
        ttk.Label(frame, text="Population Size:").pack(anchor=tk.W)
        self.pop_size = ttk.Spinbox(frame, from_=4, to=16, width=10)
        self.pop_size.set(8)
        self.pop_size.pack(anchor=tk.W, pady=(0, 10))
        
        # Generate population
        ttk.Button(frame, text="1. Generate Population", 
                  command=self.generate_population).pack(fill=tk.X, pady=2)
        
        # Render all
        ttk.Button(frame, text="2. Render Population", 
                  command=self.render_population).pack(fill=tk.X, pady=2)
        
        # Record selection
        ttk.Label(frame, text="Selected IDs (comma):").pack(anchor=tk.W, pady=(10, 5))
        self.selected_ids = ttk.Entry(frame)
        self.selected_ids.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(frame, text="3. Record Selection", 
                  command=self.record_selection).pack(fill=tk.X, pady=2)
        
        # Evolve
        ttk.Button(frame, text="4. Evolve Next Generation", 
                  command=self.evolve_next).pack(fill=tk.X, pady=2)
        
        # Save session
        ttk.Button(frame, text="Save Session", 
                  command=self.save_session).pack(fill=tk.X, pady=(10, 2))
    
    def create_session_tab(self, parent):
        """Session management tab."""
        frame = ttk.LabelFrame(parent, text="Session Management", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # New session
        ttk.Label(frame, text="Session Name:").pack(anchor=tk.W)
        self.session_name = ttk.Entry(frame)
        self.session_name.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(frame, text="Create New Session", 
                  command=self.create_new_session).pack(fill=tk.X, pady=2)
        
        # List sessions
        ttk.Button(frame, text="Refresh Session List", 
                  command=self.refresh_sessions).pack(fill=tk.X, pady=(10, 2))
        
        # Session list
        self.session_list = tk.Listbox(frame, height=8)
        self.session_list.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Session actions
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="Load", command=self.load_session).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Reset", command=self.reset_session).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Delete", command=self.delete_session).pack(side=tk.LEFT, padx=2)
        
        # Stats
        ttk.Label(frame, text="Session Stats:").pack(anchor=tk.W, pady=(10, 5))
        self.session_stats = ttk.Label(frame, text="No session loaded", foreground='gray')
        self.session_stats.pack(anchor=tk.W)
    
    def create_animation_tab(self, parent):
        """Animation tab."""
        frame = ttk.LabelFrame(parent, text="Animation", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Animation type
        ttk.Label(frame, text="Animation Type:").pack(anchor=tk.W)
        self.anim_type = ttk.Combobox(frame, values=[
            "power_morph", "camera_orbit", "color_morph"
        ], state="readonly")
        self.anim_type.current(0)
        self.anim_type.pack(fill=tk.X, pady=(0, 10))
        
        # Frames
        ttk.Label(frame, text="Frames:").pack(anchor=tk.W)
        self.anim_frames = ttk.Spinbox(frame, from_=10, to=120, width=10)
        self.anim_frames.set(30)
        self.anim_frames.pack(anchor=tk.W, pady=(0, 10))
        
        ttk.Button(frame, text="Create Animation", 
                  command=self.create_animation).pack(fill=tk.X, pady=10)
        
        ttk.Button(frame, text="Export GIF", 
                  command=self.export_gif).pack(fill=tk.X, pady=2)
    
    def create_display_panel(self, parent):
        """Create display panel for logs and previews."""
        # Log area
        log_frame = ttk.LabelFrame(parent, text="Log", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Quick preview
        preview_frame = ttk.LabelFrame(parent, text="Last Render", padding=5)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.preview_label = ttk.Label(preview_frame, text="No preview")
        self.preview_label.pack()
    
    def log(self, message: str):
        """Add message to log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.status_bar.config(text=message)
    
    # === Generate Actions ===
    
    def generate_single(self):
        """Generate a single fractal."""
        try:
            from renderers.python_3d import FractalRenderer, FractalParams
            
            ftype = self.gen_type.get()
            power = self.gen_power.get()
            
            self.log(f"Generating {ftype} (power={power})...")
            
            renderer = FractalRenderer()
            params = FractalParams(
                fractal_type=ftype,
                power=float(power),
                width=500,
                height=500,
                color_palette="rainbow"
            )
            
            img, metrics = renderer.render(params)
            
            # Save
            output_dir = Path(self.output_dir.get())
            output_dir.mkdir(parents=True, exist_ok=True)
            
            import matplotlib.pyplot as plt
            filepath = output_dir / f"{ftype}_{int(power)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.imsave(filepath, img)
            
            self.log(f"✓ Saved: {filepath.name}")
            
        except Exception as e:
            self.log(f"✗ Error: {e}")
    
    # === Evolution Actions ===
    
    def generate_population(self):
        """Generate evolution population."""
        try:
            size = int(self.pop_size.get())
            self.current_population = self.evolver.generate_population(size)
            self.log(f"✓ Generated {size} genomes")
            
            # Show types
            types = [g.fractal_type for g in self.current_population]
            self.log(f"  Types: {', '.join(types)}")
            
        except Exception as e:
            self.log(f"✗ Error: {e}")
    
    def render_population(self):
        """Render current population."""
        if not self.current_population:
            self.log("No population - generate first")
            return
        
        try:
            self.log("Rendering population...")
            paths = self.evolver.render_population(
                self.current_population, 
                "output/evolution_ui"
            )
            
            valid = sum(1 for p in paths if p)
            self.log(f"✓ Rendered {valid}/{len(paths)} images")
            
        except Exception as e:
            self.log(f"✗ Error: {e}")
    
    def record_selection(self):
        """Record user selection."""
        if not self.current_population:
            self.log("No population")
            return
        
        ids = self.selected_ids.get().split(',')
        ids = [i.strip() for i in ids if i.strip()]
        
        if not ids:
            self.log("Enter genome IDs to select")
            return
        
        try:
            self.evolver.record_selections(ids)
            self.log(f"✓ Recorded {len(ids)} selections")
            
            # Create session if not exists
            if not self.session:
                self.create_new_session()
            
            # Add to session
            for gid in ids:
                for g in self.current_population:
                    if g.genome_id == gid:
                        if self.session:
                            self.session.add_selection(
                                g.to_dict(), 
                                selection_type="still",
                                rank=1
                            )
                        break
            
        except Exception as e:
            self.log(f"✗ Error: {e}")
    
    def evolve_next(self):
        """Evolve to next generation."""
        try:
            self.current_population = self.evolver.evolve_next_generation()
            self.log(f"✓ Evolved to generation {self.evolver.generation}")
            self.log(f"  New population: {len(self.current_population)}")
            
        except Exception as e:
            self.log(f"✗ Error: {e}")
    
    def save_session(self):
        """Save current session."""
        if self.session:
            self.session_manager.save_session(self.session)
            self.log(f"✓ Saved session: {self.session.name}")
    
    # === Session Actions ===
    
    def create_new_session(self):
        """Create new session."""
        name = self.session_name.get() or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if self.evolver:
            self.session = self.session_manager.create_session(name)
            self.log(f"✓ Created session: {name}")
            self.session_name.delete(0, tk.END)
    
    def refresh_sessions(self):
        """Refresh session list."""
        sessions = self.session_manager.list_sessions()
        self.session_list.delete(0, tk.END)
        
        for s in sessions:
            text = f"{s['name']} ({s['selections']} selections)"
            self.session_list.insert(tk.END, text)
    
    def load_session(self):
        """Load selected session."""
        idx = self.session_list.curselection()
        if not idx:
            return
        
        text = self.session_list.get(idx[0])
        name = text.split('(')[0].strip()
        
        # Find session
        for s in self.session_manager.list_sessions():
            if s['name'] == name:
                self.session_manager.load_session(s['session_id'])
                self.session = self.session_manager.current_session
                self.log(f"✓ Loaded: {name}")
                self.session_stats.config(text=f"{s['selections']} selections, Gen {s['generation']}")
                break
    
    def reset_session(self):
        """Reset current session."""
        if self.session:
            self.session_manager.reset_session(self.session.session_id)
            self.log(f"✓ Reset: {self.session.name}")
            self.session_stats.config(text="Session reset")
    
    def delete_session(self):
        """Delete selected session."""
        idx = self.session_list.curselection()
        if not idx:
            return
        
        if messagebox.askyesno("Delete", "Delete this session?"):
            text = self.session_list.get(idx[0])
            name = text.split('(')[0].strip()
            
            for s in self.session_manager.list_sessions():
                if s['name'] == name:
                    self.session_manager.delete_session(s['session_id'])
                    self.log(f"✓ Deleted: {name}")
                    self.refresh_sessions()
                    break
    
    # === Animation Actions ===
    
    def create_animation(self):
        """Create animation."""
        try:
            from animation import FractalAnimator
            
            anim_type = self.anim_type.get()
            frames = int(self.anim_frames.get())
            
            self.log(f"Creating {anim_type} animation ({frames} frames)...")
            
            animator = FractalAnimator()
            
            if anim_type == "power_morph":
                animator.animate_power_morph(
                    power_start=4.0,
                    power_end=14.0,
                    num_frames=frames,
                    output_dir="output/animation_ui"
                )
            
            self.log("✓ Animation created")
            
        except Exception as e:
            self.log(f"✗ Error: {e}")
    
    def export_gif(self):
        """Export as GIF."""
        self.log("Exporting GIF...")


def main():
    root = tk.Tk()
    app = FractalGenesisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()