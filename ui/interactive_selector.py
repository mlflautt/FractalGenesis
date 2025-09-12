#!/usr/bin/env python3
"""
Interactive Fractal Selector

A pop-up window that displays fractal candidates and allows users to
click on their favorite, collecting selection data for neural network training.
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import json
import time
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import logging


class InteractiveFractalSelector:
    """
    Interactive window for fractal selection with user preference tracking.
    
    Shows 4 fractal candidates in a grid, allows click selection,
    and records all interaction data for neural network training.
    """
    
    def __init__(self, 
                 image_paths: List[str],
                 generation: int,
                 fractal_data: Optional[List[Dict[str, Any]]] = None,
                 session_id: Optional[str] = None):
        """
        Initialize the selector window.
        
        Args:
            image_paths: List of paths to fractal images to display
            generation: Current generation number
            fractal_data: Optional list of fractal parameter data for each image
            session_id: Optional session identifier for tracking
        """
        self.image_paths = image_paths[:4]  # Max 4 images
        self.generation = generation
        self.fractal_data = fractal_data or [{} for _ in self.image_paths]
        self.session_id = session_id or f"session_{int(time.time())}"
        
        self.selected_index = None
        self.selection_time = None
        self.window_open_time = time.time()
        self.hover_times = {i: 0.0 for i in range(len(self.image_paths))}
        self.hover_start_times = {i: None for i in range(len(self.image_paths))}
        
        self.root = None
        self.image_labels = []
        self.image_frames = []
        self.preview_images = []
        
        self.logger = logging.getLogger(__name__)
        
        # Ensure data collection directory exists
        self.data_dir = Path("data/user_selections")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def show_selection_window(self) -> int:
        """
        Display the selection window and wait for user choice.
        
        Returns:
            Index of selected fractal (0-3) or -1 if cancelled
        """
        self.root = tk.Toplevel() if hasattr(tk, '_default_root') and tk._default_root else tk.Tk()
        self.root.title(f"Choose Your Favorite Fractal - Generation {self.generation}")
        self.root.geometry("800x700")
        self.root.resizable(True, True)
        
        # Center the window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() - self.root.winfo_width()) // 2
        y = (self.root.winfo_screenheight() - self.root.winfo_height()) // 2
        self.root.geometry(f"+{x}+{y}")
        
        # Make window modal
        self.root.transient()
        self.root.grab_set()
        self.root.focus_set()
        
        self.setup_ui()
        self.load_images()
        
        # Wait for selection
        self.root.wait_window()
        
        # Record final interaction data
        self.record_selection_data()
        
        return self.selected_index if self.selected_index is not None else -1
    
    def setup_ui(self):
        """Create the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title and instructions
        title_label = ttk.Label(main_frame, 
                               text=f"Generation {self.generation}: Choose Your Favorite Fractal",
                               font=('TkDefaultFont', 14, 'bold'))
        title_label.pack(pady=(0, 10))
        
        instruction_label = ttk.Label(main_frame, 
                                    text="Click on the fractal you like best. Hover over images to see them larger.",
                                    font=('TkDefaultFont', 10))
        instruction_label.pack(pady=(0, 20))
        
        # Progress indicator
        progress_text = f"Generation {self.generation} • {len(self.image_paths)} candidates"
        progress_label = ttk.Label(main_frame, text=progress_text, foreground="gray")
        progress_label.pack(pady=(0, 15))
        
        # Image grid container
        grid_frame = ttk.Frame(main_frame)
        grid_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Configure grid weights for responsive layout
        grid_frame.columnconfigure(0, weight=1)
        grid_frame.columnconfigure(1, weight=1)
        grid_frame.rowconfigure(0, weight=1)
        grid_frame.rowconfigure(1, weight=1)
        
        # Create image frames in 2x2 grid
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        
        for i, (row, col) in enumerate(positions[:len(self.image_paths)]):
            # Frame for each fractal candidate
            frame = ttk.LabelFrame(grid_frame, text=f"Option {i+1}", padding="10")
            frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            
            # Image label
            image_label = ttk.Label(frame, text="Loading...", 
                                   background="white", relief="sunken", borderwidth=2)
            image_label.pack(fill=tk.BOTH, expand=True)
            
            # Info label for image details
            info_label = ttk.Label(frame, text="", font=('TkDefaultFont', 8), foreground="gray")
            info_label.pack(pady=(5, 0))
            
            # Bind click events
            image_label.bind("<Button-1>", lambda e, idx=i: self.select_fractal(idx))
            image_label.bind("<Enter>", lambda e, idx=i: self.on_hover_enter(idx))
            image_label.bind("<Leave>", lambda e, idx=i: self.on_hover_leave(idx))
            
            # Make frame also clickable
            frame.bind("<Button-1>", lambda e, idx=i: self.select_fractal(idx))
            
            # Store references
            self.image_labels.append(image_label)
            self.image_frames.append(frame)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Skip This Generation", 
                  command=self.skip_selection).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Help", 
                  command=self.show_help).pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_label = ttk.Label(main_frame, text="Select your favorite fractal by clicking on it",
                                     foreground="blue")
        self.status_label.pack(pady=(10, 0))
        
        # Keyboard shortcuts
        self.root.bind('1', lambda e: self.select_fractal(0))
        self.root.bind('2', lambda e: self.select_fractal(1))
        self.root.bind('3', lambda e: self.select_fractal(2))
        self.root.bind('4', lambda e: self.select_fractal(3))
        self.root.bind('<Escape>', lambda e: self.skip_selection())
        
        self.root.protocol("WM_DELETE_WINDOW", self.skip_selection)
        
    def load_images(self):
        """Load and display the fractal images."""
        target_size = (300, 300)  # Display size
        
        for i, image_path in enumerate(self.image_paths):
            try:
                # Load and resize image
                if os.path.exists(image_path):
                    pil_image = Image.open(image_path)
                    pil_image.thumbnail(target_size, Image.Resampling.LANCZOS)
                    
                    # Convert to PhotoImage for tkinter
                    tk_image = ImageTk.PhotoImage(pil_image)
                    self.preview_images.append(tk_image)
                    
                    # Update label
                    self.image_labels[i].configure(image=tk_image, text="")
                    
                    # Update info
                    file_info = f"{os.path.basename(image_path)}\n{pil_image.size[0]}x{pil_image.size[1]}"
                    info_label = self.image_frames[i].winfo_children()[-1]  # Last child is info label
                    info_label.configure(text=file_info)
                    
                else:
                    # Placeholder for missing image
                    self.image_labels[i].configure(text=f"Image {i+1}\nNot Available", 
                                                 background="lightgray")
                    
            except Exception as e:
                self.logger.error(f"Failed to load image {image_path}: {e}")
                self.image_labels[i].configure(text=f"Image {i+1}\nLoad Error", 
                                             background="lightcoral")
    
    def select_fractal(self, index: int):
        """Handle fractal selection."""
        if index < 0 or index >= len(self.image_paths):
            return
        
        self.selected_index = index
        self.selection_time = time.time()
        
        # Visual feedback
        for i, frame in enumerate(self.image_frames):
            if i == index:
                frame.configure(style="Selected.TLabelframe")
                self.image_labels[i].configure(relief="solid", borderwidth=3)
            else:
                frame.configure(style="TLabelframe")
                self.image_labels[i].configure(relief="sunken", borderwidth=2)
        
        self.status_label.configure(text=f"Selected Option {index + 1}! Closing in 2 seconds...",
                                   foreground="green")
        
        # Auto-close after brief delay
        self.root.after(2000, self.close_window)
    
    def skip_selection(self):
        """Skip this generation (no selection)."""
        self.selected_index = -1
        self.selection_time = time.time()
        self.close_window()
    
    def close_window(self):
        """Close the selection window."""
        if self.root:
            self.root.quit()
            self.root.destroy()
    
    def on_hover_enter(self, index: int):
        """Handle mouse hover enter."""
        self.hover_start_times[index] = time.time()
        
        # Visual feedback on hover
        if self.selected_index != index:
            self.image_labels[index].configure(relief="raised", borderwidth=2)
    
    def on_hover_leave(self, index: int):
        """Handle mouse hover leave."""
        if self.hover_start_times[index] is not None:
            hover_duration = time.time() - self.hover_start_times[index]
            self.hover_times[index] += hover_duration
            self.hover_start_times[index] = None
        
        # Restore visual state
        if self.selected_index != index:
            self.image_labels[index].configure(relief="sunken", borderwidth=2)
    
    def show_help(self):
        """Show help dialog."""
        help_text = """Fractal Selection Help

• Click on your favorite fractal to select it
• Use keyboard shortcuts: 1, 2, 3, 4 for quick selection
• Hover over images to see them highlighted
• Press Escape or click "Skip" to skip this generation
• The algorithm will use your choice to create better fractals

Your selections are recorded to train an AI that will eventually
learn your preferences and suggest fractals you'll like!"""
        
        from tkinter import messagebox
        messagebox.showinfo("Help", help_text)
    
    def record_selection_data(self):
        """Record selection data for neural network training."""
        if self.selected_index == -1:
            return  # Skip recording if no selection made
        
        selection_data = {
            'timestamp': time.time(),
            'session_id': self.session_id,
            'generation': self.generation,
            'selected_index': self.selected_index,
            'total_candidates': len(self.image_paths),
            'selection_time_seconds': self.selection_time - self.window_open_time if self.selection_time else 0,
            'hover_times': self.hover_times,
            'image_paths': self.image_paths,
            'fractal_parameters': self.fractal_data,
        }
        
        # Save to JSON file
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        filename = f"selection_{self.session_id}_gen{self.generation:02d}_{timestamp_str}.json"
        filepath = self.data_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(selection_data, f, indent=2, default=str)
            self.logger.info(f"Selection data saved: {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save selection data: {e}")


def test_selector():
    """Test the interactive selector with sample data."""
    # Create some test image paths (you can use actual images)
    test_paths = [
        "/home/mitchellflautt/FractalGenesis/test_render.00000.png",
        "/home/mitchellflautt/FractalGenesis/test_fractal.00000.png",
        "/home/mitchellflautt/FractalGenesis/test_fractal2.00000.png",
        "/home/mitchellflautt/FractalGenesis/test_fractal3.00000.png"
    ]
    
    # Filter to only existing images
    existing_paths = [p for p in test_paths if os.path.exists(p)]
    
    if not existing_paths:
        print("No test images found. Creating placeholder...")
        # Would create test images here
        existing_paths = test_paths  # Use anyway for testing UI
    
    # Test data
    test_fractal_data = [
        {"formula": "mandelbrot", "iterations": 100, "zoom": 1.0},
        {"formula": "julia", "iterations": 150, "zoom": 1.5},
        {"formula": "burning_ship", "iterations": 200, "zoom": 0.8},
        {"formula": "tricorn", "iterations": 120, "zoom": 1.2}
    ]
    
    # Create and show selector
    selector = InteractiveFractalSelector(
        image_paths=existing_paths,
        generation=1,
        fractal_data=test_fractal_data[:len(existing_paths)],
        session_id="test_session"
    )
    
    selected = selector.show_selection_window()
    print(f"User selected: {selected}")
    return selected


if __name__ == "__main__":
    # Test the selector
    test_selector()
