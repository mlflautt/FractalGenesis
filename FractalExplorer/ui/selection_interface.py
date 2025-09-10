"""
Simplified Selection Interface for Fractal Rating

A clean, simple interface that shows 4 fractal options and lets the user
pick their favorite by clicking on it. Much better UX than numeric scoring.
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SelectionInterface:
    """
    Simple 4-option fractal selection interface.
    
    Shows 4 fractal thumbnails in a grid layout and allows user to select
    their preferred option with a single click.
    """
    
    def __init__(self, window_size=(800, 600), thumbnail_size=(250, 250)):
        """
        Initialize the selection interface.
        
        Args:
            window_size: Main window dimensions (width, height)
            thumbnail_size: Size of individual fractal thumbnails
        """
        self.window_size = window_size
        self.thumbnail_size = thumbnail_size
        
        self.root = None
        self.selected_index = None
        self.is_waiting = False
        
        # UI elements
        self.image_buttons = []
        self.photo_images = []
        
    def setup_ui(self):
        """Set up the user interface."""
        if self.root:
            return
            
        self.root = tk.Tk()
        self.root.title("Fractal Genesis - Choose Your Favorite")
        self.root.geometry(f"{self.window_size[0]}x{self.window_size[1]}")
        self.root.resizable(False, False)
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, 
                               text="Choose Your Favorite Fractal",
                               font=("Arial", 18, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Instructions
        instructions = ttk.Label(main_frame,
                                text="Click on the fractal you find most visually appealing",
                                font=("Arial", 10))
        instructions.pack(pady=(0, 20))
        
        # Image grid
        grid_frame = ttk.Frame(main_frame)
        grid_frame.pack(expand=True, fill=tk.BOTH)
        
        # Create 2x2 grid of image buttons
        self.image_buttons = []
        self.photo_images = [None, None, None, None]
        
        for i in range(4):
            row = i // 2
            col = i % 2
            
            # Create frame for each option
            option_frame = ttk.LabelFrame(grid_frame, text=f"Option {i+1}", padding="10")
            option_frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            
            # Configure grid weights
            grid_frame.grid_rowconfigure(row, weight=1)
            grid_frame.grid_columnconfigure(col, weight=1)
            
            # Create image button
            btn = tk.Button(option_frame,
                           text=f"Loading...\\nOption {i+1}",
                           width=20, height=12,
                           command=lambda idx=i: self.on_selection(idx),
                           cursor="hand2",
                           font=("Arial", 10),
                           relief=tk.RAISED,
                           bg="white")
            btn.pack(expand=True, fill=tk.BOTH)
            
            self.image_buttons.append(btn)
        
        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(pady=(20, 0), fill=tk.X)
        
        # Skip button
        skip_btn = ttk.Button(control_frame,
                             text="Skip - None Appeal to Me",
                             command=lambda: self.on_selection(-1))
        skip_btn.pack(side=tk.LEFT)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Ready...")
        self.status_label.pack(side=tk.RIGHT)
        
        # Center window on screen
        self.root.update_idletasks()
        self._center_window()
    
    def _center_window(self):
        """Center the window on the screen."""
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        pos_x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        pos_y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{pos_x}+{pos_y}")
    
    def load_fractal_images(self, image_paths: List[Optional[Path]]) -> bool:
        """
        Load fractal images into the interface.
        
        Args:
            image_paths: List of 4 image file paths
            
        Returns:
            True if images loaded successfully
        """
        if len(image_paths) != 4:
            logger.error(f"Expected 4 image paths, got {len(image_paths)}")
            return False
        
        if not self.root:
            self.setup_ui()
        
        success = True
        
        for i, img_path in enumerate(image_paths):
            if img_path and img_path.exists():
                try:
                    # Load and resize image
                    with Image.open(img_path) as img:
                        # Resize maintaining aspect ratio
                        img.thumbnail(self.thumbnail_size, Image.Resampling.LANCZOS)
                        
                        # Create fixed size image with padding
                        display_img = Image.new('RGB', self.thumbnail_size, (50, 50, 50))
                        paste_x = (self.thumbnail_size[0] - img.width) // 2
                        paste_y = (self.thumbnail_size[1] - img.height) // 2
                        display_img.paste(img, (paste_x, paste_y))
                        
                        # Convert to PhotoImage
                        photo = ImageTk.PhotoImage(display_img)
                        self.photo_images[i] = photo
                        
                        # Update button
                        self.image_buttons[i].config(
                            image=photo,
                            text="",
                            compound=tk.CENTER,
                            relief=tk.RAISED,
                            bg="white"
                        )
                        
                except Exception as e:
                    logger.error(f"Failed to load image {img_path}: {e}")
                    self._set_button_placeholder(i, f"Error loading\\nOption {i+1}")
                    success = False
            else:
                self._set_button_placeholder(i, f"No Image\\nOption {i+1}")
        
        return success
    
    def _set_button_placeholder(self, index: int, text: str):
        """Set a text placeholder for a button that couldn't load an image."""
        self.image_buttons[index].config(
            image="",
            text=text,
            compound=tk.CENTER,
            relief=tk.FLAT,
            bg="#f0f0f0"
        )
        self.photo_images[index] = None
    
    def on_selection(self, index: int):
        """
        Handle user selection.
        
        Args:
            index: Selected option index (0-3) or -1 for skip
        """
        self.selected_index = index
        
        # Provide visual feedback
        if index >= 0:
            for i, btn in enumerate(self.image_buttons):
                if i == index:
                    btn.config(relief=tk.SUNKEN, bg="#90EE90")  # Light green
                else:
                    btn.config(relief=tk.RAISED, bg="white")
            
            self.status_label.config(text=f"Selected Option {index+1}")
        else:
            # Skip selection
            for btn in self.image_buttons:
                btn.config(relief=tk.FLAT, bg="#ffcccc")  # Light red
            self.status_label.config(text="Skipped - None selected")
        
        # Close window after a short delay for visual feedback
        self.root.after(1000, self._close_window)
    
    def _close_window(self):
        """Close the selection window."""
        if self.root:
            self.root.quit()
    
    def show_and_get_selection(self) -> int:
        """
        Show the interface and wait for user selection.
        
        Returns:
            Selected option index (0-3) or -1 for skip
        """
        if not self.root:
            self.setup_ui()
        
        self.selected_index = None
        self.is_waiting = True
        
        # Reset button states
        for btn in self.image_buttons:
            btn.config(relief=tk.RAISED, bg="white")
        
        self.status_label.config(text="Choose your favorite...")
        
        # Show window and wait for selection
        self.root.mainloop()
        
        return self.selected_index if self.selected_index is not None else -1
    
    def destroy(self):
        """Clean up the interface."""
        if self.root:
            self.root.destroy()
            self.root = None


# Simple usage example/test function
def test_selection_interface():
    """Test function for the selection interface."""
    interface = SelectionInterface()
    
    # For testing, create some dummy image paths
    # In real usage, these would be actual fractal render paths
    test_paths = [None, None, None, None]  # Placeholder paths
    
    interface.load_fractal_images(test_paths)
    selection = interface.show_and_get_selection()
    
    print(f"User selected option: {selection}")
    interface.destroy()
    
    return selection


if __name__ == "__main__":
    # Run test if script is executed directly
    test_selection_interface()
