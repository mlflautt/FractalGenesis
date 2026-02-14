#!/usr/bin/env python3
"""
Simple Keyframe Editor for Fractal Animations
=============================================

Basic GUI for editing fractal animation keyframes.
Allows visual editing of parameters and timing.

Usage:
    python keyframe_editor.py
    python keyframe_editor.py --load my_animation.json
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
from pathlib import Path
from typing import Optional

from animation_controller import AnimationController, AnimationKeyframe, EasingType
from renderers.python_3d.fractal_animator import FractalParams


class KeyframeEditor:
    """Simple GUI for editing animation keyframes"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Fractal Genesis - Keyframe Editor")
        self.root.geometry("900x700")
        
        self.controller = AnimationController()
        self.current_keyframe_index: Optional[int] = None
        
        self._build_ui()
        
    def _build_ui(self):
        """Build the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # === Toolbar ===
        toolbar = ttk.Frame(main_frame)
        toolbar.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(toolbar, text="New", command=self._new_animation).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Load", command=self._load_animation).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Save", command=self._save_animation).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Button(toolbar, text="Add Keyframe", command=self._add_keyframe).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Delete Keyframe", command=self._delete_keyframe).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Button(toolbar, text="Render Preview", command=self._render_preview).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Export Video", command=self._export_video).pack(side=tk.LEFT, padx=2)
        
        # === Keyframe List (Left Panel) ===
        left_frame = ttk.LabelFrame(main_frame, text="Keyframes", padding="5")
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(0, weight=1)
        
        # Keyframe listbox with scrollbar
        list_frame = ttk.Frame(left_frame)
        list_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        self.keyframe_listbox = tk.Listbox(list_frame, height=20, width=30)
        self.keyframe_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.keyframe_listbox.bind('<<ListboxSelect>>', self._on_keyframe_select)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.keyframe_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.keyframe_listbox.configure(yscrollcommand=scrollbar.set)
        
        # Keyframe controls
        controls_frame = ttk.Frame(left_frame)
        controls_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Label(controls_frame, text="Time:").grid(row=0, column=0, sticky=tk.W)
        self.time_var = tk.DoubleVar(value=0.0)
        time_entry = ttk.Spinbox(controls_frame, from_=0.0, to=1.0, increment=0.01,
                                textvariable=self.time_var, width=8)
        time_entry.grid(row=0, column=1, sticky=tk.W, padx=5)
        self.time_var.trace('w', lambda *args: self._update_keyframe_time())
        
        ttk.Label(controls_frame, text="Easing:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.easing_var = tk.StringVar(value="ease_in_out")
        easing_combo = ttk.Combobox(controls_frame, textvariable=self.easing_var,
                                   values=[e.value for e in EasingType], width=15, state="readonly")
        easing_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=(5, 0))
        self.easing_var.trace('w', lambda *args: self._update_keyframe_easing())
        
        # === Parameter Editor (Right Panel) ===
        right_frame = ttk.LabelFrame(main_frame, text="Parameters", padding="5")
        right_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_frame.columnconfigure(1, weight=1)
        
        # Create parameter controls
        self.param_vars = {}
        row = 0
        
        # Camera section
        camera_frame = ttk.LabelFrame(right_frame, text="Camera", padding="5")
        camera_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        camera_frame.columnconfigure(1, weight=1)
        
        cam_params = [
            ('camera_x', 'Position X', -10.0, 10.0, 0.0),
            ('camera_y', 'Position Y', -10.0, 10.0, 0.0),
            ('camera_z', 'Position Z', -10.0, 10.0, -3.0),
            ('target_x', 'Target X', -5.0, 5.0, 0.0),
            ('target_y', 'Target Y', -5.0, 5.0, 0.0),
            ('target_z', 'Target Z', -5.0, 5.0, 0.0),
            ('fov', 'FOV', 10.0, 120.0, 45.0),
        ]
        
        for i, (key, label, min_val, max_val, default) in enumerate(cam_params):
            ttk.Label(camera_frame, text=label + ":").grid(row=i, column=0, sticky=tk.W, padx=5)
            var = tk.DoubleVar(value=default)
            self.param_vars[key] = var
            scale = ttk.Scale(camera_frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL,
                            variable=var, length=200)
            scale.grid(row=i, column=1, sticky=(tk.W, tk.E), padx=5)
            entry = ttk.Entry(camera_frame, textvariable=var, width=8)
            entry.grid(row=i, column=2, padx=5)
            var.trace('w', lambda *args, k=key: self._update_param(k))
        
        row += 1
        
        # Fractal section
        fractal_frame = ttk.LabelFrame(right_frame, text="Fractal", padding="5")
        fractal_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        fractal_frame.columnconfigure(1, weight=1)
        
        fractal_params = [
            ('power', 'Power', 0.1, 30.0, 8.0),
            ('iterations', 'Iterations', 10, 500, 100),
            ('bailout', 'Bailout', 1.0, 100.0, 2.0),
        ]
        
        for i, (key, label, min_val, max_val, default) in enumerate(fractal_params):
            ttk.Label(fractal_frame, text=label + ":").grid(row=i, column=0, sticky=tk.W, padx=5)
            var = tk.DoubleVar(value=default)
            self.param_vars[key] = var
            scale = ttk.Scale(fractal_frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL,
                            variable=var, length=200)
            scale.grid(row=i, column=1, sticky=(tk.W, tk.E), padx=5)
            entry = ttk.Entry(fractal_frame, textvariable=var, width=8)
            entry.grid(row=i, column=2, padx=5)
            var.trace('w', lambda *args, k=key: self._update_param(k))
        
        row += 1
        
        # Color section
        color_frame = ttk.LabelFrame(right_frame, text="Color", padding="5")
        color_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E))
        color_frame.columnconfigure(1, weight=1)
        
        color_params = [
            ('color_r', 'Red', 0.0, 1.0, 0.8),
            ('color_g', 'Green', 0.0, 1.0, 0.6),
            ('color_b', 'Blue', 0.0, 1.0, 0.4),
            ('metallic', 'Metallic', 0.0, 1.0, 0.0),
            ('roughness', 'Roughness', 0.0, 1.0, 0.1),
        ]
        
        for i, (key, label, min_val, max_val, default) in enumerate(color_params):
            ttk.Label(color_frame, text=label + ":").grid(row=i, column=0, sticky=tk.W, padx=5)
            var = tk.DoubleVar(value=default)
            self.param_vars[key] = var
            scale = ttk.Scale(color_frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL,
                            variable=var, length=200)
            scale.grid(row=i, column=1, sticky=(tk.W, tk.E), padx=5)
            entry = ttk.Entry(color_frame, textvariable=var, width=8)
            entry.grid(row=i, column=2, padx=5)
            var.trace('w', lambda *args, k=key: self._update_param(k))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def _update_keyframe_list(self):
        """Refresh the keyframe listbox"""
        self.keyframe_listbox.delete(0, tk.END)
        for i, kf in enumerate(self.controller.keyframes):
            easing_name = kf.easing.value.replace('_', ' ').title()
            self.keyframe_listbox.insert(tk.END, f"{i+1}. t={kf.time:.2f} ({easing_name})")
    
    def _on_keyframe_select(self, event):
        """Handle keyframe selection"""
        selection = self.keyframe_listbox.curselection()
        if selection:
            self.current_keyframe_index = selection[0]
            self._load_keyframe_into_ui(self.current_keyframe_index)
    
    def _load_keyframe_into_ui(self, index: int):
        """Load keyframe data into UI controls"""
        if index < 0 or index >= len(self.controller.keyframes):
            return
        
        kf = self.controller.keyframes[index]
        params = kf.params
        
        # Update time and easing
        self.time_var.set(kf.time)
        self.easing_var.set(kf.easing.value)
        
        # Update parameter controls
        self.param_vars['camera_x'].set(params.camera_pos[0])
        self.param_vars['camera_y'].set(params.camera_pos[1])
        self.param_vars['camera_z'].set(params.camera_pos[2])
        self.param_vars['target_x'].set(params.target[0])
        self.param_vars['target_y'].set(params.target[1])
        self.param_vars['target_z'].set(params.target[2])
        self.param_vars['fov'].set(params.fov)
        
        self.param_vars['power'].set(params.power)
        self.param_vars['iterations'].set(params.iterations)
        self.param_vars['bailout'].set(params.bailout)
        
        self.param_vars['color_r'].set(params.base_color[0])
        self.param_vars['color_g'].set(params.base_color[1])
        self.param_vars['color_b'].set(params.base_color[2])
        self.param_vars['metallic'].set(params.metallic)
        self.param_vars['roughness'].set(params.roughness)
        
        self.status_var.set(f"Editing keyframe {index + 1}")
    
    def _get_params_from_ui(self) -> FractalParams:
        """Get current parameter values from UI"""
        params = FractalParams()
        params.camera_pos = (
            self.param_vars['camera_x'].get(),
            self.param_vars['camera_y'].get(),
            self.param_vars['camera_z'].get()
        )
        params.target = (
            self.param_vars['target_x'].get(),
            self.param_vars['target_y'].get(),
            self.param_vars['target_z'].get()
        )
        params.fov = self.param_vars['fov'].get()
        params.power = self.param_vars['power'].get()
        params.iterations = int(self.param_vars['iterations'].get())
        params.bailout = self.param_vars['bailout'].get()
        params.base_color = (
            self.param_vars['color_r'].get(),
            self.param_vars['color_g'].get(),
            self.param_vars['color_b'].get()
        )
        params.metallic = self.param_vars['metallic'].get()
        params.roughness = self.param_vars['roughness'].get()
        return params
    
    def _update_keyframe_time(self):
        """Update current keyframe's time"""
        if self.current_keyframe_index is not None:
            kf = self.controller.keyframes[self.current_keyframe_index]
            kf.time = self.time_var.get()
            self.controller.keyframes.sort(key=lambda k: k.time)
            self._update_keyframe_list()
    
    def _update_keyframe_easing(self):
        """Update current keyframe's easing"""
        if self.current_keyframe_index is not None:
            kf = self.controller.keyframes[self.current_keyframe_index]
            try:
                kf.easing = EasingType(self.easing_var.get())
            except ValueError:
                pass
    
    def _update_param(self, key: str):
        """Update parameter in current keyframe"""
        if self.current_keyframe_index is not None:
            kf = self.controller.keyframes[self.current_keyframe_index]
            kf.params = self._get_params_from_ui()
    
    def _new_animation(self):
        """Create new animation"""
        self.controller = AnimationController()
        self.current_keyframe_index = None
        self._update_keyframe_list()
        self.status_var.set("New animation created")
    
    def _load_animation(self):
        """Load animation from file"""
        filename = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.controller.load_animation(filename)
                self._update_keyframe_list()
                self.status_var.set(f"Loaded: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load animation: {e}")
    
    def _save_animation(self):
        """Save animation to file"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.controller.save_animation(filename)
                self.status_var.set(f"Saved: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save animation: {e}")
    
    def _add_keyframe(self):
        """Add new keyframe at current time"""
        time = self.time_var.get()
        params = self._get_params_from_ui()
        easing = EasingType(self.easing_var.get())
        
        self.controller.add_keyframe(time, params, easing)
        self._update_keyframe_list()
        self.status_var.set(f"Added keyframe at t={time:.2f}")
    
    def _delete_keyframe(self):
        """Delete selected keyframe"""
        if self.current_keyframe_index is not None:
            if len(self.controller.keyframes) > 1:
                self.controller.keyframes.pop(self.current_keyframe_index)
                self.current_keyframe_index = None
                self._update_keyframe_list()
                self.status_var.set("Keyframe deleted")
            else:
                messagebox.showwarning("Warning", "Cannot delete last keyframe")
    
    def _render_preview(self):
        """Render preview animation"""
        if not self.controller.keyframes:
            messagebox.showwarning("Warning", "No keyframes to render")
            return
        
        self.status_var.set("Rendering preview...")
        self.root.update()
        
        try:
            frames = self.controller.render_preview(width=320, height=240, 
                                                    fps=10, duration=3.0)
            self.status_var.set(f"Preview complete: {len(frames)} frames")
            messagebox.showinfo("Success", f"Rendered {len(frames)} preview frames")
        except Exception as e:
            messagebox.showerror("Error", f"Render failed: {e}")
            self.status_var.set("Render failed")
    
    def _export_video(self):
        """Export final video"""
        if not self.controller.keyframes:
            messagebox.showwarning("Warning", "No keyframes to export")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[
                ("MP4 video", "*.mp4"),
                ("ProRes MOV", "*.mov"),
                ("WebM video", "*.webm"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            self.status_var.set("Exporting video...")
            self.root.update()
            
            try:
                success = self.controller.export_video(
                    filename, 
                    width=1920, 
                    height=1080,
                    fps=30,
                    duration=10.0,
                    quality="high"
                )
                if success:
                    self.status_var.set(f"Exported: {filename}")
                else:
                    self.status_var.set("Export failed")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")
                self.status_var.set("Export failed")


def main():
    import sys
    
    root = tk.Tk()
    app = KeyframeEditor(root)
    
    # Load animation if specified
    if len(sys.argv) > 2 and sys.argv[1] == "--load":
        try:
            app.controller.load_animation(sys.argv[2])
            app._update_keyframe_list()
        except Exception as e:
            print(f"Could not load animation: {e}")
    else:
        # Create default animation with 2 keyframes
        default_params = FractalParams()
        app.controller.add_keyframe(0.0, default_params, EasingType.EASE_IN_OUT, "Start")
        end_params = FractalParams()
        end_params.camera_pos = (3.0, 0.0, 0.0)
        end_params.power = 12.0
        app.controller.add_keyframe(1.0, end_params, EasingType.EASE_IN_OUT, "End")
        app._update_keyframe_list()
    
    root.mainloop()


if __name__ == "__main__":
    main()
