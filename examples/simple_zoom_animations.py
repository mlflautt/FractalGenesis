#!/usr/bin/env python3
"""
Simple 3D Fractal Zoom Animations
=================================

Generate zoom animations by manually creating frames with different camera positions.
"""

import sys
from pathlib import Path
import time
import math
from PIL import Image

# Add project root for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from renderers.genesis_render.genesis_render import GenesisRender, GenesisRenderParams

def create_zoom_frames(renderer, fractal_preset, output_dir, animation_name, 
                      start_distance=3.0, end_distance=0.1, num_frames=60):
    """Create zoom animation frames manually"""
    
    print(f"Creating {animation_name}: {num_frames} frames")
    
    # Create base parameters
    base_params = renderer.create_preset(fractal_preset)
    base_params.width = 640
    base_params.height = 480
    base_params.iterations = 60  # Balance quality vs speed
    
    # Enhanced lighting
    base_params.ambient_occlusion_enabled = True
    base_params.hard_lighting_enabled = True
    base_params.ao_strength = 0.3
    base_params.light1_intensity = 1.0
    
    frame_paths = []
    
    for i in range(num_frames):
        t = i / (num_frames - 1)
        
        # Exponential zoom for dramatic effect
        zoom_t = pow(t, 1.8)
        distance = start_distance * (1 - zoom_t) + end_distance * zoom_t
        
        # Add slight rotation and vertical movement
        angle = t * math.pi * 0.3
        height_offset = math.sin(t * math.pi) * 0.2
        
        # Calculate camera position
        cam_x = math.sin(angle) * distance
        cam_y = height_offset * distance
        cam_z = math.cos(angle) * distance
        
        # Update camera position
        base_params.camera_pos = (cam_x, cam_y, cam_z)
        base_params.camera_target = (0, 0, 0)
        
        # Render frame
        print(f"  Frame {i+1}/{num_frames} (t={t:.3f}, distance={distance:.3f})")
        start_render = time.time()
        
        image, metrics = renderer.render(base_params)
        render_time = time.time() - start_render
        
        # Save frame
        frame_filename = f"{animation_name}_frame_{i:04d}.png"
        frame_path = output_dir / frame_filename
        renderer.save_image(image, frame_path, f"{animation_name} Frame {i+1}")
        
        frame_paths.append(frame_path)
        
        print(f"    Rendered in {render_time:.2f}s, {metrics['surface_pixels']:,} surface pixels")
    
    return frame_paths

def create_gif_from_frames(frame_paths, output_path, fps=12, loop=0):
    """Create animated GIF from frame images"""
    
    print(f"Creating GIF: {output_path}")
    
    # Load all frames
    frames = []
    for frame_path in frame_paths:
        img = Image.open(frame_path)
        frames.append(img)
    
    # Calculate frame duration in milliseconds
    frame_duration = int(1000 / fps)
    
    # Save as animated GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=loop,
        optimize=True
    )
    
    return output_path

def render_zoom_animations():
    """Generate various zoom animations"""
    
    renderer = GenesisRender()
    output_dir = project_root / "output" / "fractal_zooms"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Animation configurations - keeping them simple
    animations = [
        {
            "name": "mandelbulb_zoom",
            "fractal": "mandelbulb_classic",
            "frames": 48,
            "fps": 12,
            "start_distance": 3.0,
            "end_distance": 0.05
        },
        {
            "name": "julia_zoom", 
            "fractal": "julia_organic",
            "frames": 40,
            "fps": 10,
            "start_distance": 2.5,
            "end_distance": 0.08
        },
        {
            "name": "mandelbox_zoom",
            "fractal": "mandelbox_classic",
            "frames": 36,
            "fps": 12,
            "start_distance": 2.0,
            "end_distance": 0.1
        },
        {
            "name": "burning_ship_zoom",
            "fractal": "burning_ship_3d", 
            "frames": 30,
            "fps": 10,
            "start_distance": 2.5,
            "end_distance": 0.15
        }
    ]
    
    results = []
    
    for anim_config in animations:
        print(f"\n{'='*60}")
        print(f"Creating: {anim_config['name'].replace('_', ' ').title()}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Create frame directory
            frame_dir = output_dir / anim_config['name']
            frame_dir.mkdir(exist_ok=True)
            
            # Generate frames
            frame_paths = create_zoom_frames(
                renderer=renderer,
                fractal_preset=anim_config['fractal'],
                output_dir=frame_dir,
                animation_name=anim_config['name'],
                start_distance=anim_config['start_distance'],
                end_distance=anim_config['end_distance'],
                num_frames=anim_config['frames']
            )
            
            # Create GIF
            gif_path = output_dir / f"{anim_config['name']}.gif"
            create_gif_from_frames(frame_paths, gif_path, fps=anim_config['fps'])
            
            total_time = time.time() - start_time
            
            # Get file size
            gif_size_mb = gif_path.stat().st_size / (1024 * 1024)
            
            result = {
                'name': anim_config['name'],
                'frames': len(frame_paths),
                'fps': anim_config['fps'],
                'duration': len(frame_paths) / anim_config['fps'],
                'render_time': total_time,
                'avg_per_frame': total_time / len(frame_paths),
                'gif_size_mb': gif_size_mb,
                'gif_path': str(gif_path)
            }
            results.append(result)
            
            print(f"\n✅ {anim_config['name']} complete:")
            print(f"   Frames: {result['frames']}")
            print(f"   Duration: {result['duration']:.1f}s @ {result['fps']} fps")
            print(f"   Render time: {result['render_time']:.1f}s ({result['avg_per_frame']:.2f}s/frame)")
            print(f"   GIF size: {result['gif_size_mb']:.1f} MB")
            
        except Exception as e:
            print(f"❌ Error creating {anim_config['name']}: {e}")
            continue
    
    return results, output_dir

def main():
    print("Simple 3D Fractal Zoom Animations")
    print("=" * 40)
    
    # Generate animations
    results, output_dir = render_zoom_animations()
    
    # Summary
    print(f"\n{'='*60}")
    print("ZOOM ANIMATIONS COMPLETE")
    print(f"{'='*60}")
    
    if results:
        total_frames = sum(r['frames'] for r in results)
        total_time = sum(r['render_time'] for r in results)
        total_size = sum(r['gif_size_mb'] for r in results)
        
        print(f"\nGenerated {len(results)} animations:")
        for result in results:
            print(f"  • {result['name'].replace('_', ' ').title()}: "
                  f"{result['frames']} frames, {result['gif_size_mb']:.1f} MB")
        
        print(f"\nTotals:")
        print(f"  Frames: {total_frames}")
        print(f"  Render time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"  Average per frame: {total_time/total_frames:.2f}s")
        print(f"  Total size: {total_size:.1f} MB")
        print(f"\nOutput directory: {output_dir}")
        print(f"View animations by opening the .gif files")
    else:
        print("No animations were successfully created.")

if __name__ == "__main__":
    main()