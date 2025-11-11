#!/usr/bin/env python3
"""
3D Fractal Zoom Animations
==========================

Generate various zoom animations into different fractals using GenesisRender.
Creates smooth camera movements diving into fractal structures.
"""

import sys
from pathlib import Path
import time
import math

# Add project root for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from renderers.genesis_render.genesis_render import GenesisRender, GenesisRenderParams
from renderers.genesis_render.animation_system import CameraKeyframe, ParameterKeyframe, CameraController, EasingFunctions

def create_zoom_sequence(fractal_type, start_distance=3.0, end_distance=0.1, 
                        duration=6.0, fps=12, target_pos=(0, 0, 0)):
    """Create a smooth zoom animation sequence"""
    
    camera_controller = CameraController()
    camera_controller.set_easing('ease_in_out_cubic')
    
    # Calculate keyframes for smooth zoom
    num_keyframes = 8  # Fewer keyframes, let interpolation handle smoothness
    
    for i in range(num_keyframes):
        t = i / (num_keyframes - 1)
        
        # Use exponential easing for smooth zoom feel
        zoom_t = pow(t, 1.5)  # Accelerate zoom
        distance = start_distance * (1 - zoom_t) + end_distance * zoom_t
        
        # Calculate camera position
        angle = t * math.pi * 0.25  # Slight rotation during zoom
        cam_x = math.sin(angle) * distance
        cam_y = math.cos(angle * 0.7) * distance * 0.3
        cam_z = math.cos(angle) * distance
        
        keyframe = CameraKeyframe(
            time=t,
            position=(cam_x, cam_y, cam_z),
            target=target_pos
        )
        
        camera_controller.add_keyframe(keyframe)
    
    return camera_controller, duration, fps, fractal_type

def create_spiral_zoom(fractal_type, duration=8.0, fps=10):
    """Create a spiral zoom into fractal"""
    
    seq = AnimationSequence("spiral_zoom_" + fractal_type)
    
    num_frames = int(duration * fps)
    start_distance = 4.0
    end_distance = 0.05
    
    for i in range(num_frames):
        t = i / (num_frames - 1)
        
        # Exponential zoom with spiral motion
        zoom_t = pow(t, 2.0)
        distance = start_distance * (1 - zoom_t) + end_distance * zoom_t
        
        # Spiral parameters
        spiral_angle = t * math.pi * 6  # Multiple rotations
        spiral_radius = distance * 0.3
        
        cam_x = math.cos(spiral_angle) * spiral_radius
        cam_y = math.sin(spiral_angle) * spiral_radius
        cam_z = distance
        
        keyframe = Keyframe(
            time=t,
            camera_pos=(cam_x, cam_y, cam_z),
            camera_target=(0, 0, 0),
            fractal_type=fractal_type,
            fov=45 + t * 15  # Slightly widen FOV as we zoom
        )
        
        seq.add_keyframe(keyframe)
    
    seq.set_duration(duration, fps)
    return seq

def create_multi_target_zoom(fractal_type, duration=10.0, fps=8):
    """Zoom to multiple interesting points in the fractal"""
    
    seq = AnimationSequence("multi_zoom_" + fractal_type)
    
    # Define interesting points to zoom to
    targets = [
        (0, 0, 0),      # Center
        (0.5, 0.3, 0.2), # Side detail
        (-0.3, 0.4, 0.1), # Another detail
        (0.1, -0.2, 0.3), # Third point
    ]
    
    num_frames = int(duration * fps)
    segment_frames = num_frames // len(targets)
    
    for segment, target in enumerate(targets):
        for i in range(segment_frames):
            total_t = (segment * segment_frames + i) / (num_frames - 1)
            local_t = i / (segment_frames - 1) if segment_frames > 1 else 0
            
            # Distance varies by segment
            if segment == 0:
                distance = 3.0 - local_t * 1.5  # Initial zoom
            else:
                distance = 1.5 - local_t * 0.8  # Closer zooms
            
            # Camera movement with slight arc
            angle = total_t * math.pi * 0.5
            cam_x = target[0] + math.sin(angle) * distance
            cam_y = target[1] + math.cos(angle * 1.3) * distance * 0.4
            cam_z = target[2] + math.cos(angle) * distance
            
            keyframe = Keyframe(
                time=total_t,
                camera_pos=(cam_x, cam_y, cam_z),
                camera_target=target,
                fractal_type=fractal_type
            )
            
            seq.add_keyframe(keyframe)
    
    seq.set_duration(duration, fps)
    return seq

def render_zoom_animations():
    """Generate various zoom animations"""
    
    renderer = GenesisRender()
    output_dir = project_root / "output" / "fractal_zooms"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Animation configurations
    animations = [
        {
            "name": "Mandelbulb Deep Zoom",
            "fractal": "mandelbulb",
            "type": "simple_zoom",
            "params": {"duration": 6.0, "fps": 12, "start_distance": 3.0, "end_distance": 0.02}
        },
        {
            "name": "Julia Spiral Dive", 
            "fractal": "julia_3d",
            "type": "spiral_zoom",
            "params": {"duration": 8.0, "fps": 10}
        },
        {
            "name": "Mandelbox Exploration",
            "fractal": "mandelbox", 
            "type": "multi_target",
            "params": {"duration": 10.0, "fps": 8}
        },
        {
            "name": "Burning Ship Zoom",
            "fractal": "burning_ship_3d",
            "type": "simple_zoom", 
            "params": {"duration": 5.0, "fps": 12, "start_distance": 2.5, "end_distance": 0.05}
        }
    ]
    
    for anim_config in animations:
        print(f"\nCreating: {anim_config['name']}")
        print("=" * 50)
        
        # Create animation sequence based on type
        if anim_config['type'] == 'simple_zoom':
            params = anim_config['params']
            sequence = create_zoom_sequence(
                anim_config['fractal'],
                start_distance=params.get('start_distance', 3.0),
                end_distance=params.get('end_distance', 0.1),
                duration=params['duration'],
                fps=params['fps']
            )
        elif anim_config['type'] == 'spiral_zoom':
            params = anim_config['params']
            sequence = create_spiral_zoom(
                anim_config['fractal'],
                duration=params['duration'],
                fps=params['fps']
            )
        elif anim_config['type'] == 'multi_target':
            params = anim_config['params']
            sequence = create_multi_target_zoom(
                anim_config['fractal'], 
                duration=params['duration'],
                fps=params['fps']
            )
        
        # Set up render parameters
        base_params = renderer.create_preset(anim_config['fractal'] + "_classic")
        if not hasattr(base_params, 'width'):
            base_params.width = 640
            base_params.height = 480
            base_params.iterations = 80
        
        # Adjust for animation quality/speed balance
        base_params.width = 640
        base_params.height = 480
        base_params.iterations = 60  # Reduce iterations for faster rendering
        
        # Enhanced lighting for better visuals
        base_params.ambient_occlusion_enabled = True
        base_params.hard_lighting_enabled = True
        base_params.ao_strength = 0.3
        base_params.light1_intensity = 1.0
        
        # Apply sequence to base parameters
        sequence.base_params = base_params
        
        # Generate animation frames
        anim_name = anim_config['name'].lower().replace(' ', '_')
        print(f"Rendering {len(sequence.keyframes)} frames...")
        start_time = time.time()
        
        try:
            frames = renderer.create_animation(sequence, output_dir / anim_name, anim_name)
            render_time = time.time() - start_time
            
            print(f"Animation rendered: {len(frames)} frames in {render_time:.1f}s")
            print(f"Average: {render_time/len(frames):.2f}s per frame")
            
            # Create GIF
            gif_path = output_dir / f"{anim_name}.gif"
            print(f"Creating GIF: {gif_path}")
            renderer.create_gif(frames, gif_path, fps=anim_config['params']['fps'])
            
            # Report file size
            if gif_path.exists():
                size_mb = gif_path.stat().st_size / (1024 * 1024)
                print(f"GIF created: {size_mb:.1f} MB")
            
        except Exception as e:
            print(f"Error rendering {anim_config['name']}: {e}")
            continue
    
    print(f"\nAll zoom animations saved to: {output_dir}")
    return output_dir

def main():
    print("3D Fractal Zoom Animation Generator")
    print("==================================")
    
    # Generate the zoom animations
    output_path = render_zoom_animations()
    
    # Summary
    print(f"\n✅ Zoom animations complete!")
    print(f"Output directory: {output_path}")
    
    # List generated files
    gifs = list(output_path.glob("*.gif"))
    if gifs:
        print(f"\nGenerated animations:")
        for gif in gifs:
            size_mb = gif.stat().st_size / (1024 * 1024)
            print(f"  {gif.name} ({size_mb:.1f} MB)")
    
    print("\nYou can view these animations with any GIF viewer or web browser.")

if __name__ == "__main__":
    main()