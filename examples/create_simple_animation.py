#!/usr/bin/env python3
"""
Create Simple Animation Example

This script demonstrates how to create basic fractal animations without evolution.
Shows template usage, parameter customization, and animation rendering.

Usage:
    python3 examples/create_simple_animation.py [--template TEMPLATE_NAME] [--preview-only]
    
Examples:
    # Create animation with orbital template
    python3 examples/create_simple_animation.py --template orbital_mandelbulb
    
    # Create animation with zoom template (preview only)
    python3 examples/create_simple_animation.py --template zoom_into_fractal --preview-only
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from FractalAnimator import AnimationTemplates, AnimationRenderer, AnimationParameters
from FractalAnimator.animation_parameters import CameraPath, InterpolationType
from renderers.mandelbulber.renderer import MandelbulberRenderer
from renderers.mandelbulber.templates import ParameterTemplates

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_custom_animation():
    """Create a custom animation from scratch"""
    logger.info("Creating custom animation...")
    
    # Create animation parameters
    animation = AnimationParameters()
    animation.duration_seconds = 8.0
    animation.fps = 30
    
    # Set up camera path - orbital motion
    animation.camera_path = CameraPath(
        path_type="orbit",
        orbit_radius=4.5,
        orbit_height=1.5,
        orbit_speed=0.8,
        focus_point=(0.0, 0.0, 0.0)
    )
    
    # Create keyframes with different fractal parameters
    start_template = ParameterTemplates.classic_mandelbulb()
    start_template.fractal.power = 6.0
    start_template.material.surface_color_r = 0.2
    start_template.material.surface_color_g = 0.4
    start_template.material.surface_color_b = 0.9
    
    mid_template = start_template.copy()
    mid_template.fractal.power = 10.0
    mid_template.material.surface_color_r = 0.8
    mid_template.material.surface_color_g = 0.6
    mid_template.material.surface_color_b = 0.2
    
    end_template = start_template.copy()
    end_template.fractal.power = 8.0
    end_template.material.surface_color_r = 0.6
    end_template.material.surface_color_g = 0.2
    end_template.material.surface_color_b = 0.8
    
    # Add keyframes with smooth interpolation
    animation.add_keyframe(0.0, start_template, InterpolationType.EASE_IN_OUT)
    animation.add_keyframe(0.5, mid_template, InterpolationType.EASE_IN_OUT)
    animation.add_keyframe(1.0, end_template, InterpolationType.EASE_IN_OUT)
    
    logger.info(f"Custom animation created: {animation.total_frames} frames, {animation.duration_seconds}s")
    return animation


def modify_template(template_name: str) -> AnimationParameters:
    """Modify an existing template with custom parameters"""
    logger.info(f"Modifying template: {template_name}")
    
    # Get the base template
    all_templates = AnimationTemplates.get_all_templates()
    if template_name not in all_templates:
        raise ValueError(f"Template '{template_name}' not found. Available: {list(all_templates.keys())}")
    
    animation = all_templates[template_name].copy()
    
    # Customize the animation
    if template_name == "orbital_mandelbulb":
        # Make it longer and more colorful
        animation.duration_seconds = 12.0
        animation.camera_path.orbit_speed = 1.5
        
        # Enhance colors in keyframes
        for keyframe in animation.keyframes:
            keyframe.parameters.material.surface_color_r = min(1.0, keyframe.parameters.material.surface_color_r * 1.3)
            keyframe.parameters.lighting.main_light_intensity = 1.4
            
    elif template_name == "zoom_into_fractal":
        # Make zoom more dramatic
        animation.camera_path.zoom_factor = 0.9
        animation.duration_seconds = 6.0
        
        # Increase detail as we zoom
        for i, keyframe in enumerate(animation.keyframes):
            progress = i / max(1, len(animation.keyframes) - 1)
            keyframe.parameters.fractal.iterations = int(150 + progress * 200)
            keyframe.parameters.render.detail_level = 0.8 + progress * 0.7
    
    logger.info(f"Modified template: {animation.total_frames} frames")
    return animation


def render_animation(animation: AnimationParameters, 
                   output_name: str,
                   preview_only: bool = False) -> bool:
    """Render an animation to video"""
    
    # Check if Mandelbulber is available
    mandelbulber_renderer = MandelbulberRenderer()
    
    if not mandelbulber_renderer.is_available:
        logger.warning("Mandelbulber not found! Animation cannot be rendered.")
        logger.info("To install Mandelbulber:")
        logger.info("  Fedora/RHEL: sudo dnf install mandelbulber2")
        logger.info("  Ubuntu/Debian: sudo apt install mandelbulber2")
        logger.info("  Or install as Flatpak: flatpak install org.mandelbulber.Mandelbulber2")
        return False
    
    # Set up renderer
    output_dir = Path("./simple_animations")
    animation_renderer = AnimationRenderer(
        mandelbulber_renderer=mandelbulber_renderer,
        output_dir=output_dir,
        max_workers=2
    )
    
    logger.info(f"Rendering animation to: {output_dir}")
    
    try:
        if preview_only:
            # Render preview (faster, lower quality)
            logger.info("Rendering preview (fast, low quality)...")
            result = animation_renderer.render_preview(animation, max_frames=20)
        else:
            # Render full animation
            logger.info("Rendering full animation...")
            result = animation_renderer.render_animation(
                animation_params=animation,
                output_name=output_name,
                video_format='mp4'
            )
        
        if result.success:
            logger.info(f"✅ Animation rendered successfully!")
            if result.animation_path:
                logger.info(f"   Video: {result.animation_path}")
            logger.info(f"   Frames: {result.total_frames}")
            logger.info(f"   Render time: {result.render_time_seconds:.1f} seconds")
            return True
        else:
            logger.error(f"❌ Animation rendering failed: {result.error_message}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Rendering error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Create simple fractal animations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--template',
        choices=list(AnimationTemplates.get_all_templates().keys()) + ['custom'],
        default='orbital_mandelbulb',
        help='Animation template to use (default: orbital_mandelbulb)'
    )
    
    parser.add_argument(
        '--preview-only',
        action='store_true',
        help='Only render preview (faster, lower quality)'
    )
    
    parser.add_argument(
        '--list-templates',
        action='store_true',
        help='List available templates and exit'
    )
    
    args = parser.parse_args()
    
    # List templates if requested
    if args.list_templates:
        templates = AnimationTemplates.get_all_templates()
        print("Available animation templates:")
        for name, template in templates.items():
            print(f"  {name}: {template.duration_seconds}s, {template.total_frames} frames")
        return 0
    
    logger.info("=== Simple Animation Creator ===")
    logger.info(f"Template: {args.template}")
    logger.info(f"Preview only: {args.preview_only}")
    
    try:
        # Create animation
        if args.template == 'custom':
            animation = create_custom_animation()
            output_name = "custom_animation"
        else:
            animation = modify_template(args.template)
            output_name = f"{args.template}_modified"
        
        # Show animation info
        logger.info(f"Animation created:")
        logger.info(f"  Duration: {animation.duration_seconds}s")
        logger.info(f"  FPS: {animation.fps}")
        logger.info(f"  Total frames: {animation.total_frames}")
        logger.info(f"  Keyframes: {len(animation.keyframes)}")
        logger.info(f"  Camera path: {animation.camera_path.path_type}")
        
        # Render animation
        success = render_animation(animation, output_name, args.preview_only)
        
        if success:
            logger.info("🎉 Animation creation complete!")
            return 0
        else:
            logger.error("💥 Animation creation failed")
            return 1
            
    except Exception as e:
        logger.error(f"💥 Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())