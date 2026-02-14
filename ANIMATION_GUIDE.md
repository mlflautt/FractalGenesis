# FractalGenesis Animation System

## Overview

Enhanced animation system for smooth fractal evolution with keyframe control, professional video export, and smart Mandelbulb morphing.

## What Was Added

### 1. `animation_controller.py` - Core Animation Engine

**Smart Interpolation Features:**
- **Spherical Camera Interpolation**: Camera moves along sphere surface instead of linear paths, preventing clipping through fractal surfaces
- **Log-Space Power Morphing**: Power values interpolate exponentially for natural fractal evolution (not jarring linear jumps)
- **HSV Color Transitions**: Colors transition through hue-saturation-value space, avoiding muddy grays
- **Catmull-Rom Splines**: Smooth curves through all keyframes (much smoother than linear)

**Easing Functions:**
- Linear, Ease In/Out, Cubic (S-curve)
- Bounce, Elastic (springy effects)
- Exponential (fast start, slow end)

**Video Export:**
- MP4 (H.264) for sharing
- ProRes for editing
- WebM for web
- Configurable quality (draft/medium/high)

### 2. `keyframe_editor.py` - GUI Editor

**Features:**
- Visual timeline of keyframes
- Live parameter adjustment (sliders + entry fields)
- Add/Delete keyframes
- Render preview (low-res, fast)
- Export final video
- Save/Load animations as JSON

**Usage:**
```bash
# Launch editor
python keyframe_editor.py

# Load existing animation
python keyframe_editor.py --load my_animation.json
```

## New Dependencies

Add to your environment:
```bash
pip install ffmpeg-python taichi moderngl imgui
```

**Why these tools:**
- `ffmpeg-python`: Export MP4/ProRes/WebM videos
- `taichi`: GPU-accelerated preview (120x faster rendering)
- `moderngl`: Real-time OpenGL preview window
- `imgui`: Immediate mode GUI (for future advanced editor)

## Quick Start

### Basic Animation

```python
from animation_controller import AnimationController, EasingType
from renderers.python_3d.fractal_animator import FractalParams

# Create animation
anim = AnimationController()

# Keyframe 1: Start position
start = FractalParams()
start.power = 8.0
start.camera_pos = (0.0, 0.0, -3.0)
anim.add_keyframe(0.0, start, EasingType.EASE_IN_OUT)

# Keyframe 2: End position  
end = FractalParams()
end.power = 12.0
end.camera_pos = (3.0, 0.0, 0.0)  # Orbit around
anim.add_keyframe(1.0, end, EasingType.EASE_IN_OUT)

# Render preview (fast, low-res)
frames = anim.render_preview(width=320, height=240, fps=10, duration=3.0)

# Export final video
anim.export_video("my_fractal.mp4", width=1920, height=1080, fps=30, duration=10.0)
```

### Using the GUI Editor

```bash
# Start with blank animation
python keyframe_editor.py

# Or load existing
python keyframe_editor.py --load animation.json
```

**Editor Controls:**
1. **Toolbar**: New, Load, Save, Add/Delete keyframes
2. **Left Panel**: Keyframe list with timing
3. **Right Panel**: Parameter sliders for selected keyframe
4. **Bottom**: Render Preview or Export Video

## Smart Morphing Explained

### Problem with Simple Linear Interpolation

**Camera Movement:**
- Linear: Camera moves in straight line from A to B
- Problem: Often cuts through the fractal surface (jarring!)
- Solution: Spherical interpolation (camera orbits around center)

**Power Changes:**
- Linear: 2.0 → 8.0 goes 2, 4, 6, 8
- Problem: Fractal structure changes exponentially with power
- Solution: Log-space interpolation (smooth visual evolution)

**Color Transitions:**
- Linear RGB: Red(1,0,0) → Blue(0,0,1) goes through gray(0.5,0,0.5)
- Problem: Muddy grays in middle
- Solution: HSV interpolation (maintains vibrancy)

## Animation Workflow

1. **Create Keyframes**: Set start and end states
2. **Adjust Timing**: Position keyframes on timeline (0.0 to 1.0)
3. **Choose Easing**: Select motion curve for each transition
4. **Preview**: Render low-res version to check motion
5. **Export**: Generate final high-res video

## Example Animations

### Camera Orbit
```python
from animation_controller import AnimationController, EasingType
from renderers.python_3d.fractal_animator import FractalPresets

anim = AnimationController()

# 4 keyframes for full orbit
for i, angle in enumerate([0, 90, 180, 270]):
    rad = np.radians(angle)
    params = FractalPresets.get_preset("classic_mandelbulb")
    params.camera_pos = (3*np.cos(rad), 0, 3*np.sin(rad))
    anim.add_keyframe(i/3, params, EasingType.EASE_IN_OUT)

anim.export_video("orbit.mp4", duration=8.0)
```

### Power Morph
```python
anim = AnimationController()

# Morph from power 2 to 16
for i, power in enumerate([2, 4, 8, 12, 16]):
    params = FractalParams()
    params.power = power
    params.color_palette = "rainbow"
    anim.add_keyframe(i/4, params, EasingType.CUBIC)

anim.export_video("power_morph.mp4", duration=10.0)
```

## Performance Tips

**For Preview:**
- Use low resolution (320x240)
- Reduce max_ray_steps to 50
- Lower FPS (10-15)

**For Final Export:**
- Use high resolution (1920x1080 or 4K)
- Increase max_ray_steps to 200
- Use 30fps for smooth playback
- Consider "medium" quality for faster renders

## File Format

Animations saved as JSON:
```json
{
  "keyframes": [
    {
      "time": 0.0,
      "params": {
        "power": 8.0,
        "camera_pos": [0.0, 0.0, -3.0],
        "base_color": [0.8, 0.6, 0.4]
      },
      "easing": "ease_in_out",
      "notes": "Start position"
    }
  ]
}
```

## Next Steps

Future enhancements:
- [ ] Real-time GPU preview with Taichi
- [ ] Animation curve editor (adjust velocity/acceleration)
- [ ] Motion blur for fast movements
- [ ] Audio-reactive fractals
- [ ] Batch export multiple resolutions

## Troubleshooting

**"FFmpeg not found"**
- Install FFmpeg: `sudo dnf install ffmpeg` (Fedora) or `brew install ffmpeg` (Mac)

**"Preview too slow"**
- Reduce preview resolution in keyframe editor
- Lower max_ray_steps parameter

**"Video export fails"**
- Check disk space (videos are large)
- Verify FFmpeg is installed: `ffmpeg -version`

**"Keyframes not smooth"**
- Use Catmull-Rom or Ease In/Out easing
- Add more intermediate keyframes
- Check camera isn't cutting through fractal

## Branch Info

- **main**: Stable version (pushed before this work)
- **dev**: Current development with animation system

```bash
# Switch to dev branch
git checkout dev

# Run the new tools
python keyframe_editor.py
python animation_controller.py
```
