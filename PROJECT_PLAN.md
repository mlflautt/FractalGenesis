# PROJECT_PLAN.md - FractalGenesis Creative AI System

**Project:** FractalGenesis - AI-Driven Fractal Art & Music Visualization Pipeline  
**Created:** February 16, 2026  
**Model:** minimax-m2.5 (opencode)  
**Status:** Planning Phase → Implementation

---

## 1. Executive Summary

Build a comprehensive pipeline that:
- Generates high-quality fractal visuals (2D flames, 3D ray-marched fractals)
- Supports diverse fractal types: Mandelbulb, Mandelbox, Julia variants, IFS, KIFS, etc.
- Creates animations with parametric morphing and camera movement
- Uses evolutionary algorithms to evolve parameters based on user selection
- Uses AI to assist/generate fractal parameter "DNA" (genomes)
- Outputs formats ready for AI image-gen models and music video production

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FRACTALGENESIS CREATIVE PIPELINE                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────────┐ │
│  │   INPUTS     │    │   ANALYSIS   │    │      GENERATION              │ │
│  ├──────────────┤    ├──────────────┤    ├──────────────────────────────┤ │
│  │ • Parameters │───▶│ • Validation │───▶│ • Multi-formula Rendering    │ │
│  │ • Genomes    │    │ • Verification│    │ • Animation Frames          │ │
│  │ • User prefs │    │              │    │ • Video Export               │ │
│  └──────────────┘    └──────────────┘    └──────────────────────────────┘ │
│                                                  │                          │
│                                                  ▼                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────────┐ │
│  │   OUTPUTS    │◀───│   EVOLUTION  │◀───│   PARAMETER DNA              │ │
│  ├──────────────┤    ├──────────────┤    ├──────────────────────────────┤ │
│  │ • PNG frames │    │ • EA Engine  │    │ • FractalGenome class        │ │
│  │ • GIF        │    │ • Selection  │    │ • Mutate/Crossover           │ │
│  │ • MP4/WebM   │    │ • AI Assist  │    │ • NN Preference Model        │ │
│  └──────────────┘    └──────────────┘    └──────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Supported Fractal Types

### 3.1 3D Fractals (Ray-Marched)

| Formula | Parameters | Visual Character |
|---------|------------|------------------|
| **Mandelbulb** | power (2-16), bailout, iterations | Bulbous, organic, self-similar |
| **Mandelbox** | scale (-3 to 3), fold limits | Geometric, box-like, cubic symmetry |
| **Julia Bulb** | c constant, power | Symmetric, ethereal |
| **Quaternion Julia** | c (4D), slice position | Smooth, blobby, 4D cross-sections |
| **KIFS** | fold params, scale, rotation | Crystalline, tetrahedral symmetry |
| **Menger Sponge** | iterations | Cubic, hole-filled |
| **Sierpinski Tetrahedron** | iterations | Triangular, self-similar |

### 3.2 2D Fractals

| Formula | Parameters | Visual Character |
|---------|------------|------------------|
| **Flam3 Flames** | xform weights, variations | Organic, flowing, colorful |
| **Mandelbrot** | power, iterations | Classic, iconic |
| **Julia Set** | c constant | Symmetric, varies with c |

### 3.3 Hybrid Techniques

- Chain multiple formulas (up to 6 in sequence)
- Apply transforms between iterations
- Mix escapetime + dIFS approaches

---

## 4. Core Components

### 4.1 Visual Generation Module (`renderers/`)

| Renderer | Status | Output Type |
|----------|--------|-------------|
| `python_3d/` | ✅ Working | 3D Mandelbulb, Julia, Mandelbox (Numba JIT) |
| `mandelbulber/` | ✅ Fixed | 3D ray-marched fractals |
| `flam3/` | ✅ Working | 2D fractal flames |
| `genesis_render/` | ⚠️ Needs optimization | Advanced 3D renderer |

**Unified Interface:** `renderers/unified.py` provides `BaseRenderer` abstraction

### 4.2 Verification System (`renderers/verification.py`)

Every render step must pass:
- File exists and size > 1KB
- Mean pixel value > 20 (not all black)
- Black pixel ratio < 95%
- Diversity check (different from previous frames)

### 4.3 Evolution Framework (`evolution/`)

```
FractalGenome
├── CameraGenome: position, target, fov
├── FractalParams: type, power, iterations, bailout
├── MaterialGenome: color_palette, metallic, roughness
├── LightingGenome: positions, intensities, colors
└── AnimationGenome: keyframes, timing
```

### 4.4 AI Integration (`ai/`)

- `preference_learner.py` - Train NN on user selections
- `parameter_generator.py` - Use trained model to generate parameters

---

## 5. Implementation Phases

### Phase 1: Core Rendering (Priority)
- [x] Verification system
- [x] Unified renderer interface  
- [ ] Expand Python 3D renderer with more formula types
- [ ] Fix/verify all renderers produce valid output
- [ ] **DELIVERABLE:** Script rendering diverse fractal types

### Phase 2: Animation System
- [ ] Parametric interpolation between parameter states
- [ ] Camera path animations
- [ ] GIF/MP4 export
- [ ] **DELIVERABLE:** Fractal animation generator

### Phase 3: Evolution System
- [ ] Create `FractalGenome` dataclass
- [ ] Implement mutation/crossover operators for all parameter types
- [ ] Selection interface (CLI-based)
- [ ] **DELIVERABLE:** CLI evolution tool

### Phase 4: Audio Sync (Future)
- [ ] Audio analysis with librosa
- [ ] Beat-synchronized animations
- [ ] Video export with audio

### Phase 5: AI Integration (Future)
- [ ] Train preference model on genome selections
- [ ] AI-assisted parameter generation
- [ ] Automated fitness from trained model

---

## 6. File Structure

```
FractalGenesis/
├── PROJECT_PLAN.md              ← This document
├── CHANGELOG.md                 ← Version history
├── test_system.py               ← Verification system
├── renderers/
│   ├── unified.py               ← BaseRenderer abstraction
│   ├── verification.py          ← Output verification
│   ├── python_3d/               ← Native 3D renderer
│   ├── mandelbulber/            ← Mandelbulber CLI wrapper
│   ├── flam3/                   ← Fractal flames
│   └── genesis_render/          ← Advanced renderer
├── evolution/                   ← Evolution framework
├── ai/                          ← AI/ML components
├── examples/                    ← Demo scripts
└── output/                      ← Rendered results
```

---

## 7. Technical Stack

| Component | Technology |
|-----------|------------|
| 3D Rendering | Numba (Python 3D), Mandelbulber CLI, flam3 |
| Evolution | Custom GA + scikit-learn |
| AI Models | PyTorch |
| Image Processing | PIL, NumPy, Matplotlib |
| Video Export | OpenCV, imageio |
| Audio Analysis (future) | librosa, pretty_midi |

---

## 8. Design Decisions

### 8.1 Parameter DNA
Structured genome classes for type safety and clear mutation points.

### 8.2 Render Abstraction
All renderers implement `BaseRenderer`:
- `generate_random_parameters()`
- `render_fractal(params, output_path)`
- `mutate_parameters(params)`

### 8.3 Verification Required
Every render step passes verification before proceeding.

---

## 9. Questions Answered

1. **Beat-sync:** Simple implementation, lower priority than core rendering
2. **Genome:** Include animation params in genome for full evolution control
3. **Verification:** Basic validity sufficient for now, expand if needed
4. **Focus:** Get fractal rendering/animation working first, add AI later

---

## 10. Immediate Next Steps

1. Verify all renderers produce diverse, valid output
2. Expand Python 3D renderer formula types (Mandelbox, IFS)
3. Create simple animation script
4. Test batch rendering

---

**Plan Version:** 1.0  
**Signed:** minimax-m2.5 (opencode)  
**Date:** 2026-02-16  
**Git Status:** Local → Push to remote