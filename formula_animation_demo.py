#!/usr/bin/env python3
"""
Formula Animation Demo
======================

Demonstrates using the extended formula library with animations.
Shows how to:
- Use meta-parameters for formula control
- Animate between different formulas
- Create hybrid formula animations
- Random formula search with animation

Usage:
    python formula_animation_demo.py
"""

import numpy as np
from pathlib import Path

from formulas import FormulaRegistry, FormulaRandomSearch, HybridFormula, BlendMode
from animation_controller import AnimationController, EasingType
from renderers.python_3d.fractal_animator import FractalParams


def demo_formula_variants():
    """Demo: Create animation with formula parameter morphing"""
    print("="*60)
    print("Demo 1: Formula Parameter Morphing")
    print("="*60)
    
    # Initialize
    registry = FormulaRegistry()
    anim = AnimationController()
    
    # Get a formula
    formula = registry.get_formula("mandelbulb")
    
    # Keyframe 1: Standard Mandelbulb
    kf1 = FractalParams()
    kf1.power = 8.0
    kf1.width = 400
    kf1.height = 300
    anim.add_keyframe(0.0, kf1, EasingType.EASE_IN_OUT, "Standard bulb")
    
    # Keyframe 2: High power (different structure)
    kf2 = FractalParams()
    kf2.power = 16.0  # Much higher power
    kf2.base_color = (0.9, 0.4, 0.2)  # Warmer color
    kf2.width = 400
    kf2.height = 300
    anim.add_keyframe(0.5, kf2, EasingType.CUBIC, "High power")
    
    # Keyframe 3: Low power (different structure again)
    kf3 = FractalParams()
    kf3.power = 2.0  # Low power
    kf3.camera_pos = (3.0, 0.0, 0.0)  # Different view
    kf3.base_color = (0.2, 0.5, 0.9)  # Cooler color
    kf3.width = 400
    kf3.height = 300
    anim.add_keyframe(1.0, kf3, EasingType.EASE_IN_OUT, "Low power")
    
    # Export
    print("\nExporting formula morph animation...")
    output_path = "output/formula_morph.mp4"
    Path("output").mkdir(exist_ok=True)
    
    # Quick preview render
    frames = anim.render_preview(width=320, height=240, fps=10, duration=3.0)
    print(f"Rendered {len(frames)} preview frames")
    
    # Save animation
    anim.save_animation("formula_morph.json")
    print("Saved animation to formula_morph.json")


def demo_hybrid_formula_animation():
    """Demo: Animate hybrid formula blending"""
    print("\n" + "="*60)
    print("Demo 2: Hybrid Formula Animation")
    print("="*60)
    
    registry = FormulaRegistry()
    combiner = registry.combiner if hasattr(registry, 'combiner') else None
    
    if combiner is None:
        from formulas import FormulaCombiner
        combiner = FormulaCombiner(registry)
    
    # Create hybrid
    print("\nCreating Mandelbulb-Mandelbox hybrid...")
    hybrid = combiner.create_formula_pairing("mandelbulb", "mandelbox", BlendMode.MIN)
    
    # Create animation showing blend factor change
    anim = AnimationController()
    
    # Keyframes with different blend factors (simulated via meta-params)
    for i, blend in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        kf = FractalParams()
        kf.power = 8.0
        kf.width = 400
        kf.height = 300
        # In a real implementation, you'd set hybrid-specific params
        # kf.set_meta("blend_factor", blend)
        
        t = i / 4.0
        anim.add_keyframe(t, kf, EasingType.EASE_IN_OUT, f"Blend {blend}")
    
    print(f"Created {len(anim.controller.keyframes)} keyframes for hybrid animation")
    
    # Render preview
    frames = anim.render_preview(width=320, height=240, fps=10, duration=3.0)
    print(f"Rendered hybrid preview: {len(frames)} frames")


def demo_random_formula_exploration():
    """Demo: Random formula search with animation"""
    print("\n" + "="*60)
    print("Demo 3: Random Formula Exploration")
    print("="*60)
    
    registry = FormulaRegistry()
    search = FormulaRandomSearch(registry)
    
    print("\nExploring random formula combinations...")
    interesting = search.explore_random(
        num_samples=20,  # Small number for demo
        include_hybrids=True,
        quality_threshold=0.5
    )
    
    print(f"\nFound {len(interesting)} interesting formulas")
    
    # Animate the top result
    if interesting:
        top = interesting[0]
        print(f"\nAnimating top result: {top['formula_name']}")
        
        anim = AnimationController()
        
        # Get formula
        try:
            formula = registry.get_formula(top['formula_name'])
        except:
            print("Top formula not found in registry, using mandelbulb")
            formula = registry.get_formula('mandelbulb')
        
        # Create parameter variants for animation
        base_params = formula.get_default_params()
        
        # Keyframe 1
        kf1 = FractalParams()
        kf1.power = base_params.power
        kf1.width = 400
        kf1.height = 300
        anim.add_keyframe(0.0, kf1, EasingType.EASE_IN_OUT)
        
        # Mutate for keyframe 2
        mutated = formula.mutate_params(base_params, strength=0.3)
        kf2 = FractalParams()
        kf2.power = mutated.power
        kf2.camera_pos = (2.0, 1.0, -2.0)
        kf2.width = 400
        kf2.height = 300
        anim.add_keyframe(1.0, kf2, EasingType.EASE_IN_OUT)
        
        # Render
        frames = anim.render_preview(width=320, height=240, fps=10, duration=2.0)
        print(f"Rendered animation: {len(frames)} frames")


def demo_formula_evolution():
    """Demo: Evolve formulas through mutation and crossover"""
    print("\n" + "="*60)
    print("Demo 4: Formula Evolution")
    print("="*60)
    
    registry = FormulaRegistry()
    search = FormulaRandomSearch(registry)
    
    # Start with random search
    print("\nInitial random search...")
    initial = search.explore_random(
        num_samples=10,
        include_hybrids=False,
        quality_threshold=0.4
    )
    
    if len(initial) < 2:
        print("Not enough initial formulas, using defaults")
        return
    
    # Select top 2
    sorted_results = sorted(initial, key=lambda x: x['quality_score'], reverse=True)
    parent1, parent2 = sorted_results[0], sorted_results[1]
    
    print(f"\nParent 1: {parent1['formula_name']} (score: {parent1['quality_score']:.3f})")
    print(f"Parent 2: {parent2['formula_name']} (score: {parent2['quality_score']:.3f})")
    
    # Mutate parents
    print("\nCreating mutations...")
    variants1 = search.mutate_interesting_formula(parent1, num_variants=3)
    variants2 = search.mutate_interesting_formula(parent2, num_variants=3)
    
    print(f"Created {len(variants1)} variants of parent 1")
    print(f"Created {len(variants2)} variants of parent 2")
    
    # Crossover
    print("\nCrossover between parents...")
    offspring = search.crossover_formulas(parent1, parent2)
    
    if offspring:
        print(f"Offspring: {offspring['formula_name']}")
        print(f"Score: {offspring['quality_score']:.3f}")
        if offspring.get('is_hybrid'):
            print("Type: Hybrid")
    
    # Create evolution animation
    print("\nCreating evolution animation...")
    anim = AnimationController()
    
    formulas_to_animate = [parent1] + variants1[:2] + [offsspring] if offspring else [parent1] + variants1[:2]
    
    for i, formula_data in enumerate(formulas_to_animate):
        kf = FractalParams()
        kf.power = formula_data['params']['power']
        kf.iterations = formula_data['params']['iterations']
        kf.width = 400
        kf.height = 300
        
        t = i / (len(formulas_to_animate) - 1)
        label = formula_data.get('variant_num', 'parent') if isinstance(formula_data.get('variant_num'), int) else 'parent'
        if formula_data == offspring:
            label = 'offspring'
        
        anim.add_keyframe(t, kf, EasingType.EASE_IN_OUT, str(label))
    
    frames = anim.render_preview(width=320, height=240, fps=8, duration=3.0)
    print(f"Evolution animation: {len(frames)} frames")


def demo_meta_parameter_animation():
    """Demo: Animate formula meta-parameters"""
    print("\n" + "="*60)
    print("Demo 5: Meta-Parameter Animation")
    print("="*60)
    
    registry = FormulaRegistry()
    
    # Get a formula with rich meta-parameters
    try:
        formula = registry.get_formula("amazing_box")
        print(f"Using formula: {formula.name}")
        print(f"Meta-parameters: {[mp.name for mp in formula.meta_parameters]}")
    except:
        print("Amazing box not found, using mandelbox")
        formula = registry.get_formula("mandelbox")
    
    # Create animation showing meta-parameter changes
    anim = AnimationController()
    
    # Keyframes with different "virtual" meta-parameter values
    # In real implementation, these would be actual meta-parameter settings
    params_configs = [
        {"fold_x": 0.5, "fold_y": 0.5, "scale": -1.5},
        {"fold_x": 1.0, "fold_y": 1.5, "scale": -2.0},
        {"fold_x": 1.5, "fold_y": 1.0, "scale": -1.0},
        {"fold_x": 2.0, "fold_y": 2.0, "scale": -2.5},
    ]
    
    for i, config in enumerate(params_configs):
        kf = FractalParams()
        kf.power = 8.0
        kf.width = 400
        kf.height = 300
        # In real implementation:
        # kf.set_meta("fold_x", config["fold_x"])
        # kf.set_meta("fold_y", config["fold_y"])
        # kf.set_meta("scale", config["scale"])
        
        t = i / (len(params_configs) - 1)
        anim.add_keyframe(t, kf, EasingType.EASE_IN_OUT, f"Config {i+1}")
    
    print(f"Created {len(anim.controller.keyframes)} meta-parameter animation keyframes")
    
    frames = anim.render_preview(width=320, height=240, fps=8, duration=3.0)
    print(f"Meta-parameter animation: {len(frames)} frames")


def main():
    """Run all demos"""
    print("\n" + "="*60)
    print("Formula Animation System Demo")
    print("="*60)
    print("\nThis demo shows how to use the extended formula library")
    print("with the animation system for infinite fractal variety.")
    
    try:
        demo_formula_variants()
    except Exception as e:
        print(f"Demo 1 error: {e}")
    
    try:
        demo_hybrid_formula_animation()
    except Exception as e:
        print(f"Demo 2 error: {e}")
    
    try:
        demo_random_formula_exploration()
    except Exception as e:
        print(f"Demo 3 error: {e}")
    
    try:
        demo_formula_evolution()
    except Exception as e:
        print(f"Demo 4 error: {e}")
    
    try:
        demo_meta_parameter_animation()
    except Exception as e:
        print(f"Demo 5 error: {e}")
    
    print("\n" + "="*60)
    print("All demos complete!")
    print("="*60)
    print("\nYou now have:")
    print("  - 25+ individual fractal formulas")
    print("  - Hybrid formula combinations (min, max, blend, etc.)")
    print("  - Meta-parameter control for animation")
    print("  - Random formula search and exploration")
    print("  - Formula evolution through mutation/crossover")
    print("\nFor infinite variety, combine these with the")
    print("animation system's keyframe interpolation!")


if __name__ == "__main__":
    main()
