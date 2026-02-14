#!/usr/bin/env python3
"""
Hybrid Formula System - Combining Formulas for Infinite Variety
===============================================================

Enables mixing multiple fractal formulas in various ways:
- Distance blending (min, max, average, add, multiply)
- Iteration alternating (switch formulas per iteration)
- Conditional switching (based on position/radius)
- Smooth transitions (morph between formulas)

Usage:
    from formulas.hybrid_system import HybridFormula, BlendMode
    
    # Create hybrid
    hybrid = HybridFormula(formula1, formula2, mode=BlendMode.MIN)
    
    # Or use the formula combiner
    from formulas.formula_combiner import FormulaCombiner
    combiner = FormulaCombiner(registry)
    random_hybrid = combiner.create_random_hybrid()
"""

import numpy as np
from numba import jit
from typing import List, Tuple, Optional, Union
from enum import Enum
from dataclasses import dataclass
import random

from .extended_library import FractalFormula, FormulaParams, HybridParams, MetaParameter, MetaParameterType


class BlendMode(Enum):
    """Different ways to combine formula distances"""
    MIN = "min"  # Minimum distance (union)
    MAX = "max"  # Maximum distance (intersection)
    AVERAGE = "average"  # Average of distances
    ADD = "add"  # Additive blending
    MULTIPLY = "multiply"  # Multiplicative blending
    SUBTRACT = "subtract"  # Subtractive blending
    SMOOTH_MIN = "smooth_min"  # Smooth minimum (blends at intersection)
    SMOOTH_MAX = "smooth_max"  # Smooth maximum
    POWER_BLEND = "power_blend"  # Blend based on power function


class HybridType(Enum):
    """Types of hybrid combinations"""
    BLEND = "blend"  # Blend distances each iteration
    ALTERNATE = "alternate"  # Alternate formulas per iteration
    CONDITIONAL = "conditional"  # Switch based on conditions
    MORPH = "morph"  # Morph parameters between formulas
    SEQUENCE = "sequence"  # Apply formulas in sequence


@dataclass
class HybridConfig:
    """Configuration for hybrid formula"""
    mode: BlendMode = BlendMode.MIN
    hybrid_type: HybridType = HybridType.BLEND
    blend_factor: float = 0.5  # For blending modes
    smoothness: float = 0.1  # For smooth min/max
    threshold: float = 1.0  # For conditional switching
    alternate_freq: int = 1  # Iterations before alternating


class HybridFormula(FractalFormula):
    """
    Combines two or more fractal formulas into a hybrid.
    
    Supports various combination modes for creating unique fractal structures.
    """
    
    def __init__(self, formulas: List[FractalFormula], name: str = None, 
                 config: HybridConfig = None):
        """
        Create hybrid formula.
        
        Args:
            formulas: List of formulas to combine (2 or more)
            name: Optional name for the hybrid
            config: Hybrid configuration
        """
        if len(formulas) < 2:
            raise ValueError("Need at least 2 formulas to create hybrid")
        
        self.formulas = formulas
        self.config = config or HybridConfig()
        
        # Generate name if not provided
        if name is None:
            name = self._generate_name()
        
        super().__init__(name, f"Hybrid of {len(formulas)} formulas")
        
        # Define meta-parameters
        self._define_meta_parameters()
    
    def _generate_name(self) -> str:
        """Generate name from component formulas"""
        names = [f.name for f in self.formulas]
        if len(names) == 2:
            return f"{names[0]}_{names[1]}_{self.config.mode.value}"
        else:
            return f"hybrid_{len(names)}_{self.config.mode.value}"
    
    def _define_meta_parameters(self):
        """Define meta-parameters for the hybrid"""
        self.meta_parameters = [
            MetaParameter("blend_factor", MetaParameterType.BLEND_FACTOR, 
                         self.config.blend_factor, 0.0, 1.0, "Blending amount"),
            MetaParameter("smoothness", MetaParameterType.FOLDING, 
                         self.config.smoothness, 0.01, 0.5, "Smooth blend factor"),
            MetaParameter("threshold", MetaParameterType.THRESHOLD, 
                         self.config.threshold, 0.1, 5.0, "Conditional threshold"),
            MetaParameter("morph_t", MetaParameterType.BLEND_FACTOR, 
                         0.5, 0.0, 1.0, "Morph position between formulas"),
        ]
    
    def get_default_params(self) -> HybridParams:
        """Get default hybrid parameters"""
        return HybridParams(
            formula1_params=self.formulas[0].get_default_params(),
            formula2_params=self.formulas[1].get_default_params() if len(self.formulas) > 1 else FormulaParams(),
            blend_mode=self.config.mode.value,
            blend_factor=self.config.blend_factor,
            threshold=self.config.threshold
        )
    
    def distance_estimate(self, x: float, y: float, z: float, params: HybridParams) -> Tuple[float, float, int]:
        """Calculate hybrid distance estimate"""
        if self.config.hybrid_type == HybridType.BLEND:
            return self._blend_distances(x, y, z, params)
        elif self.config.hybrid_type == HybridType.ALTERNATE:
            return self._alternate_iterations(x, y, z, params)
        elif self.config.hybrid_type == HybridType.CONDITIONAL:
            return self._conditional_switch(x, y, z, params)
        elif self.config.hybrid_type == HybridType.MORPH:
            return self._morph_formulas(x, y, z, params)
        else:
            return self._blend_distances(x, y, z, params)
    
    def _blend_distances(self, x, y, z, params: HybridParams) -> Tuple[float, float, int]:
        """Blend distances from multiple formulas"""
        distances = []
        orbit_traps = []
        iterations = []
        
        # Get distances from all formulas
        for i, formula in enumerate(self.formulas):
            if i == 0:
                de, ot, it = formula.distance_estimate(x, y, z, params.formula1_params)
            elif i == 1:
                de, ot, it = formula.distance_estimate(x, y, z, params.formula2_params)
            else:
                # Use default params for additional formulas
                de, ot, it = formula.distance_estimate(x, y, z, FormulaParams())
            
            distances.append(de)
            orbit_traps.append(ot)
            iterations.append(it)
        
        # Blend based on mode
        blend_factor = params.get_meta("blend_factor", self.config.blend_factor)
        smoothness = params.get_meta("smoothness", self.config.smoothness)
        
        if self.config.mode == BlendMode.MIN:
            final_de = min(distances)
        elif self.config.mode == BlendMode.MAX:
            final_de = max(distances)
        elif self.config.mode == BlendMode.AVERAGE:
            final_de = sum(distances) / len(distances)
        elif self.config.mode == BlendMode.ADD:
            final_de = sum(distances)
        elif self.config.mode == BlendMode.MULTIPLY:
            final_de = 1.0
            for d in distances:
                final_de *= d
        elif self.config.mode == BlendMode.SUBTRACT:
            final_de = distances[0] - sum(distances[1:])
        elif self.config.mode == BlendMode.SMOOTH_MIN:
            final_de = self._smooth_min(distances, smoothness)
        elif self.config.mode == BlendMode.SMOOTH_MAX:
            final_de = self._smooth_max(distances, smoothness)
        elif self.config.mode == BlendMode.POWER_BLEND:
            final_de = self._power_blend(distances, blend_factor)
        else:
            final_de = min(distances)
        
        final_ot = min(orbit_traps)
        final_it = max(iterations)
        
        return final_de, final_ot, final_it
    
    @staticmethod
    def _smooth_min(distances: List[float], k: float) -> float:
        """Smooth minimum function for blending"""
        if not distances:
            return 0.0
        
        # Exponential smooth min
        exp_sum = sum(np.exp(-d / k) for d in distances)
        if exp_sum > 0:
            return -k * np.log(exp_sum / len(distances))
        return min(distances)
    
    @staticmethod
    def _smooth_max(distances: List[float], k: float) -> float:
        """Smooth maximum function for blending"""
        if not distances:
            return 0.0
        
        # Exponential smooth max
        exp_sum = sum(np.exp(d / k) for d in distances)
        if exp_sum > 0:
            return k * np.log(exp_sum / len(distances))
        return max(distances)
    
    @staticmethod
    def _power_blend(distances: List[float], factor: float) -> float:
        """Power-based blending"""
        if len(distances) < 2:
            return distances[0] if distances else 0.0
        
        # Blend based on power function
        d1, d2 = distances[0], distances[1]
        return d1 * (1 - factor) + d2 * factor
    
    def _alternate_iterations(self, x, y, z, params: HybridParams) -> Tuple[float, float, int]:
        """Alternate between formulas per iteration"""
        # Use the first formula's iteration method but switch
        xx, yy, zz = x, y, z
        dr = 1.0
        r = np.sqrt(xx*xx + yy*yy + zz*zz)
        orbit_trap = 1000.0
        
        alternate_freq = int(params.get_meta("alternate_freq", self.config.alternate_freq))
        
        for i in range(params.iterations):
            if r > params.bailout:
                break
            
            orbit_trap = min(orbit_trap, r)
            
            # Choose formula based on iteration
            formula_idx = (i // alternate_freq) % len(self.formulas)
            formula = self.formulas[formula_idx]
            
            # Use appropriate params
            if formula_idx == 0:
                fp = params.formula1_params
            elif formula_idx == 1:
                fp = params.formula2_params
            else:
                fp = FormulaParams()
            
            # Single iteration step (approximate)
            de, _, _ = formula.distance_estimate(xx, yy, zz, fp)
            
            if de < 0.001:
                break
            
            # March along distance field
            step = max(de, 0.001)
            # Simple approximation - move in random direction scaled by step
            xx += step * np.sin(i) * 0.1
            yy += step * np.cos(i) * 0.1
            zz += step * np.sin(i * 1.5) * 0.1
            
            r = np.sqrt(xx*xx + yy*yy + zz*zz)
            dr = dr + 1.0
        
        return 0.5 * r * np.log(r + 1e-10) / dr, orbit_trap, i
    
    def _conditional_switch(self, x, y, z, params: HybridParams) -> Tuple[float, float, int]:
        """Switch formulas based on position conditions"""
        r = np.sqrt(x*x + y*y + z*z)
        threshold = params.get_meta("threshold", self.config.threshold)
        
        # Choose formula based on radius
        if r < threshold:
            formula = self.formulas[0]
            fp = params.formula1_params
        else:
            formula = self.formulas[1] if len(self.formulas) > 1 else self.formulas[0]
            fp = params.formula2_params
        
        return formula.distance_estimate(x, y, z, fp)
    
    def _morph_formulas(self, x, y, z, params: HybridParams) -> Tuple[float, float, int]:
        """Morph between formulas based on morph_t parameter"""
        morph_t = params.get_meta("morph_t", 0.5)
        
        # Get distances from both formulas
        de1, ot1, it1 = self.formulas[0].distance_estimate(x, y, z, params.formula1_params)
        de2, ot2, it2 = self.formulas[1].distance_estimate(x, y, z, params.formula2_params) if len(self.formulas) > 1 else (de1, ot1, it1)
        
        # Morph the distances
        final_de = de1 * (1 - morph_t) + de2 * morph_t
        final_ot = min(ot1, ot2)
        final_it = max(it1, it2)
        
        return final_de, final_ot, final_it


class FormulaStack(FractalFormula):
    """
    Stack multiple formulas sequentially.
    Each formula transforms the space before the next.
    """
    
    def __init__(self, formulas: List[FractalFormula], name: str = None):
        self.formulas = formulas
        
        if name is None:
            name = "stack_" + "_".join(f.name[:4] for f in formulas)
        
        super().__init__(name, f"Stack of {len(formulas)} formulas")
    
    def distance_estimate(self, x, y, z, params: FormulaParams) -> Tuple[float, float, int]:
        """Apply formulas in sequence"""
        xx, yy, zz = x, y, z
        total_orbit_trap = 1000.0
        max_iterations = 0
        
        # Apply each formula in sequence
        for i, formula in enumerate(self.formulas):
            # Get appropriate params for this formula
            if isinstance(params, HybridParams):
                if i == 0:
                    fp = params.formula1_params
                elif i == 1:
                    fp = params.formula2_params
                else:
                    fp = FormulaParams()
            else:
                fp = params
            
            de, ot, it = formula.distance_estimate(xx, yy, zz, fp)
            
            total_orbit_trap = min(total_orbit_trap, ot)
            max_iterations = max(max_iterations, it)
            
            # Move point based on distance (simplified transformation)
            if de > 0.001:
                scale = min(de, 0.5)
                xx += scale * np.sin(i * 0.5)
                yy += scale * np.cos(i * 0.5)
                zz += scale * np.sin(i * 0.7)
        
        r = np.sqrt(xx*xx + yy*yy + zz*zz)
        return r * 0.5, total_orbit_trap, max_iterations


class FormulaCombiner:
    """
    System for combining formulas in various ways.
    Provides random hybrid generation and formula mixing.
    """
    
    def __init__(self, formula_registry):
        """
        Initialize with a formula registry.
        
        Args:
            formula_registry: Registry containing available formulas
        """
        self.registry = formula_registry
        self.blend_modes = list(BlendMode)
        self.hybrid_types = list(HybridType)
    
    def create_random_hybrid(self, num_formulas: int = 2) -> HybridFormula:
        """
        Create a random hybrid formula.
        
        Args:
            num_formulas: Number of formulas to combine (2-4)
            
        Returns:
            Random hybrid formula
        """
        num_formulas = max(2, min(4, num_formulas))
        
        # Select random formulas
        available = list(self.registry.formulas.keys())
        if len(available) < num_formulas:
            num_formulas = len(available)
        
        selected_names = random.sample(available, num_formulas)
        formulas = [self.registry.get_formula(name) for name in selected_names]
        
        # Random configuration
        mode = random.choice(self.blend_modes)
        hybrid_type = random.choice(self.hybrid_types)
        
        config = HybridConfig(
            mode=mode,
            hybrid_type=hybrid_type,
            blend_factor=random.uniform(0.1, 0.9),
            smoothness=random.uniform(0.05, 0.3),
            threshold=random.uniform(0.5, 3.0)
        )
        
        # Create hybrid
        name = f"random_{formulas[0].name}_{formulas[1].name}_{mode.value}"
        hybrid = HybridFormula(formulas, name=name, config=config)
        
        return hybrid
    
    def create_formula_pairing(self, formula1_name: str, formula2_name: str,
                              mode: BlendMode = BlendMode.MIN) -> HybridFormula:
        """Create a specific hybrid from two named formulas"""
        f1 = self.registry.get_formula(formula1_name)
        f2 = self.registry.get_formula(formula2_name)
        
        config = HybridConfig(mode=mode)
        return HybridFormula([f1, f2], config=config)
    
    def create_formula_stack(self, formula_names: List[str]) -> FormulaStack:
        """Create a formula stack from named formulas"""
        formulas = [self.registry.get_formula(name) for name in formula_names]
        return FormulaStack(formulas)
    
    def create_popular_hybrids(self) -> List[HybridFormula]:
        """Create list of popular/predefined hybrid combinations"""
        hybrids = []
        
        # Mandelbulb + Mandelbox (Bulbox)
        try:
            bulbox = self.create_formula_pairing("mandelbulb", "mandelbox", BlendMode.MIN)
            bulbox.name = "bulbox"
            bulbox.description = "Popular Mandelbulb-Mandelbox hybrid"
            hybrids.append(bulbox)
        except:
            pass
        
        # Buffalo + Celtic
        try:
            bc = self.create_formula_pairing("buffalo", "celtic", BlendMode.AVERAGE)
            bc.name = "buffalo_celtic"
            hybrids.append(bc)
        except:
            pass
        
        # Julia + Mandelbulb
        try:
            jm = self.create_formula_pairing("julia", "mandelbulb", BlendMode.MAX)
            jm.name = "julia_mandelbulb"
            hybrids.append(jm)
        except:
            pass
        
        # Burning Ship + Mandelbox
        try:
            bs_box = self.create_formula_pairing("burning_ship", "mandelbox", BlendMode.SMOOTH_MIN)
            bs_box.name = "burningship_box"
            hybrids.append(bs_box)
        except:
            pass
        
        return hybrids
    
    def mutate_hybrid(self, hybrid: HybridFormula, strength: float = 0.1) -> HybridFormula:
        """Create mutated version of a hybrid"""
        # Mutate configuration
        new_config = HybridConfig(
            mode=hybrid.config.mode,
            hybrid_type=hybrid.config.hybrid_type,
            blend_factor=np.clip(hybrid.config.blend_factor + random.gauss(0, strength), 0.0, 1.0),
            smoothness=np.clip(hybrid.config.smoothness + random.gauss(0, strength * 0.1), 0.01, 0.5),
            threshold=np.clip(hybrid.config.threshold + random.gauss(0, strength), 0.1, 5.0)
        )
        
        # Create new hybrid with mutated config
        new_hybrid = HybridFormula(
            hybrid.formulas,
            name=hybrid.name + "_mutated",
            config=new_config
        )
        
        return new_hybrid
    
    def crossover_hybrids(self, hybrid1: HybridFormula, hybrid2: HybridFormula) -> HybridFormula:
        """Crossover two hybrids to create offspring"""
        # Mix formula lists
        all_formulas = hybrid1.formulas + hybrid2.formulas
        if len(all_formulas) > 2:
            selected = random.sample(all_formulas, 2)
        else:
            selected = all_formulas
        
        # Mix configurations
        mode = random.choice([hybrid1.config.mode, hybrid2.config.mode])
        blend_factor = (hybrid1.config.blend_factor + hybrid2.config.blend_factor) / 2
        
        config = HybridConfig(
            mode=mode,
            blend_factor=blend_factor
        )
        
        name = f"cross_{hybrid1.name[:4]}_{hybrid2.name[:4]}"
        return HybridFormula(selected, name=name, config=config)
