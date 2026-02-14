#!/usr/bin/env python3
"""
Extended Fractal Formula Library - 40+ Formulas
==============================================

Comprehensive collection of 3D fractal formulas inspired by Mandelbulb3D.
Includes power variations, folding formulas, Aexion family, Benesi family,
dIFS shapes, and hybrid combinations.

Usage:
    from formulas.extended_library import FormulaRegistry
    
    registry = FormulaRegistry()
    formula = registry.get_formula("aexion_c")
    distance = formula.distance_estimate(x, y, z, params)
"""

import numpy as np
from numba import jit
from typing import Dict, List, Tuple, Any, Optional, Callable, Type, Union
from dataclasses import dataclass, field
from enum import Enum
import random
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# META-PARAMETER SYSTEM FOR FORMULA CONTROL
# =============================================================================

class MetaParameterType(Enum):
    """Types of meta-parameters for formula control"""
    POWER = "power"
    FOLDING = "folding"
    SCALING = "scaling"
    ROTATION = "rotation"
    OFFSET = "offset"
    THRESHOLD = "threshold"
    BLEND_FACTOR = "blend_factor"
    ITERATION_COUNT = "iteration_count"
    CONDITIONAL = "conditional"

@dataclass
class MetaParameter:
    """A controllable parameter for formula animation and variation"""
    name: str
    param_type: MetaParameterType
    default_value: float
    min_value: float
    max_value: float
    description: str = ""
    
    def randomize(self) -> float:
        """Generate random value within range"""
        if self.param_type == MetaParameterType.ITERATION_COUNT:
            return float(random.randint(int(self.min_value), int(self.max_value)))
        return random.uniform(self.min_value, self.max_value)
    
    def mutate(self, current_value: float, strength: float = 0.1) -> float:
        """Mutate parameter value"""
        delta = (self.max_value - self.min_value) * strength
        new_value = current_value + random.gauss(0, delta)
        return max(self.min_value, min(self.max_value, new_value))

# =============================================================================
# FORMULA PARAMETERS
# =============================================================================

@dataclass 
class FormulaParams:
    """Base parameters for any fractal formula"""
    # Core parameters
    power: float = 8.0
    iterations: int = 100
    bailout: float = 4.0
    
    # Meta-parameters dictionary for flexible control
    meta_params: Dict[str, float] = field(default_factory=dict)
    
    def get_meta(self, name: str, default: float = 0.0) -> float:
        """Get meta-parameter value"""
        return self.meta_params.get(name, default)
    
    def set_meta(self, name: str, value: float):
        """Set meta-parameter value"""
        self.meta_params[name] = value

@dataclass
class FoldingParams(FormulaParams):
    """Parameters for folding-based formulas"""
    fold_limit: float = 1.0
    fold_value: float = 2.0
    inner_radius: float = 0.5
    outer_radius: float = 1.0

@dataclass
class AexionParams(FormulaParams):
    """Parameters for Aexion formula family"""
    mode: int = 0  # Bit flags for mode variations
    phi_conditional: float = 0.0  # Conditional phi rotation

@dataclass
class BenesiParams(FormulaParams):
    """Parameters for Benesi formula family"""
    x_mul: float = 2.0
    y_mul: float = 1.0
    z_mul: float = 0.0
    conditional_enabled: bool = True

@dataclass
class dIFSParams(FormulaParams):
    """Parameters for dIFS formulas"""
    scale: float = 3.0
    offset: float = 1.0
    rotation_angle: float = 0.0
    apply_scale_add: bool = True

@dataclass
class HybridParams(FormulaParams):
    """Parameters for hybrid formulas"""
    formula1_params: FormulaParams = field(default_factory=FormulaParams)
    formula2_params: FormulaParams = field(default_factory=FormulaParams)
    blend_mode: str = "min"  # min, max, average, add, multiply
    blend_factor: float = 0.5
    threshold: float = 1.0

# =============================================================================
# CORE FORMULA CLASS
# =============================================================================

class FractalFormula:
    """Base class for all fractal formulas"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.meta_parameters: List[MetaParameter] = []
        self._define_meta_parameters()
    
    def _define_meta_parameters(self):
        """Override to define formula-specific meta-parameters"""
        self.meta_parameters = [
            MetaParameter("power", MetaParameterType.POWER, 8.0, 0.1, 30.0, "Iteration power"),
            MetaParameter("iterations", MetaParameterType.ITERATION_COUNT, 100, 10, 500, "Max iterations"),
        ]
    
    def get_default_params(self) -> FormulaParams:
        """Get default parameters"""
        return FormulaParams()
    
    def distance_estimate(self, x: float, y: float, z: float, params: FormulaParams) -> Tuple[float, float, int]:
        """Calculate distance estimate. Returns (distance, orbit_trap, iterations)"""
        raise NotImplementedError
    
    def randomize_params(self) -> FormulaParams:
        """Generate randomized parameters"""
        params = self.get_default_params()
        for meta in self.meta_parameters:
            params.set_meta(meta.name, meta.randomize())
        return params
    
    def mutate_params(self, params: FormulaParams, strength: float = 0.1) -> FormulaParams:
        """Mutate parameters"""
        new_params = FormulaParams()
        new_params.power = params.power
        new_params.iterations = params.iterations
        new_params.bailout = params.bailout
        
        for meta in self.meta_parameters:
            current = params.get_meta(meta.name, meta.default_value)
            mutated = meta.mutate(current, strength)
            new_params.set_meta(meta.name, mutated)
        
        return new_params

# =============================================================================
# POWER VARIATIONS - Extended Bulb Family
# =============================================================================

class CosineMandelbulb(FractalFormula):
    """Cosine-based Mandelbulb - uses cosine instead of polar coordinates"""
    
    def __init__(self):
        super().__init__("cosine_bulb", "Mandelbulb using cosine trigonometry")
    
    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _de(x, y, z, power, max_iter, bailout):
        xx, yy, zz = x, y, z
        dr = 1.0
        r = np.sqrt(xx*xx + yy*yy + zz*zz)
        orbit_trap = 1000.0
        i = 0
        
        for i in range(max_iter):
            if r > bailout:
                break
            
            orbit_trap = min(orbit_trap, r)
            
            # Cosine-based transformation
            r_pow = r ** power
            dr = power * r_pow * dr / r + 1.0 if r > 0.001 else 1.0
            
            # Trig calculations
            theta = np.arccos(zz / r) if r > 0.001 else 0.0
            phi = np.arctan2(yy, xx)
            
            # Cosine power formula
            xx = r_pow * np.sin(power * theta) * np.cos(power * phi) + x
            yy = r_pow * np.sin(power * theta) * np.sin(power * phi) + y
            zz = r_pow * np.cos(power * theta) + z
            
            r = np.sqrt(xx*xx + yy*yy + zz*zz)
        
        return 0.5 * np.log(r) * r / dr if dr > 0 else r, orbit_trap, i
    
    def distance_estimate(self, x, y, z, params):
        return self._de(x, y, z, params.power, params.iterations, params.bailout)

class ReciprocalMandelbulb(FractalFormula):
    """Reciprocal power Mandelbulb - negative powers create different structures"""
    
    def __init__(self):
        super().__init__("reciprocal_bulb", "Mandelbulb with negative powers")
        self.meta_parameters.append(
            MetaParameter("neg_power", MetaParameterType.POWER, -2.0, -8.0, -0.5, "Negative power")
        )
    
    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _de(x, y, z, neg_power, max_iter, bailout):
        xx, yy, zz = x, y, z
        dr = 1.0
        r = np.sqrt(xx*xx + yy*yy + zz*zz)
        orbit_trap = 1000.0
        i = 0
        
        power = abs(neg_power)
        
        for i in range(max_iter):
            if r > bailout or r < 0.001:
                break
            
            orbit_trap = min(orbit_trap, r)
            
            # Reciprocal power
            r_inv = 1.0 / (r ** power)
            dr = power * r_inv * dr / r + 1.0
            
            theta = np.arctan2(np.sqrt(xx*xx + yy*yy), zz)
            phi = np.arctan2(yy, xx)
            
            # Apply negative power
            zr = r_inv / r
            theta = theta * power
            phi = phi * power
            
            xx = zr * np.sin(theta) * np.cos(phi) + x
            yy = zr * np.sin(theta) * np.sin(phi) + y
            zz = zr * np.cos(theta) + z
            
            r = np.sqrt(xx*xx + yy*yy + zz*zz)
        
        return 0.5 * r * np.log(r + 1e-10) / dr, orbit_trap, i
    
    def distance_estimate(self, x, y, z, params):
        neg_power = params.get_meta("neg_power", -2.0)
        return self._de(x, y, z, neg_power, params.iterations, params.bailout)

class AsymmetricMandelbulb(FractalFormula):
    """Asymmetric Mandelbulb - different powers for each axis"""
    
    def __init__(self):
        super().__init__("asymmetric_bulb", "Mandelbulb with axis-specific powers")
        self.meta_parameters.extend([
            MetaParameter("power_x", MetaParameterType.POWER, 2.0, 0.1, 10.0, "Power for X axis"),
            MetaParameter("power_y", MetaParameterType.POWER, 4.0, 0.1, 10.0, "Power for Y axis"),
            MetaParameter("power_z", MetaParameterType.POWER, 8.0, 0.1, 10.0, "Power for Z axis"),
        ])
    
    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _de(x, y, z, power_x, power_y, power_z, max_iter, bailout):
        xx, yy, zz = x, y, z
        dr = 1.0
        r = np.sqrt(xx*xx + yy*yy + zz*zz)
        orbit_trap = 1000.0
        i = 0
        
        for i in range(max_iter):
            if r > bailout:
                break
            
            orbit_trap = min(orbit_trap, r)
            
            # Axis-specific powers
            theta = np.arctan2(np.sqrt(xx*xx + yy*yy), zz)
            phi = np.arctan2(yy, xx)
            
            # Apply different transformations per axis
            r_pow_x = r ** (power_x - 1.0)
            r_pow_y = r ** (power_y - 1.0)
            r_pow_z = r ** (power_z - 1.0)
            
            dr = (r_pow_x + r_pow_y + r_pow_z) * dr / 3.0 + 1.0
            
            r_pow = r ** ((power_x + power_y + power_z) / 3.0)
            theta = theta * power_z  # Use z power for angle
            phi = phi * power_z
            
            xx = r_pow * np.sin(theta) * np.cos(phi) + x
            yy = r_pow * np.sin(theta) * np.sin(phi) + y
            zz = r_pow * np.cos(theta) + z
            
            r = np.sqrt(xx*xx + yy*yy + zz*zz)
        
        return 0.5 * np.log(r) * r / dr, orbit_trap, i
    
    def distance_estimate(self, x, y, z, params):
        px = params.get_meta("power_x", 2.0)
        py = params.get_meta("power_y", 4.0)
        pz = params.get_meta("power_z", 8.0)
        return self._de(x, y, z, px, py, pz, params.iterations, params.bailout)

# =============================================================================
# FOLDING VARIATIONS
# =============================================================================

class AmazingBox(FractalFormula):
    """Amazing Box - Mandelbox variant with different fold limits"""
    
    def __init__(self):
        super().__init__("amazing_box", "Amazing Box folding fractal")
        self.meta_parameters.extend([
            MetaParameter("fold_x", MetaParameterType.FOLDING, 1.0, 0.1, 2.0, "X axis fold limit"),
            MetaParameter("fold_y", MetaParameterType.FOLDING, 1.0, 0.1, 2.0, "Y axis fold limit"),
            MetaParameter("fold_z", MetaParameterType.FOLDING, 1.0, 0.1, 2.0, "Z axis fold limit"),
            MetaParameter("scale", MetaParameterType.SCALING, -1.5, -3.0, 3.0, "Box scale factor"),
        ])
    
    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _de(x, y, z, fold_x, fold_y, fold_z, scale, max_iter, bailout):
        xx, yy, zz = x, y, z
        orbit_trap = 1000.0
        dr = 1.0
        
        for i in range(max_iter):
            r = np.sqrt(xx*xx + yy*yy + zz*zz)
            if r > bailout:
                break
            
            orbit_trap = min(orbit_trap, r)
            
            # Per-axis box folding
            if xx > fold_x:
                xx = 2.0 * fold_x - xx
            elif xx < -fold_x:
                xx = -2.0 * fold_x - xx
            
            if yy > fold_y:
                yy = 2.0 * fold_y - yy
            elif yy < -fold_y:
                yy = -2.0 * fold_y - yy
            
            if zz > fold_z:
                zz = 2.0 * fold_z - zz
            elif zz < -fold_z:
                zz = -2.0 * fold_z - zz
            
            # Sphere folding
            r = np.sqrt(xx*xx + yy*yy + zz*zz)
            if r < 0.5:
                factor = 4.0
                xx, yy, zz = xx * factor, yy * factor, zz * factor
                dr *= factor
            elif r < 1.0:
                factor = 1.0 / (r * r)
                xx, yy, zz = xx * factor, yy * factor, zz * factor
                dr *= factor
            
            # Scale and add
            xx = xx * scale + x
            yy = yy * scale + y
            zz = zz * scale + z
            dr = dr * abs(scale) + 1.0
        
        r_final = np.sqrt(xx*xx + yy*yy + zz*zz)
        return r_final * np.log(r_final + 1e-10) / dr, orbit_trap, i
    
    def distance_estimate(self, x, y, z, params):
        fx = params.get_meta("fold_x", 1.0)
        fy = params.get_meta("fold_y", 1.0)
        fz = params.get_meta("fold_z", 1.0)
        sc = params.get_meta("scale", -1.5)
        return self._de(x, y, z, fx, fy, fz, sc, params.iterations, params.bailout)

class SmoothMandelbox(FractalFormula):
    """Smooth Mandelbox - soft edges instead of sharp folds"""
    
    def __init__(self):
        super().__init__("smooth_box", "Mandelbox with smooth folds")
        self.meta_parameters.extend([
            MetaParameter("smoothness", MetaParameterType.FOLDING, 0.1, 0.01, 0.5, "Fold smoothness"),
            MetaParameter("scale", MetaParameterType.SCALING, -1.5, -3.0, 3.0, "Scale factor"),
        ])
    
    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _de(x, y, z, smoothness, scale, max_iter, bailout):
        xx, yy, zz = x, y, z
        orbit_trap = 1000.0
        dr = 1.0
        
        for i in range(max_iter):
            r = np.sqrt(xx*xx + yy*yy + zz*zz)
            if r > bailout:
                break
            
            orbit_trap = min(orbit_trap, r)
            
            # Smooth box folding using tanh
            limit = 1.0
            xx = limit * np.tanh(xx / (limit * smoothness))
            yy = limit * np.tanh(yy / (limit * smoothness))
            zz = limit * np.tanh(zz / (limit * smoothness))
            
            # Sphere folding
            r = np.sqrt(xx*xx + yy*yy + zz*zz)
            if r < 0.5:
                factor = 4.0
                xx, yy, zz = xx * factor, yy * factor, zz * factor
                dr *= factor
            elif r < 1.0:
                factor = 1.0 / (r * r)
                xx, yy, zz = xx * factor, yy * factor, zz * factor
                dr *= factor
            
            xx = xx * scale + x
            yy = yy * scale + y
            zz = zz * scale + z
            dr = dr * abs(scale) + 1.0
        
        r_final = np.sqrt(xx*xx + yy*yy + zz*zz)
        return r_final * np.log(r_final + 1e-10) / dr, orbit_trap, i
    
    def distance_estimate(self, x, y, z, params):
        sm = params.get_meta("smoothness", 0.1)
        sc = params.get_meta("scale", -1.5)
        return self._de(x, y, z, sm, sc, params.iterations, params.bailout)

# =============================================================================
# AEXION FORMULA FAMILY
# =============================================================================

class AexionC(FractalFormula):
    """Aexion C - iterates the constant instead of Z"""
    
    def __init__(self):
        super().__init__("aexion_c", "Aexion C formula - iterating the constant")
        self.meta_parameters.extend([
            MetaParameter("mode", MetaParameterType.CONDITIONAL, 0.0, 0.0, 31.0, "Mode flags (bits)"),
            MetaParameter("phi_cond", MetaParameterType.CONDITIONAL, 0.0, 0.0, 1.0, "Phi conditional"),
        ])
    
    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _de(x, y, z, power, mode, phi_cond, max_iter, bailout):
        xx, yy, zz = x, y, z
        cx, cy, cz = x, y, z  # Constant to iterate
        dr = 1.0
        r = np.sqrt(xx*xx + yy*yy + zz*zz)
        orbit_trap = 1000.0
        i = 0
        
        for i in range(max_iter):
            if r > bailout:
                break
            
            orbit_trap = min(orbit_trap, r)
            
            # Calculate angles for C
            r_c = np.sqrt(cx*cx + cy*cy + cz*cz)
            if r_c < 0.001:
                r_c = 0.001
            
            theta_c = np.arctan2(np.sqrt(cx*cx + cy*cy), cz)
            phi_c = np.arctan2(cy, cx)
            
            # Mode bit operations
            if int(mode) & 1:  # Bit 1: Flip atan theta
                theta_c = -theta_c
            if int(mode) & 2:  # Bit 2: Flip atan phi
                phi_c = -phi_c
            if int(mode) & 4:  # Bit 3: Flip both
                theta_c, phi_c = -phi_c, -theta_c
            
            # Rotate C
            r_pow = r_c ** power
            theta_c = theta_c * power
            phi_c = phi_c * power
            
            if phi_cond > 0.5:
                phi_c += r_c  # Conditional rotation
            
            # Update C
            cx = r_pow * np.sin(theta_c) * np.cos(phi_c)
            cy = r_pow * np.sin(theta_c) * np.sin(phi_c)
            cz = r_pow * np.cos(theta_c)
            
            # Iterate Z with new C
            theta = np.arctan2(np.sqrt(xx*xx + yy*yy), zz)
            phi = np.arctan2(yy, xx)
            
            zr = r ** (power - 1.0)
            dr = zr * dr * power + 1.0
            
            zr = zr * r
            theta = theta * power
            phi = phi * power
            
            xx = zr * np.sin(theta) * np.cos(phi) + cx
            yy = zr * np.sin(theta) * np.sin(phi) + cy
            zz = zr * np.cos(theta) + cz
            
            r = np.sqrt(xx*xx + yy*yy + zz*zz)
        
        return 0.5 * np.log(r) * r / dr, orbit_trap, i
    
    def distance_estimate(self, x, y, z, params):
        mode = params.get_meta("mode", 0.0)
        phi_cond = params.get_meta("phi_cond", 0.0)
        return self._de(x, y, z, params.power, mode, phi_cond, params.iterations, params.bailout)

class AexOcto(FractalFormula):
    """Aex-Octo - Octopus-like structures, best in Julia mode"""
    
    def __init__(self):
        super().__init__("aex_octo", "Aex-Octo formula - octopus structures")
        self.meta_parameters.extend([
            MetaParameter("xz_mul", MetaParameterType.SCALING, 1.0, -2.0, 2.0, "XZ multiplier"),
            MetaParameter("sq_mul", MetaParameterType.SCALING, 1.0, -2.0, 2.0, "Square multiplier"),
        ])
    
    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _de(x, y, z, xz_mul, sq_mul, max_iter, bailout):
        xx, yy, zz = x, y, z
        dr = 1.0
        r = np.sqrt(xx*xx + yy*yy + zz*zz)
        orbit_trap = 1000.0
        i = 0
        
        for i in range(max_iter):
            if r > bailout:
                break
            
            orbit_trap = min(orbit_trap, r)
            
            # Octo formula
            new_x = xz_mul * xx * zz + x
            new_y = sq_mul * (-xx*xx + zz*zz) + y
            new_z = yy + z
            
            # Derivative approximation
            dx = xz_mul * (xx + zz) + 1.0
            dy = sq_mul * (-2.0 * xx + 2.0 * zz)
            dz = 1.0
            dr = np.sqrt(dx*dx + dy*dy + dz*dz) + 1.0
            
            xx, yy, zz = new_x, new_y, new_z
            r = np.sqrt(xx*xx + yy*yy + zz*zz)
        
        return 0.5 * r * np.log(r + 1e-10) / dr, orbit_trap, i
    
    def distance_estimate(self, x, y, z, params):
        xz = params.get_meta("xz_mul", 1.0)
        sq = params.get_meta("sq_mul", 1.0)
        return self._de(x, y, z, xz, sq, params.iterations, params.bailout)

# =============================================================================
# BENESI FORMULA FAMILY
# =============================================================================

class Benesi2Pow2(FractalFormula):
    """Benesi 2 Pow 2 - Simplified but beautiful formula"""
    
    def __init__(self):
        super().__init__("benesi2pow2", "Benesi 2^2 formula - simplified beauty")
        self.meta_parameters.extend([
            MetaParameter("x_mul", MetaParameterType.SCALING, 2.0, -3.0, 3.0, "X multiplier"),
            MetaParameter("y_mul", MetaParameterType.SCALING, 1.0, -3.0, 3.0, "Y multiplier"),
            MetaParameter("z_mul", MetaParameterType.SCALING, 0.0, -3.0, 3.0, "Z multiplier"),
        ])
    
    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _de(x, y, z, x_mul, y_mul, z_mul, max_iter, bailout):
        xx, yy, zz = x, y, z
        dr = 1.0
        r = np.sqrt(xx*xx + yy*yy + zz*zz)
        orbit_trap = 1000.0
        i = 0
        
        for i in range(max_iter):
            if r > bailout:
                break
            
            orbit_trap = min(orbit_trap, r)
            
            # Benesi formula
            r1 = yy*yy + zz*zz
            
            if x < 0 or xx < np.sqrt(r1):
                xx = xx*xx - r1 + x
            else:
                xx = -xx*xx + r1 + x
            
            if r1 > 0.001:
                r1_inv = 1.0 / np.sqrt(r1)
            else:
                r1_inv = 1.0
            
            temp = -r1_inv * 2.0 * abs(xx)
            yy = temp * (yy*yy - zz*zz) * y_mul + y
            zz = temp * 2.0 * yy * zz * z_mul + z
            
            # Scale X
            xx = xx * x_mul
            
            dr = dr * 2.0 * r + 1.0
            r = np.sqrt(xx*xx + yy*yy + zz*zz)
        
        return 0.5 * np.log(r) * r / dr, orbit_trap, i
    
    def distance_estimate(self, x, y, z, params):
        xm = params.get_meta("x_mul", 2.0)
        ym = params.get_meta("y_mul", 1.0)
        zm = params.get_meta("z_mul", 0.0)
        return self._de(x, y, z, xm, ym, zm, params.iterations, params.bailout)

class Benesi3Pow2(FractalFormula):
    """Benesi 3 Pow 2 - Conditional folding variant"""
    
    def __init__(self):
        super().__init__("benesi3pow2", "Benesi 3^2 with conditional folding")
        self.meta_parameters.extend([
            MetaParameter("conditional", MetaParameterType.CONDITIONAL, 1.0, 0.0, 1.0, "Enable conditional folding"),
        ])
    
    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _de(x, y, z, conditional, max_iter, bailout):
        xx, yy, zz = x, y, z
        dr = 1.0
        r = np.sqrt(xx*xx + yy*yy + zz*zz)
        orbit_trap = 1000.0
        i = 0
        
        for i in range(max_iter):
            if r > bailout:
                break
            
            orbit_trap = min(orbit_trap, r)
            
            # Conditional folding
            if conditional > 0.5:
                r1 = yy*yy + zz*zz
                if x < 0 or xx < np.sqrt(r1):
                    xx = xx*xx - r1 + x
                else:
                    xx = -xx*xx + r1 + x
                
                # Update r1 after xx change
                r1 = yy*yy + zz*zz
                if r1 > 0.001:
                    r1 = -1.0 / np.sqrt(r1) * 2.0 * abs(xx)
                    yy = r1 * (yy*yy - zz*zz) + y
                    zz = r1 * 2.0 * yy * zz + z
            else:
                # Simplified version
                xx = xx*xx - yy*yy - zz*zz + x
                yy = 2.0 * xx * yy + y
                zz = 2.0 * xx * zz + z
            
            dr = dr * 2.0 * r + 1.0
            r = np.sqrt(xx*xx + yy*yy + zz*zz)
        
        return 0.5 * np.log(r) * r / dr, orbit_trap, i
    
    def distance_estimate(self, x, y, z, params):
        cond = params.get_meta("conditional", 1.0)
        return self._de(x, y, z, cond, params.iterations, params.bailout)

# Continue with more formulas in next files...
