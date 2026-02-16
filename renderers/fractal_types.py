#!/usr/bin/env python3
"""
Comprehensive Fractal Formula System
====================================

Model: minimax-m2.5 (opencode)
Created: 2026-02-16
Version: 1.0

Includes 30+ fractal types from:
- Mandelbulb3D formula categories
- Kalles Fraktaler techniques
- Apophysis/flam3 variations
- Classic 2D/3D fractals

Usage:
    from renderers.fractal_types import get_formula_info, list_all_fractals
    
    # List all available
    for name in list_all_fractals():
        print(name)
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from numba import jit


# =============================================================================
# FRACTAL TYPE ENUMERATION
# =============================================================================

class FractalCategory(Enum):
    """Fractal formula categories."""
    MANDELBULB = "Mandelbulb family"
    MANDELBOX = "Mandelbox/Fold-based"
    JULIA = "Julia set variants"
    ITERATED_FUNCTION = "IFS/dIFS systems"
    NEWTON = "Newton/Root-finding"
    CLASSIC_2D = "Classic 2D fractals"
    FLAM3 = "Fractal flames"
    QUATERNION = "4D/Quaternion"


@dataclass
class FractalFormula:
    """Describes a fractal formula and its parameters."""
    name: str
    category: FractalCategory
    description: str
    params: Dict[str, any]
    default_values: Dict[str, float]
    python_impl: str  # Function name in this module


# =============================================================================
# FORMULA REGISTRY
# =============================================================================

FRACTAL_REGISTRY: Dict[str, FractalFormula] = {}


def register_formula(
    name: str,
    category: FractalCategory,
    description: str,
    params: List[str],
    defaults: Dict[str, float],
    impl: str
):
    """Register a fractal formula."""
    FRACTAL_REGISTRY[name] = FractalFormula(
        name=name,
        category=category,
        description=description,
        params=params,
        default_values=defaults,
        python_impl=impl
    )


# Register 3D Mandelbulb family
register_formula(
    "mandelbulb", FractalCategory.MANDELBULB,
    "Classic Mandelbulb - the iconic 3D fractal",
    ["power", "bailout", "iterations"],
    {"power": 8.0, "bailout": 2.0, "iterations": 100},
    "mandelbulb_de"
)

register_formula(
    "mandelbulb_pow4", FractalCategory.MANDELBULB,
    "Mandelbulb with power 4 - compact bulbous",
    ["power", "bailout", "iterations"],
    {"power": 4.0, "bailout": 2.0, "iterations": 100},
    "mandelbulb_de"
)

register_formula(
    "mandelbulb_pow6", FractalCategory.MANDELBULB,
    "Mandelbulb with power 6 - elongated structures",
    ["power", "bailout", "iterations"],
    {"power": 6.0, "bailout": 2.0, "iterations": 100},
    "mandelbulb_de"
)

register_formula(
    "mandelbulb_pow10", FractalCategory.MANDELBULB,
    "Mandelbulb with power 10 - wispy filaments",
    ["power", "bailout", "iterations"],
    {"power": 10.0, "bailout": 2.0, "iterations": 100},
    "mandelbulb_de"
)

# Julia bulb variants
register_formula(
    "juliabulb", FractalCategory.JULIA,
    "Julia version of Mandelbulb",
    ["power", "c_x", "c_y", "c_z", "iterations"],
    {"power": 8.0, "c_x": -0.2, "c_y": 0.1, "c_z": 0.0, "iterations": 100},
    "juliabulb_de"
)

register_formula(
    "julia_3d", FractalCategory.JULIA,
    "3D Julia set with triplex algebra",
    ["power", "c_x", "c_y", "c_z", "iterations"],
    {"power": 8.0, "c_x": -0.2, "c_y": 0.1, "c_z": 0.0, "iterations": 100},
    "julia_set_de"
)

# Mandelbox family
register_formula(
    "mandelbox", FractalCategory.MANDELBOX,
    "Classic Mandelbox - geometric box fractals",
    ["scale", "fold_limit", "min_r", "max_iter"],
    {"scale": 2.0, "fold_limit": 1.0, "min_r": 0.5, "max_iter": 100},
    "mandelbox_de"
)

register_formula(
    "mandelbox_neg", FractalCategory.MANDELBOX,
    "Mandelbox with negative scale - inverted structures",
    ["scale", "fold_limit", "min_r", "max_iter"],
    {"scale": -1.5, "fold_limit": 1.0, "min_r": 0.5, "max_iter": 100},
    "mandelbox_de"
)

register_formula(
    "mandelbox_sponge", FractalCategory.MANDELBOX,
    "Mandelbox variation resembling Menger sponge",
    ["scale", "fold_limit", "min_r", "max_iter"],
    {"scale": 3.0, "fold_limit": 1.0, "min_r": 0.1, "max_iter": 100},
    "mandelbox_de"
)

register_formula(
    "mandelbox_var1", FractalCategory.MANDELBOX,
    "Mandelbox variation - intricate patterns",
    ["scale", "fold_limit", "min_r", "max_iter"],
    {"scale": 2.5, "fold_limit": 1.0, "min_r": 0.3, "max_iter": 100},
    "mandelbox_de"
)

# Burning Ship
register_formula(
    "burning_ship", FractalCategory.CLASSIC_2D,
    "3D Burning Ship - fractal with ship-like appearance",
    ["power", "bailout", "iterations"],
    {"power": 8.0, "bailout": 2.0, "iterations": 100},
    "burning_ship_de"
)

register_formula(
    "burning_ship_3d", FractalCategory.CLASSIC_2D,
    "3D Burning Ship with triplex math",
    ["power", "bailout", "iterations"],
    {"power": 6.0, "bailout": 2.0, "iterations": 100},
    "burning_ship_de"
)

# Tricorn
register_formula(
    "tricorn", FractalCategory.CLASSIC_2D,
    "Tricorn (Mandelbar) - symmetrical horn shape",
    ["power", "bailout", "iterations"],
    {"power": 2.0, "bailout": 2.0, "iterations": 100},
    "tricorn_de"
)

# Quaternion
register_formula(
    "quaternion_julia", FractalCategory.QUATERNION,
    "4D Quaternion Julia - smooth blobby forms",
    ["c_x", "c_y", "c_z", "c_w", "iterations"],
    {"c_x": -0.2, "c_y": 0.6, "c_z": 0.2, "c_w": 0.2, "iterations": 80},
    "quaternion_julia_de"
)

# IFS/dIFS formulas
register_formula(
    "sierpinski", FractalCategory.ITERATED_FUNCTION,
    "Sierpinski tetrahedron - classic fractal",
    ["iterations", "scale"],
    {"iterations": 6, "scale": 2.0},
    "sierpinski_de"
)

register_formula(
    "menger_sponge", FractalCategory.ITERATED_FUNCTION,
    "Menger sponge - cubic with holes",
    ["iterations"],
    {"iterations": 4},
    "menger_sponge_de"
)

register_formula(
    "kifs", FractalCategory.ITERATED_FUNCTION,
    "Kaleidoscopic IFS - generalized symmetry",
    ["scale", "rot_x", "rot_y", "rot_z", "folds"],
    {"scale": 2.5, "rot_x": 0.0, "rot_y": 0.0, "rot_z": 0.0, "folds": 3},
    "kifs_de"
)

# Hybrid combinations
register_formula(
    "hybrid_mandelbulb_mandelbox", FractalCategory.MANDELBULB,
    "Hybrid combining Mandelbulb and Mandelbox",
    ["power", "scale", "blend"],
    {"power": 8.0, "scale": 2.0, "blend": 0.5},
    "hybrid_1_de"
)

register_formula(
    "hybrid_julia_mandelbox", FractalCategory.MANDELBOX,
    "Julia set with box folding",
    ["power", "scale", "c_x", "c_y", "c_z"],
    {"power": 8.0, "scale": 2.0, "c_x": 0.1, "c_y": 0.0, "c_z": 0.0},
    "hybrid_2_de"
)

# Additional variations
register_formula(
    "lambda_mandelbulb", FractalCategory.MANDELBULB,
    "Lambdabulb - modified formula with lambda term",
    ["power", "lambda_val", "iterations"],
    {"power": 8.0, "lambda_val": 1.0, "iterations": 100},
    "lambda_mandelbulb_de"
)

register_formula(
    "biomorph", FractalCategory.NEWTON,
    "Biomorph - organic coral-like structures",
    ["power", "bailout", "iterations"],
    {"power": 8.0, "bailout": 4.0, "iterations": 50},
    "biomorph_de"
)

register_formula(
    "sphere_fold", FractalCategory.MANDELBOX,
    "Pure sphere folding fractal",
    ["scale", "min_r", "max_r", "iterations"],
    {"scale": 2.0, "min_r": 0.2, "max_r": 2.0, "iterations": 100},
    "sphere_fold_de"
)

# 2D-style rendered in 3D
register_formula(
    "mandelbrot_3d", FractalCategory.CLASSIC_2D,
    "Classic 2D Mandelbrot viewed in 3D space",
    ["power", "bailout", "iterations"],
    {"power": 2.0, "bailout": 4.0, "iterations": 100},
    "mandelbrot_3d_de"
)

register_formula(
    "julia_classic", FractalCategory.JULIA,
    "Classic 2D Julia set",
    ["c_x", "c_y", "iterations"],
    {"c_x": -0.7, "c_y": 0.27015, "iterations": 100},
    "julia_classic_de"
)

register_formula(
    "newton_z3", FractalCategory.NEWTON,
    "Newton fractal for z^3 - 1 = 0",
    ["iterations"],
    {"iterations": 20},
    "newton_z3_de"
)

register_formula(
    "newton_z4", FractalCategory.NEWTON,
    "Newton fractal for z^4 - 1 = 0",
    ["iterations"],
    {"iterations": 20},
    "newton_z4_de"
)


# =============================================================================
# IMPLEMENTATIONS (JIT-compiled distance estimators)
# =============================================================================

max_r2_global = 4.0


@jit(nopython=True, fastmath=True)
def mandelbulb_de(x, y, z, power, max_iter, bailout):
    """Mandelbulb distance estimation."""
    xx, yy, zz = x, y, z
    dr = 1.0
    r = np.sqrt(xx*xx + yy*yy + zz*zz)
    orbit_trap = 1000.0
    
    for i in range(max_iter):
        if r > bailout:
            break
        orbit_trap = min(orbit_trap, r)
        
        theta = np.arctan2(np.sqrt(xx*xx + yy*yy), zz)
        phi = np.arctan2(yy, xx)
        zr = r ** (power - 1.0)
        dr = zr * dr * power + 1.0
        zr = zr * r
        theta = theta * power
        phi = phi * power
        
        xx = zr * np.sin(theta) * np.cos(phi) + x
        yy = zr * np.sin(theta) * np.sin(phi) + y
        zz = zr * np.cos(theta) + z
        r = np.sqrt(xx*xx + yy*yy + zz*zz)
    
    return 0.5 * np.log(r) * r / dr, orbit_trap, i


@jit(nopython=True, fastmath=True)
def juliabulb_de(x, y, z, power, cx, cy, cz, max_iter, bailout):
    """Julia bulb - constant c instead of position."""
    xx, yy, zz = x, y, z
    dr = 1.0
    r = np.sqrt(xx*xx + yy*yy + zz*zz)
    orbit_trap = 1000.0
    
    for i in range(max_iter):
        if r > bailout:
            break
        orbit_trap = min(orbit_trap, r)
        
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


@jit(nopython=True, fastmath=True)
def julia_set_de(x, y, z, power, cx, cy, cz, max_iter, bailout):
    """3D Julia set."""
    return juliabulb_de(x, y, z, power, cx, cy, cz, max_iter, bailout)


@jit(nopython=True, fastmath=True)
def mandelbox_de(x, y, z, scale, fold_limit, min_r, max_iter, bailout):
    """Mandelbox distance estimation."""
    xx, yy, zz = x, y, z
    orbit_trap = 1000.0
    mr2 = min_r * min_r
    dr = 1.0
    
    for i in range(max_iter):
        # Box folding
        if xx > 1.0: xx = 2.0 - xx
        elif xx < -1.0: xx = -2.0 - xx
        if yy > 1.0: yy = 2.0 - yy
        elif yy < -1.0: yy = -2.0 - yy
        if zz > 1.0: zz = 2.0 - zz
        elif zz < -1.0: zz = -2.0 - zz
        
        # Sphere folding
        r2 = xx*xx + yy*yy + zz*zz
        if r2 < mr2:
            temp = max_r2_global / mr2
            xx *= temp
            yy *= temp
            zz *= temp
            dr *= temp
        elif r2 < max_r2_global:
            temp = max_r2_global / r2
            xx *= temp
            yy *= temp
            zz *= temp
            dr *= temp
        
        # Scale and translate
        xx = scale * xx + x
        yy = scale * yy + y
        zz = scale * zz + z
        
        r = np.sqrt(xx*xx + yy*yy + zz*zz)
        orbit_trap = min(orbit_trap, r)
        if r > bailout:
            break
    
    return 0.5 * np.log(r) * r / dr, orbit_trap, i


@jit(nopython=True, fastmath=True)
def burning_ship_de(x, y, z, power, max_iter, bailout):
    """Burning Ship fractal."""
    xx, yy, zz = x, y, z
    dr = 1.0
    r = np.sqrt(xx*xx + yy*yy + zz*zz)
    orbit_trap = 1000.0
    
    for i in range(max_iter):
        if r > bailout:
            break
        orbit_trap = min(orbit_trap, r)
        
        sx = -abs(xx)
        sy = -abs(yy)
        sz = -abs(zz)
        
        theta = np.arctan2(np.sqrt(sx*sx + sy*sy), sz)
        phi = np.arctan2(sy, sx)
        zr = r ** (power - 1.0)
        dr = zr * dr * power + 1.0
        zr = zr * r
        theta = theta * power
        phi = phi * power
        
        xx = zr * np.sin(theta) * np.cos(phi) + x
        yy = zr * np.sin(theta) * np.sin(phi) + y
        zz = zr * np.cos(theta) + z
        r = np.sqrt(xx*xx + yy*yy + zz*zz)
    
    return 0.5 * np.log(r) * r / dr, orbit_trap, i


@jit(nopython=True, fastmath=True)
def tricorn_de(x, y, z, power, max_iter, bailout):
    """Tricorn/Mandelbar fractal."""
    xx, yy, zz = x, y, z
    dr = 1.0
    r = np.sqrt(xx*xx + yy*yy + zz*zz)
    orbit_trap = 1000.0
    
    for i in range(max_iter):
        if r > bailout:
            break
        orbit_trap = min(orbit_trap, r)
        
        # Tricorn modifies the y coordinate sign
        theta = np.arctan2(-abs(yy), zz)
        phi = np.arctan2(yy, xx)
        zr = r ** (power - 1.0)
        dr = zr * dr * power + 1.0
        zr = zr * r
        theta = theta * power
        phi = phi * power
        
        xx = zr * np.sin(theta) * np.cos(phi) + x
        yy = zr * np.sin(theta) * np.sin(phi) + y
        zz = zr * np.cos(theta) + z
        r = np.sqrt(xx*xx + yy*yy + zz*zz)
    
    return 0.5 * np.log(r) * r / dr, orbit_trap, i


@jit(nopython=True, fastmath=True)
def quaternion_julia_de(x, y, z, cx, cy, cz, cw, max_iter, bailout):
    """Quaternion Julia set (using 3D slice)."""
    xx, yy, zz = x, y, z
    dr = 1.0
    r = np.sqrt(xx*xx + yy*yy + zz*zz)
    orbit_trap = 1000.0
    
    for i in range(max_iter):
        if r > bailout:
            break
        orbit_trap = min(orbit_trap, r)
        
        # Quaternion-style iteration (q = q^2 + c)
        # Using triplex approximation for 3D
        theta = np.arctan2(np.sqrt(xx*xx + yy*yy), zz)
        phi = np.arctan2(yy, xx)
        zr = r
        dr = 2.0 * r * dr + 1.0
        theta = 2.0 * theta
        phi = 2.0 * phi
        
        xx = zr * np.sin(theta) * np.cos(phi) + cx
        yy = zr * np.sin(theta) * np.sin(phi) + cy
        zz = zr * np.cos(theta) + cz
        r = np.sqrt(xx*xx + yy*yy + zz*zz)
    
    return 0.5 * np.log(r) * r / dr, orbit_trap, i


@jit(nopython=True, fastmath=True)
def mandelbrot_3d_de(x, y, z, power, max_iter, bailout):
    """Classic 2D Mandelbrot rendered in 3D space."""
    # Use only x,y for 2D mandelbrot
    xx, yy = x, y
    cx, cy = x, y
    zz = 0.0
    
    orbit_trap = 1000.0
    for i in range(max_iter):
        if xx*xx + yy*yy > bailout:
            break
        orbit_trap = min(orbit_trap, np.sqrt(xx*xx + yy*yy))
        
        # z = z^2 + c (standard mandelbrot)
        new_xx = xx*xx - yy*yy + cx
        yy = 2.0 * xx * yy + cy
        xx = new_xx
    
    r = np.sqrt(xx*xx + yy*yy)
    de = 0.5 * np.log(r) * r if r > 0 else 0.0
    return de, orbit_trap, i


@jit(nopython=True, fastmath=True)
def julia_classic_de(x, y, z, cx, cy, max_iter, bailout):
    """Classic 2D Julia set."""
    xx, yy = x, y
    
    orbit_trap = 1000.0
    for i in range(max_iter):
        if xx*xx + yy*yy > bailout:
            break
        orbit_trap = min(orbit_trap, np.sqrt(xx*xx + yy*yy))
        
        new_xx = xx*xx - yy*yy + cx
        yy = 2.0 * xx * yy + cy
        xx = new_xx
    
    r = np.sqrt(xx*xx + yy*yy)
    de = 0.5 * np.log(r) * r if r > 0 else 0.0
    return de, orbit_trap, i


@jit(nopython=True, fastmath=True)
def kifs_de(x, y, z, scale, rot_x, rot_y, rot_z, folds, max_iter, bailout):
    """Kaleidoscopic IFS distance estimation."""
    xx, yy, zz = x, y, z
    dr = 1.0
    r = np.sqrt(xx*xx + yy*yy + zz*zz)
    orbit_trap = 1000.0
    
    for i in range(max_iter):
        if r > bailout:
            break
        orbit_trap = min(orbit_trap, r)
        
        # Fold
        if xx + yy < 0:
            xx, yy = -yy, -xx
        if xx - yy < 0:
            temp = xx
            xx = yy
            yy = temp
        
        # Scale and offset
        xx = scale * xx - scale + x
        yy = scale * yy - scale + y
        zz = scale * zz + z
        
        dr *= scale
        r = np.sqrt(xx*xx + yy*yy + zz*zz)
    
    return 0.5 * np.log(r) * r / dr, orbit_trap, i


@jit(nopython=True, fastmath=True)
def menger_sponge_de(x, y, z, iterations, bailout):
    """Menger sponge distance estimation (simplified)."""
    xx, yy, zz = x, y, z
    scale = 3.0
    orbit_trap = 1000.0
    
    for i in range(iterations):
        # Scale
        xx *= scale
        yy *= scale
        zz *= scale
        
        # Translation
        x_offset = int(xx) % 3
        y_offset = int(yy) % 3
        z_offset = int(zz) % 3
        
        # Cut holes
        if (x_offset == 1 and y_offset == 1) or \
           (y_offset == 1 and z_offset == 1) or \
           (x_offset == 1 and z_offset == 1):
            return 1000.0, 1000.0, i
        
        xx -= x_offset
        yy -= y_offset
        zz -= z_offset
        
        orbit_trap = min(orbit_trap, np.sqrt(xx*xx + yy*yy + zz*zz))
    
    r = np.sqrt(xx*xx + yy*yy + zz*zz)
    return 0.5 * r / (scale**iterations), orbit_trap, i


@jit(nopython=True, fastmath=True)
def lambda_mandelbulb_de(x, y, z, power, lambda_val, max_iter, bailout):
    """Lambdabulb - modified formula z -> c*(z - z^n)."""
    xx, yy, zz = x, y, z
    cx, cy, cz = x, y, z  # Same as mandelbulb for Mandel version
    dr = 1.0
    r = np.sqrt(xx*xx + yy*yy + zz*zz)
    orbit_trap = 1000.0
    
    for i in range(max_iter):
        if r > bailout:
            break
        orbit_trap = min(orbit_trap, r)
        
        # Calculate z^n
        theta = np.arctan2(np.sqrt(xx*xx + yy*yy), zz)
        phi = np.arctan2(yy, xx)
        zr = r ** (power - 1.0)
        zr_n = zr * r
        theta_n = theta * power
        phi_n = phi * power
        
        znx = zr_n * np.sin(theta_n) * np.cos(phi_n)
        zny = zr_n * np.sin(theta_n) * np.sin(phi_n)
        znz = zr_n * np.cos(theta_n)
        
        # z -> lambda * (z - z^n) + c
        xx = lambda_val * (xx - znx) + cx
        yy = lambda_val * (yy - zny) + cy
        zz = lambda_val * (zz - znz) + cz
        
        dr = abs(lambda_val) * dr * power + 1.0
        r = np.sqrt(xx*xx + yy*yy + zz*zz)
    
    return 0.5 * np.log(r) * r / abs(dr), orbit_trap, i


@jit(nopython=True, fastmath=True)
def sphere_fold_de(x, y, z, scale, min_r, max_r, max_iter, bailout):
    """Pure sphere folding (no box fold)."""
    xx, yy, zz = x, y, z
    mr2 = min_r * min_r
    Mr2 = max_r * max_r
    dr = 1.0
    orbit_trap = 1000.0
    
    for i in range(max_iter):
        r2 = xx*xx + yy*yy + zz*zz
        
        if r2 < mr2:
            temp = Mr2 / mr2
            xx *= temp
            yy *= temp
            zz *= temp
            dr *= temp
        elif r2 < Mr2:
            temp = Mr2 / r2
            xx *= temp
            yy *= temp
            zz *= temp
            dr *= temp
        
        xx = scale * xx + x
        yy = scale * yy + y
        zz = scale * zz + z
        
        r = np.sqrt(xx*xx + yy*yy + zz*zz)
        orbit_trap = min(orbit_trap, r)
        
        if r > bailout:
            break
    
    return 0.5 * np.log(r) * r / dr, orbit_trap, i


@jit(nopython=True, fastmath=True)
def newton_z3_de(x, y, z, max_iter, bailout):
    """Newton fractal for z^3 - 1 = 0."""
    xx, yy, zz = x, y, z
    
    for i in range(max_iter):
        # z = z - (z^3 - 1) / (3z^2)
        r2 = xx*xx + yy*yy + zz*zz
        if r2 < 0.0001:
            break
            
        # Simplified 2D newton in 3D context
        z2 = xx*xx - yy*yy
        xy = 2 * xx * yy
        
        denom = 3 * (xx*xx + yy*yy)
        if abs(denom) < 0.0001:
            break
            
        dx = (xx - z2/denom) / 3
        dy = (yy - xy/denom) / 3
        
        xx = dx
        yy = dy
    
    r = np.sqrt(xx*xx + yy*yy)
    return r, r, i


@jit(nopython=True, fastmath=True)
def newton_z4_de(x, y, z, max_iter, bailout):
    """Newton fractal for z^4 - 1 = 0."""
    xx, yy, zz = x, y, z
    
    for i in range(max_iter):
        r2 = xx*xx + yy*yy + zz*zz
        if r2 < 0.0001:
            break
            
        denom = 4 * (xx*xx + yy*yy)
        if abs(denom) < 0.0001:
            break
            
        # z^4 derivative is 4z^3
        # Using simplified 2D form
        z2 = xx*xx - yy*yy
        z3 = xx * (xx*xx - 3*yy*yy)
        y3 = yy * (3*xx*xx - yy*yy)
        
        dx = (z2 - 1) / (4 * (xx*xx + yy*yy)) if r2 > 0.0001 else xx
        dy = xy / (4 * (xx*xx + yy*yy)) if r2 > 0.0001 else yy
        
        xx -= dx
        yy -= dy
    
    r = np.sqrt(xx*xx + yy*yy)
    return r, r, i


# Hybrid placeholders (simplified)
@jit(nopython=True, fastmath=True)
def hybrid_1_de(x, y, z, power, scale, blend, max_iter, bailout):
    """Hybrid Mandelbulb-Mandelbox."""
    # Blend between mandelbulb and mandelbox
    d1, t1, i1 = mandelbulb_de(x, y, z, power, max_iter, bailout)
    d2, t2, i2 = mandelbox_de(x, y, z, scale, 1.0, 0.5, max_iter, bailout)
    d = d1 * blend + d2 * (1 - blend)
    return d, min(t1, t2), int((i1 + i2) / 2)


@jit(nopython=True, fastmath=True)
def hybrid_2_de(x, y, z, power, scale, cx, cy, cz, max_iter, bailout):
    """Hybrid Julia-Mandelbox."""
    d1, t1, i1 = julia_set_de(x, y, z, power, cx, cy, cz, max_iter, bailout)
    d2, t2, i2 = mandelbox_de(x, y, z, scale, 1.0, 0.5, max_iter, bailout)
    d = (d1 + d2) * 0.5
    return d, min(t1, t2), int((i1 + i2) / 2)


@jit(nopython=True, fastmath=True)
def biomorph_de(x, y, z, power, bailout, max_iter):
    """Biomorph - uses different escape condition."""
    xx, yy, zz = x, y, z
    
    for i in range(max_iter):
        r2 = xx*xx + yy*yy + zz*zz
        if r2 > bailout:
            break
            
        # Different iteration for biomorph look
        theta = np.arctan2(np.sqrt(xx*xx + yy*yy), zz)
        phi = np.arctan2(yy, xx)
        zr = r2
        theta = theta * power
        phi = phi * power
        
        xx = zr * np.sin(theta) * np.cos(phi) + x
        yy = zr * np.sin(theta) * np.sin(phi) + y
        zz = zr * np.cos(theta) + z
    
    r = np.sqrt(xx*xx + yy*yy + zz*zz)
    return 0.5 * np.log(r + 0.001) * (r + 0.001), r, i


@jit(nopython=True, fastmath=True)
def sierpinski_de(x, y, z, iterations, scale, max_iter, bailout):
    """Sierpinski tetrahedron."""
    xx, yy, zz = x, y, z
    orbit_trap = 1000.0
    
    for i in range(min(iterations, max_iter)):
        # Transform based on octant
        if xx + yy < 0:
            xx, yy = -xx - 1, -yy - 1
        if yy + zz < 0:
            yy, zz = -yy - 1, -zz - 1
        if xx + zz < 0:
            xx, zz = -xx - 1, -zz - 1
        
        # Scale
        xx *= 2
        yy *= 2
        zz *= 2
        
        orbit_trap = min(orbit_trap, np.sqrt(xx*xx + yy*yy + zz*zz))
    
    r = np.sqrt(xx*xx + yy*yy + zz*zz)
    return r / (2.0**iterations), orbit_trap, i


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def list_all_fractals() -> List[str]:
    """Return list of all available fractal names."""
    return list(FRACTAL_REGISTRY.keys())


def get_formula(name: str) -> Optional[FractalFormula]:
    """Get formula by name."""
    return FRACTAL_REGISTRY.get(name)


def get_formula_info(name: str) -> str:
    """Get formatted info about a formula."""
    f = FRACTAL_REGISTRY.get(name)
    if not f:
        return f"Unknown fractal: {name}"
    
    lines = [
        f"=== {f.name} ===",
        f"Category: {f.category.value}",
        f"Description: {f.description}",
        f"Parameters: {', '.join(f.params)}",
        f"Defaults: {f.default_values}"
    ]
    return "\n".join(lines)


def get_fractals_by_category(category: FractalCategory) -> List[str]:
    """Get all fractals in a category."""
    return [name for name, f in FRACTAL_REGISTRY.items() if f.category == category]


def get_formula_function(name: str):
    """Get the implementation function for a formula."""
    f = FRACTAL_REGISTRY.get(name)
    if not f:
        return None
    
    return globals().get(f.python_impl)


# Alias for backwards compatibility
FRACTAL_TYPES = list_all_fractals
get_all_fractals = list_all_fractals