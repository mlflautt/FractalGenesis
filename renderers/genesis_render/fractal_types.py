#!/usr/bin/env python3
"""
GenesisRender - Comprehensive Fractal Types
==========================================

Complete collection of 3D fractal distance estimation functions:
- Mandelbulb (various powers)
- Julia Sets (3D)
- Mandelbox 
- Burning Ship 3D
- Menger Sponge
- Sierpinski Tetrahedron
- Kleinian Groups
- IFS (Iterated Function Systems)
- Hybrid fractals

All functions are JIT-compiled with Numba for maximum performance.
"""

import numpy as np
from numba import jit
from typing import Tuple
import math

# Core fractal type enumeration
FRACTAL_TYPES = {
    'mandelbulb': 0,
    'julia_3d': 1,
    'mandelbox': 2,
    'burning_ship_3d': 3,
    'menger_sponge': 4,
    'sierpinski_tetrahedron': 5,
    'kleinian': 6,
    'ifs_dodecahedron': 7,
    'mandelbrot_3d': 8,
    'tricorn_3d': 9,
    'nova_fractal': 10,
    'phoenix_3d': 11
}

@jit(nopython=True, fastmath=True)
def mandelbulb_de(pos, power, max_iter, bailout):
    """Enhanced Mandelbulb with better orbit trap tracking"""
    x, y, z = pos[0], pos[1], pos[2]
    xx, yy, zz = x, y, z
    dr = 1.0
    r = math.sqrt(xx*xx + yy*yy + zz*zz)
    orbit_trap = 1000.0
    
    for i in range(max_iter):
        if r > bailout:
            break
            
        # Track minimum distance for orbit trap coloring
        orbit_trap = min(orbit_trap, r)
        
        # Convert to spherical coordinates
        theta = math.atan2(math.sqrt(xx*xx + yy*yy), zz)
        phi = math.atan2(yy, xx)
        
        # Scale derivative
        zr = r ** (power - 1.0)
        dr = zr * dr * power + 1.0
        
        # Apply power and convert back to cartesian
        zr = zr * r
        theta = theta * power
        phi = phi * power
        
        xx = zr * math.sin(theta) * math.cos(phi) + x
        yy = zr * math.sin(theta) * math.sin(phi) + y
        zz = zr * math.cos(theta) + z
        
        r = math.sqrt(xx*xx + yy*yy + zz*zz)
    
    return 0.5 * math.log(r) * r / dr, orbit_trap, i

@jit(nopython=True, fastmath=True)
def julia_3d_de(pos, power, max_iter, bailout, julia_c):
    """3D Julia set with configurable constant"""
    x, y, z = pos[0], pos[1], pos[2]
    xx, yy, zz = x, y, z
    dr = 1.0
    r = math.sqrt(xx*xx + yy*yy + zz*zz)
    orbit_trap = 1000.0
    
    for i in range(max_iter):
        if r > bailout:
            break
            
        orbit_trap = min(orbit_trap, r)
        
        theta = math.atan2(math.sqrt(xx*xx + yy*yy), zz)
        phi = math.atan2(yy, xx)
        
        zr = r ** (power - 1.0)
        dr = zr * dr * power + 1.0
        
        zr = zr * r
        theta = theta * power
        phi = phi * power
        
        xx = zr * math.sin(theta) * math.cos(phi) + julia_c[0]
        yy = zr * math.sin(theta) * math.sin(phi) + julia_c[1]
        zz = zr * math.cos(theta) + julia_c[2]
        
        r = math.sqrt(xx*xx + yy*yy + zz*zz)
    
    return 0.5 * math.log(r) * r / dr, orbit_trap, i

@jit(nopython=True, fastmath=True)
def mandelbox_de(pos, scale, folding_limit, folding_value, max_iter):
    """Mandelbox fractal with configurable folding parameters"""
    x, y, z = pos[0], pos[1], pos[2]
    cx, cy, cz = x, y, z
    dr = 1.0
    orbit_trap = 1000.0
    
    for i in range(max_iter):
        # Box folding
        if x > folding_limit:
            x = folding_value - x
        elif x < -folding_limit:
            x = -folding_value - x
            
        if y > folding_limit:
            y = folding_value - y
        elif y < -folding_limit:
            y = -folding_value - y
            
        if z > folding_limit:
            z = folding_value - z
        elif z < -folding_limit:
            z = -folding_value - z
        
        # Sphere folding
        r2 = x*x + y*y + z*z
        orbit_trap = min(orbit_trap, math.sqrt(r2))
        
        if r2 < 0.25:  # r < 0.5
            x *= 4.0
            y *= 4.0
            z *= 4.0
            dr *= 4.0
        elif r2 < 1.0:  # r < 1.0
            factor = 1.0 / r2
            x *= factor
            y *= factor
            z *= factor
            dr *= factor
        
        # Scale and translate
        x = scale * x + cx
        y = scale * y + cy
        z = scale * z + cz
        dr = dr * abs(scale) + 1.0
        
        if r2 > 256.0:  # Bailout
            break
    
    return math.sqrt(x*x + y*y + z*z) / abs(dr), orbit_trap, i

@jit(nopython=True, fastmath=True)
def burning_ship_3d_de(pos, power, max_iter, bailout):
    """3D Burning Ship fractal"""
    x, y, z = pos[0], pos[1], pos[2]
    xx, yy, zz = x, y, z
    dr = 1.0
    r = math.sqrt(xx*xx + yy*yy + zz*zz)
    orbit_trap = 1000.0
    
    for i in range(max_iter):
        if r > bailout:
            break
            
        orbit_trap = min(orbit_trap, r)
        
        # Apply absolute values (burning ship characteristic)
        xx = abs(xx)
        yy = abs(yy)
        zz = abs(zz)
        
        theta = math.atan2(math.sqrt(xx*xx + yy*yy), zz)
        phi = math.atan2(yy, xx)
        
        zr = r ** (power - 1.0)
        dr = zr * dr * power + 1.0
        
        zr = zr * r
        theta = theta * power
        phi = phi * power
        
        xx = zr * math.sin(theta) * math.cos(phi) + x
        yy = zr * math.sin(theta) * math.sin(phi) + y
        zz = zr * math.cos(theta) + z
        
        r = math.sqrt(xx*xx + yy*yy + zz*zz)
    
    return 0.5 * math.log(r) * r / dr, orbit_trap, i

@jit(nopython=True, fastmath=True)
def menger_sponge_de(pos, iterations):
    """Menger Sponge using iterative construction"""
    x, y, z = abs(pos[0]), abs(pos[1]), abs(pos[2])
    orbit_trap = max(x, max(y, z))
    
    for i in range(iterations):
        x = abs(x)
        y = abs(y)
        z = abs(z)
        
        if x < y:
            x, y = y, x
        if x < z:
            x, z = z, x
        if y < z:
            y, z = z, y
            
        x = 3.0 * x - 2.0
        y = 3.0 * y - 2.0
        if y > 1.0:
            y = 2.0 - y
            
        z = 3.0 * z - 2.0
        if z > 1.0:
            z = 2.0 - z
            
        orbit_trap = min(orbit_trap, max(x, max(y, z)) / pow(3.0, i + 1))
    
    return (max(x, max(y, z)) - 1.0) / 3.0, orbit_trap, iterations

@jit(nopython=True, fastmath=True)
def sierpinski_tetrahedron_de(pos, iterations):
    """Sierpinski Tetrahedron (3D Sierpinski triangle)"""
    x, y, z = pos[0], pos[1], pos[2]
    orbit_trap = math.sqrt(x*x + y*y + z*z)
    scale = 2.0
    
    for i in range(iterations):
        if x + y < 0:
            x, y = -y, -x
        if x + z < 0:
            x, z = -z, -x
        if y + z < 0:
            y, z = -z, -y
            
        x = scale * x - 1.0
        y = scale * y - 1.0
        z = scale * z - 1.0
        
        orbit_trap = min(orbit_trap, math.sqrt(x*x + y*y + z*z) / pow(scale, i + 1))
    
    return (math.sqrt(x*x + y*y + z*z) - 2.0) / pow(scale, iterations), orbit_trap, iterations

@jit(nopython=True, fastmath=True)
def kleinian_de(pos, max_iter):
    """Kleinian group fractal (Apollonian gasket 3D)"""
    x, y, z = pos[0], pos[1], pos[2]
    orbit_trap = 1000.0
    
    for i in range(max_iter):
        # Inversion through spheres
        r2 = x*x + y*y + z*z
        orbit_trap = min(orbit_trap, math.sqrt(r2))
        
        if r2 < 1.0:
            inv = 1.0 / r2
            x *= inv
            y *= inv
            z *= inv
        
        # Translation and folding
        x = abs(x + 1.0) - 1.0
        y = abs(y + 1.0) - 1.0
        z = abs(z)
        
        if math.sqrt(x*x + y*y + z*z) > 2.0:
            break
    
    return (math.sqrt(x*x + y*y + z*z) - 1.0), orbit_trap, i

@jit(nopython=True, fastmath=True)
def nova_fractal_de(pos, power, c_real, c_imag, max_iter, bailout):
    """Nova fractal (Newton-Raphson variant)"""
    x, y, z = pos[0], pos[1], pos[2]
    orbit_trap = 1000.0
    
    for i in range(max_iter):
        # Treat as complex number z = x + iy, with z-component as parameter
        r2 = x*x + y*y
        orbit_trap = min(orbit_trap, math.sqrt(r2 + z*z))
        
        if r2 > bailout*bailout:
            break
        
        # Newton iteration: z = z - (z^n - 1)/(n*z^(n-1)) + c
        if r2 > 1e-10:  # Avoid division by zero
            # Calculate z^(power-1)
            zn_1 = pow(r2, (power-1)*0.5)
            angle = (power-1) * math.atan2(y, x)
            
            zn_1_real = zn_1 * math.cos(angle)
            zn_1_imag = zn_1 * math.sin(angle)
            
            # Calculate z^power - 1
            zn_real = zn_1_real * x - zn_1_imag * y - 1.0
            zn_imag = zn_1_real * y + zn_1_imag * x
            
            # Newton update
            denom = power * (zn_1_real*zn_1_real + zn_1_imag*zn_1_imag)
            if denom > 1e-10:
                x = x - zn_real/denom + c_real
                y = y - zn_imag/denom + c_imag
                z = z * 0.99 + 0.01 * math.sin(i * 0.1)  # Add 3D variation
    
    r = math.sqrt(x*x + y*y + z*z)
    return 0.5 * math.log(r) * r, orbit_trap, i

@jit(nopython=True, fastmath=True)
def get_fractal_distance(pos, fractal_type, params):
    """Dispatch function for all fractal types
    
    params array layout:
    [0] = power/scale
    [1] = max_iterations
    [2] = bailout
    [3-5] = julia_c or other parameters
    [6-8] = additional parameters
    """
    
    if fractal_type == 0:  # mandelbulb
        return mandelbulb_de(pos, params[0], int(params[1]), params[2])
    elif fractal_type == 1:  # julia_3d
        julia_c = np.array([params[3], params[4], params[5]])
        return julia_3d_de(pos, params[0], int(params[1]), params[2], julia_c)
    elif fractal_type == 2:  # mandelbox
        return mandelbox_de(pos, params[0], params[3], params[4], int(params[1]))
    elif fractal_type == 3:  # burning_ship_3d
        return burning_ship_3d_de(pos, params[0], int(params[1]), params[2])
    elif fractal_type == 4:  # menger_sponge
        return menger_sponge_de(pos, int(params[1]))
    elif fractal_type == 5:  # sierpinski_tetrahedron
        return sierpinski_tetrahedron_de(pos, int(params[1]))
    elif fractal_type == 6:  # kleinian
        return kleinian_de(pos, int(params[1]))
    elif fractal_type == 10:  # nova_fractal
        return nova_fractal_de(pos, params[0], params[3], params[4], int(params[1]), params[2])
    else:  # default to mandelbulb
        return mandelbulb_de(pos, params[0], int(params[1]), params[2])

# Fractal parameter presets for easy use
FRACTAL_PRESETS = {
    'mandelbulb_classic': {
        'type': 'mandelbulb',
        'power': 8.0,
        'iterations': 100,
        'bailout': 2.0
    },
    'mandelbulb_smooth': {
        'type': 'mandelbulb', 
        'power': 6.0,
        'iterations': 150,
        'bailout': 2.0
    },
    'julia_organic': {
        'type': 'julia_3d',
        'power': 8.0,
        'iterations': 100,
        'bailout': 2.0,
        'julia_c': (-0.2, 0.1, 0.0)
    },
    'julia_crystalline': {
        'type': 'julia_3d',
        'power': 4.0,
        'iterations': 80,
        'bailout': 2.0,
        'julia_c': (0.3, -0.1, 0.2)
    },
    'mandelbox_classic': {
        'type': 'mandelbox',
        'scale': -1.5,
        'iterations': 100,
        'folding_limit': 1.0,
        'folding_value': 2.0
    },
    'mandelbox_spiky': {
        'type': 'mandelbox',
        'scale': -2.1,
        'iterations': 120,
        'folding_limit': 0.8,
        'folding_value': 2.2
    },
    'burning_ship_3d': {
        'type': 'burning_ship_3d',
        'power': 3.0,
        'iterations': 100,
        'bailout': 2.0
    },
    'menger_sponge': {
        'type': 'menger_sponge',
        'iterations': 8
    },
    'sierpinski_3d': {
        'type': 'sierpinski_tetrahedron',
        'iterations': 10
    },
    'kleinian_gasket': {
        'type': 'kleinian',
        'iterations': 50
    },
    'nova_swirl': {
        'type': 'nova_fractal',
        'power': 3.0,
        'iterations': 100,
        'bailout': 10.0,
        'c_real': 0.1,
        'c_imag': 0.05
    }
}