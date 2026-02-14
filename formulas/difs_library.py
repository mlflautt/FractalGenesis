#!/usr/bin/env python3
"""
dIFS Formula Library - Geometric IFS Shapes
==========================================

Direct IFS (Iterated Function System) formulas for geometric fractals.
These use distance fields rather than escape-time iteration.

Includes: Sierpinski, Menger variants, crystal structures, etc.
"""

import numpy as np
from numba import jit
from typing import Tuple
from .extended_library import FractalFormula, FormulaParams, MetaParameter, MetaParameterType


class SierpinskiTetrahedron(FractalFormula):
    """Sierpinski Tetrahedron (Gasket) via dIFS"""
    
    def __init__(self):
        super().__init__("sierpinski_tetra", "Sierpinski Tetrahedron dIFS")
        self.meta_parameters.extend([
            MetaParameter("scale", MetaParameterType.SCALING, 2.0, 1.5, 3.0, "IFS Scale factor"),
            MetaParameter("offset", MetaParameterType.OFFSET, 1.0, 0.1, 2.0, "Vertex offset"),
        ])
    
    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _de(x, y, z, scale, offset, max_iter, bailout):
        xx, yy, zz = x, y, z
        
        # Tetrahedron vertices
        v1 = (offset, offset, offset)
        v2 = (-offset, -offset, offset)
        v3 = (-offset, offset, -offset)
        v4 = (offset, -offset, -offset)
        
        for i in range(min(max_iter, 20)):  # dIFS typically needs fewer iterations
            # Find closest vertex
            d1 = (xx-v1[0])**2 + (yy-v1[1])**2 + (zz-v1[2])**2
            d2 = (xx-v2[0])**2 + (yy-v2[1])**2 + (zz-v2[2])**2
            d3 = (xx-v3[0])**2 + (yy-v3[1])**2 + (zz-v3[2])**2
            d4 = (xx-v4[0])**2 + (yy-v4[1])**2 + (zz-v4[2])**2
            
            # Scale towards closest vertex
            if d1 <= d2 and d1 <= d3 and d1 <= d4:
                xx = (xx - v1[0]) * scale + v1[0]
                yy = (yy - v1[1]) * scale + v1[1]
                zz = (zz - v1[2]) * scale + v1[2]
            elif d2 <= d3 and d2 <= d4:
                xx = (xx - v2[0]) * scale + v2[0]
                yy = (yy - v2[1]) * scale + v2[1]
                zz = (zz - v2[2]) * scale + v2[2]
            elif d3 <= d4:
                xx = (xx - v3[0]) * scale + v3[0]
                yy = (yy - v3[1]) * scale + v3[1]
                zz = (zz - v3[2]) * scale + v3[2]
            else:
                xx = (xx - v4[0]) * scale + v4[0]
                yy = (yy - v4[1]) * scale + v4[1]
                zz = (zz - v4[2]) * scale + v4[2]
        
        # Distance to tetrahedron
        r = np.sqrt(xx*xx + yy*yy + zz*zz)
        return r * 0.5, r, min(max_iter, 20)
    
    def distance_estimate(self, x, y, z, params):
        sc = params.get_meta("scale", 2.0)
        off = params.get_meta("offset", 1.0)
        return self._de(x, y, z, sc, off, params.iterations, params.bailout)


class MengerSpongeIFS(FractalFormula):
    """Menger Sponge via dIFS method"""
    
    def __init__(self):
        super().__init__("menger_ifs", "Menger Sponge via dIFS")
        self.meta_parameters.extend([
            MetaParameter("scale", MetaParameterType.SCALING, 3.0, 2.0, 4.0, "Scale factor"),
            MetaParameter("offset", MetaParameterType.OFFSET, 1.0, 0.5, 2.0, "Offset"),
        ])
    
    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _de(x, y, z, scale, offset, max_iter, bailout):
        xx, yy, zz = x, y, z
        
        for i in range(min(max_iter, 15)):
            xx = abs(xx)
            yy = abs(yy)
            zz = abs(zz)
            
            # Sort to ensure xx >= yy >= zz
            if xx < yy:
                xx, yy = yy, xx
            if xx < zz:
                xx, zz = zz, xx
            if yy < zz:
                yy, zz = zz, yy
            
            # Menger operation
            xx = scale * xx - offset * (scale - 1.0)
            yy = scale * yy - offset * (scale - 1.0)
            zz = scale * zz
            
            if zz > 0.5 * offset * (scale - 1.0):
                zz -= offset * (scale - 1.0)
        
        r = np.sqrt(xx*xx + yy*yy + zz*zz)
        return r * 0.5, r, min(max_iter, 15)
    
    def distance_estimate(self, x, y, z, params):
        sc = params.get_meta("scale", 3.0)
        off = params.get_meta("offset", 1.0)
        return self._de(x, y, z, sc, off, params.iterations, params.bailout)


class CrystalIFS(FractalFormula):
    """Crystal-like structure via dIFS"""
    
    def __init__(self):
        super().__init__("crystal_ifs", "Crystal structure dIFS")
        self.meta_parameters.extend([
            MetaParameter("scale", MetaParameterType.SCALING, 2.0, 1.5, 3.0, "Scale"),
            MetaParameter("rotation", MetaParameterType.ROTATION, 0.5, 0.0, 1.0, "Rotation amount"),
            MetaParameter("symmetry", MetaParameterType.CONDITIONAL, 6.0, 3.0, 12.0, "Symmetry order"),
        ])
    
    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _de(x, y, z, scale, rotation, symmetry, max_iter, bailout):
        xx, yy, zz = x, y, z
        angle_step = 2.0 * np.pi / symmetry
        
        for i in range(min(max_iter, 20)):
            # Radial symmetry
            angle = np.arctan2(yy, xx)
            sector = int(angle / angle_step)
            angle = angle - sector * angle_step
            
            # Rotate back
            cos_a = np.cos(-angle)
            sin_a = np.sin(-angle)
            xr = xx * cos_a - yy * sin_a
            yr = xx * sin_a + yy * cos_a
            
            # Scale
            xx = xr * scale
            yy = yr * scale
            zz = zz * scale
            
            # Add rotation offset
            angle += rotation * angle_step
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            xr = xx * cos_a - yy * sin_a
            yr = xx * sin_a + yy * cos_a
            xx, yy = xr, yr
        
        r = np.sqrt(xx*xx + yy*yy + zz*zz)
        return r * 0.3, r, min(max_iter, 20)
    
    def distance_estimate(self, x, y, z, params):
        sc = params.get_meta("scale", 2.0)
        rot = params.get_meta("rotation", 0.5)
        sym = params.get_meta("symmetry", 6.0)
        return self._de(x, y, z, sc, rot, sym, params.iterations, params.bailout)


class HoneycombIFS(FractalFormula):
    """Honeycomb structure via dIFS"""
    
    def __init__(self):
        super().__init__("honeycomb_ifs", "Honeycomb structure dIFS")
        self.meta_parameters.extend([
            MetaParameter("scale", MetaParameterType.SCALING, 2.0, 1.5, 3.0, "Scale"),
            MetaParameter("hex_size", MetaParameterType.OFFSET, 1.0, 0.5, 2.0, "Hexagon size"),
        ])
    
    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _de(x, y, z, scale, hex_size, max_iter, bailout):
        xx, yy, zz = x, y, z
        
        for i in range(min(max_iter, 15)):
            # Hexagonal tiling in XY plane
            # Convert to hex coordinates
            q = (np.sqrt(3.0)/3.0 * xx - 1.0/3.0 * yy) / hex_size
            r_coord = (2.0/3.0 * yy) / hex_size
            
            # Round to nearest hex
            x_hex = q
            y_hex = (-q - r_coord)
            z_hex = r_coord
            
            rx = round(x_hex)
            ry = round(y_hex)
            rz = round(z_hex)
            
            x_diff = abs(rx - x_hex)
            y_diff = abs(ry - y_hex)
            z_diff = abs(rz - z_hex)
            
            if x_diff > y_diff and x_diff > z_diff:
                rx = -ry - rz
            elif y_diff > z_diff:
                ry = -rx - rz
            else:
                rz = -rx - ry
            
            # Hex center
            hx = hex_size * np.sqrt(3.0) * (rx + rz/2.0)
            hy = hex_size * 3.0/2.0 * rz
            
            # Scale towards hex center
            xx = (xx - hx) * scale + hx
            yy = (yy - hy) * scale + hy
            zz = zz * scale
        
        r = np.sqrt(xx*xx + yy*yy + zz*zz)
        return r * 0.4, r, min(max_iter, 15)
    
    def distance_estimate(self, x, y, z, params):
        sc = params.get_meta("scale", 2.0)
        hs = params.get_meta("hex_size", 1.0)
        return self._de(x, y, z, sc, hs, params.iterations, params.bailout)


class TreeIFS(FractalFormula):
    """Fractal tree structure via dIFS"""
    
    def __init__(self):
        super().__init__("tree_ifs", "Fractal tree dIFS")
        self.meta_parameters.extend([
            MetaParameter("branch_angle", MetaParameterType.ROTATION, 0.5, 0.2, 1.0, "Branch angle"),
            MetaParameter("branch_scale", MetaParameterType.SCALING, 0.7, 0.5, 0.9, "Branch scale"),
            MetaParameter("branch_count", MetaParameterType.CONDITIONAL, 2.0, 2.0, 4.0, "Branches per level"),
        ])
    
    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _de(x, y, z, branch_angle, branch_scale, branch_count, max_iter, bailout):
        xx, yy, zz = x, y, z
        
        # Branch directions
        cos_a = np.cos(branch_angle)
        sin_a = np.sin(branch_angle)
        
        for i in range(min(max_iter, 12)):
            # Scale down
            xx *= 1.0 / branch_scale
            yy *= 1.0 / branch_scale
            zz *= 1.0 / branch_scale
            
            # Choose branch based on position
            if branch_count >= 2:
                if yy > 0:
                    # Right branch
                    yy -= 1.0
                    yr = yy * cos_a - zz * sin_a
                    zr = yy * sin_a + zz * cos_a
                    yy, zz = yr, zr
                else:
                    # Left branch
                    yy += 1.0
                    yr = yy * cos_a + zz * sin_a
                    zr = -yy * sin_a + zz * cos_a
                    yy, zz = yr, zr
            
            if branch_count >= 3 and abs(xx) > 0.5:
                # Third branch in X
                if xx > 0:
                    xx -= 1.0
                    xr = xx * cos_a - zz * sin_a
                    zr = xx * sin_a + zz * cos_a
                    xx, zz = xr, zr
                else:
                    xx += 1.0
        
        r = np.sqrt(xx*xx + yy*yy + zz*zz)
        return r * branch_scale**min(max_iter, 12), r, min(max_iter, 12)
    
    def distance_estimate(self, x, y, z, params):
        ba = params.get_meta("branch_angle", 0.5)
        bs = params.get_meta("branch_scale", 0.7)
        bc = int(params.get_meta("branch_count", 2.0))
        return self._de(x, y, z, ba, bs, bc, params.iterations, params.bailout)
