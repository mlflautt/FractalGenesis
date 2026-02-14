#!/usr/bin/env python3
"""
Fractal Formula System - Inspired by Mandelbulb3D
===============================================

A comprehensive system for defining, combining, and managing fractal formulas.
Supports hybrid fractals, formula combinations, and extensible formula library.

Features:
- Individual fractal formulas (Mandelbulb, Mandelbox, Julia, etc.)
- Formula combinations and hybrids
- Formula registry and management
- Parameter handling for each formula
- Distance estimation functions optimized for rendering
"""

import numpy as np
from numba import jit, prange
from typing import Dict, List, Tuple, Any, Optional, Callable, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)

# =============================================================================
# FRACTAL FORMULA BASE CLASSES
# =============================================================================

@dataclass
class FormulaParameters:
    """Base class for fractal formula parameters"""
    power: float = 8.0
    iterations: int = 100
    bailout: float = 2.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'power': self.power,
            'iterations': self.iterations,
            'bailout': self.bailout
        }

@dataclass
class MandelbulbParams(FormulaParameters):
    """Parameters specific to Mandelbulb formula"""
    pass

@dataclass
class MandelboxParams(FormulaParameters):
    """Parameters specific to Mandelbox formula"""
    folding_limit: float = 1.0
    folding_value: float = 2.0
    scale: float = -1.5

@dataclass
class JuliaParams(FormulaParameters):
    """Parameters specific to Julia set"""
    cx: float = -0.2
    cy: float = 0.1
    cz: float = 0.0

@dataclass
class BurningShipParams(FormulaParameters):
    """Parameters specific to Burning Ship fractal"""
    pass

@dataclass
class MengerParams(FormulaParameters):
    """Parameters specific to Menger Sponge"""
    scale: float = 3.0
    offset: float = 1.0

class FractalFormula(ABC):
    """Abstract base class for fractal formulas"""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description

    @abstractmethod
    def distance_estimate(self, x: float, y: float, z: float, params: FormulaParameters) -> Tuple[float, float, int]:
        """Calculate distance estimate and orbit trap for given point"""
        pass

    @abstractmethod
    def get_parameters_class(self) -> Type[FormulaParameters]:
        """Return the parameter class for this formula"""
        pass

    def get_default_parameters(self) -> FormulaParameters:
        """Get default parameters for this formula"""
        return self.get_parameters_class()()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize formula to dictionary"""
        return {
            'name': self.name,
            'description': self.description,
            'type': self.__class__.__name__,
            'default_params': self.get_default_parameters().to_dict()
        }

# =============================================================================
# INDIVIDUAL FRACTAL FORMULAS
# =============================================================================

class MandelbulbFormula(FractalFormula):
    """Classic Mandelbulb fractal formula"""

    def __init__(self):
        super().__init__(
            "mandelbulb",
            "Classic 3D Mandelbulb fractal using polar coordinate transformations"
        )

    def get_parameters_class(self):
        return MandelbulbParams

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _distance_estimate(x: float, y: float, z: float, power: float, max_iter: int, bailout: float) -> Tuple[float, float, int]:
        """JIT-compiled Mandelbulb distance estimation"""
        xx, yy, zz = x, y, z
        dr = 1.0
        r = np.sqrt(xx*xx + yy*yy + zz*zz)
        orbit_trap = 1000.0
        i = 0

        for i in range(max_iter):
            if r > bailout:
                break

            orbit_trap = min(orbit_trap, r)

            # Convert to polar coordinates
            theta = np.arctan2(np.sqrt(xx*xx + yy*yy), zz)
            phi = np.arctan2(yy, xx)

            # Scale and rotate
            zr = r ** (power - 1.0)
            dr = zr * dr * power + 1.0

            # Convert back to cartesian
            zr = zr * r
            theta = theta * power
            phi = phi * power

            xx = zr * np.sin(theta) * np.cos(phi) + x
            yy = zr * np.sin(theta) * np.sin(phi) + y
            zz = zr * np.cos(theta) + z

            r = np.sqrt(xx*xx + yy*yy + zz*zz)

        return 0.5 * np.log(r) * r / dr, orbit_trap, i

    def distance_estimate(self, x: float, y: float, z: float, params: FormulaParameters) -> Tuple[float, float, int]:
        return MandelbulbFormula._distance_estimate(x, y, z, params.power, params.iterations, params.bailout)

class MandelboxFormula(FractalFormula):
    """Mandelbox fractal formula - box folding with sphere scaling"""

    def __init__(self):
        super().__init__(
            "mandelbox",
            "Mandelbox fractal using box folding and sphere scaling operations"
        )

    def get_parameters_class(self):
        return MandelboxParams

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _distance_estimate(x: float, y: float, z: float, power: float, max_iter: int, bailout: float,
                          folding_limit: float, folding_value: float, scale: float) -> Tuple[float, float, int]:
        """JIT-compiled Mandelbox distance estimation"""
        xx, yy, zz = x, y, z
        orbit_trap = 1000.0
        dr = 1.0
        i = 0

        for i in range(max_iter):
            r = np.sqrt(xx*xx + yy*yy + zz*zz)

            orbit_trap = min(orbit_trap, r)

            # Box folding
            if xx > folding_limit:
                xx = folding_value - xx
            elif xx < -folding_limit:
                xx = -folding_value - xx

            if yy > folding_limit:
                yy = folding_value - yy
            elif yy < -folding_limit:
                yy = -folding_value - yy

            if zz > folding_limit:
                zz = folding_value - zz
            elif zz < -folding_limit:
                zz = -folding_value - zz

            # Sphere folding
            r = np.sqrt(xx*xx + yy*yy + zz*zz)
            if r < 0.5:
                factor = 1.0 / (0.5 * 0.5)
                xx, yy, zz = xx * factor, yy * factor, zz * factor
                dr *= factor
            elif r < 1.0:
                factor = 1.0 / (r * r)
                xx, yy, zz = xx * factor, yy * factor, zz * factor
                dr *= factor

            # Scaling
            xx = xx * scale + x
            yy = yy * scale + y
            zz = zz * scale + z
            dr = dr * abs(scale) + 1.0

            # Check bailout after iteration
            r = np.sqrt(xx*xx + yy*yy + zz*zz)
            if r > bailout:
                break

        # Final distance calculation
        r_final = np.sqrt(xx*xx + yy*yy + zz*zz)
        return 0.5 * r_final * np.log(r_final + 1e-10) / dr, orbit_trap, i

    def distance_estimate(self, x: float, y: float, z: float, params: FormulaParameters) -> Tuple[float, float, int]:
        if isinstance(params, MandelboxParams):
            return MandelboxFormula._distance_estimate(x, y, z, params.power, params.iterations, params.bailout,
                                        params.folding_limit, params.folding_value, params.scale)
        else:
            # Fallback to basic parameters
            return MandelboxFormula._distance_estimate(x, y, z, params.power, params.iterations, params.bailout,
                                        1.0, 2.0, -1.5)

class JuliaFormula(FractalFormula):
    """3D Julia set fractal formula"""

    def __init__(self):
        super().__init__(
            "julia",
            "3D Julia set fractal with complex constant parameters"
        )

    def get_parameters_class(self):
        return JuliaParams

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _distance_estimate(x: float, y: float, z: float, power: float, max_iter: int, bailout: float,
                          cx: float, cy: float, cz: float) -> Tuple[float, float, int]:
        """JIT-compiled Julia set distance estimation"""
        xx, yy, zz = x, y, z
        dr = 1.0
        r = np.sqrt(xx*xx + yy*yy + zz*zz)
        orbit_trap = 1000.0
        i = 0

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

    def distance_estimate(self, x: float, y: float, z: float, params: FormulaParameters) -> Tuple[float, float, int]:
        if isinstance(params, JuliaParams):
            return JuliaFormula._distance_estimate(x, y, z, params.power, params.iterations, params.bailout,
                                        params.cx, params.cy, params.cz)
        else:
            # Fallback
            return JuliaFormula._distance_estimate(x, y, z, params.power, params.iterations, params.bailout,
                                        -0.2, 0.1, 0.0)

class BurningShipFormula(FractalFormula):
    """Burning Ship fractal formula"""

    def __init__(self):
        super().__init__(
            "burning_ship",
            "Burning Ship fractal using absolute values in iteration"
        )

    def get_parameters_class(self):
        return BurningShipParams

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _distance_estimate(x: float, y: float, z: float, power: float, max_iter: int, bailout: float) -> Tuple[float, float, int]:
        """JIT-compiled Burning Ship distance estimation"""
        xx, yy, zz = x, y, z
        dr = 1.0
        r = np.sqrt(xx*xx + yy*yy + zz*zz)
        orbit_trap = 1000.0
        i = 0

        for i in range(max_iter):
            if r > bailout:
                break

            orbit_trap = min(orbit_trap, r)

            # Take absolute values (Burning Ship characteristic)
            xx = abs(xx)
            yy = abs(yy)
            zz = abs(zz)

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

    def distance_estimate(self, x: float, y: float, z: float, params: FormulaParameters) -> Tuple[float, float, int]:
        return BurningShipFormula._distance_estimate(x, y, z, params.power, params.iterations, params.bailout)

class MengerFormula(FractalFormula):
    """Menger Sponge fractal formula"""

    def __init__(self):
        super().__init__(
            "menger",
            "Menger Sponge fractal using iterative cube subtraction"
        )

    def get_parameters_class(self):
        return MengerParams

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _distance_estimate(x: float, y: float, z: float, power: float, max_iter: int, bailout: float,
                          scale: float, offset: float) -> Tuple[float, float, int]:
        """JIT-compiled Menger Sponge distance estimation"""
        xx, yy, zz = x, y, z
        orbit_trap = 1000.0
        i = 0
        r = np.sqrt(xx*xx + yy*yy + zz*zz)

        for i in range(max_iter):
            r = np.sqrt(xx*xx + yy*yy + zz*zz)
            if r > bailout:
                break

            orbit_trap = min(orbit_trap, r)

            # Menger sponge iteration
            xx = abs(xx)
            yy = abs(yy)
            zz = abs(zz)

            if xx < yy:
                xx, yy = yy, xx
            if xx < zz:
                xx, zz = zz, xx
            if yy < zz:
                yy, zz = zz, yy

            xx = scale * xx - offset * (scale - 1.0)
            yy = scale * yy - offset * (scale - 1.0)
            zz = scale * zz

            if zz > 0.5 * offset * (scale - 1.0):
                zz -= offset * (scale - 1.0)

        return r * np.log(r) / 1.0, orbit_trap, i

    def distance_estimate(self, x: float, y: float, z: float, params: FormulaParameters) -> Tuple[float, float, int]:
        if isinstance(params, MengerParams):
            return MengerFormula._distance_estimate(x, y, z, params.power, params.iterations, params.bailout,
                                        params.scale, params.offset)
        else:
            # Fallback
            return MengerFormula._distance_estimate(x, y, z, params.power, params.iterations, params.bailout,
                                        3.0, 1.0)

class BuffaloFormula(FractalFormula):
    """Buffalo fractal formula - variation with absolute values"""

    def __init__(self):
        super().__init__(
            "buffalo",
            "Buffalo fractal using absolute values in the iteration"
        )

    def get_parameters_class(self):
        return FormulaParameters

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _distance_estimate(x: float, y: float, z: float, power: float, max_iter: int, bailout: float) -> Tuple[float, float, int]:
        """JIT-compiled Buffalo distance estimation"""
        xx, yy, zz = x, y, z
        dr = 1.0
        r = np.sqrt(xx*xx + yy*yy + zz*zz)
        orbit_trap = 1000.0
        i = 0

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

            # Buffalo variation: take absolute values
            xx = abs(zr * np.sin(theta) * np.cos(phi)) + x
            yy = abs(zr * np.sin(theta) * np.sin(phi)) + y
            zz = abs(zr * np.cos(theta)) + z

            r = np.sqrt(xx*xx + yy*yy + zz*zz)

        return 0.5 * np.log(r) * r / dr, orbit_trap, i

    def distance_estimate(self, x: float, y: float, z: float, params: FormulaParameters) -> Tuple[float, float, int]:
        return BuffaloFormula._distance_estimate(x, y, z, params.power, params.iterations, params.bailout)

class CelticFormula(FractalFormula):
    """Celtic fractal formula - variation with different folding"""

    def __init__(self):
        super().__init__(
            "celtic",
            "Celtic fractal using different folding operations"
        )

    def get_parameters_class(self):
        return FormulaParameters

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _distance_estimate(x: float, y: float, z: float, power: float, max_iter: int, bailout: float) -> Tuple[float, float, int]:
        """JIT-compiled Celtic distance estimation"""
        xx, yy, zz = x, y, z
        dr = 1.0
        r = np.sqrt(xx*xx + yy*yy + zz*zz)
        orbit_trap = 1000.0
        i = 0

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

            # Celtic variation: different folding
            xx = zr * np.sin(theta) * np.cos(phi)
            yy = zr * np.sin(theta) * np.sin(phi)
            zz = zr * np.cos(theta)

            # Celtic folding
            xx = abs(xx) - 1.0
            yy = abs(yy) - 1.0
            zz = abs(zz) - 1.0

            xx += x
            yy += y
            zz += z

            r = np.sqrt(xx*xx + yy*yy + zz*zz)

        return 0.5 * np.log(r) * r / dr, orbit_trap, i

    def distance_estimate(self, x: float, y: float, z: float, params: FormulaParameters) -> Tuple[float, float, int]:
        return CelticFormula._distance_estimate(x, y, z, params.power, params.iterations, params.bailout)

@dataclass
class AmazingBoxParams(FormulaParameters):
    """Parameters specific to Amazing Box formula"""
    folding_limit: float = 1.0
    folding_value: float = 2.0
    scale: float = -1.5
    min_radius: float = 0.5
    fixed_radius: float = 1.0

@dataclass
class QuaternionParams(FormulaParameters):
    """Parameters specific to Quaternion formula"""
    pass

@dataclass
class SierpinskiParams(FormulaParameters):
    """Parameters specific to Sierpinski formula"""
    scale: float = 2.0
    offset: float = 1.0

@dataclass
class TricornParams(FormulaParameters):
    """Parameters specific to Tricorn formula"""
    pass

@dataclass
class PhoenixParams(FormulaParameters):
    """Parameters specific to Phoenix formula"""
    phoenix_c: float = 0.5667

class AmazingBoxFormula(FractalFormula):
    """Amazing Box fractal formula - advanced Mandelbox variation"""

    def __init__(self):
        super().__init__(
            "amazing_box",
            "Amazing Box fractal with enhanced folding operations"
        )

    def get_parameters_class(self):
        return AmazingBoxParams

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _distance_estimate(x: float, y: float, z: float, power: float, max_iter: int, bailout: float,
                          folding_limit: float, folding_value: float, scale: float,
                          min_radius: float, fixed_radius: float) -> Tuple[float, float, int]:
        """JIT-compiled Amazing Box distance estimation"""
        xx, yy, zz = x, y, z
        orbit_trap = 1000.0
        dr = 1.0
        i = 0

        for i in range(max_iter):
            r = np.sqrt(xx*xx + yy*yy + zz*zz)

            orbit_trap = min(orbit_trap, r)

            # Amazing Box folding
            if xx > folding_limit:
                xx = folding_value - xx
            elif xx < -folding_limit:
                xx = -folding_value - xx

            if yy > folding_limit:
                yy = folding_value - yy
            elif yy < -folding_limit:
                yy = -folding_value - yy

            if zz > folding_limit:
                zz = folding_value - zz
            elif zz < -folding_limit:
                zz = -folding_value - zz

            # Sphere folding with min_radius and fixed_radius
            r = np.sqrt(xx*xx + yy*yy + zz*zz)
            if r < min_radius:
                factor = fixed_radius / min_radius
                xx, yy, zz = xx * factor, yy * factor, zz * factor
                dr *= factor
            elif r < fixed_radius:
                factor = fixed_radius / r
                xx, yy, zz = xx * factor, yy * factor, zz * factor
                dr *= factor

            # Scaling
            xx = xx * scale + x
            yy = yy * scale + y
            zz = zz * scale + z
            dr = dr * abs(scale) + 1.0

            # Check bailout after iteration
            r = np.sqrt(xx*xx + yy*yy + zz*zz)
            if r > bailout:
                break

        # Final distance calculation
        r_final = np.sqrt(xx*xx + yy*yy + zz*zz)
        return 0.5 * r_final * np.log(r_final + 1e-10) / dr, orbit_trap, i

    def distance_estimate(self, x: float, y: float, z: float, params: FormulaParameters) -> Tuple[float, float, int]:
        if isinstance(params, AmazingBoxParams):
            return AmazingBoxFormula._distance_estimate(x, y, z, params.power, params.iterations, params.bailout,
                                        params.folding_limit, params.folding_value, params.scale,
                                        params.min_radius, params.fixed_radius)
        else:
            # Fallback
            return AmazingBoxFormula._distance_estimate(x, y, z, params.power, params.iterations, params.bailout,
                                        1.0, 2.0, -1.5, 0.5, 1.0)

class QuaternionFormula(FractalFormula):
    """Quaternion fractal formula"""

    def __init__(self):
        super().__init__(
            "quaternion",
            "Quaternion fractal using quaternion mathematics"
        )

    def get_parameters_class(self):
        return QuaternionParams

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _distance_estimate(x: float, y: float, z: float, power: float, max_iter: int, bailout: float) -> Tuple[float, float, int]:
        """JIT-compiled Quaternion distance estimation"""
        # For simplicity, using a Mandelbulb-like formula with quaternion-inspired operations
        xx, yy, zz = x, y, z
        dr = 1.0
        r = np.sqrt(xx*xx + yy*yy + zz*zz)
        orbit_trap = 1000.0
        i = 0

        for i in range(max_iter):
            if r > bailout:
                break

            orbit_trap = min(orbit_trap, r)

            # Quaternion-like operations
            theta = np.arctan2(np.sqrt(xx*xx + yy*yy), zz)
            phi = np.arctan2(yy, xx)

            zr = r ** (power - 1.0)
            dr = zr * dr * power + 1.0

            zr = zr * r
            theta = theta * power
            phi = phi * power

            # Quaternion multiplication pattern
            xx = zr * np.sin(theta) * np.cos(phi) + x
            yy = zr * np.sin(theta) * np.sin(phi) + y
            zz = zr * np.cos(theta) + z

            r = np.sqrt(xx*xx + yy*yy + zz*zz)

        return 0.5 * np.log(r) * r / dr, orbit_trap, i

    def distance_estimate(self, x: float, y: float, z: float, params: FormulaParameters) -> Tuple[float, float, int]:
        return QuaternionFormula._distance_estimate(x, y, z, params.power, params.iterations, params.bailout)

class SierpinskiFormula(FractalFormula):
    """Sierpinski fractal formula"""

    def __init__(self):
        super().__init__(
            "sierpinski",
            "Sierpinski tetrahedron fractal"
        )

    def get_parameters_class(self):
        return SierpinskiParams

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _distance_estimate(x: float, y: float, z: float, power: float, max_iter: int, bailout: float,
                          scale: float, offset: float) -> Tuple[float, float, int]:
        """JIT-compiled Sierpinski distance estimation"""
        xx, yy, zz = abs(x), abs(y), abs(z)
        orbit_trap = 1000.0
        dr = 1.0
        i = 0

        for i in range(max_iter):
            # Distance to current tetrahedron approximation
            dist_to_tetra = max(xx + yy + zz - offset, xx + yy - zz - offset,
                               xx - yy + zz - offset, -xx + yy + zz - offset) / np.sqrt(3.0)

            orbit_trap = min(orbit_trap, dist_to_tetra)

            # Sierpinski tetrahedron folding
            if xx + yy < 0.0:
                xx, yy = -yy, -xx
            if xx + zz < 0.0:
                xx, zz = -zz, -xx
            if yy + zz < 0.0:
                yy, zz = -zz, -yy

            # Scale and offset
            xx = scale * xx - offset * (scale - 1.0)
            yy = scale * yy - offset * (scale - 1.0)
            zz = scale * zz - offset * (scale - 1.0)
            dr = dr * scale

            # Check bailout
            r = np.sqrt(xx*xx + yy*yy + zz*zz)
            if r > bailout:
                break

        # Final distance estimation
        r_final = np.sqrt(xx*xx + yy*yy + zz*zz)
        return r_final / dr, orbit_trap, i

    def distance_estimate(self, x: float, y: float, z: float, params: FormulaParameters) -> Tuple[float, float, int]:
        if isinstance(params, SierpinskiParams):
            return SierpinskiFormula._distance_estimate(x, y, z, params.power, params.iterations, params.bailout,
                                        params.scale, params.offset)
        else:
            # Fallback
            return SierpinskiFormula._distance_estimate(x, y, z, params.power, params.iterations, params.bailout,
                                        2.0, 1.0)

class TricornFormula(FractalFormula):
    """Tricorn fractal formula"""

    def __init__(self):
        super().__init__(
            "tricorn",
            "Tricorn fractal (Mandelbar) using conjugate iteration"
        )

    def get_parameters_class(self):
        return TricornParams

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _distance_estimate(x: float, y: float, z: float, power: float, max_iter: int, bailout: float) -> Tuple[float, float, int]:
        """JIT-compiled Tricorn distance estimation"""
        xx, yy, zz = x, y, z
        dr = 1.0
        r = np.sqrt(xx*xx + yy*yy + zz*zz)
        orbit_trap = 1000.0
        i = 0

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
            phi = -phi * power  # Tricorn uses negative phi

            xx = zr * np.sin(theta) * np.cos(phi) + x
            yy = zr * np.sin(theta) * np.sin(phi) + y
            zz = zr * np.cos(theta) + z

            r = np.sqrt(xx*xx + yy*yy + zz*zz)

        return 0.5 * np.log(r) * r / dr, orbit_trap, i

    def distance_estimate(self, x: float, y: float, z: float, params: FormulaParameters) -> Tuple[float, float, int]:
        return TricornFormula._distance_estimate(x, y, z, params.power, params.iterations, params.bailout)

class PhoenixFormula(FractalFormula):
    """Phoenix fractal formula"""

    def __init__(self):
        super().__init__(
            "phoenix",
            "Phoenix fractal using previous iteration values"
        )

    def get_parameters_class(self):
        return PhoenixParams

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def _distance_estimate(x: float, y: float, z: float, power: float, max_iter: int, bailout: float,
                          phoenix_c: float) -> Tuple[float, float, int]:
        """JIT-compiled Phoenix distance estimation"""
        xx, yy, zz = x, y, z
        prev_x, prev_y, prev_z = x, y, z
        dr = 1.0
        r = np.sqrt(xx*xx + yy*yy + zz*zz)
        orbit_trap = 1000.0
        i = 0

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

            # Phoenix uses previous iteration
            new_x = zr * np.sin(theta) * np.cos(phi) + phoenix_c * prev_x + x
            new_y = zr * np.sin(theta) * np.sin(phi) + phoenix_c * prev_y + y
            new_z = zr * np.cos(theta) + phoenix_c * prev_z + z

            prev_x, prev_y, prev_z = xx, yy, zz
            xx, yy, zz = new_x, new_y, new_z

            r = np.sqrt(xx*xx + yy*yy + zz*zz)

        return 0.5 * np.log(r) * r / dr, orbit_trap, i

    def distance_estimate(self, x: float, y: float, z: float, params: FormulaParameters) -> Tuple[float, float, int]:
        if isinstance(params, PhoenixParams):
            return PhoenixFormula._distance_estimate(x, y, z, params.power, params.iterations, params.bailout,
                                        params.phoenix_c)
        else:
            # Fallback
            return PhoenixFormula._distance_estimate(x, y, z, params.power, params.iterations, params.bailout,
                                        0.5667)

# =============================================================================
# HYBRID/COMBINED FORMULAS
# =============================================================================

@dataclass
class HybridFormulaParams(FormulaParameters):
    """Parameters for hybrid/combined formulas"""
    formula1_weight: float = 0.5
    formula2_weight: float = 0.5
    blend_mode: str = "min"  # min, max, average, multiply

class HybridFormula(FractalFormula):
    """Hybrid formula combining two base formulas"""

    def __init__(self, formula1: FractalFormula, formula2: FractalFormula,
                 name: Optional[str] = None, blend_mode: str = "min"):
        hybrid_name = name or f"{formula1.name}_{formula2.name}_hybrid"
        description = f"Hybrid of {formula1.name} and {formula2.name} using {blend_mode} blending"
        super().__init__(hybrid_name, description)

        self.formula1 = formula1
        self.formula2 = formula2
        self.blend_mode = blend_mode

    def get_parameters_class(self):
        return HybridFormulaParams

    def distance_estimate(self, x: float, y: float, z: float, params: FormulaParameters) -> Tuple[float, float, int]:
        if isinstance(params, HybridFormulaParams):
            # Get distance estimates from both formulas
            de1, trap1, iter1 = self.formula1.distance_estimate(x, y, z, params)
            de2, trap2, iter2 = self.formula2.distance_estimate(x, y, z, params)

            # Blend distance estimates
            if self.blend_mode == "min":
                blended_de = min(de1, de2)
            elif self.blend_mode == "max":
                blended_de = max(de1, de2)
            elif self.blend_mode == "average":
                blended_de = (de1 + de2) / 2.0
            elif self.blend_mode == "multiply":
                blended_de = de1 * de2
            else:
                blended_de = min(de1, de2)  # default to min

            # Blend orbit traps
            blended_trap = min(trap1, trap2)
            blended_iter = max(iter1, iter2)

            return blended_de, blended_trap, blended_iter
        else:
            # Fallback to formula1
            return self.formula1.distance_estimate(x, y, z, params)

# =============================================================================
# FORMULA REGISTRY AND MANAGEMENT
# =============================================================================

class FormulaRegistry:
    """Registry for managing fractal formulas"""

    def __init__(self):
        self.formulas: Dict[str, FractalFormula] = {}
        self._register_builtin_formulas()

    def _register_builtin_formulas(self):
        """Register all built-in fractal formulas"""
        self.register_formula(MandelbulbFormula())
        self.register_formula(MandelboxFormula())
        self.register_formula(JuliaFormula())
        self.register_formula(BurningShipFormula())
        self.register_formula(MengerFormula())
        self.register_formula(BuffaloFormula())
        self.register_formula(CelticFormula())
        self.register_formula(AmazingBoxFormula())
        self.register_formula(QuaternionFormula())
        self.register_formula(SierpinskiFormula())
        self.register_formula(TricornFormula())
        self.register_formula(PhoenixFormula())

        # Register hybrid formulas
        mandelbulb = self.get_formula("mandelbulb")
        mandelbox = self.get_formula("mandelbox")
        julia = self.get_formula("julia")

        if mandelbulb and mandelbox:
            self.register_formula(HybridFormula(mandelbulb, mandelbox, "mandelbulb_mandelbox_min", "min"))
            self.register_formula(HybridFormula(mandelbulb, mandelbox, "mandelbulb_mandelbox_average", "average"))

        if mandelbulb and julia:
            self.register_formula(HybridFormula(mandelbulb, julia, "mandelbulb_julia_min", "min"))

    def register_formula(self, formula: FractalFormula):
        """Register a new formula"""
        self.formulas[formula.name] = formula
        logger.info(f"Registered formula: {formula.name}")

    def get_formula(self, name: str) -> Optional[FractalFormula]:
        """Get a formula by name"""
        return self.formulas.get(name)

    def list_formulas(self) -> List[str]:
        """List all registered formula names"""
        return list(self.formulas.keys())

    def get_formula_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a formula"""
        formula = self.get_formula(name)
        if formula:
            return formula.to_dict()
        return None

    def save_registry(self, filepath: str):
        """Save formula registry to JSON file"""
        registry_data = {
            "formulas": {name: formula.to_dict() for name, formula in self.formulas.items()},
            "timestamp": str(np.datetime64('now'))
        }

        with open(filepath, 'w') as f:
            json.dump(registry_data, f, indent=2)

        logger.info(f"Saved formula registry to {filepath}")

    def load_registry(self, filepath: str):
        """Load formula registry from JSON file"""
        with open(filepath, 'r') as f:
            registry_data = json.load(f)

        # Note: This only loads metadata, not the actual formula objects
        # Formulas need to be re-registered programmatically
        logger.info(f"Loaded formula registry metadata from {filepath}")
        return registry_data

# =============================================================================
# RENDERING INTEGRATION
# =============================================================================

class FormulaRenderer:
    """Renderer that can handle different fractal formulas"""

    def __init__(self, registry: FormulaRegistry):
        self.registry = registry
        self.formula_type_map = {}

    def render_formula(self, formula_name: str, params: FormulaParameters,
                      width: int = 800, height: int = 600) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Render a specific formula with given parameters"""

        formula = self.registry.get_formula(formula_name)
        if not formula:
            raise ValueError(f"Formula '{formula_name}' not found in registry")

        # Create image buffer
        image = np.zeros((height, width, 3), dtype=np.float64)

        # Get formula-specific parameters
        formula_params = params if isinstance(params, formula.get_parameters_class()) else formula.get_default_parameters()

        # Render using the formula's distance estimation
        image, metrics = self._render_with_formula(image, width, height, formula, formula_params)

        # Add formula metadata to metrics
        metrics.update({
            'formula_name': formula_name,
            'formula_description': formula.description,
            'parameters': formula_params.to_dict()
        })

        return image, metrics

    def _render_with_formula(self, image: np.ndarray, width: int, height: int,
                           formula: FractalFormula, params: FormulaParameters) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Internal rendering function using formula's distance estimation"""

        # Simple ray marching implementation
        # This is a basic implementation - could be optimized further
        start_time = time.time()

        # Camera setup (simplified)
        camera_pos = np.array([0.0, 0.0, -3.0])
        target = np.array([0.0, 0.0, 0.0])
        up = np.array([0.0, 1.0, 0.0])

        forward = target - camera_pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        camera_up = np.cross(right, forward)

        fov_rad = 45.0 * np.pi / 180.0
        h = np.tan(fov_rad / 2.0)
        w = h * (width / height)

        max_steps = 100
        epsilon = 0.001
        max_dist = 10.0

        # Render each pixel
        for j in range(height):
            for i in range(width):
                # Pixel to world coordinates
                u = (2.0 * i / width - 1.0) * w
                v = (2.0 * j / height - 1.0) * h

                ray_dir = forward + u * right + v * camera_up
                ray_dir = ray_dir / np.linalg.norm(ray_dir)

                # Ray marching
                t = 0.0
                hit = False
                orbit_trap = 1000.0
                final_iterations = 0

                for step in range(max_steps):
                    pos = camera_pos + t * ray_dir
                    dist, trap, iters = formula.distance_estimate(pos[0], pos[1], pos[2], params)

                    orbit_trap = min(orbit_trap, trap)
                    final_iterations = iters

                    if abs(dist) < epsilon:
                        hit = True
                        break

                    t += max(abs(dist), epsilon)

                    if t > max_dist:
                        break

                if hit:
                    # Simple coloring based on orbit trap
                    color_t = min(1.0, orbit_trap)
                    r = 0.8 + 0.2 * np.sin(color_t * 3.14159 * 2.0)
                    g = 0.4 + 0.4 * np.sin(color_t * 3.14159 * 2.0 + 1.0)
                    b = 0.2 + 0.3 * np.sin(color_t * 3.14159 * 2.0 + 2.0)

                    image[j, i, 0] = r
                    image[j, i, 1] = g
                    image[j, i, 2] = b

        render_time = time.time() - start_time

        # Calculate basic metrics
        surface_mask = np.any(image > 0.1, axis=2)
        surface_pixels = np.sum(surface_mask)
        unique_colors = len(np.unique(image.reshape(-1, 3), axis=0))

        metrics = {
            'render_time': render_time,
            'surface_pixels': int(surface_pixels),
            'unique_colors': int(unique_colors),
            'resolution': f"{width}x{height}"
        }

        return image, metrics

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_formula_showcase(output_dir: str = "output/formula_showcase"):
    """Create a showcase of all available formulas"""

    # Initialize registry and renderer
    registry = FormulaRegistry()
    renderer = FormulaRenderer(registry)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []

    # Render each formula
    for formula_name in registry.list_formulas():
        logger.info(f"Rendering formula: {formula_name}")

        try:
            # Get formula and its default parameters
            formula = registry.get_formula(formula_name)
            if formula is None:
                continue
            params = formula.get_default_parameters()

            # Render
            image, metrics = renderer.render_formula(formula_name, params)

            # Save image
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 8))
            plt.imshow(image, origin='upper')
            plt.axis('off')
            plt.title(f"{formula_name}\n{formula.description}")

            output_file = output_path / f"{formula_name}.png"
            plt.savefig(output_file, dpi=100, bbox_inches='tight', facecolor='black')
            plt.close()

            result = {
                'formula_name': formula_name,
                'description': formula.description,
                'success': True,
                'render_time': metrics['render_time'],
                'surface_pixels': metrics['surface_pixels'],
                'output_file': str(output_file)
            }

            results.append(result)
            logger.info(f"✓ {formula_name}: {metrics['render_time']:.2f}s, {metrics['surface_pixels']} pixels")

        except Exception as e:
            logger.error(f"✗ Failed to render {formula_name}: {e}")
            results.append({
                'formula_name': formula_name,
                'success': False,
                'error': str(e)
            })

    # Save results summary
    summary_file = output_path / "render_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Save formula registry
    registry_file = output_path / "formula_registry.json"
    registry.save_registry(str(registry_file))

    logger.info(f"Formula showcase complete! Results saved to {output_path}")
    return results

if __name__ == "__main__":
    # Test the formula system
    logging.basicConfig(level=logging.INFO)

    print("🌀 Fractal Formula System Test")
    print("=" * 50)

    # Create registry and test formulas
    registry = FormulaRegistry()

    print(f"Registered formulas: {registry.list_formulas()}")

    # Test individual formula info
    mandelbulb_info = registry.get_formula_info("mandelbulb")
    if mandelbulb_info:
        print(f"\nMandelbulb info: {mandelbulb_info}")

    # Create showcase
    print("\nCreating formula showcase...")
    results = create_formula_showcase()

    successful = sum(1 for r in results if r.get('success', False))
    print(f"\n✅ Showcase complete: {successful}/{len(results)} formulas rendered successfully")