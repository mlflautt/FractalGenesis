"""
Fractal Formulas Package
========================

Comprehensive fractal formula library with:
- 40+ individual formulas
- Hybrid formula combinations  
- dIFS geometric shapes
- Meta-parameter control
- Random search and exploration

Usage:
    from formulas import FormulaRegistry, FormulaRandomSearch
    
    # Initialize
    registry = FormulaRegistry()
    search = FormulaRandomSearch(registry)
    
    # Explore
    interesting = search.explore_random(num_samples=100)
"""

from .formula_registry import FormulaRegistry, FormulaRandomSearch
from .extended_library import (
    FractalFormula, FormulaParams, MetaParameter, MetaParameterType,
    FoldingParams, AexionParams, BenesiParams, dIFSParams, HybridParams
)
from .hybrid_system import (
    HybridFormula, FormulaCombiner, FormulaStack,
    BlendMode, HybridType, HybridConfig
)

# Import dIFS formulas
from .difs_library import (
    SierpinskiTetrahedron, MengerSpongeIFS, CrystalIFS,
    HoneycombIFS, TreeIFS
)

__all__ = [
    # Registry and Search
    'FormulaRegistry',
    'FormulaRandomSearch',
    
    # Base Classes
    'FractalFormula',
    'FormulaParams',
    'MetaParameter',
    'MetaParameterType',
    
    # Parameter Types
    'FoldingParams',
    'AexionParams',
    'BenesiParams',
    'dIFSParams',
    'HybridParams',
    
    # Hybrid System
    'HybridFormula',
    'FormulaCombiner',
    'FormulaStack',
    'BlendMode',
    'HybridType',
    'HybridConfig',
    
    # dIFS Shapes
    'SierpinskiTetrahedron',
    'MengerSpongeIFS',
    'CrystalIFS',
    'HoneycombIFS',
    'TreeIFS',
]

__version__ = "1.0.0"
