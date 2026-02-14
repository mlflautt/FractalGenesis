#!/usr/bin/env python3
"""
Formula Registry and Random Search System
========================================

Central registry for managing all fractal formulas.
Includes random search and exploration capabilities.

Usage:
    from formulas.formula_registry import FormulaRegistry
    
    # Initialize registry with all formulas
    registry = FormulaRegistry()
    
    # Get a formula
    formula = registry.get_formula("mandelbulb")
    
    # Random search
    search = FormulaRandomSearch(registry)
    interesting_formula = search.explore_random(num_samples=100)
"""

import random
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import numpy as np

# Import all formula modules
from .extended_library import (
    FractalFormula, FormulaParams, MetaParameter, MetaParameterType,
    CosineMandelbulb, ReciprocalMandelbulb, AsymmetricMandelbulb,
    AmazingBox, SmoothMandelbox,
    AexionC, AexOcto,
    Benesi2Pow2, Benesi3Pow2
)

from .difs_library import (
    SierpinskiTetrahedron, MengerSpongeIFS, CrystalIFS, HoneycombIFS, TreeIFS
)

from .hybrid_system import HybridFormula, FormulaCombiner, BlendMode, HybridConfig

# Import existing formulas from the main codebase
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from renderers.python_3d.fractal_formulas import (
    MandelbulbFormula, MandelboxFormula, JuliaFormula, 
    BurningShipFormula, MengerFormula, BuffaloFormula, CelticFormula
)

logger = logging.getLogger(__name__)


class FormulaRegistry:
    """
    Central registry for all fractal formulas.
    Manages formula registration, retrieval, and metadata.
    """
    
    def __init__(self):
        self.formulas: Dict[str, FractalFormula] = {}
        self.categories: Dict[str, List[str]] = {}
        self._register_all_formulas()
    
    def _register_all_formulas(self):
        """Register all available formulas"""
        
        # Core formulas (existing)
        self.register("mandelbulb", MandelbulbFormula(), "core")
        self.register("mandelbox", MandelboxFormula(), "core")
        self.register("julia", JuliaFormula(), "core")
        self.register("burning_ship", BurningShipFormula(), "core")
        self.register("menger", MengerFormula(), "core")
        self.register("buffalo", BuffaloFormula(), "core")
        self.register("celtic", CelticFormula(), "core")
        
        # Power variations
        self.register("cosine_bulb", CosineMandelbulb(), "power")
        self.register("reciprocal_bulb", ReciprocalMandelbulb(), "power")
        self.register("asymmetric_bulb", AsymmetricMandelbulb(), "power")
        
        # Folding variations
        self.register("amazing_box", AmazingBox(), "folding")
        self.register("smooth_box", SmoothMandelbox(), "folding")
        
        # Aexion family
        self.register("aexion_c", AexionC(), "aexion")
        self.register("aex_octo", AexOcto(), "aexion")
        
        # Benesi family
        self.register("benesi2pow2", Benesi2Pow2(), "benesi")
        self.register("benesi3pow2", Benesi3Pow2(), "benesi")
        
        # dIFS shapes
        self.register("sierpinski_tetra", SierpinskiTetrahedron(), "difs")
        self.register("menger_ifs", MengerSpongeIFS(), "difs")
        self.register("crystal_ifs", CrystalIFS(), "difs")
        self.register("honeycomb_ifs", HoneycombIFS(), "difs")
        self.register("tree_ifs", TreeIFS(), "difs")
        
        logger.info(f"Registered {len(self.formulas)} formulas")
    
    def register(self, name: str, formula: FractalFormula, category: str = "general"):
        """Register a formula"""
        self.formulas[name] = formula
        
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(name)
        
        logger.debug(f"Registered formula: {name} in category {category}")
    
    def get_formula(self, name: str) -> FractalFormula:
        """Get formula by name"""
        if name not in self.formulas:
            raise ValueError(f"Formula '{name}' not found. Available: {list(self.formulas.keys())}")
        return self.formulas[name]
    
    def list_formulas(self, category: str = None) -> List[str]:
        """List all formula names, optionally filtered by category"""
        if category:
            return self.categories.get(category, [])
        return list(self.formulas.keys())
    
    def list_categories(self) -> List[str]:
        """List all formula categories"""
        return list(self.categories.keys())
    
    def get_formulas_by_category(self, category: str) -> List[FractalFormula]:
        """Get all formulas in a category"""
        names = self.categories.get(category, [])
        return [self.formulas[name] for name in names]
    
    def get_random_formula(self, category: str = None) -> Tuple[str, FractalFormula]:
        """Get a random formula"""
        names = self.list_formulas(category)
        name = random.choice(names)
        return name, self.formulas[name]
    
    def save_formula_library(self, filepath: str):
        """Save formula library to JSON"""
        data = {
            "formulas": {
                name: {
                    "name": formula.name,
                    "description": formula.description,
                    "category": self._get_category(name),
                    "meta_parameters": [
                        {
                            "name": mp.name,
                            "type": mp.param_type.value,
                            "default": mp.default_value,
                            "min": mp.min_value,
                            "max": mp.max_value,
                            "description": mp.description
                        }
                        for mp in formula.meta_parameters
                    ]
                }
                for name, formula in self.formulas.items()
            },
            "categories": self.categories
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved formula library to {filepath}")
    
    def _get_category(self, name: str) -> str:
        """Get category for a formula name"""
        for cat, names in self.categories.items():
            if name in names:
                return cat
        return "unknown"


class FormulaRandomSearch:
    """
    Random search and exploration system for fractal formulas.
    Generates random parameter combinations and evaluates results.
    """
    
    def __init__(self, registry: FormulaRegistry):
        self.registry = registry
        self.combiner = FormulaCombiner(registry)
        self.results_history: List[Dict[str, Any]] = []
    
    def explore_random(self, num_samples: int = 100, 
                      include_hybrids: bool = True,
                      quality_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Explore random formula combinations.
        
        Args:
            num_samples: Number of random formulas to generate
            include_hybrids: Whether to include hybrid combinations
            quality_threshold: Minimum quality score to keep result
            
        Returns:
            List of interesting formula configurations
        """
        interesting = []
        
        print(f"Exploring {num_samples} random formula configurations...")
        
        for i in range(num_samples):
            # Choose random formula or hybrid
            if include_hybrids and random.random() < 0.5:
                formula, params = self._generate_random_hybrid()
            else:
                formula, params = self._generate_random_single()
            
            # Evaluate (simplified evaluation - could render sample image)
            quality_score = self._evaluate_formula(formula, params)
            
            if quality_score >= quality_threshold:
                result = {
                    "formula_name": formula.name,
                    "params": self._params_to_dict(params),
                    "quality_score": quality_score,
                    "sample_num": i
                }
                interesting.append(result)
                self.results_history.append(result)
                
                print(f"  Found interesting formula #{len(interesting)}: {formula.name} (score: {quality_score:.3f})")
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{num_samples} samples, {len(interesting)} interesting")
        
        print(f"\nExploration complete: {len(interesting)} interesting formulas found")
        return interesting
    
    def _generate_random_single(self) -> Tuple[FractalFormula, FormulaParams]:
        """Generate random single formula with random parameters"""
        name, formula = self.registry.get_random_formula()
        params = formula.randomize_params()
        return formula, params
    
    def _generate_random_hybrid(self) -> Tuple[FractalFormula, FormulaParams]:
        """Generate random hybrid formula"""
        num_formulas = random.choices([2, 3, 4], weights=[0.6, 0.3, 0.1])[0]
        hybrid = self.combiner.create_random_hybrid(num_formulas)
        params = hybrid.get_default_params()
        
        # Randomize hybrid-specific params
        for meta in hybrid.meta_parameters:
            params.set_meta(meta.name, meta.randomize())
        
        return hybrid, params
    
    def _evaluate_formula(self, formula: FractalFormula, params: FormulaParams) -> float:
        """
        Evaluate formula quality (simplified version).
        
        In practice, this would render a sample and analyze the image.
        For now, uses parameter-based heuristics.
        """
        score = 0.5  # Base score
        
        # Factor 1: Power diversity (powers far from 8.0 are more interesting)
        power = params.power if hasattr(params, 'power') else params.get_meta("power", 8.0)
        power_distance = abs(power - 8.0)
        score += min(power_distance / 10.0, 0.2)  # Up to 0.2 bonus
        
        # Factor 2: Iteration count (higher = more detail, but not too high)
        iters = params.iterations
        if 50 <= iters <= 200:
            score += 0.1
        
        # Factor 3: Meta-parameter diversity
        if hasattr(params, 'meta_params') and params.meta_params:
            num_meta = len(params.meta_params)
            score += min(num_meta * 0.02, 0.1)  # Up to 0.1 bonus
        
        # Factor 4: Hybrid bonus
        if isinstance(formula, HybridFormula):
            score += 0.15  # Hybrids tend to be more interesting
        
        # Factor 5: Random variation (simulates rendering luck)
        score += random.uniform(-0.1, 0.1)
        
        return max(0.0, min(1.0, score))
    
    def _params_to_dict(self, params: FormulaParams) -> Dict[str, Any]:
        """Convert parameters to dictionary"""
        return {
            "power": params.power if hasattr(params, 'power') else params.get_meta("power", 8.0),
            "iterations": params.iterations,
            "bailout": params.bailout,
            "meta_params": params.meta_params if hasattr(params, 'meta_params') else {}
        }
    
    def mutate_interesting_formula(self, result: Dict[str, Any], 
                                   num_variants: int = 5) -> List[Dict[str, Any]]:
        """
        Create mutated variants of an interesting formula.
        
        Args:
            result: The interesting formula result to mutate
            num_variants: Number of variants to create
            
        Returns:
            List of mutated results
        """
        variants = []
        formula_name = result["formula_name"]
        
        try:
            formula = self.registry.get_formula(formula_name)
        except ValueError:
            # Might be a hybrid
            return []
        
        print(f"Creating {num_variants} variants of {formula_name}...")
        
        for i in range(num_variants):
            # Create mutated params
            base_params = self._dict_to_params(result["params"])
            mutated_params = formula.mutate_params(base_params, strength=0.15)
            
            # Evaluate
            quality_score = self._evaluate_formula(formula, mutated_params)
            
            variant = {
                "formula_name": formula_name,
                "params": self._params_to_dict(mutated_params),
                "quality_score": quality_score,
                "parent": result,
                "variant_num": i
            }
            variants.append(variant)
        
        return variants
    
    def _dict_to_params(self, data: Dict[str, Any]) -> FormulaParams:
        """Convert dictionary to parameters"""
        params = FormulaParams()
        params.power = data.get("power", 8.0)
        params.iterations = data.get("iterations", 100)
        params.bailout = data.get("bailout", 4.0)
        
        if "meta_params" in data:
            params.meta_params = data["meta_params"].copy()
        
        return params
    
    def crossover_formulas(self, result1: Dict[str, Any], 
                          result2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Crossover two formulas to create offspring.
        
        Args:
            result1: First parent formula
            result2: Second parent formula
            
        Returns:
            Offspring formula result or None if not compatible
        """
        name1 = result1["formula_name"]
        name2 = result2["formula_name"]
        
        # If same formula, interpolate parameters
        if name1 == name2:
            try:
                formula = self.registry.get_formula(name1)
                
                # Interpolate parameters
                params1 = self._dict_to_params(result1["params"])
                params2 = self._dict_to_params(result2["params"])
                
                offspring_params = FormulaParams()
                offspring_params.power = (params1.power + params2.power) / 2
                offspring_params.iterations = int((params1.iterations + params2.iterations) / 2)
                offspring_params.bailout = (params1.bailout + params2.bailout) / 2
                
                # Blend meta parameters
                for key in set(params1.meta_params.keys()) | set(params2.meta_params.keys()):
                    v1 = params1.get_meta(key, 0.0)
                    v2 = params2.get_meta(key, 0.0)
                    offspring_params.set_meta(key, (v1 + v2) / 2)
                
                quality_score = self._evaluate_formula(formula, offspring_params)
                
                return {
                    "formula_name": name1,
                    "params": self._params_to_dict(offspring_params),
                    "quality_score": quality_score,
                    "parent1": result1,
                    "parent2": result2,
                    "is_crossover": True
                }
            except:
                return None
        
        # Different formulas - create hybrid
        try:
            hybrid = self.combiner.create_formula_pairing(name1, name2)
            params = hybrid.get_default_params()
            
            quality_score = self._evaluate_formula(hybrid, params)
            
            return {
                "formula_name": hybrid.name,
                "params": self._params_to_dict(params),
                "quality_score": quality_score,
                "parent1": result1,
                "parent2": result2,
                "is_hybrid": True
            }
        except:
            return None
    
    def save_results(self, filepath: str):
        """Save search results to JSON"""
        data = {
            "num_results": len(self.results_history),
            "results": self.results_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(self.results_history)} results to {filepath}")
    
    def load_results(self, filepath: str) -> List[Dict[str, Any]]:
        """Load search results from JSON"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.results_history = data.get("results", [])
        print(f"Loaded {len(self.results_history)} results from {filepath}")
        return self.results_history


def demo_formula_exploration():
    """Demonstrate formula exploration"""
    print("="*60)
    print("Fractal Formula Exploration Demo")
    print("="*60)
    
    # Initialize registry
    registry = FormulaRegistry()
    
    print(f"\nLoaded {len(registry.formulas)} formulas:")
    for category in registry.list_categories():
        formulas = registry.list_formulas(category)
        print(f"  {category}: {len(formulas)} formulas")
    
    # Show some example formulas
    print("\nExample formulas:")
    for name in list(registry.formulas.keys())[:5]:
        formula = registry.get_formula(name)
        print(f"  - {name}: {formula.description}")
        print(f"    Meta-parameters: {[mp.name for mp in formula.meta_parameters[:3]]}")
    
    # Initialize search
    search = FormulaRandomSearch(registry)
    
    # Explore
    print("\n" + "="*60)
    print("Starting random exploration...")
    print("="*60)
    
    interesting = search.explore_random(
        num_samples=50,
        include_hybrids=True,
        quality_threshold=0.4
    )
    
    # Show top results
    print("\n" + "="*60)
    print("Top 5 Interesting Formulas:")
    print("="*60)
    
    sorted_results = sorted(interesting, key=lambda x: x["quality_score"], reverse=True)
    for i, result in enumerate(sorted_results[:5], 1):
        print(f"\n{i}. {result['formula_name']}")
        print(f"   Quality Score: {result['quality_score']:.3f}")
        print(f"   Power: {result['params']['power']:.2f}")
        print(f"   Iterations: {result['params']['iterations']}")
        if result['params'].get('meta_params'):
            print(f"   Meta-params: {result['params']['meta_params']}")
    
    # Demonstrate mutation
    if interesting:
        print("\n" + "="*60)
        print("Creating Variants of Top Formula...")
        print("="*60)
        
        variants = search.mutate_interesting_formula(sorted_results[0], num_variants=3)
        for i, variant in enumerate(variants, 1):
            print(f"{i}. Variant score: {variant['quality_score']:.3f}")
    
    # Demonstrate crossover
    if len(interesting) >= 2:
        print("\n" + "="*60)
        print("Crossover Between Top 2 Formulas...")
        print("="*60)
        
        offspring = search.crossover_formulas(sorted_results[0], sorted_results[1])
        if offspring:
            print(f"Offspring: {offspring['formula_name']}")
            print(f"Score: {offspring['quality_score']:.3f}")
            if offspring.get('is_hybrid'):
                print("Type: Hybrid formula")
    
    # Save results
    search.save_results("formula_exploration_results.json")
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)


if __name__ == "__main__":
    demo_formula_exploration()
