"""
Evolution Package
=================

Advanced evolutionary algorithms for fractal evolution using DEAP:
- Novelty Search (exploration-focused)
- CMA-ES (continuous optimization)
- NSGA-II (multi-objective)
- Island Model (parallel evolution)

Usage:
    from evolution import NoveltySearch, CMAEvolutionStrategy
    
    search = NoveltySearch(formula_registry)
    result = search.evolve(population_size=100, generations=50)
"""

from .deap_integration import (
    BaseEvolutionStrategy,
    NoveltySearch,
    CMAEvolutionStrategy,
    NSGA2Evolution,
    IslandModelEvolution,
    EvolutionResult,
    run_novelty_search,
    run_cma_es
)

__all__ = [
    'BaseEvolutionStrategy',
    'NoveltySearch',
    'CMAEvolutionStrategy',
    'NSGA2Evolution',
    'IslandModelEvolution',
    'EvolutionResult',
    'run_novelty_search',
    'run_cma_es'
]

__version__ = "1.0.0"
