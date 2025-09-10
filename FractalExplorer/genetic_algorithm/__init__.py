"""
Genetic Algorithm Engine for Fractal Evolution.

This module implements the core genetic algorithm that evolves fractal parameters
based on user preferences and AI guidance.
"""

from .evolution_engine import EvolutionEngine
from .selection import TournamentSelection, RouletteSelection
from .population import Population
from .fitness import FitnessEvaluator

__all__ = ['EvolutionEngine', 'TournamentSelection', 'RouletteSelection', 'Population', 'FitnessEvaluator']
