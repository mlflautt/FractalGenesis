"""
Evolution Package
=================

Model: minimax-m2.5 (opencode)
Created: 2026-02-16

Advanced evolutionary algorithms for fractal evolution:
- User-guided selection (user picks favorites)
- Session management (track multiple experiments)
- AI-assisted selection (train after enough data)
- Still and animation parameter evolution

Usage:
    from evolution import FractalEvolver, SessionManager
    
    # Basic evolution
    evolver = FractalEvolver()
    population = evolver.generate_population(8)
    evolver.render_population(population, "output")
    evolver.record_selections([selected_ids])
    next_gen = evolver.evolve_next_generation()
    
    # Session management
    from evolution.session_manager import SessionManager
    manager = SessionManager()
    session = manager.create_session("my_experiment")
"""

from .simple_evolution import (
    FractalEvolver,
    AnimationEvolver,
    FractalGenome,
    quick_evolution_demo
)

from .session_manager import (
    SessionManager,
    SelectionSession,
    StillParams,
    AnimationParams
)

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
    # Simple Evolution (main interface)
    'FractalEvolver',
    'AnimationEvolver', 
    'FractalGenome',
    'quick_evolution_demo',
    
    # Session Management
    'SessionManager',
    'SelectionSession',
    'StillParams',
    'AnimationParams',
    
    # Advanced (DEAP-based)
    'BaseEvolutionStrategy',
    'NoveltySearch',
    'CMAEvolutionStrategy',
    'NSGA2Evolution',
    'IslandModelEvolution',
    'EvolutionResult',
    'run_novelty_search',
    'run_cma_es'
]

__version__ = "1.1.0"
