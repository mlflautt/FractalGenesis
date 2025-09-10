"""
Fitness Evaluation for Fractal Genetic Algorithm

This module provides fitness evaluation methods that combine user preferences
with AI predictions to guide the evolution of fractal parameters.
"""

import logging
from typing import List, Dict, Any, Optional, Callable
import numpy as np
from abc import ABC, abstractmethod

from shared.genome import FractalGenome

logger = logging.getLogger(__name__)


class FitnessEvaluator(ABC):
    """Abstract base class for fitness evaluation strategies."""
    
    @abstractmethod
    def evaluate(self, genome: FractalGenome, context: Dict[str, Any] = None) -> float:
        """
        Evaluate the fitness of a genome.
        
        Args:
            genome: Genome to evaluate
            context: Optional context information
            
        Returns:
            Fitness score (higher is better)
        """
        pass


class UserPreferenceFitnessEvaluator(FitnessEvaluator):
    """
    Fitness evaluator based solely on user preferences.
    
    This is the primary fitness evaluator for the human-in-the-loop approach,
    where fitness is determined by user selections in the 4-choice interface.
    """
    
    def __init__(self):
        self.user_selections: List[Dict[str, Any]] = []
        self.selection_weights = {'selected': 1.0, 'not_selected': -0.1, 'skipped': 0.0}
    
    def record_user_selection(self, candidates: List[FractalGenome], selected_index: int):
        """
        Record a user selection for fitness calculation.
        
        Args:
            candidates: List of candidate genomes presented to user
            selected_index: Index of selected genome (or -1 for skip)
        """
        selection_record = {
            'candidates': [g.genome_id for g in candidates],
            'selected_index': selected_index,
            'selected_id': candidates[selected_index].genome_id if 0 <= selected_index < len(candidates) else None
        }
        self.user_selections.append(selection_record)
        
        # Update fitness scores immediately
        if 0 <= selected_index < len(candidates):
            selected_genome = candidates[selected_index]
            
            # Increase fitness for selected genome
            current_fitness = selected_genome.fitness or 0.0
            selected_genome.fitness = current_fitness + self.selection_weights['selected']
            
            # Slightly decrease fitness for non-selected candidates
            for i, candidate in enumerate(candidates):
                if i != selected_index:
                    current_fitness = candidate.fitness or 0.0
                    new_fitness = current_fitness + self.selection_weights['not_selected']
                    candidate.fitness = max(0.0, new_fitness)
    
    def evaluate(self, genome: FractalGenome, context: Dict[str, Any] = None) -> float:
        """Evaluate fitness based on user selection history."""
        # Return stored fitness value (updated by record_user_selection)
        return genome.fitness or 0.0
    
    def get_user_preference_summary(self) -> Dict[str, Any]:
        """Get summary of user preferences."""
        if not self.user_selections:
            return {'total_selections': 0, 'skip_rate': 0.0}
        
        total_selections = len(self.user_selections)
        skipped_selections = sum(1 for s in self.user_selections if s['selected_index'] == -1)
        skip_rate = skipped_selections / total_selections
        
        return {
            'total_selections': total_selections,
            'skip_rate': skip_rate,
            'average_selection_index': np.mean([s['selected_index'] for s in self.user_selections if s['selected_index'] >= 0])
        }


class DiversityBoostFitnessEvaluator(FitnessEvaluator):
    """
    Fitness evaluator that boosts fitness based on genetic diversity.
    
    Encourages exploration by giving higher fitness to genomes that are
    different from the rest of the population.
    """
    
    def __init__(self, diversity_weight: float = 0.3, population_ref: Optional[List[FractalGenome]] = None):
        """
        Initialize diversity boost evaluator.
        
        Args:
            diversity_weight: Weight for diversity component (0.0-1.0)
            population_ref: Reference to current population for diversity calculation
        """
        self.diversity_weight = diversity_weight
        self.population_ref = population_ref or []
    
    def update_population_reference(self, population: List[FractalGenome]):
        """Update the reference population for diversity calculation."""
        self.population_ref = population
    
    def evaluate(self, genome: FractalGenome, context: Dict[str, Any] = None) -> float:
        """Evaluate fitness with diversity boost."""
        base_fitness = genome.fitness or 0.0
        
        if not self.population_ref or len(self.population_ref) < 2:
            return base_fitness
        
        # Calculate average diversity to all other genomes in population
        total_diversity = 0.0
        comparison_count = 0
        
        for other_genome in self.population_ref:
            if other_genome.genome_id != genome.genome_id:
                diversity = genome.calculate_diversity(other_genome)
                total_diversity += diversity
                comparison_count += 1
        
        avg_diversity = total_diversity / comparison_count if comparison_count > 0 else 0.0
        
        # Combine base fitness with diversity bonus
        diversity_bonus = avg_diversity * self.diversity_weight
        total_fitness = base_fitness + diversity_bonus
        
        return total_fitness


class NoveltyFitnessEvaluator(FitnessEvaluator):
    """
    Fitness evaluator that rewards novelty and uniqueness.
    
    Maintains an archive of previously seen genomes and rewards
    genomes that are different from all previously evaluated ones.
    """
    
    def __init__(self, archive_size: int = 100, novelty_threshold: float = 0.5):
        """
        Initialize novelty evaluator.
        
        Args:
            archive_size: Maximum size of the novelty archive
            novelty_threshold: Minimum diversity required for novelty bonus
        """
        self.archive_size = archive_size
        self.novelty_threshold = novelty_threshold
        self.novelty_archive: List[FractalGenome] = []
    
    def add_to_archive(self, genome: FractalGenome):
        """Add a genome to the novelty archive."""
        # Check if similar genome already exists in archive
        for archived_genome in self.novelty_archive:
            if genome.calculate_diversity(archived_genome) < self.novelty_threshold:
                return  # Similar genome already exists
        
        # Add to archive
        self.novelty_archive.append(genome.copy())
        
        # Maintain archive size limit
        if len(self.novelty_archive) > self.archive_size:
            # Remove oldest genome
            self.novelty_archive.pop(0)
    
    def evaluate(self, genome: FractalGenome, context: Dict[str, Any] = None) -> float:
        """Evaluate fitness with novelty bonus."""
        base_fitness = genome.fitness or 0.0
        
        if not self.novelty_archive:
            novelty_bonus = 1.0  # First genome gets max novelty bonus
        else:
            # Calculate minimum distance to archived genomes
            min_distance = min(genome.calculate_diversity(archived) for archived in self.novelty_archive)
            
            # Novelty bonus based on minimum distance
            if min_distance >= self.novelty_threshold:
                novelty_bonus = min_distance
                self.add_to_archive(genome)  # Add novel genome to archive
            else:
                novelty_bonus = 0.0
        
        return base_fitness + novelty_bonus


class CombinedFitnessEvaluator(FitnessEvaluator):
    """
    Combined fitness evaluator that uses multiple evaluation strategies.
    
    Combines user preferences, diversity, and novelty for comprehensive fitness evaluation.
    """
    
    def __init__(self, 
                 user_weight: float = 0.7,
                 diversity_weight: float = 0.2,
                 novelty_weight: float = 0.1):
        """
        Initialize combined evaluator.
        
        Args:
            user_weight: Weight for user preference component
            diversity_weight: Weight for diversity component  
            novelty_weight: Weight for novelty component
        """
        # Normalize weights
        total_weight = user_weight + diversity_weight + novelty_weight
        self.user_weight = user_weight / total_weight
        self.diversity_weight = diversity_weight / total_weight
        self.novelty_weight = novelty_weight / total_weight
        
        # Component evaluators
        self.user_evaluator = UserPreferenceFitnessEvaluator()
        self.diversity_evaluator = DiversityBoostFitnessEvaluator(diversity_weight=1.0)
        self.novelty_evaluator = NoveltyFitnessEvaluator()
    
    def record_user_selection(self, candidates: List[FractalGenome], selected_index: int):
        """Record user selection for the user preference component."""
        self.user_evaluator.record_user_selection(candidates, selected_index)
    
    def update_population_reference(self, population: List[FractalGenome]):
        """Update population reference for diversity calculation."""
        self.diversity_evaluator.update_population_reference(population)
    
    def evaluate(self, genome: FractalGenome, context: Dict[str, Any] = None) -> float:
        """Evaluate fitness using combined approach."""
        # Get component fitness scores
        user_fitness = self.user_evaluator.evaluate(genome, context)
        diversity_fitness = self.diversity_evaluator.evaluate(genome, context)
        novelty_fitness = self.novelty_evaluator.evaluate(genome, context)
        
        # Combine with weights
        combined_fitness = (self.user_weight * user_fitness +
                           self.diversity_weight * diversity_fitness +
                           self.novelty_weight * novelty_fitness)
        
        return combined_fitness
    
    def get_fitness_breakdown(self, genome: FractalGenome) -> Dict[str, float]:
        """Get detailed breakdown of fitness components."""
        user_fitness = self.user_evaluator.evaluate(genome)
        diversity_fitness = self.diversity_evaluator.evaluate(genome)
        novelty_fitness = self.novelty_evaluator.evaluate(genome)
        combined_fitness = self.evaluate(genome)
        
        return {
            'user_fitness': user_fitness,
            'diversity_fitness': diversity_fitness,
            'novelty_fitness': novelty_fitness,
            'combined_fitness': combined_fitness,
            'weights': {
                'user_weight': self.user_weight,
                'diversity_weight': self.diversity_weight,
                'novelty_weight': self.novelty_weight
            }
        }


class AIGuidedFitnessEvaluator(FitnessEvaluator):
    """
    AI-guided fitness evaluator that learns from user preferences.
    
    Uses machine learning models to predict user preferences and
    guide fitness evaluation when user feedback is not available.
    """
    
    def __init__(self):
        self.ai_model = None  # Placeholder for AI model
        self.training_data = []
        self.is_trained = False
    
    def add_training_data(self, genome: FractalGenome, user_rating: float):
        """Add training data for the AI model."""
        # Convert genome to feature vector (placeholder)
        features = self._genome_to_features(genome)
        self.training_data.append((features, user_rating))
    
    def _genome_to_features(self, genome: FractalGenome) -> List[float]:
        """Convert genome to feature vector for AI model."""
        # Placeholder implementation - extract key features
        features = []
        
        # Camera features
        features.extend(genome.camera.position)
        features.extend(genome.camera.target)
        features.append(genome.camera.fov)
        
        # Fractal features
        features.append(genome.fractal.power)
        features.append(genome.fractal.bailout)
        features.append(float(genome.fractal.iterations))
        
        # Color features
        features.extend(genome.color.base_color)
        features.append(genome.color.roughness)
        
        # Lighting features
        features.extend(genome.lighting.main_light_direction)
        features.append(genome.lighting.main_light_intensity)
        
        return features
    
    def train_model(self):
        """Train the AI model on collected user preference data."""
        if len(self.training_data) < 10:
            logger.warning("Insufficient training data for AI model")
            return False
        
        # Placeholder for actual ML model training
        # In a real implementation, this would train a regression or classification model
        self.is_trained = True
        logger.info(f"AI model trained on {len(self.training_data)} samples")
        return True
    
    def evaluate(self, genome: FractalGenome, context: Dict[str, Any] = None) -> float:
        """Evaluate fitness using AI prediction."""
        if not self.is_trained:
            # Fallback to basic fitness if model not trained
            return genome.fitness or 0.0
        
        # Placeholder for AI prediction
        # In real implementation, this would use the trained model
        features = self._genome_to_features(genome)
        ai_prediction = 0.5  # Placeholder prediction
        
        # Combine with existing fitness
        base_fitness = genome.fitness or 0.0
        return base_fitness + ai_prediction
