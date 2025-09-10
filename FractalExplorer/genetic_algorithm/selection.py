"""
Selection Strategies for Genetic Algorithm

This module provides different selection methods for choosing parents
in the genetic algorithm, optimized for user preference-based evolution.
"""

import random
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from shared.genome import FractalGenome


class SelectionStrategy(ABC):
    """Abstract base class for selection strategies."""
    
    @abstractmethod
    def select(self, population: List[FractalGenome], fitness_scores: List[float]) -> FractalGenome:
        """
        Select an individual from the population.
        
        Args:
            population: List of individuals to select from
            fitness_scores: Fitness scores for each individual
            
        Returns:
            Selected individual
        """
        pass


class TournamentSelection(SelectionStrategy):
    """
    Tournament selection strategy.
    
    Randomly selects a subset of individuals and returns the fittest one.
    Good for maintaining diversity while still favoring fit individuals.
    """
    
    def __init__(self, tournament_size: int = 3):
        """
        Initialize tournament selection.
        
        Args:
            tournament_size: Number of individuals to compete in each tournament
        """
        self.tournament_size = tournament_size
    
    def select(self, population: List[FractalGenome], fitness_scores: List[float]) -> FractalGenome:
        """Select individual using tournament selection."""
        if not population:
            raise ValueError("Population is empty")
        
        tournament_size = min(self.tournament_size, len(population))
        
        # Randomly select tournament participants
        tournament_indices = random.sample(range(len(population)), tournament_size)
        
        # Find the best individual in the tournament
        best_index = max(tournament_indices, key=lambda i: fitness_scores[i])
        
        return population[best_index]


class RouletteSelection(SelectionStrategy):
    """
    Roulette wheel selection strategy.
    
    Selects individuals with probability proportional to their fitness.
    """
    
    def select(self, population: List[FractalGenome], fitness_scores: List[float]) -> FractalGenome:
        """Select individual using roulette wheel selection."""
        if not population:
            raise ValueError("Population is empty")
        
        # Handle negative fitness values by shifting
        min_fitness = min(fitness_scores)
        adjusted_scores = [score - min_fitness + 0.1 for score in fitness_scores]
        
        # Calculate cumulative probabilities
        total_fitness = sum(adjusted_scores)
        if total_fitness == 0:
            return random.choice(population)
        
        probabilities = [score / total_fitness for score in adjusted_scores]
        cumulative_probs = np.cumsum(probabilities)
        
        # Select based on random value
        random_value = random.random()
        selected_index = next(i for i, cum_prob in enumerate(cumulative_probs) if cum_prob >= random_value)
        
        return population[selected_index]


class RankSelection(SelectionStrategy):
    """
    Rank-based selection strategy.
    
    Selects individuals based on their rank rather than raw fitness values.
    Helps prevent premature convergence when fitness values vary widely.
    """
    
    def __init__(self, selection_pressure: float = 1.5):
        """
        Initialize rank selection.
        
        Args:
            selection_pressure: Controls how much better individuals are favored (1.0-2.0)
        """
        self.selection_pressure = max(1.0, min(2.0, selection_pressure))
    
    def select(self, population: List[FractalGenome], fitness_scores: List[float]) -> FractalGenome:
        """Select individual using rank-based selection."""
        if not population:
            raise ValueError("Population is empty")
        
        # Create rank-based probabilities
        n = len(population)
        sorted_indices = sorted(range(n), key=lambda i: fitness_scores[i])
        
        # Assign probabilities based on rank
        rank_probs = []
        for rank in range(n):
            prob = (2 - self.selection_pressure + 2 * (self.selection_pressure - 1) * rank / (n - 1)) / n
            rank_probs.append(prob)
        
        # Select based on rank probabilities
        cumulative_probs = np.cumsum(rank_probs)
        random_value = random.random()
        selected_rank = next(i for i, cum_prob in enumerate(cumulative_probs) if cum_prob >= random_value)
        selected_index = sorted_indices[selected_rank]
        
        return population[selected_index]


class EliteSelection(SelectionStrategy):
    """
    Elite selection strategy.
    
    Always selects the best individuals. Useful for elitism in evolutionary algorithms.
    """
    
    def select(self, population: List[FractalGenome], fitness_scores: List[float]) -> FractalGenome:
        """Select the fittest individual."""
        if not population:
            raise ValueError("Population is empty")
        
        best_index = max(range(len(population)), key=lambda i: fitness_scores[i])
        return population[best_index]


class DiversityBasedSelection(SelectionStrategy):
    """
    Diversity-based selection that favors individuals that are different from already selected ones.
    Helps maintain genetic diversity in the population.
    """
    
    def __init__(self, diversity_weight: float = 0.5):
        """
        Initialize diversity-based selection.
        
        Args:
            diversity_weight: How much to weight diversity vs. fitness (0.0-1.0)
        """
        self.diversity_weight = max(0.0, min(1.0, diversity_weight))
        self.previously_selected: List[FractalGenome] = []
    
    def select(self, population: List[FractalGenome], fitness_scores: List[float]) -> FractalGenome:
        """Select individual balancing fitness and diversity."""
        if not population:
            raise ValueError("Population is empty")
        
        if not self.previously_selected:
            # First selection - just use fitness
            best_index = max(range(len(population)), key=lambda i: fitness_scores[i])
            selected = population[best_index]
            self.previously_selected.append(selected)
            return selected
        
        # Calculate combined scores (fitness + diversity)
        combined_scores = []
        for i, individual in enumerate(population):
            fitness = fitness_scores[i]
            
            # Calculate minimum diversity to previously selected individuals
            min_diversity = min(individual.calculate_diversity(prev) 
                              for prev in self.previously_selected)
            
            # Combine fitness and diversity
            combined_score = (1 - self.diversity_weight) * fitness + self.diversity_weight * min_diversity
            combined_scores.append(combined_score)
        
        # Select best combined score
        best_index = max(range(len(population)), key=lambda i: combined_scores[i])
        selected = population[best_index]
        self.previously_selected.append(selected)
        
        # Keep only recent selections to prevent memory growth
        if len(self.previously_selected) > 10:
            self.previously_selected = self.previously_selected[-5:]
        
        return selected
    
    def reset(self):
        """Reset the selection history."""
        self.previously_selected = []
