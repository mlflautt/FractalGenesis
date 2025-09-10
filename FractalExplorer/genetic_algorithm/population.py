"""
Population Management for Genetic Algorithm

This module manages the population of fractal genomes in the evolutionary process,
including initialization, statistics, and diversity maintenance.
"""

import random
import logging
from typing import List, Optional, Dict, Any
import numpy as np

from shared.genome import FractalGenome, RendererType

logger = logging.getLogger(__name__)


class Population:
    """
    Manages a population of fractal genomes for the genetic algorithm.
    
    Handles population initialization, statistics calculation, and diversity monitoring.
    """
    
    def __init__(self, size: int):
        """
        Initialize population manager.
        
        Args:
            size: Target population size
        """
        self.size = size
        self.individuals: List[FractalGenome] = []
        self.generation = 0
    
    def initialize(self, 
                  seed_genomes: Optional[List[FractalGenome]] = None,
                  renderer_type: RendererType = RendererType.MANDELBULBER):
        """
        Initialize the population with seed genomes and/or random individuals.
        
        Args:
            seed_genomes: Optional list of seed genomes to include
            renderer_type: Renderer type for new genomes
        """
        self.individuals = []
        
        # Add seed genomes if provided
        if seed_genomes:
            for genome in seed_genomes[:self.size]:
                self.individuals.append(genome.copy())
            logger.info(f"Added {len(self.individuals)} seed genomes")
        
        # Fill remaining spots with random genomes
        while len(self.individuals) < self.size:
            genome = FractalGenome(renderer_type)
            genome.randomize()
            genome.generation = 0
            self.individuals.append(genome)
        
        logger.info(f"Initialized population with {len(self.individuals)} individuals")
    
    def add_individual(self, genome: FractalGenome):
        """Add an individual to the population."""
        if len(self.individuals) < self.size:
            self.individuals.append(genome)
        else:
            logger.warning("Population is full, cannot add individual")
    
    def remove_individual(self, genome_id: str) -> bool:
        """
        Remove an individual from the population by ID.
        
        Args:
            genome_id: ID of genome to remove
            
        Returns:
            True if individual was removed, False if not found
        """
        for i, individual in enumerate(self.individuals):
            if individual.genome_id == genome_id:
                del self.individuals[i]
                return True
        return False
    
    def get_best(self, count: int = 1) -> List[FractalGenome]:
        """
        Get the best individuals from the population.
        
        Args:
            count: Number of best individuals to return
            
        Returns:
            List of best individuals sorted by fitness
        """
        sorted_individuals = sorted(self.individuals, 
                                  key=lambda g: g.fitness or 0.0, 
                                  reverse=True)
        return sorted_individuals[:count]
    
    def get_worst(self, count: int = 1) -> List[FractalGenome]:
        """
        Get the worst individuals from the population.
        
        Args:
            count: Number of worst individuals to return
            
        Returns:
            List of worst individuals sorted by fitness
        """
        sorted_individuals = sorted(self.individuals, 
                                  key=lambda g: g.fitness or 0.0)
        return sorted_individuals[:count]
    
    def get_random_sample(self, count: int) -> List[FractalGenome]:
        """
        Get a random sample from the population.
        
        Args:
            count: Number of individuals to sample
            
        Returns:
            Random sample of individuals
        """
        sample_size = min(count, len(self.individuals))
        return random.sample(self.individuals, sample_size)
    
    def calculate_diversity(self) -> float:
        """
        Calculate the genetic diversity of the population.
        
        Returns:
            Average pairwise diversity score
        """
        if len(self.individuals) < 2:
            return 0.0
        
        total_diversity = 0.0
        comparison_count = 0
        
        for i in range(len(self.individuals)):
            for j in range(i + 1, len(self.individuals)):
                diversity = self.individuals[i].calculate_diversity(self.individuals[j])
                total_diversity += diversity
                comparison_count += 1
        
        return total_diversity / comparison_count if comparison_count > 0 else 0.0
    
    def calculate_fitness_stats(self) -> Dict[str, float]:
        """
        Calculate fitness statistics for the population.
        
        Returns:
            Dictionary with fitness statistics
        """
        fitness_values = [individual.fitness or 0.0 for individual in self.individuals]
        
        if not fitness_values:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0}
        
        return {
            'mean': np.mean(fitness_values),
            'std': np.std(fitness_values),
            'min': np.min(fitness_values),
            'max': np.max(fitness_values),
            'median': np.median(fitness_values)
        }
    
    def get_diversity_distribution(self, bins: int = 10) -> Dict[str, Any]:
        """
        Get the distribution of diversity scores in the population.
        
        Args:
            bins: Number of bins for the histogram
            
        Returns:
            Dictionary with diversity distribution information
        """
        if len(self.individuals) < 2:
            return {'histogram': [], 'bin_edges': [], 'mean_diversity': 0.0}
        
        # Calculate all pairwise diversities
        diversities = []
        for i in range(len(self.individuals)):
            for j in range(i + 1, len(self.individuals)):
                diversity = self.individuals[i].calculate_diversity(self.individuals[j])
                diversities.append(diversity)
        
        if not diversities:
            return {'histogram': [], 'bin_edges': [], 'mean_diversity': 0.0}
        
        # Create histogram
        hist, bin_edges = np.histogram(diversities, bins=bins)
        
        return {
            'histogram': hist.tolist(),
            'bin_edges': bin_edges.tolist(),
            'mean_diversity': np.mean(diversities),
            'diversity_std': np.std(diversities),
            'min_diversity': np.min(diversities),
            'max_diversity': np.max(diversities)
        }
    
    def identify_clusters(self, diversity_threshold: float = 0.5) -> List[List[FractalGenome]]:
        """
        Identify clusters of similar genomes in the population.
        
        Args:
            diversity_threshold: Maximum diversity within a cluster
            
        Returns:
            List of clusters, where each cluster is a list of similar genomes
        """
        clusters = []
        unclustered = self.individuals.copy()
        
        while unclustered:
            # Start new cluster with first unclustered individual
            cluster_seed = unclustered.pop(0)
            cluster = [cluster_seed]
            
            # Find all individuals similar to the seed
            remaining = []
            for individual in unclustered:
                if cluster_seed.calculate_diversity(individual) <= diversity_threshold:
                    cluster.append(individual)
                else:
                    remaining.append(individual)
            
            unclustered = remaining
            clusters.append(cluster)
        
        return clusters
    
    def get_representative_sample(self, count: int, diversity_weight: float = 0.5) -> List[FractalGenome]:
        """
        Get a representative sample that balances fitness and diversity.
        
        Args:
            count: Number of individuals to select
            diversity_weight: Weight for diversity vs. fitness (0.0-1.0)
            
        Returns:
            Representative sample of the population
        """
        if count >= len(self.individuals):
            return self.individuals.copy()
        
        sample = []
        remaining = self.individuals.copy()
        
        # First, select the fittest individual
        if remaining:
            best = max(remaining, key=lambda g: g.fitness or 0.0)
            sample.append(best)
            remaining.remove(best)
        
        # Then select individuals that maximize diversity from already selected
        while len(sample) < count and remaining:
            best_candidate = None
            best_score = -1
            
            for candidate in remaining:
                # Calculate fitness score
                fitness_score = candidate.fitness or 0.0
                
                # Calculate diversity score (minimum distance to selected individuals)
                if sample:
                    diversity_score = min(candidate.calculate_diversity(selected) for selected in sample)
                else:
                    diversity_score = 0.0
                
                # Combined score
                combined_score = ((1 - diversity_weight) * fitness_score + 
                                diversity_weight * diversity_score)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = candidate
            
            if best_candidate:
                sample.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break
        
        return sample
    
    def prune_duplicates(self, diversity_threshold: float = 0.01):
        """
        Remove very similar individuals from the population.
        
        Args:
            diversity_threshold: Minimum diversity required between individuals
        """
        unique_individuals = []
        
        for individual in self.individuals:
            is_unique = True
            for unique_individual in unique_individuals:
                if individual.calculate_diversity(unique_individual) < diversity_threshold:
                    is_unique = False
                    break
            
            if is_unique:
                unique_individuals.append(individual)
        
        removed_count = len(self.individuals) - len(unique_individuals)
        if removed_count > 0:
            self.individuals = unique_individuals
            logger.info(f"Pruned {removed_count} duplicate individuals")
    
    def get_population_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the population state.
        
        Returns:
            Dictionary with population statistics and information
        """
        fitness_stats = self.calculate_fitness_stats()
        diversity_info = self.get_diversity_distribution()
        clusters = self.identify_clusters()
        
        return {
            'size': len(self.individuals),
            'generation': self.generation,
            'fitness_stats': fitness_stats,
            'diversity_stats': {
                'mean': diversity_info['mean_diversity'],
                'std': diversity_info.get('diversity_std', 0.0),
                'min': diversity_info.get('min_diversity', 0.0),
                'max': diversity_info.get('max_diversity', 0.0)
            },
            'cluster_count': len(clusters),
            'cluster_sizes': [len(cluster) for cluster in clusters],
            'unique_genome_count': len(set(g.genome_id for g in self.individuals))
        }
    
    def __len__(self):
        """Return the current population size."""
        return len(self.individuals)
    
    def __iter__(self):
        """Iterate over individuals in the population."""
        return iter(self.individuals)
    
    def __getitem__(self, index):
        """Get individual by index."""
        return self.individuals[index]
