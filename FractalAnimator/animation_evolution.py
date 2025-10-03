"""
Animation Evolution Engine

This module implements evolutionary algorithms specifically for fractal animations,
including fitness functions for motion quality, visual continuity, and aesthetic appeal.
"""

import random
import logging
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

from .animation_parameters import AnimationParameters
from .animation_renderer import AnimationRenderer, AnimationRenderResult
# from FractalExplorer.genetic_algorithm.population import Individual
# from FractalExplorer.genetic_algorithm.selection import SelectionStrategy, TournamentSelection
import random


logger = logging.getLogger(__name__)


@dataclass
class AnimationFitness:
    """Comprehensive fitness metrics for animations"""
    total_score: float = 0.0
    visual_continuity: float = 0.0
    motion_smoothness: float = 0.0
    aesthetic_appeal: float = 0.0
    complexity_variation: float = 0.0
    color_harmony: float = 0.0
    technical_quality: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'total_score': self.total_score,
            'visual_continuity': self.visual_continuity,
            'motion_smoothness': self.motion_smoothness,
            'aesthetic_appeal': self.aesthetic_appeal,
            'complexity_variation': self.complexity_variation,
            'color_harmony': self.color_harmony,
            'technical_quality': self.technical_quality
        }


class AnimationIndividual:
    """Individual in animation evolution population"""
    
    def __init__(self, animation_params: AnimationParameters, fitness: Optional[AnimationFitness] = None):
        self.animation_params = animation_params
        self.fitness = fitness or AnimationFitness()
        self.age = 0
        self.generation = 0
        self.render_result: Optional[AnimationRenderResult] = None
    
    @property
    def fitness_score(self) -> float:
        """Return total fitness score"""
        return self.fitness.total_score
    
    def copy(self) -> 'AnimationIndividual':
        """Create a copy of this individual"""
        return AnimationIndividual(
            animation_params=self.animation_params.copy(),
            fitness=AnimationFitness(**self.fitness.to_dict())
        )


class AnimationFitnessEvaluator:
    """
    Evaluates fitness of animation parameters based on multiple criteria.
    
    This evaluator can work in two modes:
    1. Preview mode: Fast evaluation using low-quality preview renders
    2. Full mode: Detailed evaluation using full-quality renders
    """
    
    def __init__(self, 
                 renderer: AnimationRenderer,
                 preview_mode: bool = True,
                 max_preview_frames: int = 15):
        self.renderer = renderer
        self.preview_mode = preview_mode
        self.max_preview_frames = max_preview_frames
        
        # Fitness function weights (can be tuned)
        self.default_weights = {
            'visual_continuity': 0.3,
            'motion_smoothness': 0.25,
            'aesthetic_appeal': 0.2,
            'complexity_variation': 0.15,
            'color_harmony': 0.1
        }
    
    def evaluate_fitness(self, individual: AnimationIndividual) -> AnimationFitness:
        """
        Evaluate comprehensive fitness for an animation individual.
        
        Args:
            individual: Animation individual to evaluate
            
        Returns:
            Comprehensive fitness metrics
        """
        try:
            # Render the animation (preview or full)
            if self.preview_mode:
                render_result = self.renderer.render_preview(
                    individual.animation_params,
                    max_frames=self.max_preview_frames
                )
            else:
                render_result = self.renderer.render_animation(
                    individual.animation_params,
                    preview_mode=True  # Still use preview for fitness evaluation
                )
            
            individual.render_result = render_result
            
            if not render_result.success:
                # Failed render gets very low fitness
                return AnimationFitness(total_score=0.1)
            
            # Calculate individual fitness components
            fitness = AnimationFitness()
            
            # 1. Technical quality (did it render successfully?)
            fitness.technical_quality = self._evaluate_technical_quality(render_result)
            
            # 2. Visual continuity (smooth transitions between frames)
            fitness.visual_continuity = self._evaluate_visual_continuity(individual.animation_params)
            
            # 3. Motion smoothness (camera path and parameter changes)
            fitness.motion_smoothness = self._evaluate_motion_smoothness(individual.animation_params)
            
            # 4. Aesthetic appeal (color balance, composition)
            fitness.aesthetic_appeal = self._evaluate_aesthetic_appeal(individual.animation_params)
            
            # 5. Complexity variation (interesting changes over time)
            fitness.complexity_variation = self._evaluate_complexity_variation(individual.animation_params)
            
            # 6. Color harmony (pleasing color combinations)
            fitness.color_harmony = self._evaluate_color_harmony(individual.animation_params)
            
            # Calculate weighted total score
            weights = individual.animation_params.fitness_weights or self.default_weights
            fitness.total_score = (
                weights.get('visual_continuity', 0.3) * fitness.visual_continuity +
                weights.get('motion_smoothness', 0.25) * fitness.motion_smoothness +
                weights.get('aesthetic_appeal', 0.2) * fitness.aesthetic_appeal +
                weights.get('complexity_variation', 0.15) * fitness.complexity_variation +
                weights.get('color_harmony', 0.1) * fitness.color_harmony +
                0.05 * fitness.technical_quality  # Small bonus for successful render
            )
            
            return fitness
            
        except Exception as e:
            logger.error(f"Fitness evaluation failed: {e}")
            return AnimationFitness(total_score=0.1)
    
    def _evaluate_technical_quality(self, render_result: AnimationRenderResult) -> float:
        """Evaluate technical aspects of the render"""
        if not render_result.success:
            return 0.0
        
        # Bonus for successful complete render
        score = 0.8
        
        # Penalty for failed frames
        if render_result.failed_frames:
            failure_ratio = len(render_result.failed_frames) / max(1, render_result.total_frames)
            score *= (1.0 - failure_ratio)
        
        # Small bonus for reasonable render time
        if render_result.render_time_seconds > 0:
            # Prefer renders that complete in reasonable time
            time_per_frame = render_result.render_time_seconds / max(1, render_result.total_frames)
            if time_per_frame < 30:  # Less than 30 seconds per frame
                score += 0.2
        
        return min(1.0, score)
    
    def _evaluate_visual_continuity(self, params: AnimationParameters) -> float:
        """Evaluate smoothness of visual transitions"""
        if len(params.keyframes) < 2:
            return 0.5  # Neutral score for single keyframe
        
        score = 0.0
        comparisons = 0
        
        # Sample frames throughout the animation
        sample_count = min(10, params.total_frames // 2)
        for i in range(sample_count):
            t1 = i / max(1, sample_count - 1)
            t2 = min(1.0, t1 + (1.0 / sample_count))
            
            params1 = params.get_parameters_at_time(t1)
            params2 = params.get_parameters_at_time(t2)
            
            # Calculate parameter differences
            continuity = self._calculate_parameter_continuity(params1, params2)
            score += continuity
            comparisons += 1
        
        return score / max(1, comparisons)
    
    def _calculate_parameter_continuity(self, p1: 'MandelbulberParameters', p2: 'MandelbulberParameters') -> float:
        """Calculate continuity between two parameter sets"""
        score = 0.0
        components = 0
        
        # Camera position continuity
        cam_dist = np.sqrt(
            (p1.camera.camera_x - p2.camera.camera_x)**2 +
            (p1.camera.camera_y - p2.camera.camera_y)**2 +
            (p1.camera.camera_z - p2.camera.camera_z)**2
        )
        # Normalize and invert (closer = better)
        score += max(0, 1.0 - cam_dist / 10.0)
        components += 1
        
        # Fractal parameter continuity
        power_diff = abs(p1.fractal.power - p2.fractal.power)
        score += max(0, 1.0 - power_diff / 5.0)
        components += 1
        
        # Color continuity
        color_diff = np.sqrt(
            (p1.material.surface_color_r - p2.material.surface_color_r)**2 +
            (p1.material.surface_color_g - p2.material.surface_color_g)**2 +
            (p1.material.surface_color_b - p2.material.surface_color_b)**2
        )
        score += max(0, 1.0 - color_diff / 1.0)
        components += 1
        
        return score / max(1, components)
    
    def _evaluate_motion_smoothness(self, params: AnimationParameters) -> float:
        """Evaluate smoothness of camera and parameter motion"""
        score = 0.0
        
        # Evaluate camera path smoothness
        if params.camera_path:
            path_smoothness = self._evaluate_camera_path_smoothness(params.camera_path, params.duration_seconds)
            score += 0.6 * path_smoothness
        else:
            score += 0.3  # Neutral score for no camera path
        
        # Evaluate parameter change smoothness
        param_smoothness = self._evaluate_parameter_change_smoothness(params)
        score += 0.4 * param_smoothness
        
        return min(1.0, score)
    
    def _evaluate_camera_path_smoothness(self, path: 'CameraPath', duration: float) -> float:
        """Evaluate smoothness of camera path"""
        if path.path_type == "orbit":
            # Orbital paths are naturally smooth
            # Bonus for reasonable speed (not too fast/slow)
            if 0.2 <= path.orbit_speed <= 3.0:
                return 0.9
            else:
                return 0.6
        
        elif path.path_type == "spiral":
            # Spirals are smooth but check for reasonable parameters
            if 0.1 <= path.orbit_speed <= 2.0 and path.zoom_factor < 0.9:
                return 0.8
            else:
                return 0.5
        
        elif path.path_type == "linear":
            # Linear paths are smooth by definition
            return 0.7
        
        elif path.path_type == "zoom":
            # Zoom can be smooth if not too aggressive
            if path.zoom_factor < 0.8:
                return 0.8
            else:
                return 0.4  # Very aggressive zoom can be jarring
        
        return 0.5  # Default for unknown path types
    
    def _evaluate_parameter_change_smoothness(self, params: AnimationParameters) -> float:
        """Evaluate smoothness of parameter changes over time"""
        if len(params.keyframes) < 2:
            return 0.5
        
        # Check for excessive parameter jumps between keyframes
        smoothness_scores = []
        
        for i in range(len(params.keyframes) - 1):
            kf1 = params.keyframes[i]
            kf2 = params.keyframes[i + 1]
            
            time_gap = kf2.time - kf1.time
            if time_gap <= 0:
                continue
                
            # Calculate parameter velocity (change per time unit)
            power_velocity = abs(kf2.parameters.fractal.power - kf1.parameters.fractal.power) / time_gap
            
            # Penalize very rapid changes
            if power_velocity < 5.0:  # Reasonable power change rate
                smoothness_scores.append(0.9)
            elif power_velocity < 15.0:
                smoothness_scores.append(0.6)
            else:
                smoothness_scores.append(0.2)
        
        return statistics.mean(smoothness_scores) if smoothness_scores else 0.5
    
    def _evaluate_aesthetic_appeal(self, params: AnimationParameters) -> float:
        """Evaluate overall aesthetic appeal"""
        score = 0.0
        
        # Check color variety and balance
        colors = []
        for kf in params.keyframes:
            colors.append([
                kf.parameters.material.surface_color_r,
                kf.parameters.material.surface_color_g,
                kf.parameters.material.surface_color_b
            ])
        
        if colors:
            color_variety = self._calculate_color_variety(colors)
            score += 0.4 * color_variety
        
        # Check fractal complexity diversity
        powers = [kf.parameters.fractal.power for kf in params.keyframes]
        if len(powers) > 1:
            power_range = max(powers) - min(powers)
            # Prefer some variation but not too extreme
            if 2.0 <= power_range <= 8.0:
                score += 0.3
            elif power_range > 0:
                score += 0.15
        
        # Camera composition
        camera_positions = [
            (kf.parameters.camera.camera_x, kf.parameters.camera.camera_y, kf.parameters.camera.camera_z)
            for kf in params.keyframes
        ]
        if len(camera_positions) > 1:
            position_variety = self._calculate_position_variety(camera_positions)
            score += 0.3 * position_variety
        
        return min(1.0, score)
    
    def _calculate_color_variety(self, colors: List[List[float]]) -> float:
        """Calculate variety in color palette"""
        if len(colors) <= 1:
            return 0.5
        
        # Calculate pairwise color distances
        distances = []
        for i in range(len(colors)):
            for j in range(i + 1, len(colors)):
                dist = np.sqrt(sum((colors[i][k] - colors[j][k])**2 for k in range(3)))
                distances.append(dist)
        
        avg_distance = statistics.mean(distances) if distances else 0
        # Normalize to 0-1 range
        return min(1.0, avg_distance / 1.5)
    
    def _calculate_position_variety(self, positions: List[Tuple[float, float, float]]) -> float:
        """Calculate variety in camera positions"""
        if len(positions) <= 1:
            return 0.5
        
        # Calculate distances from origin and variety
        distances = [np.sqrt(x**2 + y**2 + z**2) for x, y, z in positions]
        distance_variety = np.std(distances) / (np.mean(distances) + 1e-8)
        
        return min(1.0, distance_variety)
    
    def _evaluate_complexity_variation(self, params: AnimationParameters) -> float:
        """Evaluate how complexity varies throughout the animation"""
        complexity_scores = []
        
        # Sample complexity at different time points
        sample_count = min(10, len(params.keyframes) * 2)
        for i in range(sample_count):
            t = i / max(1, sample_count - 1)
            frame_params = params.get_parameters_at_time(t)
            
            # Calculate complexity metrics
            iteration_complexity = frame_params.fractal.iterations / 300.0
            power_complexity = abs(frame_params.fractal.power - 2.0) / 10.0
            
            complexity = (iteration_complexity + power_complexity) / 2
            complexity_scores.append(complexity)
        
        if not complexity_scores:
            return 0.5
        
        # Prefer moderate variation in complexity
        complexity_std = np.std(complexity_scores)
        if 0.1 <= complexity_std <= 0.4:
            return 0.9
        elif complexity_std > 0:
            return 0.6
        else:
            return 0.3  # No variation is boring
    
    def _evaluate_color_harmony(self, params: AnimationParameters) -> float:
        """Evaluate color harmony across the animation"""
        if not params.keyframes:
            return 0.5
        
        # Collect all colors used
        colors = []
        for kf in params.keyframes:
            mat = kf.parameters.material
            colors.append([mat.surface_color_r, mat.surface_color_g, mat.surface_color_b])
            colors.append([mat.specular_r, mat.specular_g, mat.specular_b])
        
        return self._calculate_color_harmony(colors)
    
    def _calculate_color_harmony(self, colors: List[List[float]]) -> float:
        """Calculate color harmony score"""
        if len(colors) <= 1:
            return 0.7  # Single color is harmonious by definition
        
        # Convert to HSV for better harmony analysis
        hsv_colors = []
        for rgb in colors:
            hsv = self._rgb_to_hsv(rgb[0], rgb[1], rgb[2])
            hsv_colors.append(hsv)
        
        # Analyze hue relationships
        hues = [hsv[0] for hsv in hsv_colors]
        hue_spread = max(hues) - min(hues)
        
        # Complementary colors (opposite hues) or analogous colors (nearby hues) score well
        if hue_spread < 0.2:  # Analogous
            return 0.85
        elif 0.4 <= hue_spread <= 0.6:  # Complementary
            return 0.9
        elif hue_spread < 0.8:
            return 0.7
        else:
            return 0.5  # Too chaotic
    
    def _rgb_to_hsv(self, r: float, g: float, b: float) -> Tuple[float, float, float]:
        """Convert RGB to HSV"""
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        diff = max_val - min_val
        
        # Value
        v = max_val
        
        # Saturation
        s = 0 if max_val == 0 else diff / max_val
        
        # Hue
        if diff == 0:
            h = 0
        elif max_val == r:
            h = (60 * ((g - b) / diff) + 360) % 360
        elif max_val == g:
            h = (60 * ((b - r) / diff) + 120) % 360
        else:
            h = (60 * ((r - g) / diff) + 240) % 360
        
        return (h / 360.0, s, v)


class AnimationEvolutionEngine:
    """
    Evolutionary algorithm engine specifically for fractal animations.
    
    Manages population, selection, breeding, and fitness evaluation for animation evolution.
    """
    
    def __init__(self,
                 renderer: AnimationRenderer,
                 population_size: int = 20,
                 elite_size: int = 4,
                 mutation_rate: float = 0.15,
                 crossover_rate: float = 0.8,
                 max_generations: int = 50):
        """
        Initialize evolution engine.
        
        Args:
            renderer: Animation renderer for fitness evaluation
            population_size: Size of evolution population
            elite_size: Number of elite individuals to preserve
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            max_generations: Maximum generations to evolve
        """
        self.renderer = renderer
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        
        # Initialize fitness evaluator
        self.fitness_evaluator = AnimationFitnessEvaluator(renderer)
        
        # Simple tournament selection for animations
        self.tournament_size = 3
        
        # Evolution statistics
        self.evolution_stats = {
            'generations': [],
            'best_fitness_per_generation': [],
            'average_fitness_per_generation': [],
            'diversity_metrics': []
        }
        
        # Current population
        self.population: List[AnimationIndividual] = []
        self.generation = 0
        
        # Callbacks for progress tracking
        self.generation_callback: Optional[Callable[[int, List[AnimationIndividual]], None]] = None
        self.fitness_callback: Optional[Callable[[AnimationIndividual], None]] = None
    
    def evolve_animations(self, 
                         seed_animations: List[AnimationParameters] = None,
                         target_fitness: float = 0.9,
                         early_stopping_patience: int = 10) -> List[AnimationIndividual]:
        """
        Evolve animations using genetic algorithm.
        
        Args:
            seed_animations: Initial animation parameters (random if None)
            target_fitness: Stop evolution if this fitness is reached
            early_stopping_patience: Stop if no improvement for this many generations
            
        Returns:
            Final evolved population
        """
        logger.info(f"Starting animation evolution with {self.population_size} individuals")
        
        # Initialize population
        self._initialize_population(seed_animations)
        
        best_fitness = 0.0
        no_improvement_count = 0
        
        for generation in range(self.max_generations):
            self.generation = generation
            logger.info(f"Generation {generation + 1}/{self.max_generations}")
            
            # Evaluate fitness for all individuals
            self._evaluate_population()
            
            # Calculate statistics
            fitness_scores = [ind.fitness_score for ind in self.population]
            current_best = max(fitness_scores)
            current_avg = statistics.mean(fitness_scores)
            
            self.evolution_stats['generations'].append(generation)
            self.evolution_stats['best_fitness_per_generation'].append(current_best)
            self.evolution_stats['average_fitness_per_generation'].append(current_avg)
            
            logger.info(f"Best fitness: {current_best:.3f}, Average: {current_avg:.3f}")
            
            # Check for improvement
            if current_best > best_fitness + 1e-6:  # Small epsilon for floating point comparison
                best_fitness = current_best
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Check stopping conditions
            if current_best >= target_fitness:
                logger.info(f"Target fitness {target_fitness} reached!")
                break
            
            if no_improvement_count >= early_stopping_patience:
                logger.info(f"Early stopping: no improvement for {early_stopping_patience} generations")
                break
            
            # Create next generation
            if generation < self.max_generations - 1:
                self._create_next_generation()
            
            # Callback for progress tracking
            if self.generation_callback:
                self.generation_callback(generation, self.population[:])
        
        # Sort final population by fitness
        self.population.sort(key=lambda ind: ind.fitness_score, reverse=True)
        
        logger.info(f"Evolution complete. Best fitness: {self.population[0].fitness_score:.3f}")
        return self.population
    
    def _initialize_population(self, seed_animations: List[AnimationParameters] = None):
        """Initialize the evolution population"""
        self.population = []
        
        if seed_animations:
            # Use provided seed animations
            for i, params in enumerate(seed_animations[:self.population_size]):
                individual = AnimationIndividual(params.copy())
                individual.generation = 0
                self.population.append(individual)
        
        # Fill remaining spots with random animations
        while len(self.population) < self.population_size:
            random_params = self._create_random_animation()
            individual = AnimationIndividual(random_params)
            individual.generation = 0
            self.population.append(individual)
    
    def _create_random_animation(self) -> AnimationParameters:
        """Create random animation parameters"""
        from renderers.mandelbulber.templates import ParameterTemplates
        
        params = AnimationParameters()
        
        # Random duration and framerate
        params.duration_seconds = random.uniform(5.0, 20.0)
        params.fps = random.choice([24, 30, 60])
        
        # Random camera path
        path_types = ["orbit", "spiral", "zoom", "linear"]
        params.camera_path.path_type = random.choice(path_types)
        params.camera_path.orbit_radius = random.uniform(2.0, 8.0)
        params.camera_path.orbit_speed = random.uniform(0.3, 2.0)
        params.camera_path.orbit_height = random.uniform(-2.0, 2.0)
        
        # Create random keyframes using templates
        templates = list(ParameterTemplates.get_all_templates().values())
        num_keyframes = random.randint(2, 5)
        
        for i in range(num_keyframes):
            time = i / (num_keyframes - 1)
            base_template = random.choice(templates).copy()
            base_template.mutate(mutation_rate=0.3, mutation_strength=0.2)
            params.add_keyframe(time, base_template)
        
        return params
    
    def _evaluate_population(self):
        """Evaluate fitness for all individuals in population"""
        logger.info("Evaluating population fitness...")
        
        # Evaluate fitness in parallel for speed
        with ThreadPoolExecutor(max_workers=min(4, len(self.population))) as executor:
            future_to_individual = {
                executor.submit(self.fitness_evaluator.evaluate_fitness, individual): individual
                for individual in self.population
                if individual.fitness.total_score == 0.0  # Only evaluate if not already done
            }
            
            for future in as_completed(future_to_individual):
                individual = future_to_individual[future]
                try:
                    fitness = future.result()
                    individual.fitness = fitness
                    individual.age += 1
                    
                    if self.fitness_callback:
                        self.fitness_callback(individual)
                        
                except Exception as e:
                    logger.error(f"Fitness evaluation failed for individual: {e}")
                    individual.fitness = AnimationFitness(total_score=0.1)
    
    def _create_next_generation(self):
        """Create the next generation through selection and breeding"""
        # Sort by fitness
        self.population.sort(key=lambda ind: ind.fitness_score, reverse=True)
        
        new_population = []
        
        # Keep elite individuals
        for i in range(min(self.elite_size, len(self.population))):
            elite = self.population[i].copy()
            elite.generation = self.generation + 1
            new_population.append(elite)
        
        # Generate offspring to fill remaining population
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate:
                # Crossover
                parent1 = self._select_individual(self.population)
                parent2 = self._select_individual(self.population)
                
                child_params = parent1.animation_params.crossover(parent2.animation_params)
                child = AnimationIndividual(child_params)
            else:
                # Mutation only
                parent = self._select_individual(self.population)
                child_params = parent.animation_params.copy()
                child = AnimationIndividual(child_params)
            
            # Apply mutation
            if random.random() < self.mutation_rate:
                child.animation_params.mutate(
                    mutation_rate=self.mutation_rate,
                    mutation_strength=0.15
                )
            
            child.generation = self.generation + 1
            new_population.append(child)
        
        self.population = new_population
    
    def _select_individual(self, population: List[AnimationIndividual]) -> AnimationIndividual:
        """Simple tournament selection for animation individuals"""
        tournament_size = min(self.tournament_size, len(population))
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda ind: ind.fitness_score)
    
    def get_best_individual(self) -> Optional[AnimationIndividual]:
        """Get the best individual from current population"""
        if not self.population:
            return None
        
        return max(self.population, key=lambda ind: ind.fitness_score)
    
    def get_diversity_metrics(self) -> Dict[str, float]:
        """Calculate diversity metrics for current population"""
        if not self.population:
            return {}
        
        # Duration diversity
        durations = [ind.animation_params.duration_seconds for ind in self.population]
        duration_std = np.std(durations) if len(durations) > 1 else 0
        
        # Keyframe count diversity
        keyframe_counts = [len(ind.animation_params.keyframes) for ind in self.population]
        keyframe_std = np.std(keyframe_counts) if len(keyframe_counts) > 1 else 0
        
        # Camera path diversity
        path_types = [ind.animation_params.camera_path.path_type for ind in self.population]
        unique_paths = len(set(path_types)) / len(path_types)
        
        return {
            'duration_diversity': duration_std,
            'keyframe_diversity': keyframe_std,
            'path_diversity': unique_paths,
            'population_size': len(self.population)
        }
    
    def save_evolution_results(self, output_path: Path):
        """Save evolution results and statistics"""
        best_individual = self.get_best_individual()
        
        results = {
            'evolution_stats': self.evolution_stats,
            'final_population_size': len(self.population),
            'best_fitness': best_individual.fitness_score if best_individual else 0,
            'best_individual': {
                'fitness': best_individual.fitness.to_dict() if best_individual else {},
                'animation_params': best_individual.animation_params.to_dict() if best_individual else {}
            },
            'diversity_metrics': self.get_diversity_metrics(),
            'evolution_settings': {
                'population_size': self.population_size,
                'elite_size': self.elite_size,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'max_generations': self.max_generations
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evolution results saved to {output_path}")
    
    def set_generation_callback(self, callback: Callable[[int, List[AnimationIndividual]], None]):
        """Set callback for generation completion"""
        self.generation_callback = callback
    
    def set_fitness_callback(self, callback: Callable[[AnimationIndividual], None]):
        """Set callback for individual fitness evaluation"""
        self.fitness_callback = callback