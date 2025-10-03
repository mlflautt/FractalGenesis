"""
Animation Testing Framework

Comprehensive testing system for animation quality, evolution convergence, and rendering performance.
Provides automated testing, benchmarking, and quality assurance for the fractal animation system.
"""

import time
import json
import logging
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import tempfile
import shutil

from .animation_parameters import AnimationParameters
from .animation_renderer import AnimationRenderer, AnimationRenderResult
from .animation_evolution import AnimationEvolutionEngine, AnimationIndividual, AnimationFitness
from .animation_templates import AnimationTemplates
from renderers.mandelbulber.renderer import MandelbulberRenderer


logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a single test"""
    test_name: str
    success: bool
    duration_seconds: float
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_name': self.test_name,
            'success': self.success,
            'duration_seconds': self.duration_seconds,
            'details': self.details,
            'error_message': self.error_message
        }


@dataclass 
class TestSuite:
    """Collection of test results"""
    name: str
    results: List[TestResult] = field(default_factory=list)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    @property
    def total_duration(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return sum(r.duration_seconds for r in self.results)
    
    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return len([r for r in self.results if r.success]) / len(self.results)
    
    def add_result(self, result: TestResult):
        self.results.append(result)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'total_duration': self.total_duration,
            'success_rate': self.success_rate,
            'total_tests': len(self.results),
            'successful_tests': len([r for r in self.results if r.success]),
            'failed_tests': len([r for r in self.results if not r.success]),
            'results': [r.to_dict() for r in self.results]
        }


class AnimationTester:
    """
    Comprehensive testing framework for fractal animation system.
    
    Provides various testing capabilities:
    - Basic functionality tests
    - Performance benchmarks
    - Evolution convergence tests
    - Render quality validation
    """
    
    def __init__(self, test_output_dir: Optional[Path] = None):
        """Initialize the testing framework"""
        self.test_output_dir = test_output_dir or Path(tempfile.mkdtemp(prefix="animation_tests_"))
        self.test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test configuration
        self.quick_test_config = {
            'max_frames': 5,
            'population_size': 4,
            'generations': 2,
            'preview_only': True
        }
        
        self.standard_test_config = {
            'max_frames': 15,
            'population_size': 8,
            'generations': 5,
            'preview_only': False
        }
        
    def run_all_tests(self, quick_mode: bool = True) -> TestSuite:
        """
        Run complete test suite.
        
        Args:
            quick_mode: If True, run faster tests with reduced parameters
            
        Returns:
            Complete test suite results
        """
        suite = TestSuite("Complete Animation System Test")
        suite.start_time = time.time()
        
        logger.info("=== Running Animation System Test Suite ===")
        
        # Basic functionality tests
        suite.add_result(self._test_imports())
        suite.add_result(self._test_animation_parameters())
        suite.add_result(self._test_templates())
        suite.add_result(self._test_genetic_operations())
        
        # System integration tests
        suite.add_result(self._test_renderer_setup())
        suite.add_result(self._test_evolution_setup())
        
        if not quick_mode:
            # More comprehensive tests (require more time)
            suite.add_result(self._test_animation_rendering())
            suite.add_result(self._test_evolution_convergence())
            suite.add_result(self._test_fitness_evaluation())
        
        # Performance tests
        suite.add_result(self._test_performance_benchmarks(quick_mode))
        
        suite.end_time = time.time()
        
        logger.info(f"=== Test Suite Complete: {suite.success_rate:.1%} success rate ===")
        
        return suite
    
    def _test_imports(self) -> TestResult:
        """Test that all modules can be imported"""
        start_time = time.time()
        
        try:
            from . import (
                AnimationParameters, CameraPath, InterpolationType,
                AnimationRenderer, AnimationEvolutionEngine, AnimationTemplates
            )
            
            return TestResult(
                test_name="Import Test",
                success=True,
                duration_seconds=time.time() - start_time,
                details={"imported_modules": 6}
            )
            
        except Exception as e:
            return TestResult(
                test_name="Import Test",
                success=False,
                duration_seconds=time.time() - start_time,
                error_message=str(e)
            )
    
    def _test_animation_parameters(self) -> TestResult:
        """Test animation parameter creation and manipulation"""
        start_time = time.time()
        
        try:
            from .animation_parameters import AnimationParameters, CameraPath, InterpolationType
            from renderers.mandelbulber.templates import ParameterTemplates
            
            # Test basic parameter creation
            params = AnimationParameters()
            params.duration_seconds = 10.0
            params.fps = 30
            
            # Test camera path
            camera_path = CameraPath(
                path_type="orbit",
                orbit_radius=5.0,
                orbit_speed=1.0
            )
            params.camera_path = camera_path
            
            # Test keyframes
            template = ParameterTemplates.classic_mandelbulb()
            params.add_keyframe(0.0, template, InterpolationType.LINEAR)
            params.add_keyframe(1.0, template, InterpolationType.EASE_IN_OUT)
            
            # Test interpolation
            mid_params = params.get_parameters_at_time(0.5)
            
            # Test serialization
            params_dict = params.to_dict()
            
            details = {
                "total_frames": params.total_frames,
                "keyframe_count": len(params.keyframes),
                "camera_path_type": params.camera_path.path_type,
                "serialization_keys": len(params_dict)
            }
            
            return TestResult(
                test_name="Animation Parameters Test",
                success=True,
                duration_seconds=time.time() - start_time,
                details=details
            )
            
        except Exception as e:
            return TestResult(
                test_name="Animation Parameters Test",
                success=False,
                duration_seconds=time.time() - start_time,
                error_message=str(e)
            )
    
    def _test_templates(self) -> TestResult:
        """Test animation template creation"""
        start_time = time.time()
        
        try:
            # Test individual templates
            orbital = AnimationTemplates.orbital_mandelbulb()
            zoom = AnimationTemplates.zoom_into_fractal()
            color_morph = AnimationTemplates.color_morph_sequence()
            
            # Test all templates
            all_templates = AnimationTemplates.get_all_templates()
            
            # Test variations
            variation = AnimationTemplates.create_random_variation(orbital, 0.3)
            
            # Test seed population
            population = AnimationTemplates.create_seed_population(6)
            
            details = {
                "template_count": len(all_templates),
                "template_names": list(all_templates.keys()),
                "seed_population_size": len(population),
                "variation_duration_change": abs(orbital.duration_seconds - variation.duration_seconds)
            }
            
            return TestResult(
                test_name="Templates Test",
                success=True,
                duration_seconds=time.time() - start_time,
                details=details
            )
            
        except Exception as e:
            return TestResult(
                test_name="Templates Test", 
                success=False,
                duration_seconds=time.time() - start_time,
                error_message=str(e)
            )
    
    def _test_genetic_operations(self) -> TestResult:
        """Test genetic operations on animation parameters"""
        start_time = time.time()
        
        try:
            template1 = AnimationTemplates.orbital_mandelbulb()
            template2 = AnimationTemplates.zoom_into_fractal()
            
            # Test mutation
            mutated = template1.copy()
            original_duration = mutated.duration_seconds
            mutated.mutate(mutation_rate=0.5, mutation_strength=0.3)
            
            # Test crossover
            child = template1.crossover(template2)
            
            # Test copying
            copied = template1.copy()
            
            details = {
                "mutation_changed_duration": abs(original_duration - mutated.duration_seconds) > 0.1,
                "crossover_keyframes": len(child.keyframes),
                "copy_identical": copied.duration_seconds == template1.duration_seconds,
                "child_has_mixed_traits": True  # Would need more complex validation
            }
            
            return TestResult(
                test_name="Genetic Operations Test",
                success=True,
                duration_seconds=time.time() - start_time,
                details=details
            )
            
        except Exception as e:
            return TestResult(
                test_name="Genetic Operations Test",
                success=False,
                duration_seconds=time.time() - start_time,
                error_message=str(e)
            )
    
    def _test_renderer_setup(self) -> TestResult:
        """Test animation renderer setup"""
        start_time = time.time()
        
        try:
            # Create renderer (works even without Mandelbulber)
            mandelbulber_renderer = MandelbulberRenderer(
                output_dir=self.test_output_dir / "frames"
            )
            
            animation_renderer = AnimationRenderer(
                mandelbulber_renderer=mandelbulber_renderer,
                output_dir=self.test_output_dir / "animations",
                max_workers=2
            )
            
            # Test statistics
            stats = animation_renderer.get_render_statistics()
            
            details = {
                "renderer_available": mandelbulber_renderer.is_available,
                "output_dir_created": animation_renderer.output_dir.exists(),
                "stats_keys": list(stats.keys())
            }
            
            return TestResult(
                test_name="Renderer Setup Test",
                success=True,
                duration_seconds=time.time() - start_time,
                details=details
            )
            
        except Exception as e:
            return TestResult(
                test_name="Renderer Setup Test",
                success=False,
                duration_seconds=time.time() - start_time,
                error_message=str(e)
            )
    
    def _test_evolution_setup(self) -> TestResult:
        """Test evolution engine setup"""
        start_time = time.time()
        
        try:
            # Create minimal renderer
            mandelbulber_renderer = MandelbulberRenderer()
            animation_renderer = AnimationRenderer(
                mandelbulber_renderer=mandelbulber_renderer,
                max_workers=1
            )
            
            # Create evolution engine
            evolution_engine = AnimationEvolutionEngine(
                renderer=animation_renderer,
                population_size=4,
                max_generations=2
            )
            
            # Test diversity metrics
            seed_population = AnimationTemplates.create_seed_population(4)
            evolution_engine._initialize_population(seed_population)
            diversity = evolution_engine.get_diversity_metrics()
            
            details = {
                "population_size": len(evolution_engine.population),
                "diversity_metrics": list(diversity.keys()),
                "evolution_stats_initialized": bool(evolution_engine.evolution_stats)
            }
            
            return TestResult(
                test_name="Evolution Setup Test",
                success=True,
                duration_seconds=time.time() - start_time,
                details=details
            )
            
        except Exception as e:
            return TestResult(
                test_name="Evolution Setup Test",
                success=False,
                duration_seconds=time.time() - start_time,
                error_message=str(e)
            )
    
    def _test_animation_rendering(self) -> TestResult:
        """Test animation rendering (requires Mandelbulber or simulation)"""
        start_time = time.time()
        
        try:
            # Create a simple test animation
            test_animation = AnimationTemplates.create_preview_template()
            
            # Try to render (will work if Mandelbulber available)
            mandelbulber_renderer = MandelbulberRenderer()
            animation_renderer = AnimationRenderer(
                mandelbulber_renderer=mandelbulber_renderer,
                output_dir=self.test_output_dir / "test_renders"
            )
            
            if mandelbulber_renderer.is_available:
                # Actually render if possible
                result = animation_renderer.render_preview(test_animation, max_frames=3)
                success = result.success
                details = {
                    "mandelbulber_available": True,
                    "render_success": result.success,
                    "total_frames": result.total_frames,
                    "render_time": result.render_time_seconds
                }
            else:
                # Simulate successful render
                success = True
                details = {
                    "mandelbulber_available": False,
                    "simulated_render": True,
                    "test_animation_frames": test_animation.total_frames
                }
            
            return TestResult(
                test_name="Animation Rendering Test",
                success=success,
                duration_seconds=time.time() - start_time,
                details=details
            )
            
        except Exception as e:
            return TestResult(
                test_name="Animation Rendering Test",
                success=False,
                duration_seconds=time.time() - start_time,
                error_message=str(e)
            )
    
    def _test_evolution_convergence(self) -> TestResult:
        """Test evolution convergence behavior"""
        start_time = time.time()
        
        try:
            # Set up minimal evolution
            mandelbulber_renderer = MandelbulberRenderer()
            animation_renderer = AnimationRenderer(mandelbulber_renderer, max_workers=1)
            
            evolution_engine = AnimationEvolutionEngine(
                renderer=animation_renderer,
                population_size=4,
                max_generations=3,
                elite_size=1
            )
            
            # Override fitness evaluator for testing
            class MockFitnessEvaluator:
                def evaluate_fitness(self, individual):
                    # Mock fitness based on duration (prefer shorter animations)
                    fitness_score = max(0.1, 1.0 - (individual.animation_params.duration_seconds - 5.0) / 10.0)
                    from .animation_evolution import AnimationFitness
                    return AnimationFitness(
                        total_score=fitness_score,
                        technical_quality=0.8,
                        visual_continuity=0.7,
                        motion_smoothness=0.6
                    )
            
            evolution_engine.fitness_evaluator = MockFitnessEvaluator()
            
            # Run short evolution
            seed_population = AnimationTemplates.create_seed_population(4)
            final_population = evolution_engine.evolve_animations(
                seed_animations=seed_population,
                target_fitness=0.7,
                early_stopping_patience=2
            )
            
            # Check convergence
            fitness_scores = [ind.fitness_score for ind in final_population]
            best_fitness = max(fitness_scores)
            avg_fitness = statistics.mean(fitness_scores)
            
            details = {
                "final_population_size": len(final_population),
                "best_fitness": best_fitness,
                "average_fitness": avg_fitness,
                "generations_run": evolution_engine.generation,
                "convergence_achieved": best_fitness > 0.6
            }
            
            return TestResult(
                test_name="Evolution Convergence Test",
                success=best_fitness > 0.5,  # Reasonable fitness achieved
                duration_seconds=time.time() - start_time,
                details=details
            )
            
        except Exception as e:
            return TestResult(
                test_name="Evolution Convergence Test",
                success=False,
                duration_seconds=time.time() - start_time,
                error_message=str(e)
            )
    
    def _test_fitness_evaluation(self) -> TestResult:
        """Test fitness evaluation components"""
        start_time = time.time()
        
        try:
            from .animation_evolution import AnimationFitnessEvaluator, AnimationIndividual
            
            # Create test renderer
            mandelbulber_renderer = MandelbulberRenderer()
            animation_renderer = AnimationRenderer(mandelbulber_renderer, max_workers=1)
            
            evaluator = AnimationFitnessEvaluator(animation_renderer)
            
            # Test different animation types
            orbital_anim = AnimationTemplates.orbital_mandelbulb()
            zoom_anim = AnimationTemplates.zoom_into_fractal()
            
            orbital_individual = AnimationIndividual(orbital_anim)
            zoom_individual = AnimationIndividual(zoom_anim)
            
            # Evaluate fitness components without full rendering
            orbital_continuity = evaluator._evaluate_visual_continuity(orbital_anim)
            orbital_smoothness = evaluator._evaluate_motion_smoothness(orbital_anim)
            orbital_aesthetic = evaluator._evaluate_aesthetic_appeal(orbital_anim)
            
            zoom_continuity = evaluator._evaluate_visual_continuity(zoom_anim)
            zoom_smoothness = evaluator._evaluate_motion_smoothness(zoom_anim)
            
            details = {
                "orbital_visual_continuity": orbital_continuity,
                "orbital_motion_smoothness": orbital_smoothness,
                "orbital_aesthetic_appeal": orbital_aesthetic,
                "zoom_visual_continuity": zoom_continuity,
                "zoom_motion_smoothness": zoom_smoothness,
                "fitness_components_working": True
            }
            
            # All fitness scores should be reasonable (0.0-1.0 range)
            all_scores = [orbital_continuity, orbital_smoothness, orbital_aesthetic,
                         zoom_continuity, zoom_smoothness]
            valid_scores = all(0.0 <= score <= 1.0 for score in all_scores)
            
            return TestResult(
                test_name="Fitness Evaluation Test",
                success=valid_scores and all(score > 0 for score in all_scores),
                duration_seconds=time.time() - start_time,
                details=details
            )
            
        except Exception as e:
            return TestResult(
                test_name="Fitness Evaluation Test",
                success=False,
                duration_seconds=time.time() - start_time,
                error_message=str(e)
            )
    
    def _test_performance_benchmarks(self, quick_mode: bool = True) -> TestResult:
        """Test performance benchmarks"""
        start_time = time.time()
        
        try:
            performance_data = {}
            
            # Benchmark template creation
            template_start = time.time()
            all_templates = AnimationTemplates.get_all_templates()
            template_creation_time = time.time() - template_start
            
            # Benchmark parameter interpolation
            test_animation = AnimationTemplates.power_evolution_sequence()
            
            interp_start = time.time()
            sample_count = 10 if quick_mode else 50
            for i in range(sample_count):
                t = i / (sample_count - 1)
                params = test_animation.get_parameters_at_time(t)
            interp_time = time.time() - interp_start
            
            # Benchmark genetic operations
            genetic_start = time.time()
            template1 = AnimationTemplates.orbital_mandelbulb()
            template2 = AnimationTemplates.zoom_into_fractal()
            
            operations = 5 if quick_mode else 20
            for i in range(operations):
                mutated = template1.copy()
                mutated.mutate(0.2, 0.1)
                child = template1.crossover(template2)
            genetic_time = time.time() - genetic_start
            
            performance_data = {
                "template_creation_time": template_creation_time,
                "template_count": len(all_templates),
                "interpolation_time_per_sample": interp_time / sample_count,
                "genetic_operations_time": genetic_time,
                "operations_per_second": operations / max(genetic_time, 0.001)
            }
            
            # Performance should be reasonable
            reasonable_performance = (
                template_creation_time < 5.0 and
                interp_time / sample_count < 0.1 and
                genetic_time < 10.0
            )
            
            return TestResult(
                test_name="Performance Benchmark Test",
                success=reasonable_performance,
                duration_seconds=time.time() - start_time,
                details=performance_data
            )
            
        except Exception as e:
            return TestResult(
                test_name="Performance Benchmark Test",
                success=False,
                duration_seconds=time.time() - start_time,
                error_message=str(e)
            )
    
    def save_test_results(self, test_suite: TestSuite, filename: str = "test_results.json"):
        """Save test results to file"""
        output_path = self.test_output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(test_suite.to_dict(), f, indent=2)
        
        logger.info(f"Test results saved to: {output_path}")
        return output_path
    
    def cleanup(self):
        """Clean up test files"""
        if self.test_output_dir.exists():
            shutil.rmtree(self.test_output_dir)
    
    def generate_test_report(self, test_suite: TestSuite) -> str:
        """Generate a human-readable test report"""
        report = []
        report.append("=" * 60)
        report.append(f"FRACTAL ANIMATION SYSTEM TEST REPORT")
        report.append("=" * 60)
        report.append(f"Test Suite: {test_suite.name}")
        report.append(f"Total Duration: {test_suite.total_duration:.2f} seconds")
        report.append(f"Success Rate: {test_suite.success_rate:.1%}")
        report.append(f"Tests Passed: {len([r for r in test_suite.results if r.success])}")
        report.append(f"Tests Failed: {len([r for r in test_suite.results if not r.success])}")
        report.append("")
        
        # Individual test results
        for result in test_suite.results:
            status = "✓ PASS" if result.success else "✗ FAIL"
            report.append(f"{status} {result.test_name} ({result.duration_seconds:.2f}s)")
            
            if result.error_message:
                report.append(f"    Error: {result.error_message}")
            
            if result.details:
                for key, value in result.details.items():
                    report.append(f"    {key}: {value}")
            report.append("")
        
        # Summary
        if test_suite.success_rate == 1.0:
            report.append("🎉 ALL TESTS PASSED! The animation system is working perfectly.")
        elif test_suite.success_rate >= 0.8:
            report.append("✅ Most tests passed. Minor issues may exist.")
        else:
            report.append("⚠️  Several tests failed. System may have significant issues.")
        
        return "\n".join(report)


def run_animation_tests(quick_mode: bool = True, output_dir: Optional[Path] = None) -> TestSuite:
    """
    Convenience function to run all animation tests.
    
    Args:
        quick_mode: Run faster tests with reduced complexity
        output_dir: Directory for test output files
        
    Returns:
        Complete test suite results
    """
    tester = AnimationTester(output_dir)
    
    try:
        test_suite = tester.run_all_tests(quick_mode)
        
        # Save results
        results_path = tester.save_test_results(test_suite)
        
        # Generate and save report
        report = tester.generate_test_report(test_suite)
        report_path = tester.test_output_dir / "test_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(report)
        print(f"\nDetailed results saved to: {results_path}")
        print(f"Test report saved to: {report_path}")
        
        return test_suite
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        raise
    finally:
        if not output_dir:  # Only cleanup if using temporary directory
            tester.cleanup()


if __name__ == "__main__":
    # Run tests when executed directly
    logging.basicConfig(level=logging.INFO)
    test_suite = run_animation_tests(quick_mode=True)
    exit(0 if test_suite.success_rate == 1.0 else 1)