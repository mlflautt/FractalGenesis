#!/usr/bin/env python3
"""
Automated 3D Mandelbulb Fractal Image Generation System
======================================================

This script provides automated generation, testing, and verification of 3D mandelbulb
fractal images using the Python 3D renderer. It includes:

- Batch generation with parameter randomization
- Quality verification and assessment
- Systematic testing and validation
- Organized output folder structure
- Performance monitoring and troubleshooting

Usage:
    python3 automated_mandelbulb_generation.py --num_samples 50 --output_dir output/mandelbulb_batch
    python3 automated_mandelbulb_generation.py --test_mode --verify_quality
    python3 automated_mandelbulb_generation.py --troubleshoot --parameter_range_check
"""

import sys
import os
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging
import hashlib
import warnings

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from renderers.python_3d.fractal_formulas import FormulaRegistry, FormulaRenderer, FormulaParameters

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mandelbulb_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for automated mandelbulb generation"""
    num_samples: int = 100
    output_dir: str = "output/mandelbulb_generation"
    image_size: Tuple[int, int] = (800, 600)
    quality_threshold: float = 0.7
    enable_verification: bool = True
    batch_size: int = 10
    random_seed: Optional[int] = None
    parameter_ranges: Optional[Dict[str, Tuple[float, float]]] = None

    def __post_init__(self):
        if self.parameter_ranges is None:
            self.parameter_ranges = {
                'power': (2.0, 16.0),
                'iterations': (50, 150),
                'bailout': (1.5, 4.0),
                'camera_distance': (2.0, 6.0),
                'camera_angle_xy': (0.0, 2 * np.pi),
                'camera_angle_z': (-np.pi/4, np.pi/4),
                'color_intensity': (0.5, 2.0),
                'ambient': (0.05, 0.25),
                'metallic': (0.0, 0.8),
                'roughness': (0.1, 0.9)
            }

@dataclass
class QualityMetrics:
    """Quality assessment metrics for generated images"""
    surface_coverage: float  # Percentage of pixels showing fractal surface
    detail_level: float     # Measure of fine structure complexity
    color_diversity: float  # Range and distribution of colors
    brightness_uniformity: float  # Evenness of brightness distribution
    artifact_score: float   # Measure of rendering artifacts (lower is better)
    overall_quality: float  # Combined quality score

    def __post_init__(self):
        # Convert numpy types to Python types
        self.surface_coverage = float(self.surface_coverage)
        self.detail_level = float(self.detail_level)
        self.color_diversity = float(self.color_diversity)
        self.brightness_uniformity = float(self.brightness_uniformity)
        self.artifact_score = float(self.artifact_score)
        self.overall_quality = float(self.overall_quality)

@dataclass
class GenerationResult:
    """Result of a single fractal generation"""
    sample_id: int
    formula_name: str
    parameters: Dict[str, Any]
    render_time: float
    quality_metrics: QualityMetrics
    file_path: Optional[str]
    success: bool
    error_message: Optional[str] = None

class FractalVerifier:
    """Comprehensive verification system for 3D mandelbulb images"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def assess_quality(self, image: np.ndarray, parameters: Dict[str, Any]) -> QualityMetrics:
        """Assess the quality of a generated mandelbulb image"""

        # Convert to float for analysis
        img_float = image.astype(np.float64) / 255.0

        # Surface coverage: percentage of non-background pixels
        # Background is typically dark blue/black gradient
        background_mask = np.all(img_float < 0.1, axis=2)
        surface_pixels = np.sum(~background_mask)
        total_pixels = image.shape[0] * image.shape[1]
        surface_coverage = surface_pixels / total_pixels

        # Detail level: measure of high-frequency content using Laplacian variance
        gray = np.mean(img_float, axis=2)
        laplacian = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=np.float64)

        detail_level = np.var(self._convolve2d(gray, laplacian))

        # Color diversity: measure of color distribution
        pixels = img_float.reshape(-1, 3)
        # Remove background pixels for diversity calculation
        non_bg_pixels = pixels[~np.all(pixels < 0.1, axis=1)]
        if len(non_bg_pixels) > 0:
            color_diversity = np.std(non_bg_pixels, axis=0).mean()
        else:
            color_diversity = 0.0

        # Brightness uniformity: coefficient of variation of brightness
        brightness = np.mean(img_float, axis=2)
        brightness_std = np.std(brightness)
        brightness_mean = np.mean(brightness)
        brightness_uniformity = 1.0 - (brightness_std / (brightness_mean + 1e-6))

        # Artifact detection: look for unnatural patterns
        # Simple artifact score based on unusual brightness spikes
        diff_rows = np.abs(np.diff(brightness, axis=0))
        diff_cols = np.abs(np.diff(brightness, axis=1))
        # Use the minimum size to avoid broadcasting issues
        min_rows = min(diff_rows.shape[0], diff_cols.shape[0])
        min_cols = min(diff_rows.shape[1], diff_cols.shape[1])
        brightness_diff = diff_rows[:min_rows, :min_cols] + diff_cols[:min_rows, :min_cols]
        artifact_score = np.mean(brightness_diff > 0.5)  # Percentage of high-contrast edges

        # Overall quality score (weighted combination)
        overall_quality = (
            0.3 * surface_coverage +
            0.2 * min(detail_level / 1000.0, 1.0) +  # Normalize detail level
            0.2 * min(color_diversity * 5.0, 1.0) +   # Normalize color diversity
            0.2 * brightness_uniformity +
            0.1 * (1.0 - artifact_score)  # Lower artifacts = higher quality
        )

        return QualityMetrics(
            surface_coverage=float(surface_coverage),
            detail_level=float(detail_level),
            color_diversity=float(color_diversity),
            brightness_uniformity=float(brightness_uniformity),
            artifact_score=float(artifact_score),
            overall_quality=float(overall_quality)
        )

    def _convolve2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Simple 2D convolution for edge detection"""
        from scipy.signal import convolve2d
        return convolve2d(image, kernel, mode='same', boundary='symm')

    def verify_parameter_ranges(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Verify that parameters are within reasonable ranges"""
        issues = []

        # Power should be reasonable for mandelbulb
        if not (2.0 <= parameters.get('power', 8.0) <= 20.0):
            issues.append(f"Power {parameters['power']} outside recommended range [2.0, 20.0]")

        # Iterations should be sufficient but not excessive
        if not (20 <= parameters.get('iterations', 100) <= 300):
            issues.append(f"Iterations {parameters['iterations']} outside recommended range [20, 300]")

        # Camera distance should be reasonable
        camera_pos = parameters.get('camera_pos', (0, 0, -3))
        distance = np.sqrt(sum(x**2 for x in camera_pos))
        if not (1.0 <= distance <= 10.0):
            issues.append(f"Camera distance {distance:.2f} outside recommended range [1.0, 10.0]")

        # Color intensity should be reasonable
        if not (0.1 <= parameters.get('color_intensity', 1.0) <= 3.0):
            issues.append(f"Color intensity {parameters['color_intensity']} outside recommended range [0.1, 3.0]")

        return len(issues) == 0, issues

class FractalBatchGenerator:
    """Automated batch generation system for 3D fractals using formula system"""

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.registry = FormulaRegistry()
        self.renderer = FormulaRenderer(self.registry)
        self.verifier = FractalVerifier()
        self.output_dir = Path(config.output_dir)

        # Set random seed for reproducibility
        if config.random_seed is not None:
            np.random.seed(config.random_seed)

        # Create output directory structure
        self._create_output_structure()

        self.logger = logging.getLogger(__name__)

    def _create_output_structure(self):
        """Create organized output directory structure"""
        dirs = [
            self.output_dir / "images",
            self.output_dir / "metadata",
            self.output_dir / "verification",
            self.output_dir / "logs"
        ]

        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

    def generate_random_parameters(self) -> Tuple[str, FormulaParameters]:
        """Generate random parameters for a randomly selected fractal formula"""
        # Select random formula
        available_formulas = self.registry.list_formulas()
        formula_name = np.random.choice(available_formulas)
        formula = self.registry.get_formula(formula_name)

        if formula is None:
            # Fallback to mandelbulb
            formula_name = "mandelbulb"
            formula = self.registry.get_formula(formula_name)

        # Get default parameters and randomize them
        params = formula.get_default_parameters()

        # Randomize common parameters
        ranges = self.config.parameter_ranges or self.config.__dataclass_fields__['parameter_ranges'].default

        params.power = np.random.uniform(ranges['power'][0], ranges['power'][1])
        params.iterations = int(np.random.uniform(ranges['iterations'][0], ranges['iterations'][1]))
        params.bailout = np.random.uniform(ranges['bailout'][0], ranges['bailout'][1])

        # Randomize formula-specific parameters
        if hasattr(params, 'folding_limit'):
            params.folding_limit = np.random.uniform(0.5, 2.0)
        if hasattr(params, 'folding_value'):
            params.folding_value = np.random.uniform(1.5, 3.0)
        if hasattr(params, 'scale'):
            params.scale = np.random.uniform(-2.0, 2.0)
        if hasattr(params, 'cx'):
            params.cx = np.random.uniform(-1.0, 1.0)
        if hasattr(params, 'cy'):
            params.cy = np.random.uniform(-1.0, 1.0)
        if hasattr(params, 'cz'):
            params.cz = np.random.uniform(-1.0, 1.0)

        return formula_name, params

        return params

    def generate_sample(self, sample_id: int) -> GenerationResult:
        """Generate a single mandelbulb sample"""
        start_time = time.time()

        try:
            # Generate random parameters
            formula_name, params = self.generate_random_parameters()

            # Render the fractal
            image, render_metrics = self.renderer.render_formula(formula_name, params)

            # Assess quality
            quality_metrics = self.verifier.assess_quality(image, params.__dict__)

            # Verify parameter ranges
            params_valid, param_issues = self.verifier.verify_parameter_ranges(params.__dict__)

            # Determine success based on quality threshold
            success = (
                quality_metrics.overall_quality >= self.config.quality_threshold and
                params_valid and
                render_metrics['surface_pixels'] > 1000  # At least some surface visible
            )

            # Save image if successful or if we're saving all attempts
            if success or not self.config.enable_verification:
                filename = f"mandelbulb_{sample_id:04d}.png"
                image_path = self.output_dir / "images" / filename
                self._save_image(image, image_path)
            else:
                image_path = None

            render_time = time.time() - start_time

            result = GenerationResult(
                sample_id=sample_id,
                formula_name=formula_name,
                parameters=params.to_dict(),
                render_time=render_time,
                quality_metrics=quality_metrics,
                file_path=str(image_path) if image_path else None,
                success=success
            )

            if not success:
                issues = []
                if quality_metrics.overall_quality < self.config.quality_threshold:
                    issues.append(".2f")
                if not params_valid:
                    issues.extend(param_issues)
                if render_metrics['surface_pixels'] <= 1000:
                    issues.append(f"Too few surface pixels: {render_metrics['surface_pixels']}")
                result.error_message = "; ".join(issues)

            return result

        except Exception as e:
            render_time = time.time() - start_time
            self.logger.error(f"Error generating sample {sample_id}: {e}")

            return GenerationResult(
                sample_id=sample_id,
                formula_name="unknown",
                parameters={},
                render_time=render_time,
                quality_metrics=QualityMetrics(0, 0, 0, 0, 1, 0),
                file_path=None,
                success=False,
                error_message=str(e)
            )

    def _save_image(self, image: np.ndarray, filepath: Path):
        """Save image with metadata"""
        plt.figure(figsize=(12, 9))
        plt.imshow(image, origin='upper')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filepath, dpi=100, bbox_inches='tight', facecolor='black')
        plt.close()

    def generate_batch(self) -> Dict[str, Any]:
        """Generate a batch of mandelbulb samples"""
        self.logger.info(f"Starting batch generation of {self.config.num_samples} samples")
        self.logger.info(f"Output directory: {self.output_dir.absolute()}")

        results = []
        successful = 0
        total_time = 0

        # Progress tracking
        start_batch_time = time.time()

        for i in range(self.config.num_samples):
            if (i + 1) % self.config.batch_size == 0:
                self.logger.info(f"Generated {i + 1}/{self.config.num_samples} samples...")

            result = self.generate_sample(i + 1)
            results.append(result)

            if result.success:
                successful += 1

            total_time += result.render_time

            # Save intermediate results every 10 samples
            if (i + 1) % 10 == 0:
                self._save_intermediate_results(results[:i+1])

        # Final results
        batch_time = time.time() - start_batch_time
        success_rate = successful / self.config.num_samples

        self.logger.info(f"Batch generation complete!")
        self.logger.info(f"Success rate: {success_rate:.1%} ({successful}/{self.config.num_samples})")
        self.logger.info(f"Total time: {batch_time:.2f}s, Average time per sample: {total_time/self.config.num_samples:.2f}s")

        # Save final results
        final_results = self._save_final_results(results)

        return final_results

    def _save_intermediate_results(self, results: List[GenerationResult]):
        """Save intermediate results for monitoring"""
        intermediate_file = self.output_dir / "metadata" / "intermediate_results.json"

        results_dict = []
        for result in results:
            result_dict = {
                'sample_id': result.sample_id,
                'success': result.success,
                'render_time': result.render_time,
                'quality_score': result.quality_metrics.overall_quality,
                'file_path': result.file_path,
                'error_message': result.error_message
            }
            results_dict.append(result_dict)

        with open(intermediate_file, 'w') as f:
            json.dump(results_dict, f, indent=2)

    def _save_final_results(self, results: List[GenerationResult]) -> Dict[str, Any]:
        """Save comprehensive final results"""
        # Convert results to dictionaries
        results_dict = []
        for result in results:
            result_dict = {
                'sample_id': result.sample_id,
                'success': result.success,
                'render_time': result.render_time,
                'parameters': result.parameters,
                'quality_metrics': {
                    'surface_coverage': result.quality_metrics.surface_coverage,
                    'detail_level': result.quality_metrics.detail_level,
                    'color_diversity': result.quality_metrics.color_diversity,
                    'brightness_uniformity': result.quality_metrics.brightness_uniformity,
                    'artifact_score': result.quality_metrics.artifact_score,
                    'overall_quality': result.quality_metrics.overall_quality
                },
                'file_path': result.file_path,
                'error_message': result.error_message
            }
            results_dict.append(result_dict)

        # Save detailed results
        results_file = self.output_dir / "metadata" / "generation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2, default=self._json_serializer)

        # Calculate statistics
        successful_results = [r for r in results if r.success]
        quality_scores = [r.quality_metrics.overall_quality for r in successful_results]
        render_times = [r.render_time for r in results]

        stats = {
            'total_samples': len(results),
            'successful_samples': len(successful_results),
            'success_rate': len(successful_results) / len(results),
            'average_quality': float(np.mean(quality_scores) if quality_scores else 0),
            'quality_std': float(np.std(quality_scores) if quality_scores else 0),
            'average_render_time': float(np.mean(render_times)),
            'total_time': float(sum(render_times)),
            'config': {
                'num_samples': self.config.num_samples,
                'output_dir': self.config.output_dir,
                'image_size': self.config.image_size,
                'quality_threshold': self.config.quality_threshold,
                'enable_verification': self.config.enable_verification,
                'batch_size': self.config.batch_size,
                'random_seed': self.config.random_seed
            },
            'timestamp': time.time(),
            'results': results_dict
        }

        # Save statistics
        stats_file = self.output_dir / "metadata" / "generation_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=self._json_serializer)

        # Generate quality report
        self._generate_quality_report(successful_results)

        return stats

    def _generate_quality_report(self, successful_results: List[GenerationResult]):
        """Generate detailed quality analysis report"""
        if not successful_results:
            return

        quality_data = []
        for result in successful_results:
            quality_data.append({
                'sample_id': result.sample_id,
                'quality_score': result.quality_metrics.overall_quality,
                'surface_coverage': result.quality_metrics.surface_coverage,
                'detail_level': result.quality_metrics.detail_level,
                'color_diversity': result.quality_metrics.color_diversity,
                'brightness_uniformity': result.quality_metrics.brightness_uniformity,
                'artifact_score': result.quality_metrics.artifact_score,
                'render_time': result.render_time
            })

        # Quality distribution analysis
        quality_scores = [d['quality_score'] for d in quality_data]

        report = {
            'total_successful': len(successful_results),
            'quality_distribution': {
                'mean': np.mean(quality_scores),
                'median': np.median(quality_scores),
                'std': np.std(quality_scores),
                'min': np.min(quality_scores),
                'max': np.max(quality_scores),
                'percentiles': {
                    '25th': np.percentile(quality_scores, 25),
                    '75th': np.percentile(quality_scores, 75),
                    '90th': np.percentile(quality_scores, 90)
                }
            },
            'correlations': self._calculate_correlations(quality_data),
            'quality_data': quality_data
        }

        report_file = self.output_dir / "verification" / "quality_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=self._json_serializer)

    def _calculate_correlations(self, quality_data: List[Dict]) -> Dict[str, float]:
        """Calculate correlations between quality metrics"""
        if len(quality_data) < 2:
            return {}

        metrics = ['quality_score', 'surface_coverage', 'detail_level',
                  'color_diversity', 'brightness_uniformity', 'render_time']

        correlations = {}
        for i, metric1 in enumerate(metrics):
            for metric2 in metrics[i+1:]:
                values1 = [d[metric1] for d in quality_data]
                values2 = [d[metric2] for d in quality_data]

                if len(values1) > 1 and np.std(values1) > 0 and np.std(values2) > 0:
                    corr = np.corrcoef(values1, values2)[0, 1]
                    correlations[f"{metric1}_vs_{metric2}"] = corr

        return correlations

    def _json_serializer(self, obj):
        """JSON serializer for non-serializable objects"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bool):
            return obj
        else:
            return str(obj)

def run_system_tests():
    """Run comprehensive system tests for mandelbulb generation"""
    logger.info("Running system tests for automated mandelbulb generation...")

    config = GenerationConfig(
        num_samples=20,  # Small test batch
        output_dir="output/mandelbulb_system_test",
        enable_verification=True,
        quality_threshold=0.5  # Lower threshold for testing
    )

    generator = FractalBatchGenerator(config)

    # Test parameter generation
    logger.info("Testing parameter generation...")
    params = generator.generate_random_parameters()
    verifier = FractalVerifier()
    params_valid, issues = verifier.verify_parameter_ranges(params.__dict__)

    if not params_valid:
        logger.warning(f"Parameter validation issues: {issues}")

    # Test single generation
    logger.info("Testing single sample generation...")
    result = generator.generate_sample(1)

    if result.success:
        logger.info(f"✓ Single generation successful, quality: {result.quality_metrics.overall_quality:.3f}")
    else:
        logger.error(f"✗ Single generation failed: {result.error_message}")

    # Test batch generation
    logger.info("Testing batch generation...")
    stats = generator.generate_batch()

    logger.info("System tests complete!")
    logger.info(f"Batch results: {stats['successful_samples']}/{stats['total_samples']} successful")
    logger.info(f"Average quality: {stats['average_quality']:.3f}")

    return stats

def run_troubleshooting(parameter_range_check: bool = False):
    """Run troubleshooting diagnostics"""
    logger.info("Running troubleshooting diagnostics...")

    verifier = FractalVerifier()

    if parameter_range_check:
        logger.info("Testing parameter range validation...")

        # Test various parameter combinations
        test_cases = [
            {"power": 8.0, "iterations": 100, "camera_pos": (0, 0, -3)},  # Valid
            {"power": 25.0, "iterations": 100, "camera_pos": (0, 0, -3)}, # Invalid power
            {"power": 8.0, "iterations": 500, "camera_pos": (0, 0, -3)},  # Invalid iterations
            {"power": 8.0, "iterations": 100, "camera_pos": (0, 0, -20)}, # Invalid distance
        ]

        for i, params in enumerate(test_cases):
            valid, issues = verifier.verify_parameter_ranges(params)
            status = "✓" if valid else "✗"
            logger.info(f"Test case {i+1}: {status} - {issues if issues else 'Valid'}")

def main():
    parser = argparse.ArgumentParser(description="Automated 3D Mandelbulb Fractal Generation")
    parser.add_argument("--num_samples", type=int, default=50,
                       help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="output/mandelbulb_generation",
                       help="Output directory for generated images")
    parser.add_argument("--image_width", type=int, default=800,
                       help="Image width")
    parser.add_argument("--image_height", type=int, default=600,
                       help="Image height")
    parser.add_argument("--quality_threshold", type=float, default=0.7,
                       help="Minimum quality threshold for successful generation")
    parser.add_argument("--batch_size", type=int, default=10,
                       help="Batch size for progress reporting")
    parser.add_argument("--random_seed", type=int, default=None,
                       help="Random seed for reproducible generation")
    parser.add_argument("--test_mode", action="store_true",
                       help="Run system tests instead of generation")
    parser.add_argument("--verify_quality", action="store_true",
                       help="Enable quality verification")
    parser.add_argument("--troubleshoot", action="store_true",
                       help="Run troubleshooting diagnostics")
    parser.add_argument("--parameter_range_check", action="store_true",
                       help="Check parameter range validation")

    args = parser.parse_args()

    # Configure logging level
    if args.test_mode or args.troubleshoot:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        if args.test_mode:
            run_system_tests()
        elif args.troubleshoot:
            run_troubleshooting(args.parameter_range_check)
        else:
            # Configure generation
            config = GenerationConfig(
                num_samples=args.num_samples,
                output_dir=args.output_dir,
                image_size=(args.image_width, args.image_height),
                quality_threshold=args.quality_threshold,
                enable_verification=args.verify_quality,
                batch_size=args.batch_size,
                random_seed=args.random_seed
            )

            # Run generation
            generator = FractalBatchGenerator(config)
            stats = generator.generate_batch()

            print("\n🎯 Generation Complete!")
            print(f"Success Rate: {stats['success_rate']:.1%}")
            print(f"Average Quality: {stats['average_quality']:.3f}")
            print(f"Output Directory: {config.output_dir}")
            print(f"Results: {stats['successful_samples']}/{stats['total_samples']} successful")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()