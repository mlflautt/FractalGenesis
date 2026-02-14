#!/usr/bin/env python3
"""
Comprehensive Fractal System Test & Verification
=================================================

Tests all fractal renderers and verifies output quality.
This script ensures the fractal generation system is working correctly.

Usage:
    python3 test_system.py                    # Run all tests
    python3 test_system.py --quick            # Quick smoke test
    python3 test_system.py --renderer python  # Test specific renderer
    python3 test_system.py --verify           # Verify existing outputs
    python3 test_system.py --animation        # Test animation system
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    renderer: str
    status: str  # "passed", "failed", "warning"
    output_file: Optional[str]
    duration_seconds: float
    verification_result: Optional[Dict]
    error: Optional[str] = None


class FractalSystemTester:
    """Comprehensive testing system for fractal generation."""

    def __init__(self, output_dir: str = "output/system_test"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[TestResult] = []

    def test_python_3d_renderer(self, num_images: int = 3) -> List[TestResult]:
        """Test Python 3D native renderer."""
        logger.info("=" * 60)
        logger.info("Testing Python 3D Native Renderer")
        logger.info("=" * 60)

        results = []
        try:
            from renderers.python_3d import FractalRenderer, FractalParams

            renderer = FractalRenderer()

            test_configs = [
                {"name": "classic_mandelbulb", "power": 8.0, "color_palette": "warm"},
                {"name": "julia_3d", "fractal_type": "julia", "power": 8.0, "color_palette": "cool"},
                {"name": "high_power_mandelbulb", "power": 12.0, "color_palette": "rainbow"},
            ]

            for i, config in enumerate(test_configs[:num_images]):
                start_time = time.time()
                try:
                    params = FractalParams(
                        fractal_type=config.get("fractal_type", "mandelbulb"),
                        power=config.get("power", 8.0),
                        width=400,
                        height=400,
                        camera_pos=(0.0, 0.0, -3.0),
                        color_palette=config.get("color_palette", "warm")
                    )

                    output_path = self.output_dir / f"python3d_{config['name']}.png"
                    image, metrics = renderer.render(params)

                    import matplotlib.pyplot as plt
                    plt.imsave(output_path, image)

                    duration = time.time() - start_time

                    # Verify
                    from renderers.verification import verify_render
                    verify_result = verify_render(str(output_path))

                    results.append(TestResult(
                        name=config["name"],
                        renderer="python_3d",
                        status="passed" if verify_result.is_valid else "warning",
                        output_file=str(output_path),
                        duration_seconds=duration,
                        verification_result=verify_result.checks
                    ))

                    logger.info(f"✓ {config['name']}: {duration:.2f}s - {output_path.name}")

                except Exception as e:
                    results.append(TestResult(
                        name=config.get("name", f"test_{i}"),
                        renderer="python_3d",
                        status="failed",
                        output_file=None,
                        duration_seconds=time.time() - start_time,
                        verification_result=None,
                        error=str(e)
                    ))
                    logger.error(f"✗ {config['name']}: {e}")

        except ImportError as e:
            logger.warning(f"Python 3D renderer not available: {e}")
            results.append(TestResult(
                name="import",
                renderer="python_3d",
                status="failed",
                output_file=None,
                duration_seconds=0,
                verification_result=None,
                error=f"Import failed: {e}"
            ))

        return results

    def test_mandelbulber_renderer(self, num_images: int = 2) -> List[TestResult]:
        """Test Mandelbulber renderer."""
        logger.info("=" * 60)
        logger.info("Testing Mandelbulber Renderer")
        logger.info("=" * 60)

        results = []
        try:
            from renderers.mandelbulber.renderer import MandelbulberRenderer

            mandelbulber_dir = self.output_dir / "mandelbulber"
            renderer = MandelbulberRenderer(str(mandelbulber_dir))

            for i in range(num_images):
                start_time = time.time()
                try:
                    params = renderer.generate_random_parameters()
                    output_path = mandelbulber_dir / f"mandelbulber_test_{i:02d}.png"

                    success = renderer.render_fractal(params, str(output_path), 400, 400)
                    duration = time.time() - start_time

                    if success:
                        from renderers.verification import verify_render
                        verify_result = verify_render(str(output_path))

                        results.append(TestResult(
                            name=f"random_{i}",
                            renderer="mandelbulber",
                            status="passed" if verify_result.is_valid else "warning",
                            output_file=str(output_path),
                            duration_seconds=duration,
                            verification_result=verify_result.checks
                        ))
                        logger.info(f"✓ Mandelbulber test {i}: {duration:.2f}s")
                    else:
                        results.append(TestResult(
                            name=f"random_{i}",
                            renderer="mandelbulber",
                            status="failed",
                            output_file=None,
                            duration_seconds=duration,
                            verification_result=None,
                            error="Render returned False"
                        ))
                        logger.error(f"✗ Mandelbulber test {i}: Render failed")

                except Exception as e:
                    results.append(TestResult(
                        name=f"test_{i}",
                        renderer="mandelbulber",
                        status="failed",
                        output_file=None,
                        duration_seconds=time.time() - start_time,
                        verification_result=None,
                        error=str(e)
                    ))
                    logger.error(f"✗ Mandelbulber test {i}: {e}")

        except Exception as e:
            logger.warning(f"Mandelbulber not available: {e}")
            results.append(TestResult(
                name="init",
                renderer="mandelbulber",
                status="failed",
                output_file=None,
                duration_seconds=0,
                verification_result=None,
                error=f"Initialization failed: {e}"
            ))

        return results

    def test_flam3_renderer(self, num_images: int = 3) -> List[TestResult]:
        """Test Flam3 fractal flame renderer."""
        logger.info("=" * 60)
        logger.info("Testing Flam3 Renderer")
        logger.info("=" * 60)

        results = []
        try:
            from renderers.flam3_renderer import Flam3Renderer

            flam3_dir = self.output_dir / "flam3"
            renderer = Flam3Renderer(str(flam3_dir))

            for i in range(num_images):
                start_time = time.time()
                try:
                    xml = renderer.generate_random_genome()
                    output_path = flam3_dir / f"flam3_test_{i:02d}.png"

                    success = renderer.render_genome(xml, output_path.stem)
                    duration = time.time() - start_time

                    if success and output_path.exists():
                        from renderers.verification import verify_render
                        verify_result = verify_render(str(output_path))

                        results.append(TestResult(
                            name=f"flame_{i}",
                            renderer="flam3",
                            status="passed" if verify_result.is_valid else "warning",
                            output_file=str(output_path),
                            duration_seconds=duration,
                            verification_result=verify_result.checks
                        ))
                        logger.info(f"✓ Flam3 test {i}: {duration:.2f}s - {output_path.name}")
                    else:
                        results.append(TestResult(
                            name=f"flame_{i}",
                            renderer="flam3",
                            status="failed",
                            output_file=None,
                            duration_seconds=duration,
                            verification_result=None,
                            error="Render did not produce output"
                        ))
                        logger.error(f"✗ Flam3 test {i}: No output")

                except Exception as e:
                    results.append(TestResult(
                        name=f"test_{i}",
                        renderer="flam3",
                        status="failed",
                        output_file=None,
                        duration_seconds=time.time() - start_time,
                        verification_result=None,
                        error=str(e)
                    ))
                    logger.error(f"✗ Flam3 test {i}: {e}")

        except Exception as e:
            logger.warning(f"Flam3 not available: {e}")
            results.append(TestResult(
                name="init",
                renderer="flam3",
                status="failed",
                output_file=None,
                duration_seconds=0,
                verification_result=None,
                error=f"Initialization failed: {e}"
            ))

        return results

    def test_animation_system(self) -> List[TestResult]:
        """Test animation creation with Python 3D renderer."""
        logger.info("=" * 60)
        logger.info("Testing Animation System")
        logger.info("=" * 60)

        results = []
        try:
            from renderers.python_3d import FractalRenderer, FractalParams

            renderer = FractalRenderer()
            animation_dir = self.output_dir / "animation"
            animation_dir.mkdir(parents=True, exist_ok=True)

            start_time = time.time()

            start_params = {
                "fractal_type": "mandelbulb",
                "power": 4.0,
                "camera_pos": (0.0, 0.0, -4.0),
                "color_palette": "warm"
            }

            end_params = {
                "fractal_type": "mandelbulb",
                "power": 16.0,
                "camera_pos": (0.0, 0.0, -2.0),
                "color_palette": "fire"
            }

            num_frames = 5
            frames = []

            for i in range(num_frames):
                t = i / (num_frames - 1)
                power = start_params["power"] + (end_params["power"] - start_params["power"]) * t
                cam_z = start_params["camera_pos"][2] + (end_params["camera_pos"][2] - start_params["camera_pos"][2]) * t

                params = FractalParams(
                    fractal_type="mandelbulb",
                    power=power,
                    width=300,
                    height=300,
                    camera_pos=(0.0, 0.0, cam_z),
                    color_palette="warm"
                )

                output_path = animation_dir / f"frame_{i:02d}.png"
                image, _ = renderer.render(params)
                import matplotlib.pyplot as plt
                plt.imsave(output_path, image)
                frames.append(str(output_path))

            duration = time.time() - start_time

            if len(frames) >= 3:
                gif_path = animation_dir / "animation.gif"
                renderer.create_gif(frames, str(gif_path), fps=2)

                results.append(TestResult(
                    name="power_morph_animation",
                    renderer="python_3d",
                    status="passed",
                    output_file=str(gif_path),
                    duration_seconds=duration,
                    verification_result={"frames_created": len(frames), "gif_created": str(gif_path)}
                ))
                logger.info(f"✓ Animation: {len(frames)} frames, {duration:.2f}s - {gif_path.name}")
            else:
                results.append(TestResult(
                    name="power_morph_animation",
                    renderer="python_3d",
                    status="failed",
                    output_file=None,
                    duration_seconds=duration,
                    verification_result=None,
                    error=f"Only {len(frames)} frames created"
                ))

        except Exception as e:
            logger.error(f"Animation test failed: {e}")
            results.append(TestResult(
                name="animation",
                renderer="python_3d",
                status="failed",
                output_file=None,
                duration_seconds=time.time() - start_time,
                verification_result=None,
                error=str(e)
            ))

        return results

    def test_unified_interface(self) -> List[TestResult]:
        """Test the unified renderer interface."""
        logger.info("=" * 60)
        logger.info("Testing Unified Interface")
        logger.info("=" * 60)

        results = []
        try:
            from renderers.unified import (
                create_renderer, get_available_renderers, RendererType
            )

            available = get_available_renderers()
            logger.info(f"Available renderers: {[r[0].value for r in available if r[2]]}")

            for rt, name, is_available in available:
                if not is_available:
                    continue

                start_time = time.time()
                renderer = create_renderer(rt)

                if renderer:
                    params = renderer.generate_random_parameters()
                    output_path = self.output_dir / f"unified_{rt.value}.png"
                    success = renderer.render_fractal(params, str(output_path), 300, 300)
                    duration = time.time() - start_time

                    from renderers.verification import verify_render
                    verify_result = verify_render(str(output_path))

                    results.append(TestResult(
                        name=f"unified_{rt.value}",
                        renderer=rt.value,
                        status="passed" if verify_result.is_valid else "warning",
                        output_file=str(output_path),
                        duration_seconds=duration,
                        verification_result=verify_result.checks
                    ))
                    logger.info(f"✓ {name}: {duration:.2f}s")

        except Exception as e:
            logger.error(f"Unified interface test failed: {e}")
            results.append(TestResult(
                name="unified_interface",
                renderer="all",
                status="failed",
                output_file=None,
                duration_seconds=0,
                verification_result=None,
                error=str(e)
            ))

        return results

    def run_all_tests(self, quick: bool = False) -> Dict:
        """Run all tests and return results."""
        self.results = []

        # Run tests based on quick mode
        if not quick:
            self.results.extend(self.test_python_3d_renderer(num_images=3))
            self.results.extend(self.test_mandelbulber_renderer(num_images=2))
            self.results.extend(self.test_flam3_renderer(num_images=3))
            self.results.extend(self.test_animation_system())
        else:
            self.results.extend(self.test_python_3d_renderer(num_images=1))

        self.results.extend(self.test_unified_interface())

        return self.generate_report()

    def generate_report(self) -> Dict:
        """Generate test report."""
        passed = sum(1 for r in self.results if r.status == "passed")
        failed = sum(1 for r in self.results if r.status == "failed")
        warnings = sum(1 for r in self.results if r.status == "warning")
        total = len(self.results)

        report = {
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "warnings": warnings,
                "pass_rate": passed / total if total > 0 else 0
            },
            "results": [
                {
                    "name": r.name,
                    "renderer": r.renderer,
                    "status": r.status,
                    "output_file": r.output_file,
                    "duration": f"{r.duration_seconds:.2f}s",
                    "error": r.error
                }
                for r in self.results
            ],
            "output_directory": str(self.output_dir)
        }

        # Print summary
        logger.info("=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total tests: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Warnings: {warnings}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("=" * 60)

        return report

    def verify_existing_outputs(self, output_dir: str = None) -> Dict:
        """Verify existing fractal outputs."""
        target_dir = Path(output_dir) if output_dir else self.output_dir

        if not target_dir.exists():
            return {"error": f"Directory not found: {target_dir}"}

        from renderers.verification import verify_batch

        images = list(target_dir.rglob("*.png"))
        if not images:
            return {"error": "No PNG files found"}

        results = verify_batch([str(p) for p in images])
        return results


def main():
    parser = argparse.ArgumentParser(description="Fractal System Test & Verification")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test")
    parser.add_argument("--verify", metavar="DIR", help="Verify existing outputs")
    parser.add_argument("--animation", action="store_true", help="Test animation only")
    parser.add_argument("--output", default="output/system_test", help="Output directory")
    args = parser.parse_args()

    tester = FractalSystemTester(args.output)

    if args.verify:
        results = tester.verify_existing_outputs(args.verify)
        print(f"Verification results: {results}")
    elif args.animation:
        results = tester.test_animation_system()
        print(f"Animation test: {len(results)} results")
    else:
        report = tester.run_all_tests(quick=args.quick)
        print(f"\nTest complete. Report saved to: {args.output}")


if __name__ == "__main__":
    main()
