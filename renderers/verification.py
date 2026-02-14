#!/usr/bin/env python3
"""
Fractal Rendering Verification System
=====================================

This module provides automated verification that fractal renders are:
1. Successful (file created)
2. Valid (proper image format, not corrupted)
3. Non-trivial (not all black/solid color - indicates rendering issues)
4. Diverse (different from other renders - indicates parameter variation worked)

Usage:
    from renderers.verification import verify_render, verify_batch

    result = verify_render("/path/to/image.png", min_diversity=0.01)
    print(result.status)  # "passed", "warning", "failed"
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of a render verification check."""
    status: str  # "passed", "warning", "failed"
    image_path: str
    checks: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.status in ("passed", "warning")

    @property
    def summary(self) -> str:
        if self.status == "passed":
            return "✓ All checks passed"
        elif self.status == "warning":
            return f"⚠ {len(self.warnings)} warnings: {'; '.join(self.warnings[:2])}"
        else:
            return f"✗ Failed: {'; '.join(self.errors[:2])}"


def verify_render(
    image_path: str,
    min_file_size: int = 1000,
    max_black_ratio: float = 0.95,
    min_mean_value: int = 20,
    compare_to: Optional[str] = None
) -> VerificationResult:
    """
    Verify a single rendered fractal image.

    Args:
        image_path: Path to the rendered image
        min_file_size: Minimum file size in bytes (default: 1KB)
        max_black_ratio: Maximum ratio of black pixels allowed (default: 95%)
        min_mean_value: Minimum mean pixel value (0-255, default: 20)
        compare_to: Optional path to compare diversity against

    Returns:
        VerificationResult with status and details
    """
    errors = []
    warnings = []
    checks = {}

    # Check 1: File exists
    if not os.path.exists(image_path):
        return VerificationResult(
            status="failed",
            image_path=image_path,
            errors=[f"File does not exist: {image_path}"]
        )

    checks["file_exists"] = True

    # Check 2: File size
    file_size = os.path.getsize(image_path)
    checks["file_size_bytes"] = file_size
    if file_size < min_file_size:
        errors.append(f"File too small ({file_size} < {min_file_size} bytes)")
    elif file_size < min_file_size * 10:
        warnings.append(f"File unusually small")

    # Check 3: Valid image format
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            mode = img.mode
            checks["dimensions"] = (width, height)
            checks["mode"] = mode

            # Convert to array for pixel analysis
            arr = np.array(img)

            # Check 4: Not all black (black silhouette check)
            if mode == 'RGBA':
                # For RGBA, check RGB channels
                rgb_arr = arr[:, :, :3]
            else:
                rgb_arr = arr

            # Calculate black pixel ratio
            black_pixels = np.sum(np.all(rgb_arr < 10, axis=2))
            total_pixels = rgb_arr.shape[0] * rgb_arr.shape[1]
            black_ratio = black_pixels / total_pixels
            checks["black_pixel_ratio"] = black_ratio

            if black_ratio > max_black_ratio:
                errors.append(f"Image is too dark ({black_ratio*100:.1f}% black pixels)")
            elif black_ratio > 0.8:
                warnings.append(f"Image is mostly dark ({black_ratio*100:.1f}%)")

            # Check 5: Mean value (overall brightness)
            mean_value = rgb_arr.mean()
            checks["mean_pixel_value"] = mean_value

            if mean_value < min_mean_value:
                errors.append(f"Image too dim (mean={mean_value:.1f} < {min_mean_value})")

            # Check 6: Pixel variance (not solid color)
            std_value = rgb_arr.std()
            checks["pixel_std_dev"] = std_value

            if std_value < 5:
                warnings.append(f"Low pixel variance (std={std_value:.1f})")

            # Check 7: Diversity vs comparison image
            if compare_to and os.path.exists(compare_to):
                with Image.open(compare_to) as comp_img:
                    comp_arr = np.array(comp_img)

                    # Resize if needed
                    if comp_arr.shape != rgb_arr.shape:
                        comp_arr = np.array(comp_img.resize((width, height), Image.Resampling.LANCZOS))

                    # Calculate structural similarity
                    diff = np.abs(rgb_arr.astype(float) - comp_arr.astype(float))
                    mean_diff = diff.mean()
                    max_diff = diff.max()

                    checks["diversity_vs_previous"] = {
                        "mean_diff": mean_diff,
                        "max_diff": max_diff
                    }

                    # Warn if too similar to previous
                    if mean_diff < 5:
                        warnings.append(f"Very similar to previous render (diff={mean_diff:.1f})")

    except Exception as e:
        return VerificationResult(
            status="failed",
            image_path=image_path,
            errors=[f"Failed to read image: {str(e)}"]
        )

    # Determine final status
    if errors:
        status = "failed"
    elif warnings:
        status = "warning"
    else:
        status = "passed"

    return VerificationResult(
        status=status,
        image_path=image_path,
        checks=checks,
        errors=errors,
        warnings=warnings
    )


def verify_batch(image_paths: List[str], required_ratio: float = 0.8) -> Dict[str, Any]:
    """
    Verify a batch of rendered images.

    Args:
        image_paths: List of paths to verify
        required_ratio: Minimum ratio of images that must pass (default: 80%)

    Returns:
        Dict with batch verification results
    """
    results = []
    for path in image_paths:
        result = verify_render(path)
        results.append({
            "path": path,
            "status": result.status,
            "summary": result.summary,
            "checks": result.checks
        })

    passed = sum(1 for r in results if r["status"] == "passed")
    failed = sum(1 for r in results if r["status"] == "failed")
    warnings = sum(1 for r in results if r["status"] == "warning")

    total = len(results)
    pass_rate = passed / total if total > 0 else 0

    # Determine batch status
    if pass_rate >= required_ratio:
        batch_status = "passed"
    elif pass_rate >= 0.5:
        batch_status = "warning"
    else:
        batch_status = "failed"

    return {
        "batch_status": batch_status,
        "total_images": total,
        "passed": passed,
        "failed": failed,
        "warnings": warnings,
        "pass_rate": pass_rate,
        "required_rate": required_ratio,
        "results": results,
        "all_images_diverse": _check_all_diverse(results)
    }


def _check_all_diverse(results: List[Dict]) -> bool:
    """Check if all rendered images have meaningful diversity."""
    if len(results) < 2:
        return True

    # Get mean pixel values
    means = []
    for r in results:
        if "checks" in r and "mean_pixel_value" in r["checks"]:
            means.append(r["checks"]["mean_pixel_value"])

    if len(means) < 2:
        return True

    # Check if there's meaningful variation in means
    mean_of_means = np.mean(means)
    std_of_means = np.std(means)

    # If all images have very similar mean values, they might be duplicates
    return std_of_means > 5 or mean_of_means > 30


def render_with_verification(
    render_func,
    output_path: str,
    render_args: Tuple = (),
    render_kwargs: Dict = None,
    **verify_kwargs
) -> Tuple[bool, VerificationResult]:
    """
    Helper to wrap a render function with verification.

    Args:
        render_func: Function that takes output_path and returns success bool
        output_path: Where to save the rendered image
        render_args: Positional args for render_func
        render_kwargs: Keyword args for render_func
        verify_kwargs: Additional args for verify_render

    Returns:
        Tuple of (render_success, verification_result)
    """
    if render_kwargs is None:
        render_kwargs = {}

    # Remove old file if exists
    if os.path.exists(output_path):
        os.remove(output_path)

    # Run render
    try:
        success = render_func(output_path, *render_args, **render_kwargs)
    except Exception as e:
        success = False
        logger.error(f"Render function failed: {e}")

    # Verify result
    verify_result = verify_render(output_path, **verify_kwargs)

    return success, verify_result


def generate_verification_report(batch_results: Dict, output_file: str = None) -> str:
    """Generate a text report of verification results."""
    lines = [
        "=" * 60,
        "FRACTAL RENDER VERIFICATION REPORT",
        "=" * 60,
        "",
        f"Batch Status: {batch_results['batch_status'].upper()}",
        f"Total Images: {batch_results['total_images']}",
        f"Passed: {batch_results['passed']}",
        f"Failed: {batch_results['failed']}",
        f"Warnings: {batch_results['warnings']}",
        f"Pass Rate: {batch_results['pass_rate']*100:.1f}%",
        f"Diversity Check: {'PASSED' if batch_results['all_images_diverse'] else 'FAILED'}",
        "",
        "-" * 60,
        "Individual Results:",
        "-" * 60,
    ]

    for result in batch_results['results']:
        lines.append(f"[{result['status'].upper():7}] {Path(result['path']).name}")
        if result['status'] != 'passed':
            lines.append(f"           {result['summary']}")

    report = "\n".join(lines)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)

    return report


if __name__ == "__main__":
    # Demo usage
    import argparse

    parser = argparse.ArgumentParser(description="Verify fractal renders")
    parser.add_argument("image", nargs="+", help="Image(s) to verify")
    parser.add_argument("--batch", action="store_true", help="Batch verify all images")
    parser.add_argument("--report", help="Write report to file")
    args = parser.parse_args()

    if args.batch:
        results = verify_batch(args.image)
        print(generate_verification_report(results, args.report))
    else:
        for img in args.image:
            result = verify_render(img)
            print(f"[{result.status.upper():7}] {img}")
            print(f"           {result.summary}")
