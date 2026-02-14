#!/usr/bin/env python3
"""
Diversity Verification Engine
=============================

Detects "same silhouette" problems and ensures creative diversity in fractal renders.

Usage:
    from verification.diversity_engine import DiversityEngine
    
    engine = DiversityEngine()
    result = engine.verify_batch_diversity(image_paths)
    
    if not result['passed']:
        print("Diversity check failed:", result['recommendations'])
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageFilter
import hashlib
from collections import defaultdict


@dataclass
class DiversityMetrics:
    """Comprehensive metrics for analyzing fractal diversity"""
    # File info
    file_path: str
    file_size: int
    dimensions: Tuple[int, int]
    
    # Visual composition
    compositional_hash: str      # Perceptual hash of structure
    brightness_mean: float       # Average brightness
    brightness_std: float        # Brightness variation
    edge_density: float          # Amount of edges/detail
    
    # Color analysis
    dominant_colors: List[Tuple[int, int, int]]  # Top 5 colors
    color_entropy: float         # Color variety (Shannon entropy)
    
    # Geometric properties
    center_of_mass: Tuple[float, float]  # Where the fractal is centered
    symmetry_score: float        # How symmetric the image is
    

class DiversityEngine:
    """
    Engine for detecting lack of diversity in fractal renders.
    Catches the "same silhouette" problem where different parameters
    produce visually identical results.
    """
    
    def __init__(self):
        self.metrics_cache: Dict[str, DiversityMetrics] = {}
        
    def analyze_image(self, image_path: str) -> DiversityMetrics:
        """Extract comprehensive diversity metrics from image"""
        
        if image_path in self.metrics_cache:
            return self.metrics_cache[image_path]
        
        img = Image.open(image_path)
        arr = np.array(img)
        
        # Convert to RGB if necessary
        if len(arr.shape) == 2:
            arr = np.stack([arr, arr, arr], axis=2)
        elif arr.shape[2] == 4:
            arr = arr[:, :, :3]
        
        # Basic info
        file_size = os.path.getsize(image_path)
        height, width = arr.shape[:2]
        
        # Compositional hash (perceptual)
        comp_hash = self._compute_perceptual_hash(arr)
        
        # Brightness stats
        gray = np.mean(arr, axis=2)
        brightness_mean = np.mean(gray)
        brightness_std = np.std(gray)
        
        # Edge density using gradient
        grad_x = np.abs(np.gradient(gray, axis=1))
        grad_y = np.abs(np.gradient(gray, axis=0))
        edges = np.sqrt(grad_x**2 + grad_y**2)
        edge_density = np.mean(edges) / 255.0
        
        # Dominant colors (k-means-like clustering)
        dominant_colors = self._extract_dominant_colors(arr, k=5)
        
        # Color entropy
        color_entropy = self._compute_color_entropy(arr)
        
        # Center of mass
        total_mass = np.sum(gray)
        if total_mass > 0:
            y_indices, x_indices = np.indices(gray.shape)
            center_y = np.sum(y_indices * gray) / total_mass
            center_x = np.sum(x_indices * gray) / total_mass
            center_of_mass = (center_x / width, center_y / height)
        else:
            center_of_mass = (0.5, 0.5)
        
        # Symmetry score
        symmetry_score = self._compute_symmetry(arr)
        
        metrics = DiversityMetrics(
            file_path=image_path,
            file_size=file_size,
            dimensions=(width, height),
            compositional_hash=comp_hash,
            brightness_mean=float(brightness_mean),
            brightness_std=float(brightness_std),
            edge_density=float(edge_density),
            dominant_colors=dominant_colors,
            color_entropy=float(color_entropy),
            center_of_mass=center_of_mass,
            symmetry_score=float(symmetry_score)
        )
        
        self.metrics_cache[image_path] = metrics
        return metrics
    
    def _compute_perceptual_hash(self, arr: np.ndarray) -> str:
        """Compute perceptual hash of image structure"""
        # Convert to grayscale and resize to 8x8
        gray = np.mean(arr, axis=2)
        img = Image.fromarray(gray.astype(np.uint8))
        img_small = img.resize((8, 8), Image.Resampling.LANCZOS)
        pixels = list(img_small.getdata())
        
        # Compute average
        avg = sum(pixels) / len(pixels)
        
        # Create hash based on whether each pixel is above/below average
        bits = ''.join('1' if p > avg else '0' for p in pixels)
        return hex(int(bits, 2))[2:].zfill(16)
    
    def _extract_dominant_colors(
        self, 
        arr: np.ndarray, 
        k: int = 5
    ) -> List[Tuple[int, int, int]]:
        """Extract k dominant colors using simple binning"""
        # Reshape to list of RGB pixels
        pixels = arr.reshape(-1, 3)
        
        # Quantize to 8-color bins
        quantized = (pixels // 32) * 32 + 16
        
        # Count occurrences
        color_counts = defaultdict(int)
        for pixel in quantized:
            color_counts[tuple(pixel)] += 1
        
        # Get top k colors
        top_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)[:k]
        return [color for color, _ in top_colors]
    
    def _compute_color_entropy(self, arr: np.ndarray) -> float:
        """Compute Shannon entropy of color distribution"""
        # Quantize colors
        quantized = (arr // 16).astype(np.uint8)
        
        # Count unique colors
        pixels = quantized.reshape(-1, 3)
        unique_colors = len(np.unique(pixels, axis=0))
        
        # Normalize by total possible colors (16^3 bins)
        max_colors = 16**3
        entropy = unique_colors / max_colors
        
        return entropy * 100  # Scale to 0-100
    
    def _compute_symmetry(self, arr: np.ndarray) -> float:
        """Compute horizontal symmetry score"""
        gray = np.mean(arr, axis=2)
        height, width = gray.shape
        
        # Compare left and right halves
        left = gray[:, :width//2]
        right = np.fliplr(gray[:, width//2:width//2*2])
        
        # Compute similarity
        min_h = min(left.shape[1], right.shape[1])
        diff = np.abs(left[:, :min_h] - right[:, :min_h])
        similarity = 1.0 - (np.mean(diff) / 255.0)
        
        return similarity
    
    def compute_diversity_score(
        self, 
        metrics1: DiversityMetrics, 
        metrics2: DiversityMetrics
    ) -> float:
        """
        Compute diversity score between two fractals.
        Returns 0-1 where 1 is completely different, 0 is identical.
        """
        scores = []
        
        # Compositional difference (perceptual hash)
        hash_diff = self._hash_distance(
            metrics1.compositional_hash, 
            metrics2.compositional_hash
        )
        scores.append(hash_diff * 0.35)  # 35% weight
        
        # Brightness difference
        brightness_diff = abs(metrics1.brightness_mean - metrics2.brightness_mean) / 255.0
        scores.append(min(brightness_diff * 2, 1.0) * 0.15)  # 15% weight
        
        # Edge density difference
        edge_diff = abs(metrics1.edge_density - metrics2.edge_density)
        scores.append(edge_diff * 0.20)  # 20% weight
        
        # Color difference (compare dominant colors)
        color_diff = self._color_distance(
            metrics1.dominant_colors,
            metrics2.dominant_colors
        )
        scores.append(color_diff * 0.20)  # 20% weight
        
        # Center of mass difference (composition shift)
        center_diff = np.sqrt(
            (metrics1.center_of_mass[0] - metrics2.center_of_mass[0])**2 +
            (metrics1.center_of_mass[1] - metrics2.center_of_mass[1])**2
        )
        scores.append(min(center_diff * 2, 1.0) * 0.10)  # 10% weight
        
        return sum(scores)
    
    def _hash_distance(self, hash1: str, hash2: str) -> float:
        """Compute distance between two perceptual hashes"""
        if len(hash1) != len(hash2):
            return 1.0
        
        # Convert hex to binary
        bin1 = bin(int(hash1, 16))[2:].zfill(64)
        bin2 = bin(int(hash2, 16))[2:].zfill(64)
        
        # Hamming distance
        diff_bits = sum(c1 != c2 for c1, c2 in zip(bin1, bin2))
        return diff_bits / 64.0
    
    def _color_distance(
        self, 
        colors1: List[Tuple[int, int, int]], 
        colors2: List[Tuple[int, int, int]]
    ) -> float:
        """Compute distance between two color palettes"""
        if not colors1 or not colors2:
            return 1.0
        
        # Compare each color in palette1 to closest in palette2
        total_dist = 0.0
        for c1 in colors1:
            min_dist = min(
                np.sqrt(sum((a - b)**2 for a, b in zip(c1, c2))) / 441.67  # Max distance in RGB
                for c2 in colors2
            )
            total_dist += min_dist
        
        return total_dist / len(colors1)
    
    def verify_batch_diversity(
        self,
        image_paths: List[str],
        min_diversity_threshold: float = 0.25,
        max_similar_pairs_ratio: float = 0.3
    ) -> Dict[str, Any]:
        """
        Verify that a batch of renders has sufficient diversity.
        
        Args:
            image_paths: List of image file paths
            min_diversity_threshold: Minimum pairwise diversity score (0-1)
            max_similar_pairs_ratio: Maximum fraction of pairs that can be too similar
        
        Returns:
            Dictionary with verification results
        """
        if len(image_paths) < 2:
            return {
                'passed': True,
                'message': 'Only one image, cannot compute diversity',
                'diversity_matrix': None,
                'statistics': {}
            }
        
        # Analyze all images
        print("  Analyzing images for diversity...")
        metrics_list = [self.analyze_image(p) for p in image_paths]
        
        # Compute pairwise diversity matrix
        n = len(image_paths)
        diversity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                diversity = self.compute_diversity_score(metrics_list[i], metrics_list[j])
                diversity_matrix[i, j] = diversity_matrix[j, i] = diversity
        
        # Find too-similar pairs
        too_similar_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if diversity_matrix[i, j] < min_diversity_threshold:
                    too_similar_pairs.append((
                        i, j, 
                        Path(image_paths[i]).name,
                        Path(image_paths[j]).name,
                        diversity_matrix[i, j]
                    ))
        
        # Compute statistics
        total_pairs = n * (n - 1) / 2
        similarity_ratio = len(too_similar_pairs) / total_pairs if total_pairs > 0 else 0
        
        diversities = diversity_matrix[np.triu_indices(n, k=1)]
        
        statistics = {
            'num_images': n,
            'mean_diversity': float(np.mean(diversities)),
            'min_diversity': float(np.min(diversities)),
            'max_diversity': float(np.max(diversities)),
            'std_diversity': float(np.std(diversities)),
            'num_similar_pairs': len(too_similar_pairs),
            'similarity_ratio': similarity_ratio,
            'threshold': min_diversity_threshold
        }
        
        # Generate recommendations
        recommendations = []
        
        if similarity_ratio > max_similar_pairs_ratio:
            recommendations.append(
                f"⚠️  Too many similar pairs ({similarity_ratio*100:.1f}% > {max_similar_pairs_ratio*100:.1f}%). "
                "Increase mutation rate or randomize camera positions more."
            )
        
        if statistics['mean_diversity'] < min_diversity_threshold * 1.5:
            recommendations.append(
                f"⚠️  Low overall diversity ({statistics['mean_diversity']:.3f}). "
                "Consider increasing population size or enabling novelty search."
            )
        
        if statistics['min_diversity'] < 0.1:
            recommendations.append(
                f"⚠️  Some images are nearly identical (min diversity: {statistics['min_diversity']:.3f}). "
                "Check for parameter convergence or stuck evolution."
            )
        
        # Check for common issues
        brightnesses = [m.brightness_mean for m in metrics_list]
        if np.std(brightnesses) < 10:
            recommendations.append(
                "⚠️  All images have similar brightness. Vary lighting parameters more."
            )
        
        centers = [m.center_of_mass for m in metrics_list]
        center_std = np.std([c[0] for c in centers]) + np.std([c[1] for c in centers])
        if center_std < 0.1:
            recommendations.append(
                "⚠️  All fractals are centered similarly. Vary camera target positions."
            )
        
        passed = similarity_ratio <= max_similar_pairs_ratio and len(recommendations) == 0
        
        return {
            'passed': passed,
            'image_paths': image_paths,
            'diversity_matrix': diversity_matrix,
            'too_similar_pairs': too_similar_pairs,
            'statistics': statistics,
            'recommendations': recommendations,
            'metrics': [
                {
                    'path': m.file_path,
                    'hash': m.compositional_hash[:8],
                    'brightness': round(m.brightness_mean, 1),
                    'edge_density': round(m.edge_density, 3),
                    'center': (round(m.center_of_mass[0], 2), round(m.center_of_mass[1], 2))
                }
                for m in metrics_list
            ]
        }
    
    def generate_report(self, result: Dict[str, Any]) -> str:
        """Generate human-readable verification report"""
        lines = [
            "=" * 60,
            "DIVERSITY VERIFICATION REPORT",
            "=" * 60,
            "",
            f"Status: {'✅ PASSED' if result['passed'] else '❌ FAILED'}",
            f"Images analyzed: {result['statistics']['num_images']}",
            "",
            "Statistics:",
            f"  Mean diversity: {result['statistics']['mean_diversity']:.3f}",
            f"  Min diversity:  {result['statistics']['min_diversity']:.3f}",
            f"  Max diversity:  {result['statistics']['max_diversity']:.3f}",
            f"  Similar pairs:  {result['statistics']['num_similar_pairs']}",
            "",
        ]
        
        if result['recommendations']:
            lines.append("Recommendations:")
            for rec in result['recommendations']:
                lines.append(f"  {rec}")
            lines.append("")
        
        if result['too_similar_pairs']:
            lines.append("Too-similar pairs:")
            for i, j, name1, name2, dist in result['too_similar_pairs']:
                lines.append(f"  {name1} ↔ {name2}: {dist:.3f}")
            lines.append("")
        
        lines.append("Image details:")
        for m in result['metrics']:
            lines.append(f"  {Path(m['path']).name}")
            lines.append(f"    Hash: {m['hash']} | Brightness: {m['brightness']:.0f} | Center: {m['center']}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)


def test_diversity_engine():
    """Test the diversity engine with sample images"""
    print("Testing Diversity Engine")
    print("=" * 60)
    
    # Create test images if needed
    test_dir = Path("test_diversity")
    test_dir.mkdir(exist_ok=True)
    
    # Check for existing test images
    existing_images = list(test_dir.glob("*.png"))
    
    if len(existing_images) < 4:
        print("Generating test fractals...")
        from renderers.unified import create_renderer, RendererType
        
        renderer = create_renderer(RendererType.PYTHON_3D)
        
        for i in range(5):
            params = renderer.generate_random_parameters()
            output_path = test_dir / f"test_{i}.png"
            renderer.render_fractal(params, str(output_path), 200, 200)
            print(f"  Generated: {output_path.name}")
        
        existing_images = list(test_dir.glob("*.png"))
    
    # Test diversity verification
    engine = DiversityEngine()
    result = engine.verify_batch_diversity(
        [str(p) for p in existing_images],
        min_diversity_threshold=0.25,
        max_similar_pairs_ratio=0.3
    )
    
    print("\n" + engine.generate_report(result))
    
    return result


if __name__ == "__main__":
    test_diversity_engine()
