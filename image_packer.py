#!/usr/bin/env python3
"""
Image Packer - Optimally pack multiple images into a single canvas.

This script takes all images from a folder and arranges them efficiently
into a canvas of specified dimensions, maximizing image sizes while
minimizing whitespace.
"""

import os
import argparse
from typing import List, Tuple, Optional
from dataclasses import dataclass
from PIL import Image
import math
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import random


@dataclass
class ImageInfo:
    """Information about an image to be packed."""
    path: str
    original_width: int
    original_height: int
    aspect_ratio: float
    image: Optional[Image.Image] = None


@dataclass
class PackedImage:
    """Information about a packed image with its position and size."""
    info: ImageInfo
    x: int
    y: int
    width: int
    height: int


@dataclass
class Rectangle:
    """A rectangle representing available space."""
    x: int
    y: int
    width: int
    height: int


# Maximum images per batch to prevent memory crashes
# Lower limit is critical for area uniformity optimization which uses O(n^2) constraints
MAX_IMAGES_PER_BATCH = 50


class ImagePacker:
    """Packs images into a canvas using a guillotine-based algorithm."""

    def __init__(self, canvas_width: int, canvas_height: int,
                 respect_original_size: bool = False,
                 max_size_variation: float = 15.0,
                 overlap_percent: float = 10.0,
                 no_uniformity: bool = False,
                 randomize: bool = False):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.respect_original_size = respect_original_size
        self.max_size_variation = max_size_variation / 100.0  # Convert to decimal
        self.overlap_percent = overlap_percent / 100.0  # Convert to decimal
        self.no_uniformity = no_uniformity
        self.randomize = randomize
        self.free_rectangles: List[Rectangle] = [Rectangle(0, 0, canvas_width, canvas_height)]
        self.packed_images: List[PackedImage] = []

    def load_images(self, folder_path: str) -> List[ImageInfo]:
        """Load all images from a folder (excludes GIFs)."""
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        images = []

        if not os.path.exists(folder_path):
            raise ValueError(f"Folder not found: {folder_path}")

        # Get list of files first to show progress
        all_files = sorted(os.listdir(folder_path))
        image_files = [f for f in all_files if os.path.isfile(os.path.join(folder_path, f)) and
                      os.path.splitext(f)[1].lower() in supported_formats]

        for filename in tqdm(image_files, desc="Loading images", unit="img"):
            file_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(file_path)
                width, height = img.size
                images.append(ImageInfo(
                    path=file_path,
                    original_width=width,
                    original_height=height,
                    aspect_ratio=width / height,
                    image=img
                ))
            except Exception as e:
                tqdm.write(f"Warning: Could not load {filename}: {e}")

        return images

    def find_best_fit(self, width: int, height: int) -> Optional[Tuple[int, Rectangle]]:
        """Find the best rectangle to fit the given dimensions."""
        best_index = None
        best_rect = None
        best_score = float('inf')

        for i, rect in enumerate(self.free_rectangles):
            if rect.width >= width and rect.height >= height:
                # Calculate waste (remaining space)
                waste = (rect.width * rect.height) - (width * height)
                if waste < best_score:
                    best_score = waste
                    best_index = i
                    best_rect = rect

        if best_index is not None:
            return best_index, best_rect
        return None

    def split_rectangle(self, rect: Rectangle, used_width: int, used_height: int) -> List[Rectangle]:
        """Split a rectangle after placing an image, creating new free rectangles."""
        new_rects = []

        # Right remainder
        if rect.width > used_width:
            new_rects.append(Rectangle(
                rect.x + used_width,
                rect.y,
                rect.width - used_width,
                rect.height
            ))

        # Bottom remainder
        if rect.height > used_height:
            new_rects.append(Rectangle(
                rect.x,
                rect.y + used_height,
                used_width,
                rect.height - used_height
            ))

        return new_rects

    def remove_redundant_rectangles(self):
        """Remove rectangles that are contained within other rectangles."""
        i = 0
        while i < len(self.free_rectangles):
            j = i + 1
            while j < len(self.free_rectangles):
                rect_i = self.free_rectangles[i]
                rect_j = self.free_rectangles[j]

                # Check if rect_j is inside rect_i
                if (rect_j.x >= rect_i.x and rect_j.y >= rect_i.y and
                    rect_j.x + rect_j.width <= rect_i.x + rect_i.width and
                    rect_j.y + rect_j.height <= rect_i.y + rect_i.height):
                    self.free_rectangles.pop(j)
                # Check if rect_i is inside rect_j
                elif (rect_i.x >= rect_j.x and rect_i.y >= rect_j.y and
                      rect_i.x + rect_i.width <= rect_j.x + rect_j.width and
                      rect_i.y + rect_i.height <= rect_j.y + rect_j.height):
                    self.free_rectangles.pop(i)
                    i -= 1
                    break
                else:
                    j += 1
            i += 1

    def calculate_scale_factor(self, images: List[ImageInfo]) -> float:
        """Calculate initial scale factor estimate for binary search."""
        total_area = sum(img.original_width * img.original_height for img in images)
        canvas_area = self.canvas_width * self.canvas_height

        # Start with an optimistic scale - binary search will find the actual maximum
        # For equal sizing mode, we target full canvas utilization
        if not self.respect_original_size:
            # Each image will be roughly canvas_area / len(images)
            # So we need minimal scaling - use 1.0 as estimate
            estimated_scale = 1.0
        else:
            # For proportional sizing, estimate based on total area
            estimated_scale = math.sqrt(canvas_area / total_area)

        return estimated_scale

    def try_pack_with_target_areas(self, images: List[ImageInfo], area_scale: float = 1.0) -> bool:
        """
        Pack images where each has a target area of (canvas_area / n) * area_scale.

        Args:
            area_scale: Multiplier for target area (1.0 = perfect equal areas, <1.0 = smaller for safety)
        """
        self.free_rectangles = [Rectangle(0, 0, self.canvas_width, self.canvas_height)]
        self.packed_images = []

        # Sort images by area (largest first) for better packing, unless randomize is enabled
        if self.randomize:
            sorted_images = images.copy()
            random.shuffle(sorted_images)
        else:
            sorted_images = sorted(images,
                                 key=lambda img: img.original_width * img.original_height,
                                 reverse=True)

        # Calculate target area per image
        canvas_area = self.canvas_width * self.canvas_height
        target_area_per_image = (canvas_area / len(images)) * area_scale

        for img_info in sorted_images:
            if self.respect_original_size:
                # Use proportional sizing based on original areas
                original_area = img_info.original_width * img_info.original_height
                scale = math.sqrt(target_area_per_image / original_area)
                scaled_width = int(img_info.original_width * scale)
                scaled_height = int(img_info.original_height * scale)
                # Do not exceed original dimensions
                scaled_width = min(scaled_width, img_info.original_width)
                scaled_height = min(scaled_height, img_info.original_height)
            else:
                # Size each image to exactly target_area while respecting aspect ratio
                original_area = img_info.original_width * img_info.original_height
                aspect_ratio = img_info.aspect_ratio

                # Calculate dimensions for target area with this aspect ratio
                # area = width * height, and width = height * aspect_ratio
                # So: area = height^2 * aspect_ratio
                # Therefore: height = sqrt(area / aspect_ratio)
                height = math.sqrt(target_area_per_image / aspect_ratio)
                width = height * aspect_ratio

                scaled_width = int(width)
                scaled_height = int(height)

            # Ensure minimum size
            scaled_width = max(1, scaled_width)
            scaled_height = max(1, scaled_height)

            # Find a place to fit this image
            fit = self.find_best_fit(scaled_width, scaled_height)

            if fit is None:
                return False

            rect_index, rect = fit

            # Place the image
            self.packed_images.append(PackedImage(
                info=img_info,
                x=rect.x,
                y=rect.y,
                width=scaled_width,
                height=scaled_height
            ))

            # Remove the used rectangle
            self.free_rectangles.pop(rect_index)

            # Add new rectangles from splitting
            new_rects = self.split_rectangle(rect, scaled_width, scaled_height)
            self.free_rectangles.extend(new_rects)

            # Clean up redundant rectangles
            self.remove_redundant_rectangles()

        return True

    def _optimize_image_size_for_space(self, img_info: ImageInfo, base_width: int,
                                        base_height: int, target_area: float) -> Tuple[int, int]:
        """
        Optimize image size to fill available space better during initial packing.
        Tries multiple size variations to find the best fit.
        """
        aspect_ratio = img_info.aspect_ratio
        best_width, best_height = base_width, base_height
        best_fit_score = float('inf')

        # Try different size multipliers to fill space better
        # Use max_size_variation to determine how much we can vary
        variation_factors = [1.0]  # Base size

        # Add more variation levels based on max_size_variation
        if self.max_size_variation > 0:
            # Try sizes from base to max variation in steps
            for i in range(1, 6):  # Try 5 different sizes above base
                factor = 1.0 + (self.max_size_variation * i / 5)
                variation_factors.append(factor)

        for factor in variation_factors:
            # Calculate new dimensions while maintaining aspect ratio
            test_area = target_area * factor
            test_dimension = math.sqrt(test_area)

            if aspect_ratio >= 1:
                test_height = int(test_dimension / math.sqrt(aspect_ratio))
                test_width = int(test_height * aspect_ratio)
            else:
                test_width = int(test_dimension * math.sqrt(aspect_ratio))
                test_height = int(test_width / aspect_ratio)

            # Ensure minimum size
            test_width = max(1, test_width)
            test_height = max(1, test_height)

            # Check if this size can fit in any available rectangle
            fit = self.find_best_fit(test_width, test_height)
            if fit is not None:
                _, rect = fit
                # Calculate waste for this fit
                waste = (rect.width * rect.height) - (test_width * test_height)
                # Prefer larger images with less waste
                # Score combines size (negative, larger is better) and waste
                fit_score = waste - (test_width * test_height * 0.5)

                if fit_score < best_fit_score:
                    best_fit_score = fit_score
                    best_width, best_height = test_width, test_height

        return best_width, best_height

    def grow_images_to_fill_space(self):
        """
        After initial packing, aggressively grow images to fill remaining whitespace.
        Allows size variations up to max_size_variation and overlaps up to overlap_percent.
        """
        if not self.packed_images:
            return

        # Calculate canvas area for absolute size limits
        canvas_area = self.canvas_width * self.canvas_height

        # Calculate base area - this is what we consider "normal" size
        # Use the median area rather than average to avoid skew from outliers
        areas = sorted([p.width * p.height for p in self.packed_images])
        median_area = areas[len(areas) // 2]

        # Maximum allowed area is based on canvas area and number of images
        # With high variation, some images can be much larger
        # When overlap is very low, be MUCH more aggressive since we won't be able
        # to enforce uniformity later - prioritize coverage over uniformity
        if self.overlap_percent < 0.05:
            # No uniformity enforcement later, so allow images to grow much larger
            max_allowed_area = (canvas_area / len(self.packed_images)) * 5.0  # Very aggressive
            num_passes = 10  # More passes to fill space
        else:
            # Will enforce uniformity later, so moderate growth
            max_allowed_area = (canvas_area / len(self.packed_images)) * (1 + self.max_size_variation)
            num_passes = 5

        # Multiple aggressive growth passes - more passes for better filling
        with tqdm(total=num_passes, desc="Growing images to fill space", unit="pass", leave=False) as pbar:
            for growth_pass in range(num_passes):
                # Track if any changes were made
                any_changes = False

                # Sort by current size (smallest first) to help smaller images catch up
                sorted_indices = sorted(range(len(self.packed_images)),
                                      key=lambda i: self.packed_images[i].width * self.packed_images[i].height)

                for i in sorted_indices:
                    packed = self.packed_images[i]
                    current_area = packed.width * packed.height
                    initial_size = (packed.width, packed.height)

                    # Allow continued growth, but with diminishing returns
                    # Smaller images can grow more aggressively
                    size_ratio = current_area / median_area if median_area > 0 else 1.0
                    adjusted_max_area = max_allowed_area

                    # If already large, reduce max allowed area
                    if size_ratio > 1.0:
                        adjusted_max_area = current_area * (1 + (self.max_size_variation * 0.5))

                    aspect_ratio = packed.info.aspect_ratio

                    # AGGRESSIVE STRATEGY 1: Maximum width growth with overlap allowed
                    max_width = self.canvas_width - packed.x
                    new_width = max_width
                    new_height = int(new_width / aspect_ratio)

                    # Check if within canvas and size variation limits
                    if new_height <= self.canvas_height - packed.y:
                        new_area = new_width * new_height
                        if new_area <= adjusted_max_area:
                            if self._check_space_available_with_overlap(packed.x, packed.y, new_width, new_height, i):
                                packed.width = new_width
                                packed.height = new_height
                                any_changes = True
                                continue

                    # AGGRESSIVE STRATEGY 2: Maximum height growth with overlap allowed
                    max_height = self.canvas_height - packed.y
                    new_height = max_height
                    new_width = int(new_height * aspect_ratio)

                    if new_width <= self.canvas_width - packed.x:
                        new_area = new_width * new_height
                        if new_area <= adjusted_max_area:
                            if self._check_space_available_with_overlap(packed.x, packed.y, new_width, new_height, i):
                                packed.width = new_width
                                packed.height = new_height
                                any_changes = True
                                continue

                    # AGGRESSIVE STRATEGY 3: Grow to fill available space
                    available_width = self.canvas_width - packed.x
                    available_height = self.canvas_height - packed.y
                    available_area = available_width * available_height

                    target_area = min(adjusted_max_area, available_area * 0.9)  # Use 90% of available
                    target_dimension = math.sqrt(target_area)

                    # Calculate dimensions based on aspect ratio
                    if aspect_ratio >= 1:
                        new_height = int(target_dimension / math.sqrt(aspect_ratio))
                        new_width = int(new_height * aspect_ratio)
                    else:
                        new_width = int(target_dimension * math.sqrt(aspect_ratio))
                        new_height = int(new_width / aspect_ratio)

                    # Make sure we fit in canvas
                    new_width = min(new_width, self.canvas_width - packed.x)
                    new_height = min(new_height, self.canvas_height - packed.y)

                    if new_width > packed.width and new_height > packed.height:
                        if self._check_space_available_with_overlap(packed.x, packed.y, new_width, new_height, i):
                            packed.width = new_width
                            packed.height = new_height
                            any_changes = True
                            continue

                    # AGGRESSIVE STRATEGY 4: Incremental growth with overlap
                    # Try to grow by 30% each iteration
                    growth_factor = 1.3
                    new_width = int(packed.width * growth_factor)
                    new_height = int(packed.height * growth_factor)

                    # Constrain to canvas
                    new_width = min(new_width, self.canvas_width - packed.x)
                    new_height = min(new_height, self.canvas_height - packed.y)

                    new_area = new_width * new_height
                    if new_area <= adjusted_max_area and (new_width > packed.width or new_height > packed.height):
                        if self._check_space_available_with_overlap(packed.x, packed.y, new_width, new_height, i):
                            packed.width = new_width
                            packed.height = new_height
                            any_changes = True

                # If no changes were made in this pass, we're done
                if not any_changes:
                    break

                pbar.update(1)

    def enforce_area_uniformity(self, max_area_variation: float = 0.05):
        """
        Enforce area uniformity using iterative constrained optimization.

        This is a HARD constraint that iterates until all images have areas within
        max_area_variation of the average area.

        Balances multiple objectives:
        - Minimize area deviation from average (uniformity)
        - Maximize total coverage

        Subject to HARD constraints:
        - No overlaps > overlap_percent
        - Stay within canvas bounds
        - Area within variation limits (ENFORCED)
        - Preserve aspect ratios

        Args:
            max_area_variation: Maximum allowed variation in area from average (default 0.05 = 5%)
        """
        if not self.packed_images:
            return

        n = len(self.packed_images)

        # Memory safety: skip uniformity for very large batches (O(n^2) constraints)
        if n > MAX_IMAGES_PER_BATCH:
            tqdm.write(f"⚠ Skipping area uniformity for {n} images (exceeds limit of {MAX_IMAGES_PER_BATCH})")
            tqdm.write(f"   This batch has too many images for memory-safe optimization")
            tqdm.write(f"   Use --no-uniformity flag or reduce images per canvas")
            return

        max_iterations = 10  # Reduced from 50 for performance

        # For large numbers of images, optimization can be slow
        if n > 10:
            tqdm.write(f"Enforcing area uniformity for {n} images (this may take a moment)...")
            tqdm.write(f"Tip: Use --no-uniformity to skip this step for faster processing")

        # Store original sizes for reference
        original_widths = np.array([p.width for p in self.packed_images])
        original_heights = np.array([p.height for p in self.packed_images])
        aspect_ratios = np.array([p.info.aspect_ratio for p in self.packed_images])
        positions_x = np.array([p.x for p in self.packed_images])
        positions_y = np.array([p.y for p in self.packed_images])

        # Iteratively enforce area uniformity
        with tqdm(total=max_iterations, desc="Enforcing area uniformity", unit="iter") as pbar:
            for iteration in range(max_iterations):
                # Calculate current average area
                current_areas = np.array([p.width * p.height for p in self.packed_images])
                avg_area = np.mean(current_areas)
                min_area = max(1, avg_area * (1 - max_area_variation))
                max_area = avg_area * (1 + max_area_variation)

                # Check if all images meet the area constraint
                areas_in_range = np.all((current_areas >= min_area) & (current_areas <= max_area))

                # Update progress bar with current deviation
                max_deviation = np.max(np.abs(current_areas - avg_area) / avg_area) * 100
                pbar.set_postfix({'max_dev': f'{max_deviation:.2f}%'})
                pbar.update(1)

                if areas_in_range:
                    # Success! All images meet the area constraint
                    pbar.set_description(f"✓ Area uniformity achieved")
                    break

                # Early termination: if deviation is acceptable (within 2x the target), good enough
                if max_deviation <= max_area_variation * 200:  # 2x tolerance
                    pbar.set_description(f"✓ Area uniformity acceptable")
                    break

                # Update reference sizes for this iteration
                current_widths = np.array([p.width for p in self.packed_images])
                current_heights = np.array([p.height for p in self.packed_images])

                # Initial guess: scale factors to bring areas closer to average
                x0 = np.sqrt(avg_area / current_areas)

                def get_dimensions(scales):
                    """Calculate widths and heights from scale factors."""
                    widths = current_widths * scales
                    heights = current_heights * scales
                    return widths, heights

                def objective(scales):
                    """Minimize deviation from uniform area."""
                    widths, heights = get_dimensions(scales)
                    areas = widths * heights
                    # Strongly penalize deviation from average area
                    uniformity_penalty = np.sum((areas - avg_area) ** 2)
                    return uniformity_penalty

                def overlap_constraint(scales, i, j):
                    """Constraint: overlap between images i and j must be <= overlap_percent."""
                    widths, heights = get_dimensions(scales)

                    # Calculate overlap
                    x1, y1, w1, h1 = positions_x[i], positions_y[i], widths[i], heights[i]
                    x2, y2, w2, h2 = positions_x[j], positions_y[j], widths[j], heights[j]

                    overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                    overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

                    if overlap_x > 0 and overlap_y > 0:
                        overlap_area = overlap_x * overlap_y
                        area1 = w1 * h1
                        area2 = w2 * h2

                        # Constraint: overlap_area / area1 <= overlap_percent (return non-negative when satisfied)
                        max_overlap_1 = area1 * self.overlap_percent - overlap_area
                        max_overlap_2 = area2 * self.overlap_percent - overlap_area

                        return min(max_overlap_1, max_overlap_2)

                    return 0.0  # No overlap is always OK

                def canvas_constraint_width(scales, i):
                    """Constraint: image i must fit within canvas width."""
                    widths, _ = get_dimensions(scales)
                    return self.canvas_width - (positions_x[i] + widths[i])

                def canvas_constraint_height(scales, i):
                    """Constraint: image i must fit within canvas height."""
                    _, heights = get_dimensions(scales)
                    return self.canvas_height - (positions_y[i] + heights[i])

                def area_constraint_min(scales, i):
                    """Constraint: image i area must be >= min_area."""
                    widths, heights = get_dimensions(scales)
                    area = widths[i] * heights[i]
                    return area - min_area

                def area_constraint_max(scales, i):
                    """Constraint: image i area must be <= max_area."""
                    widths, heights = get_dimensions(scales)
                    area = widths[i] * heights[i]
                    return max_area - area

                # Build constraints list
                constraints = []

                # Overlap constraints for all pairs
                for i in range(n):
                    for j in range(i + 1, n):
                        constraints.append({
                            'type': 'ineq',
                            'fun': overlap_constraint,
                            'args': (i, j)
                        })

                # Canvas bounds and area constraints
                for i in range(n):
                    constraints.append({'type': 'ineq', 'fun': canvas_constraint_width, 'args': (i,)})
                    constraints.append({'type': 'ineq', 'fun': canvas_constraint_height, 'args': (i,)})
                    constraints.append({'type': 'ineq', 'fun': area_constraint_min, 'args': (i,)})
                    constraints.append({'type': 'ineq', 'fun': area_constraint_max, 'args': (i,)})

                # Bounds: scales must be positive, allow wider range for adjustment
                bounds = [(0.1, 3.0) for _ in range(n)]

                # Solve optimization problem - reduced iterations for speed
                result = minimize(
                    objective,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 100, 'ftol': 1e-4}  # Much faster: 100 iter, looser tolerance
                )

                # Apply optimized scales
                if result.success:
                    optimal_scales = result.x
                    for i, packed in enumerate(self.packed_images):
                        new_width = int(current_widths[i] * optimal_scales[i])
                        new_height = int(current_heights[i] * optimal_scales[i])
                        packed.width = max(1, new_width)
                        packed.height = max(1, new_height)
                else:
                    # If optimization fails, try aggressive shrinking to meet constraints
                    pbar.write(f"Optimization failed at iteration {iteration}, trying fallback strategy")
                    for i, packed in enumerate(self.packed_images):
                        current_area = current_areas[i]

                        # If too large, shrink to max_area
                        if current_area > max_area:
                            scale = math.sqrt(max_area / current_area) * 0.95  # 95% to ensure we're under
                            new_width = int(current_widths[i] * scale)
                            new_height = int(current_heights[i] * scale)

                            if (new_width <= self.canvas_width - packed.x and
                                new_height <= self.canvas_height - packed.y):
                                if self._check_space_available_with_overlap(packed.x, packed.y, new_width, new_height, i):
                                    packed.width = max(1, new_width)
                                    packed.height = max(1, new_height)

                        # If too small, try to grow to min_area
                        elif current_area < min_area:
                            scale = math.sqrt(min_area / current_area) * 1.05  # 105% to ensure we're over
                            new_width = int(current_widths[i] * scale)
                            new_height = int(current_heights[i] * scale)

                            if (new_width <= self.canvas_width - packed.x and
                                new_height <= self.canvas_height - packed.y):
                                if self._check_space_available_with_overlap(packed.x, packed.y, new_width, new_height, i):
                                    packed.width = max(1, new_width)
                                    packed.height = max(1, new_height)

        # Final check
        final_areas = np.array([p.width * p.height for p in self.packed_images])
        final_avg_area = np.mean(final_areas)
        final_min_area = final_avg_area * (1 - max_area_variation)
        final_max_area = final_avg_area * (1 + max_area_variation)

        if not np.all((final_areas >= final_min_area) & (final_areas <= final_max_area)):
            max_deviation = np.max(np.abs(final_areas - final_avg_area) / final_avg_area)
            # Only warn if deviation is significantly over target
            if max_deviation > max_area_variation * 1.5:
                tqdm.write(f"Note: Area deviation {max_deviation * 100:.1f}% exceeds target {max_area_variation * 100:.0f}%")
                tqdm.write(f"Tip: Use --no-uniformity for faster processing with higher coverage")

    def _check_space_available_with_overlap(self, x: int, y: int, width: int, height: int, exclude_index: int) -> bool:
        """
        Check if a rectangle at (x, y) with given dimensions is valid.
        Ensures that no more than overlap_percent of EACH image's own area overlaps with any other.
        """
        # Check canvas boundaries
        if x + width > self.canvas_width or y + height > self.canvas_height:
            return False

        # Check overlap with other images - ensure no more than overlap_percent of each image's own area overlaps
        for i, other in enumerate(self.packed_images):
            if i == exclude_index:
                continue

            # Calculate overlap area
            overlap_x = max(0, min(x + width, other.x + other.width) - max(x, other.x))
            overlap_y = max(0, min(y + height, other.y + other.height) - max(y, other.y))

            if overlap_x > 0 and overlap_y > 0:
                overlap_area = overlap_x * overlap_y
                this_area = width * height
                other_area = other.width * other.height

                # Check if overlap exceeds allowed percentage of THIS image's area
                if overlap_area >= this_area * self.overlap_percent:
                    return False

                # Also check if overlap exceeds allowed percentage of OTHER image's area
                if overlap_area >= other_area * self.overlap_percent:
                    return False

        return True

    def _check_space_available(self, x: int, y: int, width: int, height: int, exclude_index: int) -> bool:
        """Check if a rectangle at (x, y) with given dimensions overlaps with any packed images (no overlap allowed)."""
        # Check canvas boundaries
        if x + width > self.canvas_width or y + height > self.canvas_height:
            return False

        # Check overlap with other images - no overlap allowed
        for i, other in enumerate(self.packed_images):
            if i == exclude_index:
                continue

            # Check for any overlap
            if not (x + width <= other.x or
                    x >= other.x + other.width or
                    y + height <= other.y or
                    y >= other.y + other.height):
                return False

        return True

    def rebuild_free_rectangles_from_canvas(self):
        """
        Rebuild the free_rectangles list by scanning the actual canvas.

        This is needed because after growth and overlap adjustments, the guillotine
        algorithm's tracked free_rectangles may not reflect the actual free space.
        """
        # Create a simple occupancy grid (much more accurate than tracking splits)
        # Use a resolution that balances accuracy and performance
        grid_resolution = 10  # pixels per grid cell
        grid_width = (self.canvas_width + grid_resolution - 1) // grid_resolution
        grid_height = (self.canvas_height + grid_resolution - 1) // grid_resolution

        # Initialize grid as all free
        occupied = [[False for _ in range(grid_width)] for _ in range(grid_height)]

        # Mark occupied cells
        for packed in self.packed_images:
            x1_cell = packed.x // grid_resolution
            y1_cell = packed.y // grid_resolution
            x2_cell = min((packed.x + packed.width + grid_resolution - 1) // grid_resolution, grid_width)
            y2_cell = min((packed.y + packed.height + grid_resolution - 1) // grid_resolution, grid_height)

            for y in range(y1_cell, y2_cell):
                for x in range(x1_cell, x2_cell):
                    occupied[y][x] = True

        # Find maximal free rectangles using a greedy approach
        new_free_rects = []
        used_cells = [[False for _ in range(grid_width)] for _ in range(grid_height)]

        # Scan for rectangles
        for start_y in range(grid_height):
            for start_x in range(grid_width):
                # Skip if occupied or already used in a rectangle
                if occupied[start_y][start_x] or used_cells[start_y][start_x]:
                    continue

                # Try to expand a rectangle from this point
                # First, find maximum width for this row
                max_width = 0
                for x in range(start_x, grid_width):
                    if occupied[start_y][x] or used_cells[start_y][x]:
                        break
                    max_width += 1

                if max_width == 0:
                    continue

                # Now try to expand downward while maintaining this width
                max_height = 0
                for y in range(start_y, grid_height):
                    # Check if this entire row is free
                    row_free = True
                    for x in range(start_x, start_x + max_width):
                        if occupied[y][x] or used_cells[y][x]:
                            row_free = False
                            break

                    if not row_free:
                        break
                    max_height += 1

                if max_height == 0:
                    continue

                # Found a rectangle! Convert from grid to pixels
                rect_x = start_x * grid_resolution
                rect_y = start_y * grid_resolution
                rect_width = min(max_width * grid_resolution, self.canvas_width - rect_x)
                rect_height = min(max_height * grid_resolution, self.canvas_height - rect_y)

                # Only keep rectangles that are reasonably large (at least 10x10 pixels)
                if rect_width >= 10 and rect_height >= 10:
                    new_free_rects.append(Rectangle(rect_x, rect_y, rect_width, rect_height))

                # Mark these cells as used
                for y in range(start_y, start_y + max_height):
                    for x in range(start_x, start_x + max_width):
                        used_cells[y][x] = True

        # Replace the free_rectangles list
        self.free_rectangles = new_free_rects

        # Log what we found
        if new_free_rects:
            total_free_area = sum(r.width * r.height for r in new_free_rects)
            canvas_area = self.canvas_width * self.canvas_height
            free_percent = (total_free_area / canvas_area) * 100
            tqdm.write(f"Rebuilt free rectangles: found {len(new_free_rects)} gaps covering {free_percent:.1f}% of canvas")
            # Show largest gaps
            sorted_by_area = sorted(new_free_rects, key=lambda r: r.width * r.height, reverse=True)
            for i, rect in enumerate(sorted_by_area[:5]):
                tqdm.write(f"  Gap {i+1}: {rect.width}x{rect.height} at ({rect.x}, {rect.y})")

    def fill_gaps_with_repeats(self, candidate_images: List[ImageInfo],
                               target_coverage: float,
                               used_image_ids: set,
                               used_aspects: set = None,
                               no_repeats_tolerance: float = 0,
                               freezer: set = None,
                               unfreeze_count: list = None) -> int:
        """
        ULTRA-AGGRESSIVE gap filling: Iteratively fill ALL gaps with scaled repeats until target is achieved.

        This method fills EVERY possible gap, no matter how small. Images can be repeated
        multiple times and scaled down as much as needed. NO SIZE LIMITS.

        Args:
            candidate_images: List of images to try (should be sorted by size)
            target_coverage: Target coverage percentage to achieve
            used_image_ids: Set of image IDs - IGNORED for gap filling, allows unlimited repeats
            used_aspects: Set of aspect ratios already used (for no-repeats constraint)
            no_repeats_tolerance: Tolerance for aspect ratio matching
            freezer: SHARED set of frozen image IDs across all collages (universal freezer)
            unfreeze_count: SHARED list with single int element tracking unfreeze cycles

        Returns:
            Number of images added
        """
        total_added = 0
        canvas_area = self.canvas_width * self.canvas_height
        max_iterations = 100  # Increased limit for aggressive filling

        tqdm.write(f"Starting aggressive gap-filling to achieve {target_coverage:.1f}% coverage...")

        # UNIVERSAL FREEZER MECHANISM: Shared across ALL collages
        # Images used as repeats go into freezer, when all frozen, unfreeze all and continue
        # Initialize freezer if not provided (backward compatibility)
        if freezer is None:
            freezer = set()
        if unfreeze_count is None:
            unfreeze_count = [0]

        available_repeats = [img for img in candidate_images]  # Images available as repeats

        # Keep trying to add images until we can't add any more or reach target
        with tqdm(total=max_iterations, desc="Filling gaps with repeats", unit="pass", leave=False) as pbar:
            for iteration in range(max_iterations):
                # Calculate current coverage
                current_coverage = sum(p.width * p.height for p in self.packed_images) / canvas_area * 100

                pbar.set_postfix({'coverage': f'{current_coverage:.1f}%', 'added': total_added, 'frozen': len(freezer)})

                # Stop if we've achieved target coverage
                if current_coverage >= target_coverage:
                    pbar.write(f"✓ Target coverage {target_coverage:.1f}% achieved!")
                    break

                # Track images added in this iteration
                iteration_added = 0

                # Rebuild free rectangles every iteration to catch newly exposed gaps
                # This is critical for finding big gaps after placing images
                self.rebuild_free_rectangles_from_canvas()

                # If no gaps remain, we're done
                if not self.free_rectangles:
                    pbar.write(f"No more gaps to fill")
                    break

                # Check if all images are frozen - if so, unfreeze them all
                available_for_repeats = [img for img in available_repeats if id(img) not in freezer]
                if not available_for_repeats:
                    freezer.clear()
                    unfreeze_count[0] += 1
                    pbar.write(f"🔄 UNIVERSAL UNFREEZE: All {len(available_repeats)} images used across all collages - cycle {unfreeze_count[0]}")
                    available_for_repeats = available_repeats.copy()

                # NEW STRATEGY: Place ONE image per iteration, as LARGE as possible
                # Try multiple positions from largest gaps and choose best
                best_placement = None
                best_coverage = 0

                # Try top 3 largest gaps as potential placement positions
                test_positions = sorted(self.free_rectangles,
                                       key=lambda r: r.width * r.height,
                                       reverse=True)[:3]

                # PASS 1: Try unused images first (diversity)
                for candidate in candidate_images:
                    # Skip if already used
                    if id(candidate) in used_image_ids:
                        continue

                    # Check aspect ratio constraint
                    if used_aspects is not None and no_repeats_tolerance > 0:
                        if aspect_ratio_in_set(candidate.aspect_ratio, used_aspects, no_repeats_tolerance):
                            continue

                    # Try placing at each test position, maximize size
                    for gap in test_positions:
                        aspect = candidate.aspect_ratio

                        # Try to make image as LARGE as possible
                        # Start with largest dimension that fits the gap
                        if gap.width / gap.height > aspect:
                            test_height = gap.height
                            test_width = int(test_height * aspect)
                        else:
                            test_width = gap.width
                            test_height = int(test_width / aspect)

                        # Ensure minimum size
                        if test_width < 1 or test_height < 1:
                            continue

                        # Check if valid (no excessive overlaps)
                        if self._check_space_available_with_overlap(
                            gap.x, gap.y, test_width, test_height, -1
                        ):
                            coverage = test_width * test_height
                            if coverage > best_coverage:
                                best_coverage = coverage
                                best_placement = (candidate, gap.x, gap.y, test_width, test_height, False)

                # PASS 2: If no unused image works, try repeats (from available_for_repeats only)
                if not best_placement:
                    for candidate in available_for_repeats:
                        # Check aspect ratio constraint
                        if used_aspects is not None and no_repeats_tolerance > 0:
                            if aspect_ratio_in_set(candidate.aspect_ratio, used_aspects, no_repeats_tolerance):
                                continue

                        # Try placing at each test position
                        for gap in test_positions:
                            aspect = candidate.aspect_ratio

                            if gap.width / gap.height > aspect:
                                test_height = gap.height
                                test_width = int(test_height * aspect)
                            else:
                                test_width = gap.width
                                test_height = int(test_width / aspect)

                            if test_width < 1 or test_height < 1:
                                continue

                            if self._check_space_available_with_overlap(
                                gap.x, gap.y, test_width, test_height, -1
                            ):
                                coverage = test_width * test_height
                                if coverage > best_coverage:
                                    best_coverage = coverage
                                    best_placement = (candidate, gap.x, gap.y, test_width, test_height, True)

                # Place the best image found
                if best_placement:
                    candidate, x, y, width, height, is_repeat = best_placement

                    self.packed_images.append(PackedImage(
                        info=candidate,
                        x=x,
                        y=y,
                        width=width,
                        height=height
                    ))

                    used_image_ids.add(id(candidate))
                    iteration_added = 1

                    # If this was a repeat, add to freezer
                    if is_repeat:
                        freezer.add(id(candidate))

                total_added += iteration_added
                pbar.update(1)

                # If we didn't add any images this iteration, we're done
                if iteration_added == 0:
                    final_coverage = sum(p.width * p.height for p in self.packed_images) / canvas_area * 100
                    pbar.write(f"Gap-filling complete: {final_coverage:.2f}% coverage (no more gaps can be filled)")
                    break

        if unfreeze_count[0] > 0:
            tqdm.write(f"🔄 Total cycles: Went through all {len(available_repeats)} images {unfreeze_count[0]} time(s) across all collages")

        return total_added

    def pack(self, images: List[ImageInfo]) -> List[PackedImage]:
        """
        Pack all images into the canvas.

        NEW APPROACH: Start with each image having area = canvas_area / n,
        then iterate to find the best fit.
        """
        if not images:
            return []

        # Start with target area = canvas_area / n images
        # Use slightly less (95%) to account for packing inefficiency
        area_scale = 0.95
        best_scale = area_scale

        # Try progressively smaller sizes until packing succeeds
        # This is much faster than binary search since we start close to optimal
        with tqdm(total=10, desc="Finding optimal packing", unit="try", leave=False) as pbar:
            for attempt in range(10):
                if self.try_pack_with_target_areas(images, area_scale):
                    best_scale = area_scale
                    pbar.set_postfix({'coverage': f'{area_scale*100:.0f}%'})
                    pbar.update(1)
                    break
                else:
                    # Reduce target area by 5% and try again
                    area_scale *= 0.95
                    pbar.set_postfix({'trying': f'{area_scale*100:.0f}%'})
                    pbar.update(1)

        # Pack with the best scale found
        if not self.packed_images:
            self.try_pack_with_target_areas(images, best_scale)

        # Optionally grow images to fill remaining whitespace
        # Since we start with equal areas, growth should be minimal
        if not self.respect_original_size and not self.no_uniformity:
            # Light growth pass to fill small gaps
            self.grow_images_to_fill_space()

            # Enforce area uniformity if conditions allow
            # With 0% or very low overlap, we can't adjust sizes without overlaps
            if self.overlap_percent < 0.05:  # Less than 5% overlap
                tqdm.write(f"Skipping area uniformity enforcement (overlap={self.overlap_percent*100:.0f}%)")
                tqdm.write(f"Tip: Use --overlap-percent 5 for better uniformity, or --no-uniformity for max coverage")
            else:
                # Since we started with equal areas, enforcement should converge quickly
                self.enforce_area_uniformity(max_area_variation=self.max_size_variation)
        elif self.no_uniformity and not self.respect_original_size:
            # Maximum coverage mode - aggressive growth
            self.grow_images_to_fill_space()
            tqdm.write("Skipped area uniformity enforcement (--no-uniformity flag)")

        return self.packed_images

    def create_collage(self, background_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
        """Create the final collage image."""
        canvas = Image.new('RGB', (self.canvas_width, self.canvas_height), background_color)

        for packed in self.packed_images:
            # Resize image
            resized = packed.info.image.resize(
                (packed.width, packed.height),
                Image.Resampling.LANCZOS
            )

            # Convert to RGB if necessary
            if resized.mode != 'RGB':
                resized = resized.convert('RGB')

            # Paste onto canvas
            canvas.paste(resized, (packed.x, packed.y))

        return canvas


def ensure_all_images_used(batches: List[List[ImageInfo]], all_images: List[ImageInfo]) -> List[List[ImageInfo]]:
    """
    AGGRESSIVELY ensure every single image appears in at least one batch.

    This is critical - no image should be left out!

    Args:
        batches: Current image batches
        all_images: All available images

    Returns:
        Updated batches with all images included
    """
    # Track which images are already used
    used_images = set()
    for batch in batches:
        for img in batch:
            used_images.add(id(img))

    # Find missing images
    missing_images = [img for img in all_images if id(img) not in used_images]

    if missing_images:
        tqdm.write(f"⚠ WARNING: {len(missing_images)} images not in any batch!")
        tqdm.write(f"AGGRESSIVELY forcing them into batches...")

        # Add missing images to batches in round-robin fashion
        for i, img in enumerate(missing_images):
            batch_idx = i % len(batches)
            batches[batch_idx].append(img)
            tqdm.write(f"  Added missing image to batch {batch_idx + 1}")

        tqdm.write(f"✓ All {len(all_images)} images now assigned to at least one batch")
    else:
        tqdm.write(f"✓ All {len(all_images)} images already in batches")

    return batches


def merge_single_image_batches(batches: List[List[ImageInfo]]) -> List[List[ImageInfo]]:
    """
    Merge any batches with only 1 image into adjacent batches.
    Never allow a collage with just a single image.

    Args:
        batches: List of image batches

    Returns:
        List of batches with no single-image batches
    """
    if len(batches) <= 1:
        return batches

    # Keep merging until no single-image batches remain
    while True:
        # Find first single-image batch
        single_idx = None
        for i, batch in enumerate(batches):
            if len(batch) == 1:
                single_idx = i
                break

        # If no single-image batches, we're done
        if single_idx is None:
            break

        # Merge with adjacent batch (prefer next, fallback to previous)
        if single_idx < len(batches) - 1:
            # Merge with next batch
            batches[single_idx + 1].extend(batches[single_idx])
            batches.pop(single_idx)
            tqdm.write(f"Merged single-image batch {single_idx + 1} into batch {single_idx + 2}")
        elif single_idx > 0:
            # Merge with previous batch
            batches[single_idx - 1].extend(batches[single_idx])
            batches.pop(single_idx)
            tqdm.write(f"Merged single-image batch {single_idx + 1} into batch {single_idx}")
        else:
            # Only one batch with one image - can't merge
            tqdm.write(f"Warning: Only 1 image available, cannot create multiple collages")
            break

    return batches


def aspect_ratios_are_equivalent(ar1: float, ar2: float, tolerance_percent: float) -> bool:
    """
    Check if two aspect ratios are within the specified tolerance percentage.

    Args:
        ar1: First aspect ratio
        ar2: Second aspect ratio
        tolerance_percent: Tolerance as percentage (e.g., 5 for 5%)

    Returns:
        True if aspect ratios are within tolerance, False otherwise
    """
    if tolerance_percent <= 0:
        # Exact matching with rounding to 3 decimal places
        return round(ar1, 3) == round(ar2, 3)

    # Calculate percentage difference
    if ar1 == 0 or ar2 == 0:
        return ar1 == ar2

    diff_percent = abs(ar1 - ar2) / min(ar1, ar2) * 100
    return diff_percent <= tolerance_percent


def aspect_ratio_in_set(ar: float, ar_set: set, tolerance_percent: float) -> bool:
    """
    Check if an aspect ratio is equivalent to any aspect ratio in the set.

    Args:
        ar: Aspect ratio to check
        ar_set: Set of aspect ratios to check against
        tolerance_percent: Tolerance as percentage

    Returns:
        True if any aspect ratio in set is within tolerance
    """
    for existing_ar in ar_set:
        if aspect_ratios_are_equivalent(ar, existing_ar, tolerance_percent):
            return True
    return False


def pre_distribute_forced_duplicates(images: List[ImageInfo], num_batches: int,
                                     no_repeats_tolerance: float) -> tuple:
    """
    Pre-distribute images when there are more images with the same aspect ratio
    than available batches. This ensures ALL images are used by distributing
    duplicates evenly across batches as a hard rule before optimization.

    Args:
        images: List of images to distribute
        num_batches: Number of batches to create
        no_repeats_tolerance: Tolerance for aspect ratio matching

    Returns:
        Tuple of (pre_allocated_batches, remaining_images, batch_aspect_counts)
    """
    if no_repeats_tolerance <= 0:
        return None, images, None

    # Group images by aspect ratio (within tolerance)
    aspect_groups = {}
    for img in images:
        found_group = False
        for ar_key in aspect_groups.keys():
            if aspect_ratios_are_equivalent(img.aspect_ratio, ar_key, no_repeats_tolerance):
                aspect_groups[ar_key].append(img)
                found_group = True
                break
        if not found_group:
            aspect_groups[img.aspect_ratio] = [img]

    # Initialize batches
    batches = [[] for _ in range(num_batches)]
    batch_aspect_counts = [{} for _ in range(num_batches)]  # Track count of each aspect ratio per batch

    # Pre-distribute groups that require duplicates (more images than batches)
    pre_distributed_images = set()

    for ar, group_images in aspect_groups.items():
        if len(group_images) > num_batches:
            # Must distribute duplicates - do it evenly as a hard rule
            images_per_batch = len(group_images) // num_batches
            extras = len(group_images) % num_batches

            tqdm.write(f"Pre-distributing {len(group_images)} images with aspect ratio {ar:.3f} "
                      f"across {num_batches} batches ({images_per_batch}-{images_per_batch + 1} per batch)")

            img_idx = 0
            for batch_idx in range(num_batches):
                # Each batch gets base amount, first 'extras' batches get one more
                count = images_per_batch + (1 if batch_idx < extras else 0)
                for _ in range(count):
                    if img_idx < len(group_images):
                        batches[batch_idx].append(group_images[img_idx])
                        pre_distributed_images.add(id(group_images[img_idx]))
                        # Track this aspect ratio in this batch
                        batch_aspect_counts[batch_idx][ar] = batch_aspect_counts[batch_idx].get(ar, 0) + 1
                        img_idx += 1

    # Remaining images are those not pre-distributed
    remaining_images = [img for img in images if id(img) not in pre_distributed_images]

    return batches, remaining_images, batch_aspect_counts


def optimize_image_distribution_with_tolerance(images: List[ImageInfo], target_per_batch: int,
                                               tolerance_percent: float, canvas_width: int,
                                               canvas_height: int, packer_params: dict,
                                               no_repeats_tolerance: float = 0, allow_repeats: bool = False) -> List[List[ImageInfo]]:
    """
    Distribute images with flexible batch sizes to maximize coverage.

    Args:
        images: List of images to distribute
        target_per_batch: Target number of images per batch
        tolerance_percent: Allowed percentage deviation (e.g., 20 = ±20%)
        canvas_width: Canvas width for testing
        canvas_height: Canvas height for testing
        packer_params: Parameters for ImagePacker
        no_repeats_tolerance: Tolerance percentage for aspect ratio matching (0 = disabled)
        allow_repeats: If True, allow same image to appear in multiple batches (but not within same batch)

    Returns:
        List of image batches optimized for maximum coverage
    """
    if tolerance_percent <= 0:
        # No tolerance, use standard distribution
        num_batches = (len(images) + target_per_batch - 1) // target_per_batch
        return optimize_image_distribution(images, num_batches, no_repeats_tolerance=no_repeats_tolerance, allow_repeats=allow_repeats)

    # Calculate min/max images per batch
    tolerance = tolerance_percent / 100.0
    min_per_batch = max(1, int(target_per_batch * (1 - tolerance)))
    max_per_batch = int(target_per_batch * (1 + tolerance))

    tqdm.write(f"Split tolerance: {tolerance_percent}% → {min_per_batch}-{max_per_batch} images per canvas")

    # Sort images by area for consistent distribution
    sorted_images = sorted(images, key=lambda img: img.original_width * img.original_height, reverse=True)

    batches = []
    remaining_images = sorted_images.copy()

    # Greedy algorithm: for each batch, try different image counts and pick the best coverage
    with tqdm(desc="Optimizing batch sizes", unit="batch", leave=False) as pbar:
        while remaining_images:
            best_batch = None
            best_coverage = 0
            best_count = target_per_batch

            # Try different batch sizes within tolerance
            for batch_size in range(min_per_batch, min(max_per_batch + 1, len(remaining_images) + 1)):
                # Create test batch
                if no_repeats_tolerance > 0:
                    # Build batch while avoiding equivalent aspect ratios WITHIN THIS BATCH
                    # (same aspect ratio CAN appear in different batches)
                    test_batch = []
                    batch_aspects = set()
                    for img in remaining_images:
                        img_aspect = img.aspect_ratio
                        # Only check within this batch, not globally
                        if not aspect_ratio_in_set(img_aspect, batch_aspects, no_repeats_tolerance):
                            test_batch.append(img)
                            batch_aspects.add(img_aspect)
                            if len(test_batch) == batch_size:
                                break

                    # If we couldn't get enough unique images, skip this batch size
                    # But allow smaller batches to enforce the hard rule
                    if len(test_batch) == 0:
                        continue
                else:
                    test_batch = remaining_images[:batch_size]

                # Quick test: try to pack and measure coverage
                test_packer = ImagePacker(canvas_width, canvas_height, **packer_params)
                test_packer.pack(test_batch)

                if test_packer.packed_images:
                    # Calculate coverage
                    total_area = sum(p.width * p.height for p in test_packer.packed_images)
                    canvas_area = canvas_width * canvas_height
                    coverage = total_area / canvas_area

                    # Prefer higher coverage, but also consider using more images
                    # Score = coverage + small bonus for using more images (within reason)
                    score = coverage + (batch_size / target_per_batch) * 0.1

                    if score > best_coverage:
                        best_coverage = score
                        best_batch = test_batch
                        best_count = batch_size

            if best_batch:
                batches.append(best_batch)

                # Remove used images from remaining pool
                if no_repeats_tolerance > 0:
                    remaining_images = [img for img in remaining_images if img not in best_batch]
                else:
                    remaining_images = remaining_images[best_count:]

                pbar.set_postfix({'images': len(best_batch), 'coverage': f'{best_coverage*100:.1f}%'})
                pbar.update(1)
            else:
                # Fallback: try to create a batch respecting the constraint even if smaller
                if no_repeats_tolerance > 0:
                    # Build fallback batch while avoiding equivalent aspect ratios WITHIN THE BATCH
                    fallback_batch = []
                    batch_aspects = set()
                    for img in remaining_images:
                        img_aspect = img.aspect_ratio
                        # Only check within this batch
                        if not aspect_ratio_in_set(img_aspect, batch_aspects, no_repeats_tolerance):
                            fallback_batch.append(img)
                            batch_aspects.add(img_aspect)

                    if fallback_batch:
                        batches.append(fallback_batch)
                        remaining_images = [img for img in remaining_images if img not in fallback_batch]
                        pbar.update(1)
                    else:
                        # No more images can be added without violating constraint within a single batch
                        # This means all remaining images have the same aspect ratio
                        tqdm.write(f"Warning: {len(remaining_images)} images remaining all have the same aspect ratio - adding to new batches")
                        # Add each remaining image to its own batch to satisfy constraint
                        for img in remaining_images:
                            batches.append([img])
                        break
                else:
                    # No constraint, just take minimum batch size
                    batch_size = min(min_per_batch, len(remaining_images))
                    fallback_batch = remaining_images[:batch_size]
                    batches.append(fallback_batch)
                    remaining_images = remaining_images[batch_size:]
                    pbar.update(1)

    # Ensure no single-image batches
    batches = merge_single_image_batches(batches)

    batch_sizes = [len(b) for b in batches]
    tqdm.write(f"Optimized distribution: {batch_sizes} images per canvas")

    return batches


def optimize_image_distribution(images: List[ImageInfo], num_batches: int, no_repeats_tolerance: float = 0, allow_repeats: bool = False) -> List[List[ImageInfo]]:
    """
    Distribute images across multiple batches to maximize coverage.

    Strategy: Alternate distribution by size and aspect ratio for balanced batches.
    This ensures each collage gets a mix of large/small and wide/tall images.

    Args:
        images: List of images to distribute
        num_batches: Number of batches to create
        no_repeats_tolerance: Tolerance percentage for aspect ratio matching (0 = disabled)
        allow_repeats: If True, allow same image to appear in multiple batches (but not within same batch)

    Returns:
        List of image batches optimized for packing
    """
    if num_batches == 1:
        return [images]

    # Pre-distribute forced duplicates if not using allow_repeats
    # (images with same aspect ratio that exceed batch count)
    if not allow_repeats:
        batches, remaining_images, batch_aspect_counts = pre_distribute_forced_duplicates(
            images, num_batches, no_repeats_tolerance
        )

        # If no pre-distribution happened, initialize from scratch
        if batches is None:
            batches = [[] for _ in range(num_batches)]
            remaining_images = images
            batch_aspect_counts = None
    else:
        # With allow_repeats, we start fresh and distribute all images first
        batches = [[] for _ in range(num_batches)]
        remaining_images = images
        batch_aspect_counts = None
        tqdm.write(f"Using allow-repeats mode: will optimize base images first, then add repeats to fill")

    # Sort images by area (largest first)
    sorted_by_size = sorted(remaining_images,
                           key=lambda img: img.original_width * img.original_height,
                           reverse=True)

    # Track which images are in each batch (used for both allow_repeats and tracking)
    batch_image_sets = [set() for _ in range(num_batches)]

    # Initialize batch_image_sets with pre-distributed images
    for batch_idx, batch in enumerate(batches):
        for img in batch:
            batch_image_sets[batch_idx].add(id(img))

    # Track aspect ratios in each batch if no_repeats_tolerance is enabled
    if no_repeats_tolerance > 0:
        # Convert batch_aspect_counts to sets for easier checking
        batch_aspect_ratios = []
        for batch, count_dict in zip(batches, batch_aspect_counts if batch_aspect_counts else [{}] * num_batches):
            ar_set = set(count_dict.keys()) if count_dict else set()
            # Also add aspect ratios from already allocated images in this batch
            for img in batch:
                ar_set.add(img.aspect_ratio)
            batch_aspect_ratios.append(ar_set)
    else:
        batch_aspect_ratios = None

    # FIRST PASS: Distribute each image once (base optimization)
    for idx, img in enumerate(sorted_by_size):
        if no_repeats_tolerance > 0:
            # Find a batch that doesn't already have this aspect ratio (within tolerance)
            img_aspect = img.aspect_ratio
            batch_idx = idx % num_batches
            attempts = 0

            # Try round-robin assignment, avoiding batches with equivalent aspect ratios
            while attempts < num_batches:
                if len(batches[batch_idx]) < MAX_IMAGES_PER_BATCH and \
                   not aspect_ratio_in_set(img_aspect, batch_aspect_ratios[batch_idx], no_repeats_tolerance):
                    batches[batch_idx].append(img)
                    batch_aspect_ratios[batch_idx].add(img_aspect)
                    batch_image_sets[batch_idx].add(id(img))
                    break
                batch_idx = (batch_idx + 1) % num_batches
                attempts += 1

            # HARD RULE: If we couldn't find a batch, create a new one (only if not allow_repeats)
            if attempts == num_batches and not allow_repeats:
                # Create a new batch for this image
                batches.append([img])
                batch_aspect_ratios.append({img_aspect})
                batch_image_sets.append({id(img)})
                num_batches += 1
                tqdm.write(f"Created additional batch to enforce --no-repeats constraint (aspect ratio {img_aspect:.3f})")
        else:
            # No aspect ratio constraints, simple round-robin
            batch_idx = idx % num_batches
            if len(batches[batch_idx]) < MAX_IMAGES_PER_BATCH:
                batches[batch_idx].append(img)
                batch_image_sets[batch_idx].add(id(img))


    # Further optimize by swapping images to balance aspect ratio diversity
    # Calculate average aspect ratio per batch
    batch_aspects = []
    for batch in batches:
        if batch:
            avg_aspect = sum(img.aspect_ratio for img in batch) / len(batch)
            batch_aspects.append(avg_aspect)
        else:
            batch_aspects.append(1.0)

    # Ensure each batch has similar total area for even coverage
    batch_areas = [sum(img.original_width * img.original_height for img in batch) for batch in batches]
    total_area = sum(batch_areas)
    target_area_per_batch = total_area / num_batches

    # Ensure no single-image batches
    batches = merge_single_image_batches(batches)

    tqdm.write(f"Optimized image distribution: {[len(b) for b in batches]} images per collage")

    # Recalculate stats after potential merges
    batch_areas = [sum(img.original_width * img.original_height for img in batch) for batch in batches]
    total_area = sum(batch_areas)
    if len(batches) > 0:
        target_area_per_batch = total_area / len(batches)
        tqdm.write(f"Area balance: {min(batch_areas)/target_area_per_batch*100:.1f}%-{max(batch_areas)/target_area_per_batch*100:.1f}% of target")

    if allow_repeats:
        tqdm.write(f"Note: Repeats will be added after packing to fill blank areas")

    return batches



def process_single_collage(args_tuple):
    """
    Process a single collage batch. Designed to be called in parallel.

    Args:
        args_tuple: Tuple of (batch_idx, batch, output_path, canvas_width, canvas_height,
                             respect_original_size, max_size_variation, overlap_percent,
                             no_uniformity, randomize, bg_color, save_to_file,
                             allow_repeats, all_images, no_repeats_tolerance, min_coverage)

    Returns:
        Dictionary with results, statistics, and optionally the canvas
    """
    (batch_idx, batch, output_path, canvas_width, canvas_height,
     respect_original_size, max_size_variation, overlap_percent, no_uniformity, randomize, bg_color,
     save_to_file, allow_repeats, all_images, no_repeats_tolerance, min_coverage) = args_tuple

    # Create a new packer instance for this batch
    packer = ImagePacker(
        canvas_width,
        canvas_height,
        respect_original_size=respect_original_size,
        max_size_variation=max_size_variation,
        overlap_percent=overlap_percent,
        no_uniformity=no_uniformity,
        randomize=randomize
    )

    # Pack images
    packed = packer.pack(batch)

    # Aggressively fill blank areas with repeats if enabled
    if allow_repeats and all_images:
        initial_coverage = sum(p.width * p.height for p in packed) / (canvas_width * canvas_height) * 100

        # Determine target coverage: use min_coverage if set, otherwise default to 90%
        target_coverage = min_coverage if min_coverage else 90.0

        if initial_coverage < target_coverage:  # Only try to fill if below target
            # Track which images are already used in this collage
            used_image_ids = {id(img) for img in batch}

            # Track aspect ratios if no_repeats_tolerance is enabled
            used_aspects = None
            if no_repeats_tolerance > 0:
                used_aspects = {img.aspect_ratio for img in batch}

            # Sort all images by size (largest first for better coverage)
            candidates = sorted(all_images,
                               key=lambda img: img.original_width * img.original_height,
                               reverse=True)

            # CRITICAL: Rebuild free rectangles before gap-filling to find ALL gaps
            tqdm.write(f"Collage {batch_idx}: Scanning canvas to detect all gaps...")
            packer.rebuild_free_rectangles_from_canvas()

            # Use the new direct gap-filling method
            added_count = packer.fill_gaps_with_repeats(
                candidate_images=candidates,
                target_coverage=target_coverage,
                used_image_ids=used_image_ids,
                used_aspects=used_aspects,
                no_repeats_tolerance=no_repeats_tolerance
            )

            # Update packed images from packer
            packed = packer.packed_images

            if added_count > 0:
                final_coverage = sum(p.width * p.height for p in packed) / (canvas_width * canvas_height) * 100
                tqdm.write(f"Collage {batch_idx}: Added {added_count} repeat images to fill gaps "
                          f"({initial_coverage:.1f}% → {final_coverage:.1f}% coverage)")


    # Create collage
    collage = packer.create_collage(background_color=bg_color)

    # Save or keep in memory
    if save_to_file:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        collage.save(output_path, quality=95)

    # Calculate statistics
    total_image_area = sum(p.width * p.height for p in packed)
    canvas_area = canvas_width * canvas_height
    coverage = (total_image_area / canvas_area) * 100

    stats = {
        'batch_idx': batch_idx,
        'output_path': output_path,
        'num_images': len(batch),
        'num_packed': len(packed),
        'coverage': coverage,
    }

    # Area uniformity statistics
    if packed and not respect_original_size:
        areas = [p.width * p.height for p in packed]
        avg_area = sum(areas) / len(areas)
        max_area_diff = max(abs(area - avg_area) / avg_area * 100 for area in areas)
        min_area = min(areas)
        max_area = max(areas)

        stats.update({
            'avg_area': avg_area,
            'min_area': min_area,
            'max_area': max_area,
            'max_area_deviation': max_area_diff,
        })

        # Aspect ratio preservation
        max_aspect_error = 0.0
        for p in packed:
            actual_aspect = p.width / p.height
            original_aspect = p.info.aspect_ratio
            aspect_error = abs(actual_aspect - original_aspect) / original_aspect * 100
            max_aspect_error = max(max_aspect_error, aspect_error)
        stats['max_aspect_error'] = max_aspect_error

        # Overlap statistics
        max_overlap_percent = 0.0
        for i, p in enumerate(packed):
            p_area = p.width * p.height
            for j, other in enumerate(packed):
                if i == j:
                    continue
                overlap_x = max(0, min(p.x + p.width, other.x + other.width) - max(p.x, other.x))
                overlap_y = max(0, min(p.y + p.height, other.y + other.height) - max(p.y, other.y))
                if overlap_x > 0 and overlap_y > 0:
                    overlap_area = overlap_x * overlap_y
                    overlap_percent = (overlap_area / p_area) * 100
                    max_overlap_percent = max(max_overlap_percent, overlap_percent)
        stats['max_overlap'] = max_overlap_percent

    # Include canvas if not saving to file (for PDF generation)
    if not save_to_file:
        stats['canvas'] = collage

    return stats


def print_collage_stats(stats, total_batches):
    """Print statistics for a completed collage."""
    print(f"\n{'='*60}")
    print(f"Collage {stats['batch_idx']}/{total_batches}: {stats['num_images']} images -> {stats['output_path']}")
    print(f"{'='*60}")

    if stats['num_packed'] < stats['num_images']:
        print(f"Warning: Could only pack {stats['num_packed']} out of {stats['num_images']} images.")
        print("Try increasing canvas size or enabling --respect-original-size")

    print(f"Collage saved to {stats['output_path']}")
    print(f"Canvas coverage: {stats['coverage']:.1f}%")

    if 'avg_area' in stats:
        print(f"Area uniformity - Average area: {stats['avg_area']:.0f}, "
              f"Min: {stats['min_area']:.0f}, Max: {stats['max_area']:.0f}")
        print(f"Area uniformity - Max deviation from average: {stats['max_area_deviation']:.2f}%")
        print(f"Aspect ratio preservation - Max deviation: {stats['max_aspect_error']:.2f}%")
        print(f"Image overlap - Max overlap: {stats['max_overlap']:.2f}% of any image's area")


def save_canvases_to_pdf(canvases: List, output_path: str):
    """
    Save multiple canvases (PIL Images) to a single PDF file.

    Args:
        canvases: List of PIL Image objects (one per page)
        output_path: Path to save the PDF file
    """
    if not canvases:
        print("Error: No canvases to save to PDF")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Convert all images to RGB mode (PDF requires RGB)
    rgb_canvases = []
    for canvas in canvases:
        if canvas.mode != 'RGB':
            rgb_canvases.append(canvas.convert('RGB'))
        else:
            rgb_canvases.append(canvas)

    # Save first image with remaining images as additional pages
    if len(rgb_canvases) == 1:
        rgb_canvases[0].save(output_path, 'PDF', quality=95)
    else:
        rgb_canvases[0].save(
            output_path,
            'PDF',
            save_all=True,
            append_images=rgb_canvases[1:],
            quality=95
        )

    print(f"\nPDF saved with {len(canvases)} page(s) to {output_path}")


def optimize_canvas_count_for_coverage(images: List[ImageInfo], target_images_per_canvas: int,
                                       canvas_width: int, canvas_height: int,
                                       packer_params: dict, no_repeats_tolerance: float) -> tuple:
    """
    Determine optimal number of canvases to maximize average coverage.

    Tests different canvas counts around a target images-per-canvas value,
    running quick packing tests to find the configuration with best coverage.

    Args:
        images: List of images to distribute
        target_images_per_canvas: Target images per canvas (tests configurations around this)
        canvas_width: Canvas width
        canvas_height: Canvas height
        packer_params: Parameters for ImagePacker
        no_repeats_tolerance: Tolerance for no-repeats constraint

    Returns:
        Tuple of (optimal_num_canvases, optimal_images_per_canvas, best_coverage)
    """
    total_images = len(images)

    # Calculate ideal canvas count based on target
    ideal_canvases = max(1, round(total_images / target_images_per_canvas))

    # Test a tight range around the ideal (80% to 120% of ideal)
    # This is aggressive to stay close to target, especially important with --allow-repeats
    min_canvases = max(1, int(ideal_canvases * 0.8))
    max_canvases = min(total_images, int(ideal_canvases * 1.2))

    # Ensure reasonable bounds
    if max_canvases < min_canvases:
        max_canvases = min_canvases

    total_configs = max_canvases - min_canvases + 1

    print(f"\nOptimizing canvas count for maximum coverage...")
    print(f"Target: ~{target_images_per_canvas} images per canvas")
    print(f"Ideal canvas count: {ideal_canvases} (testing range: {min_canvases}-{max_canvases})")

    best_config = None
    best_avg_coverage = 0
    results = []

    # Try different numbers of canvases
    with tqdm(desc="Testing configurations", total=total_configs, unit="config") as pbar:
        for num_canvases in range(min_canvases, max_canvases + 1):
            images_per_canvas = total_images // num_canvases

            # Distribute images for this configuration
            if no_repeats_tolerance > 0:
                test_batches, _, _ = pre_distribute_forced_duplicates(images, num_canvases, no_repeats_tolerance)
                if test_batches is None:
                    test_batches = [[] for _ in range(num_canvases)]
                # Simple round-robin for remaining to get quick estimate
                remaining = [img for img in images if not any(img in batch for batch in test_batches)]
                for idx, img in enumerate(remaining):
                    test_batches[idx % num_canvases].append(img)
            else:
                # Simple round-robin distribution
                test_batches = [[] for _ in range(num_canvases)]
                for idx, img in enumerate(images):
                    test_batches[idx % num_canvases].append(img)

            # Quick packing test on each batch
            coverages = []
            for batch in test_batches:
                if not batch:
                    continue
                test_packer = ImagePacker(canvas_width, canvas_height, **packer_params)
                test_packer.pack(batch)

                if test_packer.packed_images:
                    total_area = sum(p.width * p.height for p in test_packer.packed_images)
                    canvas_area = canvas_width * canvas_height
                    coverage = (total_area / canvas_area) * 100
                    coverages.append(coverage)

            if coverages:
                avg_coverage = sum(coverages) / len(coverages)
                min_coverage = min(coverages)
                max_coverage = max(coverages)

                results.append({
                    'num_canvases': num_canvases,
                    'images_per_canvas': images_per_canvas,
                    'avg_coverage': avg_coverage,
                    'min_coverage': min_coverage,
                    'max_coverage': max_coverage
                })

                if avg_coverage > best_avg_coverage:
                    best_avg_coverage = avg_coverage
                    best_config = (num_canvases, images_per_canvas)

                pbar.set_postfix({
                    'canvases': num_canvases,
                    'per_canvas': images_per_canvas,
                    'avg_cov': f'{avg_coverage:.1f}%'
                })

            pbar.update(1)

    # Display results
    print(f"\n{'='*80}")
    print(f"Coverage optimization results:")
    print(f"{'='*80}")
    print(f"{'Canvases':<10} {'Images/Canvas':<15} {'Avg Coverage':<15} {'Min Coverage':<15} {'Max Coverage':<15}")
    print(f"{'-'*80}")

    for result in sorted(results, key=lambda x: x['avg_coverage'], reverse=True)[:10]:
        marker = " <-- BEST" if (result['num_canvases'], result['images_per_canvas']) == best_config else ""
        print(f"{result['num_canvases']:<10} {result['images_per_canvas']:<15} "
              f"{result['avg_coverage']:<15.1f} {result['min_coverage']:<15.1f} "
              f"{result['max_coverage']:<15.1f}{marker}")

    if best_config:
        print(f"{'='*80}")
        print(f"Selected: {best_config[0]} canvases with ~{best_config[1]} images each")
        print(f"Average coverage: {best_avg_coverage:.1f}%")
        print(f"{'='*80}\n")
        return best_config[0], best_config[1], best_avg_coverage
    else:
        # Fallback to single canvas
        return 1, total_images, 0.0


def main():
    parser = argparse.ArgumentParser(
        description='Pack multiple images into a single canvas optimally.'
    )
    parser.add_argument(
        'folder',
        nargs='?',
        default='input_images',
        help='Path to folder containing images (default: input_images)'
    )
    parser.add_argument(
        '-W', '--width',
        type=int,
        required=True,
        help='Width of output canvas'
    )
    parser.add_argument(
        '-H', '--height',
        type=int,
        required=True,
        help='Height of output canvas'
    )
    parser.add_argument(
        '-o', '--output',
        default='output_images/collage.png',
        help='Output file path (default: output_images/collage.png)'
    )
    parser.add_argument(
        '--pdf',
        action='store_true',
        help='Save all collages as a single PDF file (one collage per page)'
    )
    parser.add_argument(
        '--respect-original-size',
        action='store_true',
        help='Respect original image sizes and proportions (default: make all images roughly equal in size)'
    )
    parser.add_argument(
        '--max-size-variation',
        type=float,
        default=15.0,
        help='Maximum percentage that any image area can differ from average area (default: 15.0)'
    )
    parser.add_argument(
        '--overlap-percent',
        type=float,
        default=10.0,
        help='Maximum percentage of any image\'s area that can overlap with others (default: 10.0)'
    )
    parser.add_argument(
        '--background-color',
        default='255,255,255',
        help='Background color as R,G,B (default: 255,255,255 for white)'
    )
    parser.add_argument(
        '--no-uniformity',
        action='store_true',
        help='Skip area uniformity enforcement to maximize coverage (images may vary more in size)'
    )
    parser.add_argument(
        '--randomize',
        action='store_true',
        help='Randomize image order (default: sort by size for better packing)'
    )
    parser.add_argument(
        '--no-repeats',
        type=float,
        default=0,
        metavar='TOLERANCE',
        help='Prevent images with similar aspect ratios from appearing in the same collage. '
             'Value is tolerance percentage (e.g., 5 means aspect ratios within 5%% are considered the same). '
             'Use 0 for exact matching (default: 0 = disabled)'
    )
    parser.add_argument(
        '--allow-repeats',
        action='store_true',
        help='Allow the same image to appear in multiple canvases (but never twice in the same canvas). '
             'After initial packing, fills blank areas with repeated images. Use with --min-coverage to set target.'
    )
    parser.add_argument(
        '--min-coverage',
        type=float,
        metavar='PERCENT',
        help='Minimum coverage percentage to achieve by adding repeated images (e.g., 90 for 90%%). '
             'Only works with --allow-repeats. Adds images from largest to smallest until target is reached.'
    )
    parser.add_argument(
        '-n', '--images-per-collage',
        type=int,
        help='Number of images per collage (creates multiple collages if needed)'
    )
    parser.add_argument(
        '-p', '--num-collages',
        type=int,
        help='Number of collages to create (divides images evenly)'
    )
    parser.add_argument(
        '--split-tolerance',
        type=float,
        default=0,
        help='Percentage flexibility in images per canvas to maximize coverage (e.g., 20 allows ±20%% from target)'
    )
    parser.add_argument(
        '--max-coverage',
        type=int,
        metavar='TARGET_IMAGES',
        help='Automatically determine optimal number of canvases to maximize coverage. '
             'Value is target images per canvas (e.g., 8 means aim for ~8 images per canvas). '
             'Will test different canvas counts around this target and pick the one with best coverage.'
    )
    parser.add_argument(
        '-j', '--jobs',
        type=int,
        default=None,
        help=f'Number of parallel jobs for creating collages (default: auto-detect, max: {cpu_count()})'
    )

    args = parser.parse_args()

    # Validate flags
    if sum([bool(args.images_per_collage), bool(args.num_collages), bool(args.max_coverage)]) > 1:
        print("Error: Can only specify one of -n/--images-per-collage, -p/--num-collages, or --max-coverage")
        return 1

    if args.max_coverage and args.max_coverage < 1:
        print("Error: --max-coverage target images per canvas must be at least 1")
        return 1

    if args.split_tolerance < 0 or args.split_tolerance > 100:
        print("Error: --split-tolerance must be between 0 and 100")
        return 1

    if args.no_repeats < 0 or args.no_repeats > 100:
        print("Error: --no-repeats tolerance must be between 0 and 100")
        return 1

    if args.min_coverage:
        if not args.allow_repeats:
            print("Error: --min-coverage requires --allow-repeats flag")
            return 1
        if args.min_coverage < 0 or args.min_coverage > 100:
            print("Error: --min-coverage must be between 0 and 100")
            return 1

    if args.split_tolerance > 0 and not (args.images_per_collage or args.num_collages):
        print("Warning: --split-tolerance requires -n or -p flag to have effect")

    if args.no_repeats > 0 and not (args.images_per_collage or args.num_collages):
        print("Warning: --no-repeats requires -n or -p flag to have effect (does nothing for single collage)")


    # Parse background color
    try:
        bg_color = tuple(map(int, args.background_color.split(',')))
        if len(bg_color) != 3 or not all(0 <= c <= 255 for c in bg_color):
            raise ValueError
    except ValueError:
        print("Error: Background color must be in format R,G,B with values 0-255")
        return 1

    # Create packer
    packer = ImagePacker(
        args.width,
        args.height,
        respect_original_size=args.respect_original_size,
        max_size_variation=args.max_size_variation,
        overlap_percent=args.overlap_percent,
        no_uniformity=args.no_uniformity,
        randomize=args.randomize
    )

    # Load images
    print(f"Loading images from {args.folder}...")
    try:
        images = packer.load_images(args.folder)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    if not images:
        print("No images found in the specified folder.")
        return 1

    print(f"Found {len(images)} images.")

    # Determine batching strategy with optimized distribution
    image_batches = []
    if args.images_per_collage:
        # Create batches with flexible sizes using split-tolerance
        if args.split_tolerance > 0:
            # Use tolerance-based distribution for maximum coverage
            packer_params = {
                'respect_original_size': args.respect_original_size,
                'max_size_variation': args.max_size_variation,
                'overlap_percent': args.overlap_percent,
                'no_uniformity': args.no_uniformity,
                'randomize': args.randomize
            }
            image_batches = optimize_image_distribution_with_tolerance(
                images, args.images_per_collage, args.split_tolerance,
                args.width, args.height, packer_params, no_repeats_tolerance=args.no_repeats, allow_repeats=args.allow_repeats
            )
            print(f"Creating {len(image_batches)} collage(s) with flexible sizing (±{args.split_tolerance}%)")
        else:
            # Standard fixed-size distribution
            num_collages = (len(images) + args.images_per_collage - 1) // args.images_per_collage
            image_batches = optimize_image_distribution(images, num_collages, no_repeats_tolerance=args.no_repeats, allow_repeats=args.allow_repeats)
            print(f"Creating {len(image_batches)} collage(s) with optimized image distribution")
    elif args.num_collages:
        # Divide images evenly into P collages with optimized distribution
        num_collages = args.num_collages
        if num_collages > len(images):
            print(f"Warning: Requested {num_collages} collages but only {len(images)} images available")
            num_collages = len(images)

        if args.split_tolerance > 0:
            # Calculate target per collage and use tolerance-based distribution
            target_per_collage = len(images) // num_collages
            packer_params = {
                'respect_original_size': args.respect_original_size,
                'max_size_variation': args.max_size_variation,
                'overlap_percent': args.overlap_percent,
                'no_uniformity': args.no_uniformity,
                'randomize': args.randomize
            }
            image_batches = optimize_image_distribution_with_tolerance(
                images, target_per_collage, args.split_tolerance,
                args.width, args.height, packer_params, no_repeats_tolerance=args.no_repeats, allow_repeats=args.allow_repeats
            )
            # Adjust if we created more/fewer batches than requested
            if len(image_batches) != num_collages:
                print(f"Note: Created {len(image_batches)} collages (target was {num_collages}) for optimal coverage")
            else:
                print(f"Creating {num_collages} collage(s) with flexible sizing (±{args.split_tolerance}%)")
        else:
            # Standard fixed-size distribution
            image_batches = optimize_image_distribution(images, num_collages, no_repeats_tolerance=args.no_repeats, allow_repeats=args.allow_repeats)
            print(f"Creating {num_collages} collage(s) with optimized image distribution")
    elif args.max_coverage:
        # Optimize canvas count for maximum coverage
        packer_params = {
            'respect_original_size': args.respect_original_size,
            'max_size_variation': args.max_size_variation,
            'overlap_percent': args.overlap_percent,
            'no_uniformity': args.no_uniformity,
            'randomize': args.randomize
        }
        optimal_canvases, optimal_per_canvas, avg_coverage = optimize_canvas_count_for_coverage(
            images, args.max_coverage, args.width, args.height, packer_params, args.no_repeats
        )

        # Select base images for distribution: target_per_canvas * num_canvases
        # Use optimal_per_canvas from the optimization (not all images)
        target_total_images = optimal_canvases * args.max_coverage

        # Select the best images for base distribution (largest first for better coverage)
        base_images = sorted(images,
                            key=lambda img: img.original_width * img.original_height,
                            reverse=True)[:target_total_images]

        print(f"Base distribution: {len(base_images)} images across {optimal_canvases} canvases (~{len(base_images)//optimal_canvases} per canvas)")

        # Distribute ONLY the base images (not all images)
        image_batches = optimize_image_distribution(base_images, optimal_canvases, no_repeats_tolerance=args.no_repeats, allow_repeats=False)
        print(f"Creating {len(image_batches)} collage(s) with optimized distribution for maximum coverage")
        if args.allow_repeats:
            print(f"Note: Repeats will be added after collage creation to fill blank areas (from full pool of {len(images)} images)")
    else:
        # Single collage with all images
        image_batches = [images]

    # CRITICAL: Ensure EVERY image appears in at least one batch
    print(f"\n{'='*60}")
    print(f"ENSURING ALL IMAGES USED")
    print(f"{'='*60}")
    image_batches = ensure_all_images_used(image_batches, images)

    # CRITICAL: Final safety check - merge any single-image batches
    # This catches cases that might have been missed by distribution functions
    print(f"\nFinal batch check: {len(image_batches)} collage(s) with {[len(b) for b in image_batches]} images each")
    if len(image_batches) > 1:
        image_batches = merge_single_image_batches(image_batches)
        print(f"After merge: {len(image_batches)} collage(s) with {[len(b) for b in image_batches]} images each")

    # Generate output filenames
    output_files = []
    if len(image_batches) > 1:
        # Multiple collages - generate numbered filenames
        output_dir = os.path.dirname(args.output) or '.'
        output_base = os.path.basename(args.output)
        output_name, output_ext = os.path.splitext(output_base)

        for i in range(len(image_batches)):
            numbered_filename = f"{output_name}_{i+1:03d}{output_ext}"
            output_files.append(os.path.join(output_dir, numbered_filename))
    else:
        output_files = [args.output]

    # Determine number of parallel jobs
    num_jobs = args.jobs if args.jobs else min(cpu_count(), len(image_batches))
    if num_jobs > len(image_batches):
        num_jobs = len(image_batches)

    # Create collages
    if len(image_batches) > 1 and num_jobs > 1:
        # Parallel processing for multiple collages
        print(f"\nProcessing {len(image_batches)} collages using {num_jobs} parallel workers...")

        if args.allow_repeats:
            print(f"⚠ NOTE: Parallel mode cannot share freezer across collages.")
            print(f"   Repeat images may be less evenly distributed.")
            print(f"   For best repeat distribution, use -j 1 (sequential mode).")

        # Prepare arguments for parallel processing
        job_args = []
        for batch_idx, (batch, output_path) in enumerate(zip(image_batches, output_files), 1):
            job_args.append((
                batch_idx,
                batch,
                output_path,
                args.width,
                args.height,
                args.respect_original_size,
                args.max_size_variation,
                args.overlap_percent,
                args.no_uniformity,
                args.randomize,
                bg_color,
                not args.pdf,  # save_to_file: False when creating PDF
                args.allow_repeats,  # allow_repeats
                images,  # all_images: full image pool for repeat filling
                args.no_repeats,  # no_repeats_tolerance
                args.min_coverage  # min_coverage
            ))

        # Process in parallel with progress bar
        with Pool(processes=num_jobs) as pool:
            results = list(tqdm(
                pool.imap(process_single_collage, job_args),
                total=len(job_args),
                desc="Creating collages",
                unit="collage"
            ))

        # Print statistics for all collages
        results.sort(key=lambda x: x['batch_idx'])

        # Check for single-image collages and flag them for retry
        failed_batches = []
        for stats in results:
            print_collage_stats(stats, len(image_batches))
            # Check if this resulted in a single-image collage
            if stats['num_images'] >= 2 and stats['num_packed'] == 1:
                failed_batches.append(stats['batch_idx'])
                print(f"\n⚠ WARNING: Batch {stats['batch_idx']} would create single-image collage!")

        # If any batches failed, suggest retry with sequential mode
        if failed_batches:
            print(f"\n{'='*60}")
            print(f"⚠ RETRY NEEDED: {len(failed_batches)} batch(es) resulted in single-image collages")
            print(f"Batches: {failed_batches}")
            print(f"Suggestion: These batches need more images to pack successfully.")
            print(f"Re-run with -j 1 to use sequential mode with automatic retry.")
            print(f"{'='*60}")
            # Don't save PDF if there are failures
            return 1

        # If PDF mode, save all canvases to a single PDF
        if args.pdf:
            canvases = [result['canvas'] for result in results]
            # Change output extension to .pdf
            pdf_output = os.path.splitext(args.output)[0] + '.pdf'
            save_canvases_to_pdf(canvases, pdf_output)

    else:
        # Sequential processing (single collage or single job)
        if len(image_batches) == 1:
            print("\nCreating single collage...")
        else:
            print(f"\nCreating {len(image_batches)} collages sequentially...")

        # UNIVERSAL FREEZER: Shared across ALL collages in this session
        # Ensures no image is used as a repeat twice until ALL images have been used
        universal_freezer = set()
        universal_unfreeze_count = [0]
        if args.allow_repeats and len(image_batches) > 1:
            print(f"🧊 Universal freezer enabled: Each image will be used once as repeat before any repeats")

        canvases = []  # For PDF mode
        batch_idx = 0
        while batch_idx < len(image_batches):
            batch = image_batches[batch_idx]
            output_path = output_files[batch_idx]

            print(f"\n{'='*60}")
            print(f"Collage {batch_idx+1}/{len(image_batches)}: {len(batch)} images -> {output_path}")
            print(f"{'='*60}")

            # Pack images
            print("Packing images...")
            packed = packer.pack(batch)

            # CRITICAL: Check if packing failed to fit enough images
            # If batch has 2+ images but only 1 packed, we need to retry with more images
            if len(batch) >= 2 and len(packed) == 1:
                print(f"\n⚠ WARNING: Only {len(packed)} out of {len(batch)} images fit on canvas!")
                print(f"This would create a single-image collage, which is not allowed.")

                # Try to merge with next batch and retry
                if batch_idx < len(image_batches) - 1:
                    print(f"Merging batch {batch_idx+1} with batch {batch_idx+2} and retrying...")
                    # Merge current batch with next batch
                    image_batches[batch_idx].extend(image_batches[batch_idx + 1])
                    # Remove the next batch
                    image_batches.pop(batch_idx + 1)
                    output_files.pop(batch_idx + 1)
                    # Retry this batch (don't increment batch_idx)
                    print(f"Retry: Batch now has {len(image_batches[batch_idx])} images")
                    continue
                else:
                    # Last batch - try to merge with previous
                    if batch_idx > 0:
                        print(f"Merging batch {batch_idx+1} with batch {batch_idx} and retrying...")
                        # Merge with previous batch
                        image_batches[batch_idx - 1].extend(image_batches[batch_idx])
                        # Remove current batch
                        image_batches.pop(batch_idx)
                        output_files.pop(batch_idx)
                        # Go back and retry the previous batch
                        batch_idx -= 1
                        print(f"Retry: Batch {batch_idx+1} now has {len(image_batches[batch_idx])} images")
                        continue
                    else:
                        print(f"ERROR: Cannot merge - this is the only batch!")
                        print(f"Try increasing canvas size with -W and -H")

            # Aggressively fill blank areas with repeats if enabled
            if args.allow_repeats:
                initial_coverage = sum(p.width * p.height for p in packed) / (args.width * args.height) * 100

                # Determine target coverage: use min_coverage if set, otherwise default to 90%
                target_coverage = args.min_coverage if args.min_coverage else 90.0

                if initial_coverage < target_coverage:  # Only try to fill if below target
                    print(f"Filling blank areas with repeats (initial coverage: {initial_coverage:.1f}%, target: {target_coverage:.1f}%)...")

                    # Track which images are already used in this collage
                    used_image_ids = {id(img) for img in batch}

                    # Track aspect ratios if no_repeats_tolerance is enabled
                    used_aspects = None
                    if args.no_repeats > 0:
                        used_aspects = {img.aspect_ratio for img in batch}

                    # Sort all images by size (largest first for better coverage)
                    candidates = sorted(images,
                                       key=lambda img: img.original_width * img.original_height,
                                       reverse=True)

                    # CRITICAL: Rebuild free rectangles before gap-filling to find ALL gaps
                    print(f"Scanning canvas to detect all gaps...")
                    packer.rebuild_free_rectangles_from_canvas()

                    # Use the new direct gap-filling method with UNIVERSAL freezer
                    added_count = packer.fill_gaps_with_repeats(
                        candidate_images=candidates,
                        target_coverage=target_coverage,
                        used_image_ids=used_image_ids,
                        used_aspects=used_aspects,
                        no_repeats_tolerance=args.no_repeats,
                        freezer=universal_freezer,
                        unfreeze_count=universal_unfreeze_count
                    )

                    # Update packed images from packer
                    packed = packer.packed_images

                    if added_count > 0:
                        final_coverage = sum(p.width * p.height for p in packed) / (args.width * args.height) * 100
                        print(f"Added {added_count} repeat images to fill gaps "
                              f"({initial_coverage:.1f}% → {final_coverage:.1f}% coverage)")


            if not packed or len(packed) < len(batch):
                print(f"Warning: Could only pack {len(packed)} out of {len(batch)} images.")
                print("Try increasing canvas size or enabling --respect-original-size")

            # Create collage
            print("Creating collage...")
            collage = packer.create_collage(background_color=bg_color)

            # Save or collect for PDF
            if args.pdf:
                canvases.append(collage)
            else:
                os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
                collage.save(output_path, quality=95)
                print(f"Collage saved to {output_path}")

            # Print statistics
            total_image_area = sum(p.width * p.height for p in packed)
            canvas_area = args.width * args.height
            coverage = (total_image_area / canvas_area) * 100
            print(f"Canvas coverage: {coverage:.1f}%")

            # Print area uniformity statistics
            if packed and not args.respect_original_size:
                areas = [p.width * p.height for p in packed]
                avg_area = sum(areas) / len(areas)
                max_area_diff = max(abs(area - avg_area) / avg_area * 100 for area in areas)
                min_area = min(areas)
                max_area = max(areas)
                print(f"Area uniformity - Average area: {avg_area:.0f}, Min: {min_area:.0f}, Max: {max_area:.0f}")
                print(f"Area uniformity - Max deviation from average: {max_area_diff:.2f}%")

                # Check aspect ratio preservation
                max_aspect_error = 0.0
                for p in packed:
                    actual_aspect = p.width / p.height
                    original_aspect = p.info.aspect_ratio
                    aspect_error = abs(actual_aspect - original_aspect) / original_aspect * 100
                    max_aspect_error = max(max_aspect_error, aspect_error)
                print(f"Aspect ratio preservation - Max deviation: {max_aspect_error:.2f}%")

                # Check overlap statistics
                max_overlap_percent = 0.0
                for i, p in enumerate(packed):
                    p_area = p.width * p.height
                    for j, other in enumerate(packed):
                        if i == j:
                            continue
                        overlap_x = max(0, min(p.x + p.width, other.x + other.width) - max(p.x, other.x))
                        overlap_y = max(0, min(p.y + p.height, other.y + other.height) - max(p.y, other.y))
                        if overlap_x > 0 and overlap_y > 0:
                            overlap_area = overlap_x * overlap_y
                            overlap_percent = (overlap_area / p_area) * 100
                            max_overlap_percent = max(max_overlap_percent, overlap_percent)
                print(f"Image overlap - Max overlap: {max_overlap_percent:.2f}% of any image's area")

            # Move to next batch
            batch_idx += 1

        # AGGRESSIVE: Check if any images were never used and force them in as repeats
        print(f"\n{'='*60}")
        print(f"Checking for unused images...")
        print(f"{'='*60}")

        all_used_images = set()
        for batch in image_batches:
            for img in batch:
                all_used_images.add(id(img))

        never_used = [img for img in images if id(img) not in all_used_images]

        if never_used:
            print(f"🚨 FOUND {len(never_used)} images that NEVER appeared!")
            print(f"AGGRESSIVELY forcing them into collages as repeats...")

            # Force them into the last collage(s) as repeats
            for img in never_used:
                # Add to the last collage
                if image_batches:
                    image_batches[-1].append(img)
                    print(f"  Forced missing image into last collage: {img.path}")

            # Re-create the last collage(s) with the forced images
            print(f"\nRe-creating collages with forced images...")
            # Go back and recreate affected batches
            batch_idx = len(image_batches) - 1
            while batch_idx >= 0 and batch_idx >= len(canvases):
                batch = image_batches[batch_idx]
                output_path = output_files[batch_idx]

                print(f"\n{'='*60}")
                print(f"RE-CREATING Collage {batch_idx+1}/{len(image_batches)}: {len(batch)} images -> {output_path}")
                print(f"{'='*60}")

                # Pack images
                print("Packing images...")
                packed = packer.pack(batch)

                # Create collage
                print("Creating collage...")
                collage = packer.create_collage(background_color=bg_color)

                # Update canvas
                if args.pdf:
                    if batch_idx < len(canvases):
                        canvases[batch_idx] = collage
                    else:
                        canvases.append(collage)
                else:
                    collage.save(output_path, quality=95)
                    print(f"Collage saved to {output_path}")

                batch_idx -= 1
                break  # Only recreate last one for now

        # If PDF mode in sequential processing, save all canvases to a single PDF
        if args.pdf and canvases:
            pdf_output = os.path.splitext(args.output)[0] + '.pdf'
            save_canvases_to_pdf(canvases, pdf_output)

    # FINAL VERIFICATION: Check if every image appeared in at least one collage
    print(f"\n{'='*60}")
    print(f"FINAL VERIFICATION: Checking all images were used")
    print(f"{'='*60}")

    all_used_images = set()
    for batch in image_batches:
        for img in batch:
            all_used_images.add(id(img))

    missing_images = [img for img in images if id(img) not in all_used_images]

    if missing_images:
        print(f"🚨 CRITICAL ERROR: {len(missing_images)} images NEVER appeared in any collage!")
        print(f"   This should not happen after forcing - please report this bug!")
        for i, img in enumerate(missing_images[:10]):  # Show first 10
            print(f"   Missing: {img.path}")
        return 1
    else:
        print(f"✓ VERIFIED: All {len(images)} images appeared in at least one collage!")

    print(f"\n{'='*60}")
    print(f"✓ Created {len(image_batches)} collage(s) successfully")
    print(f"{'='*60}")

    return 0


if __name__ == '__main__':
    exit(main())
