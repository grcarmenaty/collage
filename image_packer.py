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
        """Load all images from a folder."""
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
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


def optimize_image_distribution(images: List[ImageInfo], num_batches: int) -> List[List[ImageInfo]]:
    """
    Distribute images across multiple batches to maximize coverage.

    Strategy: Alternate distribution by size and aspect ratio for balanced batches.
    This ensures each collage gets a mix of large/small and wide/tall images.

    Args:
        images: List of images to distribute
        num_batches: Number of batches to create

    Returns:
        List of image batches optimized for packing
    """
    if num_batches == 1:
        return [images]

    # Sort images by area (largest first)
    sorted_by_size = sorted(images,
                           key=lambda img: img.original_width * img.original_height,
                           reverse=True)

    # Initialize batches
    batches = [[] for _ in range(num_batches)]

    # Distribute images round-robin style, alternating between batches
    # This ensures each batch gets a mix of large and small images
    for idx, img in enumerate(sorted_by_size):
        batch_idx = idx % num_batches
        batches[batch_idx].append(img)

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

    tqdm.write(f"Optimized image distribution: {[len(b) for b in batches]} images per collage")
    tqdm.write(f"Area balance: {min(batch_areas)/target_area_per_batch*100:.1f}%-{max(batch_areas)/target_area_per_batch*100:.1f}% of target")

    return batches


def process_single_collage(args_tuple):
    """
    Process a single collage batch. Designed to be called in parallel.

    Args:
        args_tuple: Tuple of (batch_idx, batch, output_path, canvas_width, canvas_height,
                             respect_original_size, max_size_variation, overlap_percent,
                             no_uniformity, randomize, bg_color)

    Returns:
        Dictionary with results and statistics
    """
    (batch_idx, batch, output_path, canvas_width, canvas_height,
     respect_original_size, max_size_variation, overlap_percent, no_uniformity, randomize, bg_color) = args_tuple

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

    # Create collage
    collage = packer.create_collage(background_color=bg_color)

    # Save
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
        '-j', '--jobs',
        type=int,
        default=None,
        help=f'Number of parallel jobs for creating collages (default: auto-detect, max: {cpu_count()})'
    )

    args = parser.parse_args()

    # Validate flags
    if args.images_per_collage and args.num_collages:
        print("Error: Cannot specify both -n/--images-per-collage and -p/--num-collages")
        return 1

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
        # Create batches of N images each with optimized distribution
        num_collages = (len(images) + args.images_per_collage - 1) // args.images_per_collage
        image_batches = optimize_image_distribution(images, num_collages)
        print(f"Creating {len(image_batches)} collage(s) with optimized image distribution")
    elif args.num_collages:
        # Divide images evenly into P collages with optimized distribution
        num_collages = args.num_collages
        if num_collages > len(images):
            print(f"Warning: Requested {num_collages} collages but only {len(images)} images available")
            num_collages = len(images)

        image_batches = optimize_image_distribution(images, num_collages)
        print(f"Creating {num_collages} collage(s) with optimized image distribution")
    else:
        # Single collage with all images
        image_batches = [images]

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
                bg_color
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
        for stats in results:
            print_collage_stats(stats, len(image_batches))

    else:
        # Sequential processing (single collage or single job)
        if len(image_batches) == 1:
            print("\nCreating single collage...")
        else:
            print(f"\nCreating {len(image_batches)} collages sequentially...")

        for batch_idx, (batch, output_path) in enumerate(zip(image_batches, output_files), 1):
            print(f"\n{'='*60}")
            print(f"Collage {batch_idx}/{len(image_batches)}: {len(batch)} images -> {output_path}")
            print(f"{'='*60}")

            # Pack images
            print("Packing images...")
            packed = packer.pack(batch)

            if not packed or len(packed) < len(batch):
                print(f"Warning: Could only pack {len(packed)} out of {len(batch)} images.")
                print("Try increasing canvas size or enabling --respect-original-size")

            # Create collage
            print("Creating collage...")
            collage = packer.create_collage(background_color=bg_color)

            # Save
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

    print(f"\n{'='*60}")
    print(f"✓ Created {len(image_batches)} collage(s) successfully")
    print(f"{'='*60}")

    return 0


if __name__ == '__main__':
    exit(main())
