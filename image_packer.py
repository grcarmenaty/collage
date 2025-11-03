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
                 max_size_variation: float = 10.0,
                 overlap_percent: float = 5.0):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.respect_original_size = respect_original_size
        self.max_size_variation = max_size_variation / 100.0  # Convert to decimal
        self.overlap_percent = overlap_percent / 100.0  # Convert to decimal
        self.free_rectangles: List[Rectangle] = [Rectangle(0, 0, canvas_width, canvas_height)]
        self.packed_images: List[PackedImage] = []

    def load_images(self, folder_path: str) -> List[ImageInfo]:
        """Load all images from a folder."""
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        images = []

        if not os.path.exists(folder_path):
            raise ValueError(f"Folder not found: {folder_path}")

        for filename in sorted(os.listdir(folder_path)):
            file_path = os.path.join(folder_path, filename)
            if not os.path.isfile(file_path):
                continue

            ext = os.path.splitext(filename)[1].lower()
            if ext in supported_formats:
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
                    print(f"Warning: Could not load {filename}: {e}")

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

    def try_pack_with_scale(self, images: List[ImageInfo], scale: float) -> bool:
        """Try to pack all images with a given scale factor."""
        self.free_rectangles = [Rectangle(0, 0, self.canvas_width, self.canvas_height)]
        self.packed_images = []

        # Sort images by area (largest first) for better packing
        sorted_images = sorted(images,
                             key=lambda img: img.original_width * img.original_height,
                             reverse=True)

        # Calculate target area per image for equal sizing (when not respecting original size)
        if not self.respect_original_size:
            canvas_area = self.canvas_width * self.canvas_height
            # Start optimistic - binary search will find the actual maximum
            target_area_per_image = (canvas_area * 1.0) / len(images)

        for img_info in sorted_images:
            if self.respect_original_size:
                # Use uniform scaling, respecting original size
                scaled_width = int(img_info.original_width * scale)
                scaled_height = int(img_info.original_height * scale)

                # Do not exceed original dimensions
                scaled_width = min(scaled_width, img_info.original_width)
                scaled_height = min(scaled_height, img_info.original_height)
            else:
                # Calculate individual scale to make images roughly equal in size
                original_area = img_info.original_width * img_info.original_height

                # Scale to achieve target area
                individual_scale = math.sqrt(target_area_per_image / original_area)

                # Apply global adjustment factor
                individual_scale *= scale

                scaled_width = int(img_info.original_width * individual_scale)
                scaled_height = int(img_info.original_height * individual_scale)

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

    def grow_images_to_fill_space(self):
        """
        After initial packing, aggressively grow images to fill remaining whitespace.
        Allows size variations up to max_size_variation and overlaps up to overlap_percent.
        """
        if not self.packed_images:
            return

        # Calculate average initial size for variation limits
        avg_area = sum(p.width * p.height for p in self.packed_images) / len(self.packed_images)
        avg_dimension = math.sqrt(avg_area)

        # Multiple aggressive growth passes
        for growth_pass in range(3):
            for i, packed in enumerate(self.packed_images):
                current_area = packed.width * packed.height

                # Calculate maximum allowed size based on variation
                max_allowed_area = avg_area * (1 + self.max_size_variation)

                # Stop growing this image if it's already at or above the max variation
                if current_area >= max_allowed_area:
                    continue

                aspect_ratio = packed.info.aspect_ratio

                # AGGRESSIVE STRATEGY 1: Maximum width growth with overlap allowed
                max_width = self.canvas_width - packed.x
                new_width = max_width
                new_height = int(new_width / aspect_ratio)

                # Check if within canvas and size variation limits
                if new_height <= self.canvas_height - packed.y:
                    new_area = new_width * new_height
                    if new_area <= max_allowed_area:
                        if self._check_space_available_with_overlap(packed.x, packed.y, new_width, new_height, i):
                            packed.width = new_width
                            packed.height = new_height
                            continue

                # AGGRESSIVE STRATEGY 2: Maximum height growth with overlap allowed
                max_height = self.canvas_height - packed.y
                new_height = max_height
                new_width = int(new_height * aspect_ratio)

                if new_width <= self.canvas_width - packed.x:
                    new_area = new_width * new_height
                    if new_area <= max_allowed_area:
                        if self._check_space_available_with_overlap(packed.x, packed.y, new_width, new_height, i):
                            packed.width = new_width
                            packed.height = new_height
                            continue

                # AGGRESSIVE STRATEGY 3: Grow to maximum allowed area
                target_area = min(max_allowed_area, (self.canvas_width - packed.x) * (self.canvas_height - packed.y))
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

                if self._check_space_available_with_overlap(packed.x, packed.y, new_width, new_height, i):
                    packed.width = new_width
                    packed.height = new_height
                    continue

                # AGGRESSIVE STRATEGY 4: Incremental growth with overlap
                # Try to grow by 20% of remaining space each iteration
                growth_factor = 1.2
                new_width = int(packed.width * growth_factor)
                new_height = int(packed.height * growth_factor)

                # Constrain to canvas
                new_width = min(new_width, self.canvas_width - packed.x)
                new_height = min(new_height, self.canvas_height - packed.y)

                new_area = new_width * new_height
                if new_area <= max_allowed_area:
                    if self._check_space_available_with_overlap(packed.x, packed.y, new_width, new_height, i):
                        packed.width = new_width
                        packed.height = new_height

    def _check_space_available_with_overlap(self, x: int, y: int, width: int, height: int, exclude_index: int) -> bool:
        """
        Check if a rectangle at (x, y) with given dimensions is valid.
        Allows overlaps up to overlap_percent of the smaller dimension.
        """
        # Check canvas boundaries
        if x + width > self.canvas_width or y + height > self.canvas_height:
            return False

        # Check overlap with other images - allow up to overlap_percent
        for i, other in enumerate(self.packed_images):
            if i == exclude_index:
                continue

            # Calculate overlap area
            overlap_x = max(0, min(x + width, other.x + other.width) - max(x, other.x))
            overlap_y = max(0, min(y + height, other.y + other.height) - max(y, other.y))

            if overlap_x > 0 and overlap_y > 0:
                # Calculate overlap as percentage of the smaller image
                overlap_area = overlap_x * overlap_y
                this_area = width * height
                other_area = other.width * other.height
                smaller_area = min(this_area, other_area)

                # If overlap exceeds allowed percentage, reject
                if overlap_area > smaller_area * self.overlap_percent:
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
        """Pack all images into the canvas, finding the optimal scale."""
        if not images:
            return []

        # Binary search for the best scale factor
        min_scale = 0.01
        max_scale = self.calculate_scale_factor(images) * 3  # More aggressive upper bound
        best_scale = min_scale

        # Try to find the largest scale that fits all images
        # More iterations for better precision = better space utilization
        for _ in range(30):
            mid_scale = (min_scale + max_scale) / 2

            if self.try_pack_with_scale(images, mid_scale):
                best_scale = mid_scale
                min_scale = mid_scale
            else:
                max_scale = mid_scale

        # Pack with the best scale found
        self.try_pack_with_scale(images, best_scale)

        # Grow images to fill remaining whitespace (allows slight size variations)
        # This prioritizes space utilization over perfect size equality
        if not self.respect_original_size:
            self.grow_images_to_fill_space()

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
        default=10.0,
        help='Maximum percentage variation in image sizes (default: 10.0)'
    )
    parser.add_argument(
        '--overlap-percent',
        type=float,
        default=5.0,
        help='Percentage of overlap allowed between images (default: 5.0)'
    )
    parser.add_argument(
        '--background-color',
        default='255,255,255',
        help='Background color as R,G,B (default: 255,255,255 for white)'
    )

    args = parser.parse_args()

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
        overlap_percent=args.overlap_percent
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

    # Pack images
    print("Packing images...")
    packed = packer.pack(images)

    if not packed or len(packed) < len(images):
        print(f"Warning: Could only pack {len(packed)} out of {len(images)} images.")
        print("Try increasing canvas size or enabling --respect-original-size")

    # Create collage
    print("Creating collage...")
    collage = packer.create_collage(background_color=bg_color)

    # Save
    collage.save(args.output, quality=95)
    print(f"Collage saved to {args.output}")

    # Print statistics
    total_image_area = sum(p.width * p.height for p in packed)
    canvas_area = args.width * args.height
    coverage = (total_image_area / canvas_area) * 100
    print(f"Canvas coverage: {coverage:.1f}%")

    return 0


if __name__ == '__main__':
    exit(main())
