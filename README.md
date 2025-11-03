# Image Packer

A Python tool that optimally packs multiple images into a single canvas with specified dimensions. The algorithm **maximizes space occupation** by making images as large as possible while maintaining equal sizes and respecting original aspect ratios.

## Features

- **Maximum Space Utilization**: Aggressively fills the canvas to maximize image sizes
- **Post-Packing Growth**: After initial packing, images are grown to fill remaining whitespace (allows slight size variations to eliminate gaps)
- **Optimal Packing**: Uses a guillotine-based bin packing algorithm with precise binary search
- **Aspect Ratio Preservation**: All images maintain their original aspect ratios
- **Equal Sizing**: By default, all images are scaled to roughly equal sizes for a uniform look
- **Flexible Scaling Modes**:
  - Default mode: Makes all images roughly equal in size, then grows them to fill whitespace (prioritizes space over perfect equality)
  - `--respect-original-size`: Maintains relative size differences between images
- **Whitespace Minimization**: Intelligently arranges images to minimize gaps
- **Multiple Format Support**: Supports JPG, PNG, BMP, GIF, TIFF, and WebP formats

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

```bash
# Sync dependencies (creates venv and installs packages automatically)
uv sync
```

## Usage

Basic usage (uses default folders):

```bash
uv run image_packer.py -W 1920 -H 1080
```

This will read images from `input_images/` and save to `output_images/collage.png`.

### Arguments

- `folder`: Path to folder containing images (default: input_images)
- `-W`, `--width`: Width of output canvas in pixels (required)
- `-H`, `--height`: Height of output canvas in pixels (required)
- `-o`, `--output`: Output file path (default: output_images/collage.png)
- `--respect-original-size`: Maintain relative size differences between images (default: make all images roughly equal in size, then grow to fill whitespace)
- `--background-color`: Background color as R,G,B (default: 255,255,255 for white)

### Examples

Create a collage with default folders:
```bash
# Place images in input_images/ folder first
uv run image_packer.py -W 1920 -H 1080
```

Specify custom input folder:
```bash
uv run image_packer.py /path/to/images -W 1920 -H 1080
```

Preserve relative size differences between images (no post-packing growth):
```bash
uv run image_packer.py -W 1920 -H 1080 --respect-original-size
```

Custom output path and background color:
```bash
uv run image_packer.py -W 2560 -H 1440 -o my_collage.jpg --background-color 0,0,0
```

## Algorithm

The tool uses a guillotine-based rectangle packing algorithm with the following approach:

1. **Load Images**: Reads all supported image formats from the folder
2. **Calculate Scale**: Uses binary search to find the optimal scale factor that fits all images
3. **Pack Images**: Places images using best-fit strategy (minimizing wasted space)
4. **Split Rectangles**: After placing each image, splits remaining space into new rectangles
5. **Optimize**: Removes redundant rectangles to maintain efficiency

Images are sorted by area (largest first) for better packing efficiency.

## How It Works

1. The algorithm maintains a list of free rectangles representing available space
2. **Scaling Mode**:
   - **Default**: Calculates a target area per image (canvas_area / num_images) and scales each image individually to achieve roughly equal sizes
   - **With `--respect-original-size`**: Applies uniform scaling to maintain relative size differences
3. For each image, it finds the best-fitting free rectangle using a best-fit strategy
4. After placing an image, it splits the used rectangle into new free rectangles
5. **Binary search with 30 iterations** precisely finds the maximum scale factor that allows all images to fit, maximizing space occupation
6. **Post-packing growth** (default mode only): Each image is grown to fill adjacent whitespace while maintaining aspect ratio, allowing slight size variations to eliminate gaps
7. The final collage is rendered with all images at their optimized positions and sizes

### Space Maximization

The algorithm uses several techniques to maximize canvas utilization:
- **Optimistic targeting**: Aims for 100% canvas coverage, letting binary search find the practical maximum
- **Precise binary search**: 30 iterations ensure near-optimal scaling
- **Aggressive upper bounds**: Explores up to 3x the estimated scale to find all possible solutions
- **Best-fit placement**: Minimizes wasted space when placing each image
- **Post-packing growth**: Images are expanded into remaining whitespace after initial packing (prioritizes space utilization over perfect size equality)

## Project Structure

- `input_images/` - Default folder for source images
- `output_images/` - Default folder for generated collages

## Output

The script creates a single image file containing all input images arranged optimally. It also prints statistics including:
- Number of images processed
- Canvas coverage percentage
- Warning if not all images could be packed
