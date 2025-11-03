# Image Packer

A Python tool that optimally packs multiple images into a single canvas with specified dimensions. The algorithm maximizes image sizes while minimizing whitespace, respecting original aspect ratios.

## Features

- **Optimal Packing**: Uses a guillotine-based bin packing algorithm to efficiently arrange images
- **Aspect Ratio Preservation**: All images maintain their original aspect ratios
- **Equal Sizing**: By default, all images are scaled to roughly equal sizes for a uniform look
- **Flexible Scaling Modes**:
  - Default mode: Makes all images roughly equal in size
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

Basic usage:

```bash
uv run image_packer.py /path/to/images -W 1920 -H 1080
```

### Arguments

- `folder`: Path to folder containing images (required)
- `-W`, `--width`: Width of output canvas in pixels (required)
- `-H`, `--height`: Height of output canvas in pixels (required)
- `-o`, `--output`: Output file path (default: collage.png)
- `--respect-original-size`: Maintain relative size differences between images (default: make all images roughly equal in size)
- `--background-color`: Background color as R,G,B (default: 255,255,255 for white)

### Examples

Create a collage with equal-sized images (default behavior):
```bash
uv run image_packer.py input_images -W 1920 -H 1080
```

Preserve relative size differences between images:
```bash
uv run image_packer.py input_images -W 1920 -H 1080 --respect-original-size
```

Custom output path and background color:
```bash
uv run image_packer.py input_images -W 2560 -H 1440 -o output_images/collage.jpg --background-color 0,0,0
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
   - **Default**: Calculates a target area per image and scales each image individually to achieve roughly equal sizes
   - **With `--respect-original-size`**: Applies uniform scaling to maintain relative size differences
3. For each image, it finds the best-fitting free rectangle
4. After placing an image, it splits the used rectangle into new free rectangles
5. Binary search is used to find the maximum scale adjustment factor that allows all images to fit
6. The final collage is rendered with all images at their optimized positions and sizes

## Project Structure

- `input_images/` - Place your source images here
- `output_images/` - Generated collages will be saved here (by default in the root directory)

## Output

The script creates a single image file containing all input images arranged optimally. It also prints statistics including:
- Number of images processed
- Canvas coverage percentage
- Warning if not all images could be packed
