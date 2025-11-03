# Image Packer

A Python tool that optimally packs multiple images into a single canvas with specified dimensions. The algorithm maximizes image sizes while minimizing whitespace, respecting original aspect ratios.

## Features

- **Optimal Packing**: Uses a guillotine-based bin packing algorithm to efficiently arrange images
- **Aspect Ratio Preservation**: All images maintain their original aspect ratios
- **Automatic Scaling**: Images are scaled up as large as possible to fill the canvas
- **Original Size Respect**: Optional flag to prevent scaling images beyond their original dimensions
- **Whitespace Minimization**: Intelligently arranges images to minimize gaps
- **Multiple Format Support**: Supports JPG, PNG, BMP, GIF, TIFF, and WebP formats

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

```bash
# Install dependencies
uv pip install -r requirements.txt
```

Or if you have a virtual environment:

```bash
# Create a virtual environment and install dependencies
uv venv
uv pip install -r requirements.txt
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
- `--respect-original-size`: Do not scale images beyond their original size
- `--background-color`: Background color as R,G,B (default: 255,255,255 for white)

### Examples

Create a collage with default settings:
```bash
uv run image_packer.py ./my_images -W 1920 -H 1080
```

Respect original image sizes (no upscaling):
```bash
uv run image_packer.py ./my_images -W 1920 -H 1080 --respect-original-size
```

Custom output path and background color:
```bash
uv run image_packer.py ./my_images -W 2560 -H 1440 -o output/collage.jpg --background-color 0,0,0
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
2. For each image, it finds the best-fitting free rectangle
3. After placing an image, it splits the used rectangle into new free rectangles
4. Binary search is used to find the maximum scale factor that allows all images to fit
5. The final collage is rendered with all images at their optimized positions and sizes

## Output

The script creates a single image file containing all input images arranged optimally. It also prints statistics including:
- Number of images processed
- Canvas coverage percentage
- Warning if not all images could be packed
