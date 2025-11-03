# Image Packer

A Python tool that optimally packs multiple images into a single canvas with specified dimensions. The algorithm **maximizes space occupation** by making images as large as possible while maintaining equal sizes and respecting original aspect ratios.

## Features

- **Maximum Space Utilization**: Aggressively fills the canvas to maximize image sizes
- **Configurable Size Variation**: Allow up to 10% size variation by default (configurable) to eliminate whitespace
- **Smart Overlap**: Permits 5% overlap between images by default (configurable) for better space utilization
- **Aggressive Post-Packing Growth**: Multiple growth passes with 4 different strategies to fill remaining whitespace
- **Optimal Packing**: Uses a guillotine-based bin packing algorithm with precise binary search
- **Aspect Ratio Preservation**: All images maintain their original aspect ratios
- **Equal Sizing**: By default, all images are scaled to roughly equal sizes for a uniform look
- **Flexible Scaling Modes**:
  - Default mode: Makes all images roughly equal in size, then aggressively grows them to fill whitespace (prioritizes space over perfect equality)
  - `--respect-original-size`: Maintains relative size differences between images (no growth phase)
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
- `-n`, `--images-per-collage`: Number of images per collage (creates multiple collages if needed)
- `-p`, `--num-collages`: Number of collages to create (divides images evenly)
- `-j`, `--jobs`: Number of parallel workers for creating collages (default: auto-detect number of CPUs)
- `--respect-original-size`: Maintain relative size differences between images (default: make all images roughly equal in size, then grow to fill whitespace)
- `--max-size-variation`: Maximum percentage variation in image area from average (default: 15.0)
- `--overlap-percent`: Percentage of overlap allowed between images (default: 10.0)
- `--no-uniformity`: Skip area uniformity enforcement to maximize coverage (images may vary more in size)
- `--background-color`: Background color as R,G,B (default: 255,255,255 for white)

**Notes:**
- You cannot specify both `-n` and `-p` at the same time
- Parallel processing is automatically enabled when creating multiple collages
- Use `-j 1` to force sequential processing
- With `--overlap-percent 0`, area uniformity enforcement is automatically skipped for better coverage
- Use `--no-uniformity` to prioritize maximum coverage over uniform image sizes

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

Allow more size variation (20%) for even better space utilization:
```bash
uv run image_packer.py -W 1920 -H 1080 --max-size-variation 20
```

Allow more overlap (10%) for denser packing:
```bash
uv run image_packer.py -W 1920 -H 1080 --overlap-percent 10
```

No overlap, minimal size variation for stricter uniformity:
```bash
uv run image_packer.py -W 1920 -H 1080 --max-size-variation 5 --overlap-percent 0
```

Create multiple collages with 5 images each:
```bash
uv run image_packer.py -W 1920 -H 1080 -n 5
# If you have 23 images, this creates 5 collages (4 with 5 images, 1 with 3 images)
# Output: collage_001.png, collage_002.png, ..., collage_005.png
```

Create exactly 3 collages (divides images evenly):
```bash
uv run image_packer.py -W 1920 -H 1080 -p 3
# If you have 23 images, this creates 3 collages with 8, 8, and 7 images
# Output: collage_001.png, collage_002.png, collage_003.png
```

Create 4-image collages with custom output directory:
```bash
uv run image_packer.py -W 1920 -H 1080 -n 4 -o output_images/grid.png
# Output: output_images/grid_001.png, grid_002.png, grid_003.png, etc.
```

Use all available CPUs for parallel processing (automatic):
```bash
uv run image_packer.py -W 1920 -H 1080 -n 5
# Automatically uses all available CPU cores to process collages in parallel
```

Limit parallel workers to 8:
```bash
uv run image_packer.py -W 1920 -H 1080 -n 5 -j 8
# Uses maximum of 8 parallel workers
```

Force sequential processing:
```bash
uv run image_packer.py -W 1920 -H 1080 -n 5 -j 1
# Process collages one at a time (useful for debugging)
```

Maximize coverage with no overlap (best for filling the canvas):
```bash
uv run image_packer.py -W 1920 -H 1080 --overlap-percent 0 -n 8
# Automatically skips area uniformity enforcement for better coverage
# Images grow aggressively to fill space (may vary more in size)
```

Force maximum coverage mode:
```bash
uv run image_packer.py -W 1920 -H 1080 --no-uniformity
# Skip area uniformity enforcement entirely
# Prioritizes filling canvas over uniform image sizes
```

## Performance

### Parallel Processing

When creating multiple collages (using `-n` or `-p` flags), the tool automatically parallelizes the work across all available CPU cores:

- **Automatic**: By default, uses all available CPUs
- **Scalable**: Near-linear speedup for multiple collages (e.g., 32 CPUs ≈ 32x faster)
- **Efficient**: Each collage is processed independently in parallel
- **Progress tracking**: Real-time progress bar shows completed collages

**Example performance gains:**
- Creating 32 collages on a 32-core CPU: ~32x faster than sequential
- Creating 100 collages with `-j 16`: processes 16 at a time

### Optimization Features

- **Iterative area uniformity enforcement**: Converges quickly to meet strict area constraints
- **Binary search scaling**: 30 iterations for precise space optimization
- **Progress bars**: Visual feedback for all long-running operations
- **Efficient constraint checking**: Optimized overlap and bounds validation

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
6. **Aggressive post-packing growth** (default mode only): Multiple passes with 4 strategies to fill whitespace:
   - **Strategy 1**: Maximize width growth (fill horizontally)
   - **Strategy 2**: Maximize height growth (fill vertically)
   - **Strategy 3**: Grow to maximum allowed area based on size variation limit
   - **Strategy 4**: Incremental growth (20% per pass) with overlap allowance
7. The final collage is rendered with all images at their optimized positions and sizes

### Space Maximization

The algorithm uses several techniques to maximize canvas utilization:
- **Optimistic targeting**: Aims for 100% canvas coverage, letting binary search find the practical maximum
- **Precise binary search**: 30 iterations ensure near-optimal scaling
- **Aggressive upper bounds**: Explores up to 3x the estimated scale to find all possible solutions
- **Best-fit placement**: Minimizes wasted space when placing each image
- **Multi-pass aggressive growth**: 3 growth passes, each trying 4 different strategies
- **Size variation allowance**: Permits up to 10% variation (configurable) from average size to fill gaps
- **Smart overlap**: Allows up to 5% overlap (configurable) between images, calculated as percentage of smaller image
- **Priority**: Space utilization > perfect size equality (configurable via flags)

## Project Structure

- `input_images/` - Default folder for source images
- `output_images/` - Default folder for generated collages

## Output

The script creates a single image file containing all input images arranged optimally. It also prints statistics including:
- Number of images processed
- Canvas coverage percentage
- Warning if not all images could be packed
