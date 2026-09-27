import cv2
import numpy as np
import json
import argparse
from skimage import color
import os
import requests


# ============================================================
# OPTIONAL: DOWNLOAD SAMPLE IMAGE
# ============================================================

def download_sample_image(save_path):
    print("Downloading sample image...")

    url = (
        "https://images.unsplash.com/"
        "photo-1543852786-1cf6624b9987"
        "?q=80&w=1200&auto=format&fit=crop"
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    with open(save_path, "wb") as f:
        f.write(response.content)

    print(f"Sample image saved to: {save_path}")


# ============================================================
# GRID ORIENTATION
# ============================================================

def get_target_dimensions(img_shape, base_grid_size):
    """
    Automatically adjusts the grid orientation according
    to the input image orientation.

    Example:
        landscape image -> 90 x 60
        portrait image  -> 60 x 90
    """

    h, w = img_shape[:2]
    img_aspect = w / h

    dim1, dim2 = base_grid_size

    if img_aspect >= 1.0:
        # Landscape
        grid_w = max(dim1, dim2)
        grid_h = min(dim1, dim2)
    else:
        # Portrait
        grid_w = min(dim1, dim2)
        grid_h = max(dim1, dim2)

    return grid_w, grid_h


# ============================================================
# CENTER CROP
# ============================================================

def crop_center_to_aspect(img, target_aspect):
    """
    Crops the image from the center so that its aspect ratio
    matches the target mosaic aspect ratio.
    """

    h, w = img.shape[:2]

    image_aspect = w / h

    if abs(image_aspect - target_aspect) < 0.001:
        return img

    if image_aspect > target_aspect:
        # Image is too wide
        new_w = int(h * target_aspect)

        offset = (w - new_w) // 2

        return img[:, offset:offset + new_w]

    else:
        # Image is too tall
        new_h = int(w / target_aspect)

        offset = (h - new_h) // 2

        return img[offset:offset + new_h, :]


# ============================================================
# LOAD PALETTE
# ============================================================

def load_palette(palette_path):
    """
    Loads brick palette from JSON.

    Expected format:

    {
        "B001": {
            "name": "Black",
            "rgb": [20, 20, 20]
        },

        "W001": {
            "name": "White",
            "rgb": [245, 245, 245]
        }
    }
    """

    if not os.path.exists(palette_path):
        raise FileNotFoundError(
            f"Palette file not found: {palette_path}"
        )

    with open(palette_path, "r") as f:
        palette_data = json.load(f)

    if not palette_data:
        raise ValueError("Palette JSON is empty.")

    brick_ids = list(palette_data.keys())

    palette_rgb = np.array(
        [
            palette_data[brick_id]["rgb"]
            for brick_id in brick_ids
        ],
        dtype=np.float32
    )

    # Convert palette RGB -> LAB ONCE
    palette_lab = color.rgb2lab(
        (palette_rgb / 255.0).reshape(1, -1, 3)
    ).reshape(-1, 3)

    return palette_data, brick_ids, palette_rgb, palette_lab


# ============================================================
# FIND CLOSEST COLOR
# ============================================================

def match_closest_color_cie2000(pixel_lab, palette_lab):
    """
    Finds the closest brick color using CIEDE2000.
    """

    pixel_tile = np.tile(
        pixel_lab,
        (palette_lab.shape[0], 1)
    )

    distances = color.deltaE_ciede2000(
        pixel_tile,
        palette_lab
    )

    return int(np.argmin(distances))


# ============================================================
# MAIN MOSAIC ENGINE
# ============================================================

def process_mosaic_dithered(
    image_path,
    palette_path,
    grid_size=(90, 60),
    enable_dithering=True,
    output_dir="output"
):

    print("\n============================================")
    print("      PHOTOBRICKS MOSAIC ENGINE")
    print("============================================\n")

    # --------------------------------------------------------
    # Validate image
    # --------------------------------------------------------

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    print(f"Input image:")
    print(f"  {image_path}")

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    print("\nLoading image...")

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(
            f"Could not read image:\n{image_path}"
        )

    print(
        f"Original resolution: "
        f"{img.shape[1]} x {img.shape[0]}"
    )

    # BGR -> RGB
    img_rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Determine grid orientation
    # --------------------------------------------------------

    grid_w, grid_h = get_target_dimensions(
        img_rgb.shape,
        grid_size
    )

    target_aspect = grid_w / grid_h

    print(
        f"\nResolved grid: "
        f"{grid_w} x {grid_h}"
    )

    print(
        f"Grid aspect ratio: "
        f"{target_aspect:.3f}"
    )

    # --------------------------------------------------------
    # Crop image
    # --------------------------------------------------------

    print("\nCropping image...")

    img_cropped = crop_center_to_aspect(
        img_rgb,
        target_aspect
    )

    print(
        f"Cropped resolution: "
        f"{img_cropped.shape[1]} x "
        f"{img_cropped.shape[0]}"
    )

    # --------------------------------------------------------
    # Resize to brick grid
    # --------------------------------------------------------

    print("\nResizing image to brick grid...")

    img_resized = cv2.resize(
        img_cropped,
        (grid_w, grid_h),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # Load palette
    # --------------------------------------------------------

    print("\nLoading brick palette...")

    (
        palette_data,
        brick_ids,
        palette_rgb,
        palette_lab
    ) = load_palette(palette_path)

    print(
        f"Number of brick colors: "
        f"{len(brick_ids)}"
    )

    # --------------------------------------------------------
    # Working buffer
    # --------------------------------------------------------

    img_float = img_resized.astype(
        np.float32
    )

    output_indices = np.zeros(
        (grid_h, grid_w),
        dtype=np.int32
    )

    # --------------------------------------------------------
    # Mosaic generation
    # --------------------------------------------------------

    print("\nGenerating brick mosaic...")

    if enable_dithering:
        print("Mode: CIEDE2000 + Floyd-Steinberg dithering")
    else:
        print("Mode: CIEDE2000 without dithering")

    total_cells = grid_w * grid_h

    processed_cells = 0

    # --------------------------------------------------------
    # Process every brick cell
    # --------------------------------------------------------

    for y in range(grid_h):

        for x in range(grid_w):

            # Current RGB pixel
            old_pixel_rgb = np.clip(
                img_float[y, x],
                0,
                255
            )

            # ------------------------------------------------
            # RGB -> LAB
            # ------------------------------------------------

            pixel_lab = color.rgb2lab(
                old_pixel_rgb.reshape(1, 1, 3) / 255.0
            ).reshape(3)

            # ------------------------------------------------
            # Find closest physical brick color
            # ------------------------------------------------

            best_idx = match_closest_color_cie2000(
                pixel_lab,
                palette_lab
            )

            output_indices[y, x] = best_idx

            # ------------------------------------------------
            # Selected brick RGB
            # ------------------------------------------------

            new_pixel_rgb = palette_rgb[best_idx]

            # ------------------------------------------------
            # Floyd-Steinberg dithering
            # ------------------------------------------------

            if enable_dithering:

                quant_error = (
                    old_pixel_rgb -
                    new_pixel_rgb
                )

                # Right
                if x + 1 < grid_w:

                    img_float[y, x + 1] += (
                        quant_error * (7 / 16)
                    )

                # Bottom-left
                if y + 1 < grid_h:

                    if x > 0:

                        img_float[y + 1, x - 1] += (
                            quant_error * (3 / 16)
                        )

                    # Bottom
                    img_float[y + 1, x] += (
                        quant_error * (5 / 16)
                    )

                    # Bottom-right
                    if x + 1 < grid_w:

                        img_float[y + 1, x + 1] += (
                            quant_error * (1 / 16)
                        )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            processed_cells += 1

            if processed_cells % 500 == 0:

                progress = (
                    processed_cells /
                    total_cells
                ) * 100

                print(
                    f"Progress: "
                    f"{progress:.1f}%",
                    end="\r"
                )

    print("\n\nMosaic generation complete.")

    # ========================================================
    # GENERATE FINAL MOSAIC
    # ========================================================

    mosaic_rgb = palette_rgb[
        output_indices
    ].astype(np.uint8)

    # ========================================================
    # SAVE BASIC MOSAIC
    # ========================================================

    mosaic_path = os.path.join(
        output_dir,
        "mosaic.png"
    )

    mosaic_bgr = cv2.cvtColor(
        mosaic_rgb,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        mosaic_path,
        mosaic_bgr
    )

    print(
        f"\nMosaic saved:"
        f"\n  {mosaic_path}"
    )

    # ========================================================
    # GENERATE LARGE PREVIEW
    # ========================================================

    scale_factor = 10

    preview_img = cv2.resize(
        mosaic_rgb,
        (
            grid_w * scale_factor,
            grid_h * scale_factor
        ),
        interpolation=cv2.INTER_NEAREST
    )

    # ========================================================
    # GRID OVERLAY
    # ========================================================

    grid_color = (30, 30, 30)

    # Vertical lines

    for i in range(grid_w + 1):

        x = i * scale_factor

        cv2.line(
            preview_img,
            (x, 0),
            (
                x,
                grid_h * scale_factor
            ),
            grid_color,
            1
        )

    # Horizontal lines

    for j in range(grid_h + 1):

        y = j * scale_factor

        cv2.line(
            preview_img,
            (0, y),
            (
                grid_w * scale_factor,
                y
            ),
            grid_color,
            1
        )

    # ========================================================
    # SAVE PREVIEW
    # ========================================================

    preview_path = os.path.join(
        output_dir,
        "mosaic_preview.png"
    )

    preview_bgr = cv2.cvtColor(
        preview_img,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        preview_path,
        preview_bgr
    )

    print(
        f"Preview saved:"
        f"\n  {preview_path}"
    )

    # ========================================================
    # BUILD BOM
    # ========================================================

    print("\nCalculating brick quantities...")

    quantities = {}

    cells = []

    for row in range(grid_h):

        row_cells = []

        for col in range(grid_w):

            brick_index = output_indices[
                row,
                col
            ]

            brick_id = brick_ids[
                brick_index
            ]

            row_cells.append(brick_id)

            quantities[brick_id] = (
                quantities.get(brick_id, 0) + 1
            )

        cells.append(row_cells)

    # ========================================================
    # TOTAL BRICKS
    # ========================================================

    total_bricks = grid_w * grid_h

    # ========================================================
    # CREATE QUANTITY DATA
    # ========================================================

    quantity_data = {}

    for brick_id, count in quantities.items():

        quantity_data[brick_id] = {

            "name":
                palette_data[brick_id]["name"],

            "count":
                count,

            "rgb":
                palette_data[brick_id]["rgb"]
        }

    # ========================================================
    # COMPLETE JSON
    # ========================================================

    output_data = {

        "project": "PhotoBricks POC",

        "image": {
            "path": image_path,
            "original_width": int(img.shape[1]),
            "original_height": int(img.shape[0])
        },

        "mosaic": {

            "width": grid_w,

            "height": grid_h,

            "total_bricks": total_bricks,

            "dithering":
                enable_dithering
        },

        "quantities":
            quantity_data,

        "cells":
            cells
    }

    # ========================================================
    # SAVE JSON
    # ========================================================

    json_path = os.path.join(
        output_dir,
        "mosaic_data.json"
    )

    with open(
        json_path,
        "w"
    ) as f:

        json.dump(
            output_data,
            f,
            indent=2
        )

    print(
        f"\nMosaic data saved:"
        f"\n  {json_path}"
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n============================================")
    print("              COMPLETE")
    print("============================================")

    print(
        f"Grid:           {grid_w} x {grid_h}"
    )

    print(
        f"Total bricks:   {total_bricks}"
    )

    print(
        f"Colors used:    {len(quantities)}"
    )

    print(
        f"Dithering:      "
        f"{'Enabled' if enable_dithering else 'Disabled'}"
    )

    print("\nOutput files:")

    print(
        f"  1. {mosaic_path}"
    )

    print(
        f"  2. {preview_path}"
    )

    print(
        f"  3. {json_path}"
    )

    print("============================================\n")

    return output_data


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "PhotoBricks Advanced "
            "Mosaic Engine POC"
        )
    )

    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    parser.add_argument(
        "--image",
        required=True,
        help=(
            "Path to input image. "
            "Example: C:\\PhotoBricks\\images\\cat.jpg"
        )
    )

    # --------------------------------------------------------
    # Palette
    # --------------------------------------------------------

    parser.add_argument(
        "--palette",
        default="palette.json",
        help="Path to brick palette JSON"
    )

    # --------------------------------------------------------
    # Grid width
    # --------------------------------------------------------

    parser.add_argument(
        "--width",
        type=int,
        default=90,
        help="Base grid width"
    )

    # --------------------------------------------------------
    # Grid height
    # --------------------------------------------------------

    parser.add_argument(
        "--height",
        type=int,
        default=60,
        help="Base grid height"
    )

    # --------------------------------------------------------
    # Disable dithering
    # --------------------------------------------------------

    parser.add_argument(
        "--no-dither",
        action="store_true",
        help="Disable Floyd-Steinberg dithering"
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    parser.add_argument(
        "--output",
        default="output",
        help="Output directory"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Normalize Windows path
    # --------------------------------------------------------

    image_path = os.path.abspath(
        os.path.expanduser(
            args.image
        )
    )

    palette_path = os.path.abspath(
        os.path.expanduser(
            args.palette
        )
    )

    # --------------------------------------------------------
    # Run engine
    # --------------------------------------------------------

    process_mosaic_dithered(

        image_path=image_path,

        palette_path=palette_path,

        grid_size=(
            args.width,
            args.height
        ),

        enable_dithering=(
            not args.no_dither
        ),

        output_dir=args.output
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()