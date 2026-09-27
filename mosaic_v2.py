import cv2
import numpy as np
import json
import argparse
import os
from skimage import color


# ============================================================
# PHOTOBRICKS MOSAIC ENGINE V3
# ============================================================
#
# Main idea:
#
#   Photo
#     ↓
#   Crop
#     ↓
#   Resize to brick grid
#     ↓
#   RGB → LAB
#     ↓
#   CIEDE2000 color candidates
#     ↓
#   Spatial optimization
#     ↓
#   Edge preservation
#     ↓
#   Region consistency
#     ↓
#   Final brick mosaic
#
# No global dithering by default.
# ============================================================


# ============================================================
# DEFAULT SETTINGS
# ============================================================

DEFAULT_WIDTH = 90
DEFAULT_HEIGHT = 60

DEFAULT_ITERATIONS = 4
DEFAULT_TOP_K = 4

# Color accuracy
COLOR_WEIGHT = 1.0

# Neighbor consistency
SMOOTHNESS_WEIGHT = 0.75

# Edge protection
EDGE_THRESHOLD = 0.18

# Penalty for changing from similar neighboring colors
SWITCH_PENALTY = 1.2

# Only use dithering if explicitly requested
DEFAULT_DITHER = False


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(image_path):

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Could not read image:\n{image_path}"
        )

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return image_rgb


# ============================================================
# DETERMINE GRID ORIENTATION
# ============================================================

def get_target_dimensions(
    image_shape,
    base_grid_size
):

    h, w = image_shape[:2]

    image_aspect = w / h

    dim1, dim2 = base_grid_size

    if image_aspect >= 1.0:

        grid_w = max(dim1, dim2)
        grid_h = min(dim1, dim2)

    else:

        grid_w = min(dim1, dim2)
        grid_h = max(dim1, dim2)

    return grid_w, grid_h


# ============================================================
# CROP TO TARGET ASPECT RATIO
# ============================================================

def crop_center_to_aspect(
    image,
    target_aspect
):

    h, w = image.shape[:2]

    image_aspect = w / h

    if abs(image_aspect - target_aspect) < 0.0001:

        return image

    # Image is too wide
    if image_aspect > target_aspect:

        new_width = int(
            h * target_aspect
        )

        offset = (
            w - new_width
        ) // 2

        return image[
            :,
            offset:offset + new_width
        ]

    # Image is too tall
    else:

        new_height = int(
            w / target_aspect
        )

        offset = (
            h - new_height
        ) // 2

        return image[
            offset:offset + new_height,
            :
        ]


# ============================================================
# LOAD PALETTE
# ============================================================

def load_palette(
    palette_path
):

    if not os.path.exists(palette_path):
        raise FileNotFoundError(
            f"Palette not found:\n{palette_path}"
        )

    with open(
        palette_path,
        "r",
        encoding="utf-8"
    ) as f:

        palette_data = json.load(f)

    if len(palette_data) == 0:

        raise ValueError(
            "Palette contains no colors."
        )

    brick_ids = list(
        palette_data.keys()
    )

    palette_rgb = np.array(
        [
            palette_data[brick_id]["rgb"]
            for brick_id in brick_ids
        ],
        dtype=np.float32
    )

    # Convert palette RGB → LAB once.
    # This is much faster than recalculating it
    # for every single brick cell.
    palette_lab = color.rgb2lab(
        (
            palette_rgb / 255.0
        ).reshape(1, -1, 3)
    ).reshape(-1, 3)

    return (
        palette_data,
        brick_ids,
        palette_rgb,
        palette_lab
    )


# ============================================================
# CALCULATE EDGE MAP
# ============================================================

def calculate_edge_map(
    image_rgb,
    grid_w,
    grid_h
):

    gray = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY
    )

    # Slight blur before edge detection.
    # This prevents tiny photographic noise
    # from becoming brick-level edges.
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    magnitude = cv2.magnitude(
        gx,
        gy
    )

    magnitude = cv2.normalize(
        magnitude,
        None,
        0.0,
        1.0,
        cv2.NORM_MINMAX
    )

    # Resize edge map to brick resolution.
    edge_small = cv2.resize(
        magnitude,
        (grid_w, grid_h),
        interpolation=cv2.INTER_AREA
    )

    return edge_small


# ============================================================
# CALCULATE INITIAL COLOR ASSIGNMENT
# ============================================================

def initial_assignment(
    image_lab,
    palette_lab,
    top_k
):

    grid_h, grid_w = (
        image_lab.shape[:2]
    )

    pixels = image_lab.reshape(
        -1,
        3
    )

    # --------------------------------------------------------
    # CIEDE2000 distance
    #
    # pixels:
    #     N x 3
    #
    # palette:
    #     K x 3
    #
    # result:
    #     N x K
    # --------------------------------------------------------

    pixel_expanded = pixels[:, None, :]

    palette_expanded = palette_lab[
        None, :, :
    ]

    distances = color.deltaE_ciede2000(
        pixel_expanded,
        palette_expanded
    )

    distances = distances.reshape(
        grid_h,
        grid_w,
        -1
    )

    # Top K colors for every cell
    k = min(
        top_k,
        palette_lab.shape[0]
    )

    candidates = np.argsort(
        distances,
        axis=2
    )[:, :, :k]

    # Best initial color
    initial_indices = candidates[
        :, :, 0
    ]

    return (
        initial_indices,
        candidates,
        distances
    )


# ============================================================
# COLOR DISTANCE BETWEEN TWO PALETTE COLORS
# ============================================================

def palette_distance(
    palette_lab,
    index_a,
    index_b
):

    return np.linalg.norm(
        palette_lab[index_a]
        -
        palette_lab[index_b]
    )


# ============================================================
# GET NEIGHBORS
# ============================================================

def get_neighbors(
    indices,
    x,
    y
):

    h, w = indices.shape

    neighbors = []

    if y > 0:
        neighbors.append(
            indices[y - 1, x]
        )

    if y < h - 1:
        neighbors.append(
            indices[y + 1, x]
        )

    if x > 0:
        neighbors.append(
            indices[y, x - 1]
        )

    if x < w - 1:
        neighbors.append(
            indices[y, x + 1]
        )

    return neighbors


# ============================================================
# SPATIAL OPTIMIZATION
# ============================================================

def optimize_mosaic(
    indices,
    candidates,
    distances,
    palette_lab,
    edge_map,
    iterations
):

    h, w = indices.shape

    current = indices.copy()

    total_cells = h * w

    print(
        f"\nOptimizing "
        f"{total_cells} brick positions..."
    )

    for iteration in range(
        iterations
    ):

        changed = 0

        # Alternate scan direction.
        # This reduces directional bias.
        if iteration % 2 == 0:

            y_range = range(h)
            x_range = range(w)

        else:

            y_range = range(
                h - 1,
                -1,
                -1
            )

            x_range = range(
                w - 1,
                -1,
                -1
            )

        for y in y_range:

            for x in x_range:

                current_index = int(
                    current[y, x]
                )

                candidate_list = candidates[
                    y, x
                ]

                # ------------------------------------------------
                # Determine whether this is an edge.
                #
                # At edges:
                #     prioritize original color
                #
                # In flat areas:
                #     prioritize neighboring consistency
                # ------------------------------------------------

                edge_strength = float(
                    edge_map[y, x]
                )

                if edge_strength > EDGE_THRESHOLD:

                    smooth_weight = (
                        SMOOTHNESS_WEIGHT
                        * 0.25
                    )

                else:

                    smooth_weight = (
                        SMOOTHNESS_WEIGHT
                    )

                neighbors = get_neighbors(
                    current,
                    x,
                    y
                )

                best_index = current_index

                best_cost = float(
                    "inf"
                )

                # ------------------------------------------------
                # Test top K candidate colors
                # ------------------------------------------------

                for candidate in candidate_list:

                    candidate = int(
                        candidate
                    )

                    # --------------------------------------------
                    # 1. COLOR COST
                    # --------------------------------------------

                    color_cost = (
                        float(
                            distances[
                                y,
                                x,
                                candidate
                            ]
                        )
                        *
                        COLOR_WEIGHT
                    )

                    # --------------------------------------------
                    # 2. NEIGHBOR CONSISTENCY
                    # --------------------------------------------

                    smooth_cost = 0.0

                    for neighbor in neighbors:

                        neighbor = int(
                            neighbor
                        )

                        d = palette_distance(
                            palette_lab,
                            candidate,
                            neighbor
                        )

                        # Normalize perceptual difference.
                        #
                        # Nearby colors:
                        #     low penalty
                        #
                        # Very different colors:
                        #     high penalty
                        #
                        smooth_cost += min(
                            d / 20.0,
                            3.0
                        )

                    smooth_cost *= (
                        smooth_weight
                    )

                    # --------------------------------------------
                    # 3. SWITCHING PENALTY
                    # --------------------------------------------

                    switch_cost = 0.0

                    if (
                        candidate
                        !=
                        current_index
                    ):

                        # Only penalize switching when
                        # the candidate isn't dramatically
                        # better in color.
                        current_color_cost = float(
                            distances[
                                y,
                                x,
                                current_index
                            ]
                        )

                        improvement = (
                            current_color_cost
                            -
                            float(
                                distances[
                                    y,
                                    x,
                                    candidate
                                ]
                            )
                        )

                        if improvement < 3.0:

                            switch_cost = (
                                SWITCH_PENALTY
                            )

                    # --------------------------------------------
                    # TOTAL COST
                    # --------------------------------------------

                    total_cost = (
                        color_cost
                        +
                        smooth_cost
                        +
                        switch_cost
                    )

                    if total_cost < best_cost:

                        best_cost = total_cost

                        best_index = candidate

                # ------------------------------------------------
                # Apply best candidate
                # ------------------------------------------------

                if (
                    best_index
                    !=
                    current_index
                ):

                    current[y, x] = (
                        best_index
                    )

                    changed += 1

        percentage = (
            changed
            /
            total_cells
            *
            100.0
        )

        print(
            f"Iteration "
            f"{iteration + 1}/{iterations} "
            f"→ {changed} cells changed "
            f"({percentage:.2f}%)"
        )

        # Stop early if almost nothing changed.
        if changed < max(
            2,
            int(total_cells * 0.001)
        ):

            print(
                "Optimization converged."
            )

            break

    return current


# ============================================================
# OPTIONAL VERY WEAK SELECTIVE DITHERING
# ============================================================

def selective_dithering(
    grid_rgb,
    indices,
    palette_rgb,
    edge_map
):

    """
    This is intentionally much weaker than the original
    Floyd-Steinberg implementation.

    It is ONLY applied to low-edge cells.

    Do not use this for the first comparison.
    """

    h, w = indices.shape

    working = grid_rgb.astype(
        np.float32
    ).copy()

    result = indices.copy()

    # Convert palette once
    palette_lab = color.rgb2lab(
        (
            palette_rgb / 255.0
        ).reshape(1, -1, 3)
    ).reshape(-1, 3)

    for y in range(h):

        for x in range(w):

            # Strong edges should remain clean.
            if edge_map[y, x] > 0.20:

                continue

            old_pixel = np.clip(
                working[y, x],
                0,
                255
            )

            pixel_lab = color.rgb2lab(
                old_pixel.reshape(
                    1, 1, 3
                ) / 255.0
            ).reshape(3)

            pixel_tile = np.tile(
                pixel_lab,
                (
                    palette_lab.shape[0],
                    1
                )
            )

            distances = (
                color.deltaE_ciede2000(
                    pixel_tile,
                    palette_lab
                )
            )

            best = int(
                np.argmin(distances)
            )

            result[y, x] = best

            new_pixel = palette_rgb[
                best
            ]

            error = (
                old_pixel
                -
                new_pixel
            )

            # Weak diffusion only.
            if x + 1 < w:

                working[
                    y,
                    x + 1
                ] += (
                    error * 0.20
                )

            if y + 1 < h:

                working[
                    y + 1,
                    x
                ] += (
                    error * 0.15
                )

    return result


# ============================================================
# CREATE MOSAIC
# ============================================================

def create_mosaic(
    indices,
    palette_rgb
):

    return palette_rgb[
        indices
    ].astype(
        np.uint8
    )


# ============================================================
# CREATE GRID PREVIEW
# ============================================================

def create_grid_preview(
    mosaic_rgb,
    cell_size=10
):

    h, w = mosaic_rgb.shape[:2]

    preview = cv2.resize(
        mosaic_rgb,
        (
            w * cell_size,
            h * cell_size
        ),
        interpolation=cv2.INTER_NEAREST
    )

    # Dark grid
    line_color = (
        35,
        35,
        35
    )

    # Vertical lines
    for x in range(
        0,
        w * cell_size + 1,
        cell_size
    ):

        cv2.line(
            preview,
            (x, 0),
            (
                x,
                h * cell_size
            ),
            line_color,
            1
        )

    # Horizontal lines
    for y in range(
        0,
        h * cell_size + 1,
        cell_size
    ):

        cv2.line(
            preview,
            (0, y),
            (
                w * cell_size,
                y
            ),
            line_color,
            1
        )

    return preview


# ============================================================
# SAVE RGB IMAGE
# ============================================================

def save_rgb_image(
    path,
    image_rgb
):

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        path,
        image_bgr
    )


# ============================================================
# BUILD BRICK QUANTITY LIST
# ============================================================

def build_bom(
    indices,
    brick_ids,
    palette_data
):

    h, w = indices.shape

    quantities = {}

    cells = []

    for y in range(h):

        row = []

        for x in range(w):

            index = int(
                indices[y, x]
            )

            brick_id = brick_ids[
                index
            ]

            row.append(
                brick_id
            )

            quantities[
                brick_id
            ] = (
                quantities.get(
                    brick_id,
                    0
                )
                +
                1
            )

        cells.append(
            row
        )

    quantity_data = {}

    for brick_id, count in quantities.items():

        quantity_data[
            brick_id
        ] = {

            "name":
                palette_data[
                    brick_id
                ]["name"],

            "rgb":
                palette_data[
                    brick_id
                ]["rgb"],

            "count":
                count
        }

    return (
        quantities,
        quantity_data,
        cells
    )


# ============================================================
# PROCESS MOSAIC
# ============================================================

def process_mosaic(
    image_path,
    palette_path,
    width,
    height,
    output_dir,
    iterations,
    top_k,
    enable_dither
):

    print()
    print("=" * 65)
    print("             PHOTOBRICKS MOSAIC ENGINE V3")
    print("=" * 65)

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 1. LOAD IMAGE
    # --------------------------------------------------------

    print("\n[1] Loading image...")

    original = load_image(
        image_path
    )

    print(
        f"Original size: "
        f"{original.shape[1]} x "
        f"{original.shape[0]}"
    )

    # --------------------------------------------------------
    # 2. DETERMINE GRID
    # --------------------------------------------------------

    print(
        "\n[2] Determining grid..."
    )

    grid_w, grid_h = (
        get_target_dimensions(
            original.shape,
            (width, height)
        )
    )

    print(
        f"Brick grid: "
        f"{grid_w} x {grid_h}"
    )

    target_aspect = (
        grid_w / grid_h
    )

    # --------------------------------------------------------
    # 3. CROP
    # --------------------------------------------------------

    print(
        "\n[3] Cropping..."
    )

    cropped = crop_center_to_aspect(
        original,
        target_aspect
    )

    save_rgb_image(
        os.path.join(
            output_dir,
            "cropped.png"
        ),
        cropped
    )

    # --------------------------------------------------------
    # 4. RESIZE
    # --------------------------------------------------------

    print(
        "\n[4] Creating brick-resolution image..."
    )

    grid_rgb = cv2.resize(
        cropped,
        (grid_w, grid_h),
        interpolation=cv2.INTER_AREA
    )

    save_rgb_image(
        os.path.join(
            output_dir,
            "grid_source.png"
        ),
        grid_rgb
    )

    # --------------------------------------------------------
    # 5. EDGE MAP
    # --------------------------------------------------------

    print(
        "\n[5] Calculating edge structure..."
    )

    edge_map = calculate_edge_map(
        cropped,
        grid_w,
        grid_h
    )

    # --------------------------------------------------------
    # 6. LOAD PALETTE
    # --------------------------------------------------------

    print(
        "\n[6] Loading brick palette..."
    )

    (
        palette_data,
        brick_ids,
        palette_rgb,
        palette_lab
    ) = load_palette(
        palette_path
    )

    print(
        f"Palette colors: "
        f"{len(brick_ids)}"
    )

    # --------------------------------------------------------
    # 7. RGB → LAB
    # --------------------------------------------------------

    print(
        "\n[7] Converting image to LAB..."
    )

    image_lab = color.rgb2lab(
        grid_rgb.astype(
            np.float32
        )
        / 255.0
    )

    # --------------------------------------------------------
    # 8. INITIAL MATCHING
    # --------------------------------------------------------

    print(
        "\n[8] Finding perceptual color candidates..."
    )

    (
        initial_indices,
        candidates,
        distances
    ) = initial_assignment(
        image_lab,
        palette_lab,
        top_k
    )

    # --------------------------------------------------------
    # 9. SPATIAL OPTIMIZATION
    # --------------------------------------------------------

    print(
        "\n[9] Spatial optimization..."
    )

    optimized_indices = (
        optimize_mosaic(
            initial_indices,
            candidates,
            distances,
            palette_lab,
            edge_map,
            iterations
        )
    )

    # --------------------------------------------------------
    # 10. OPTIONAL DITHER
    # --------------------------------------------------------

    if enable_dither:

        print(
            "\n[10] Applying selective dithering..."
        )

        final_indices = (
            selective_dithering(
                grid_rgb,
                optimized_indices,
                palette_rgb,
                edge_map
            )
        )

    else:

        print(
            "\n[10] Dithering disabled."
        )

        final_indices = (
            optimized_indices
        )

    # --------------------------------------------------------
    # 11. FINAL MOSAIC
    # --------------------------------------------------------

    print(
        "\n[11] Rendering mosaic..."
    )

    mosaic_rgb = create_mosaic(
        final_indices,
        palette_rgb
    )

    save_rgb_image(
        os.path.join(
            output_dir,
            "mosaic.png"
        ),
        mosaic_rgb
    )

    # --------------------------------------------------------
    # 12. GRID PREVIEW
    # --------------------------------------------------------

    print(
        "\n[12] Creating grid preview..."
    )

    preview = create_grid_preview(
        mosaic_rgb,
        cell_size=10
    )

    save_rgb_image(
        os.path.join(
            output_dir,
            "mosaic_preview.png"
        ),
        preview
    )

    # --------------------------------------------------------
    # 13. BOM
    # --------------------------------------------------------

    print(
        "\n[13] Creating brick quantity list..."
    )

    (
        quantities,
        quantity_data,
        cells
    ) = build_bom(
        final_indices,
        brick_ids,
        palette_data
    )

    total_bricks = (
        grid_w * grid_h
    )

    # --------------------------------------------------------
    # 14. JSON OUTPUT
    # --------------------------------------------------------

    output_data = {

        "engine": {
            "name":
                "PhotoBricks Mosaic Engine V3",

            "grid_width":
                grid_w,

            "grid_height":
                grid_h,

            "total_bricks":
                total_bricks,

            "palette_colors":
                len(brick_ids),

            "top_k":
                top_k,

            "optimization_iterations":
                iterations,

            "dithering":
                enable_dither
        },

        "source": {

            "filename":
                os.path.basename(
                    image_path
                ),

            "original_width":
                int(
                    original.shape[1]
                ),

            "original_height":
                int(
                    original.shape[0]
                )
        },

        "quantities":
            quantity_data,

        "cells":
            cells
    }

    json_path = os.path.join(
        output_dir,
        "mosaic_data.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output_data,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("                    COMPLETE")
    print("=" * 65)

    print(
        f"Grid:              {grid_w} x {grid_h}"
    )

    print(
        f"Total bricks:      {total_bricks}"
    )

    print(
        f"Palette colors:    {len(brick_ids)}"
    )

    print(
        f"Optimization:      {iterations} iterations"
    )

    print(
        f"Top candidates:    {top_k}"
    )

    print(
        f"Dithering:         "
        f"{'ON' if enable_dither else 'OFF'}"
    )

    print("\nFiles generated:")

    print(
        "  cropped.png"
    )

    print(
        "  grid_source.png"
    )

    print(
        "  mosaic.png"
    )

    print(
        "  mosaic_preview.png"
    )

    print(
        "  mosaic_data.json"
    )

    print("=" * 65)


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "PhotoBricks Mosaic Engine V3"
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image"
    )

    parser.add_argument(
        "--palette",
        default="palette.json",
        help="Path to palette JSON"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help="Grid width"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help="Grid height"
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help="Spatial optimization iterations"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of candidate colors"
    )

    parser.add_argument(
        "--output",
        default="output_v3",
        help="Output directory"
    )

    parser.add_argument(
        "--dither",
        action="store_true",
        help="Enable weak selective dithering"
    )

    args = parser.parse_args()

    image_path = os.path.abspath(
        args.image
    )

    palette_path = os.path.abspath(
        args.palette
    )

    process_mosaic(

        image_path=image_path,

        palette_path=palette_path,

        width=args.width,

        height=args.height,

        output_dir=args.output,

        iterations=args.iterations,

        top_k=args.top_k,

        enable_dither=args.dither
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()