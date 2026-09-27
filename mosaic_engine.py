import cv2
import numpy as np
import json
import argparse
from skimage import color
import os
import requests

def download_sample_image(save_path):
    print("Downloading sample image...")
    url = "https://images.unsplash.com/photo-1543852786-1cf6624b9987?q=80&w=800&auto=format&fit=crop"
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)
    print(f"Sample image saved to {save_path}")

def get_target_dimensions(img_shape, base_grid_size):
    """
    Detects whether the image is Portrait or Landscape and aligns
    the base grid dimensions automatically to prevent subject cropping.
    """
    h, w = img_shape[:2]
    img_aspect = w / h
    dim1, dim2 = base_grid_size
    
    if img_aspect >= 1.0:
        # Landscape orientation
        grid_w = max(dim1, dim2)
        grid_h = min(dim1, dim2)
    else:
        # Portrait orientation
        grid_w = min(dim1, dim2)
        grid_h = max(dim1, dim2)
        
    return grid_w, grid_h

def crop_center_to_aspect(img, target_aspect):
    """
    Minimal smart cropping that keeps the center subject in frame.
    """
    h, w = img.shape[:2]
    image_aspect = w / h
    
    if image_aspect > target_aspect:
        new_w = int(h * target_aspect)
        offset = (w - new_w) // 2
        return img[:, offset:offset+new_w]
    else:
        new_h = int(w / target_aspect)
        offset = (h - new_h) // 2
        return img[offset:offset+new_h, :]

def match_closest_color_cie2000(pixel_lab, palette_lab):
    """
    Calculates perceptual distance in CIE2000 space for a single pixel against the palette.
    """
    # tile pixel to match palette array shape for vector calculation
    pixel_tile = np.tile(pixel_lab, (palette_lab.shape[0], 1))
    distances = color.deltaE_ciede2000(pixel_tile, palette_lab)
    return np.argmin(distances)

def process_mosaic_dithered(image_path, palette_path, grid_size=(90, 60), enable_dithering=True):
    print(f"Loading image: {image_path}...")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 1. Determine Grid Orientation
    grid_w, grid_h = get_target_dimensions(img_rgb.shape, grid_size)
    target_aspect = grid_w / grid_h
    print(f"Resolved Grid Dimensions: {grid_w}x{grid_h} (Aspect Ratio: {target_aspect:.2f})")
    
    # 2. Crop & Resize
    img_cropped = crop_center_to_aspect(img_rgb, target_aspect)
    img_resized = cv2.resize(img_cropped, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
    
    # 3. Load Palette Data
    with open(palette_path, 'r') as f:
        palette_data = json.load(f)
        
    brick_ids = list(palette_data.keys())
    palette_rgb = np.array([palette_data[bid]['rgb'] for bid in brick_ids], dtype=np.float32)
    
    # Prepare floating-point image working buffer in [0, 255]
    img_float = img_resized.astype(np.float32)
    
    output_indices = np.zeros((grid_h, grid_w), dtype=int)
    
    print("Quantizing and processing grid (CIE2000 + Error Diffusion)...")
    for y in range(grid_h):
        for x in range(grid_w):
            old_pixel_rgb = np.clip(img_float[y, x], 0, 255)
            
            # Convert single pixel and full palette to LAB for CIE2000 distance
            pixel_lab = color.rgb2lab(old_pixel_rgb.reshape(1, 1, 3) / 255.0).reshape(3)
            palette_lab = color.rgb2lab(palette_rgb.reshape(1, -1, 3) / 255.0).reshape(-1, 3)
            
            best_idx = match_closest_color_cie2000(pixel_lab, palette_lab)
            output_indices[y, x] = best_idx
            
            new_pixel_rgb = palette_rgb[best_idx]
            
            if enable_dithering:
                # Calculate color error to distribute to neighboring pixels
                quant_error = old_pixel_rgb - new_pixel_rgb
                
                # Floyd-Steinberg Error Diffusion distribution coefficients
                if x + 1 < grid_w:
                    img_float[y, x + 1] += quant_error * (7 / 16)
                if y + 1 < grid_h:
                    if x > 0:
                        img_float[y + 1, x - 1] += quant_error * (3 / 16)
                    img_float[y + 1, x] += quant_error * (5 / 16)
                    if x + 1 < grid_w:
                        img_float[y + 1, x + 1] += quant_error * (1 / 16)

    # 4. Generate Visual Output
    mosaic_rgb = palette_rgb[output_indices].astype(np.uint8)
    
    scale_factor = 10
    preview_img = cv2.resize(mosaic_rgb, (grid_w * scale_factor, grid_h * scale_factor), interpolation=cv2.INTER_NEAREST)
    
    # Draw grid overlay
    for i in range(grid_w):
        cv2.line(preview_img, (i * scale_factor, 0), (i * scale_factor, grid_h * scale_factor), (30, 30, 30), 1)
    for j in range(grid_h):
        cv2.line(preview_img, (0, j * scale_factor), (grid_w * scale_factor, j * scale_factor), (30, 30, 30), 1)
        
    preview_bgr = cv2.cvtColor(preview_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("mosaic_preview.png", preview_bgr)
    
    # 5. Build BOM Data
    quantities = {}
    total_bricks = 0
    cells = []
    
    for row in range(grid_h):
        row_cells = []
        for col in range(grid_w):
            bid = brick_ids[output_indices[row, col]]
            row_cells.append(bid)
            quantities[bid] = quantities.get(bid, 0) + 1
            total_bricks += 1
        cells.append(row_cells)
        
    output_data = {
        "width": grid_w,
        "height": grid_h,
        "total_bricks": total_bricks,
        "quantities": {bid: {"name": palette_data[bid]["name"], "count": count} for bid, count in quantities.items()},
        "cells": cells
    }
    
    with open("mosaic_data.json", 'w') as f:
        json.dump(output_data, f, indent=2)
        
    print("\nMosaic engine execution complete! Generated 'mosaic_preview.png' and 'mosaic_data.json'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PhotoBricks Advanced POC Mosaic Engine")
    parser.add_argument("--image", default="sample.jpg", help="Path to input image")
    parser.add_argument("--palette", default="palette.json", help="Path to palette JSON")
    parser.add_argument("--width", type=int, default=90, help="Grid dimension 1")
    parser.add_argument("--height", type=int, default=60, help="Grid dimension 2")
    parser.add_argument("--no-dither", action="store_true", help="Disable Floyd-Steinberg dithering")
    
    args = parser.parse_args()
    
    if args.image == "sample.jpg" and not os.path.exists("sample.jpg"):
        download_sample_image("sample.jpg")
        
    process_mosaic_dithered(
        args.image, 
        args.palette, 
        grid_size=(args.width, args.height),
        enable_dithering=not args.no_dither
    )