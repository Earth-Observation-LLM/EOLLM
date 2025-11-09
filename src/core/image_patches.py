"""Image patch extraction utilities for zoom tool.

This module provides functions to extract specific patches (regions) from images
for detailed analysis by the LLM during the labeling process.
"""

from pathlib import Path
from typing import Dict, Tuple
import io

from PIL import Image


# Patch name to coordinate calculator mapping
PATCH_NAMES = ['northwest', 'northeast', 'southwest', 'southeast', 'center']


def get_patch_coordinates(
    image_width: int,
    image_height: int,
    patch_name: str
) -> Tuple[int, int, int, int]:
    """Calculate pixel coordinates for a named patch.
    
    Args:
        image_width: Width of the full image in pixels
        image_height: Height of the full image in pixels
        patch_name: Name of patch (northwest, northeast, southwest, southeast, center)
        
    Returns:
        Tuple of (x1, y1, x2, y2) pixel coordinates
        
    Raises:
        ValueError: If patch_name is invalid
    """
    patch_name = patch_name.lower()
    
    if patch_name not in PATCH_NAMES:
        raise ValueError(
            f"Invalid patch name '{patch_name}'. "
            f"Must be one of: {', '.join(PATCH_NAMES)}"
        )
    
    # Calculate dimensions
    half_w = image_width // 2
    half_h = image_height // 2
    quarter_w = image_width // 4
    quarter_h = image_height // 4
    three_quarter_w = quarter_w * 3
    three_quarter_h = quarter_h * 3
    
    # Define patch coordinates
    patches = {
        'northwest': (0, 0, half_w, half_h),
        'northeast': (half_w, 0, image_width, half_h),
        'southwest': (0, half_h, half_w, image_height),
        'southeast': (half_w, half_h, image_width, image_height),
        'center': (quarter_w, quarter_h, three_quarter_w, three_quarter_h),
    }
    
    return patches[patch_name]


def extract_patch(image_path: str, patch_name: str) -> bytes:
    """Extract a specific patch from an image and return as bytes.
    
    Args:
        image_path: Path to the source image
        patch_name: Name of patch to extract
        
    Returns:
        Image bytes in PNG format
        
    Raises:
        FileNotFoundError: If image doesn't exist
        ValueError: If patch name is invalid
        IOError: If image cannot be read
    """
    # Validate image exists
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    try:
        # Open image
        with Image.open(img_path) as img:
            # Get dimensions
            width, height = img.size
            
            # Calculate patch coordinates
            x1, y1, x2, y2 = get_patch_coordinates(width, height, patch_name)
            
            # Extract patch
            patch = img.crop((x1, y1, x2, y2))
            
            # Convert to bytes
            buffer = io.BytesIO()
            patch.save(buffer, format='PNG')
            return buffer.getvalue()
            
    except Exception as e:
        raise IOError(f"Failed to extract patch from {image_path}: {e}")


def save_patch_image(patch_bytes: bytes, output_path: Path) -> Path:
    """Save patch bytes to a file.
    
    Args:
        patch_bytes: Image bytes in PNG format
        output_path: Path where to save the patch
        
    Returns:
        Path to saved file
        
    Raises:
        IOError: If save fails
    """
    try:
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write bytes to file
        with open(output_path, 'wb') as f:
            f.write(patch_bytes)
        
        return output_path
        
    except Exception as e:
        raise IOError(f"Failed to save patch to {output_path}: {e}")


def get_patch_info(image_path: str, patch_name: str) -> Dict:
    """Get information about a patch without extracting it.
    
    Args:
        image_path: Path to the source image
        patch_name: Name of patch
        
    Returns:
        Dictionary with patch metadata including coordinates and dimensions
        
    Raises:
        FileNotFoundError: If image doesn't exist
        ValueError: If patch name is invalid
    """
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    try:
        with Image.open(img_path) as img:
            width, height = img.size
            x1, y1, x2, y2 = get_patch_coordinates(width, height, patch_name)
            
            return {
                'patch_name': patch_name,
                'source_image': str(image_path),
                'source_dimensions': (width, height),
                'coordinates': (x1, y1, x2, y2),
                'patch_dimensions': (x2 - x1, y2 - y1),
                'coverage_percent': ((x2 - x1) * (y2 - y1) / (width * height)) * 100
            }
    except Exception as e:
        raise IOError(f"Failed to get patch info: {e}")


def extract_and_save_patch(
    image_path: str,
    patch_name: str,
    output_dir: Path,
    filename: str
) -> Tuple[Path, Dict]:
    """Extract a patch and save it in one operation.
    
    Convenience function that combines extraction and saving.
    
    Args:
        image_path: Path to source image
        patch_name: Name of patch to extract
        output_dir: Directory where to save patch
        filename: Filename for the saved patch (without extension)
        
    Returns:
        Tuple of (saved_path, patch_info)
    """
    # Extract patch
    patch_bytes = extract_patch(image_path, patch_name)
    
    # Save patch
    output_path = output_dir / f"{filename}.png"
    saved_path = save_patch_image(patch_bytes, output_path)
    
    # Get info
    patch_info = get_patch_info(image_path, patch_name)
    
    return saved_path, patch_info

