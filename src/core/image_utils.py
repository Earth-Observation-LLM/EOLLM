"""Image processing utilities for encoding and validation.

This module provides utilities for encoding images to base64 format
for transmission to multimodal LLM APIs.
"""

import base64
from pathlib import Path
from typing import Optional


def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string.
    
    This function reads an image file and encodes it to base64 format,
    which is required by most multimodal LLM APIs for image transmission.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64-encoded string of the image
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        IOError: If image file cannot be read
        
    Example:
        >>> base64_str = encode_image_to_base64("/path/to/image.png")
        >>> # Use in API call
        >>> image_url = f"data:image/png;base64,{base64_str}"
    """
    image_file = Path(image_path)
    
    if not image_file.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if not image_file.is_file():
        raise ValueError(f"Path is not a file: {image_path}")
    
    try:
        with open(image_file, "rb") as f:
            image_data = f.read()
            return base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        raise IOError(f"Failed to read image file {image_path}: {e}")


def validate_image_path(image_path: str) -> bool:
    """Validate that an image file exists and is readable.
    
    Args:
        image_path: Path to image file to validate
        
    Returns:
        True if image exists and is readable, False otherwise
    """
    try:
        image_file = Path(image_path)
        return image_file.exists() and image_file.is_file()
    except Exception:
        return False


def get_image_mime_type(image_path: str) -> Optional[str]:
    """Detect MIME type of an image file based on extension.
    
    Args:
        image_path: Path to image file
        
    Returns:
        MIME type string (e.g., 'image/png') or None if unknown
    """
    extension_map = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp',
    }
    
    ext = Path(image_path).suffix.lower()
    return extension_map.get(ext)


def create_data_uri(image_path: str) -> str:
    """Create a data URI for an image file.
    
    This creates a complete data URI that can be directly used in
    API calls that expect image URLs.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Data URI string (e.g., "data:image/png;base64,...")
        
    Raises:
        FileNotFoundError: If image doesn't exist
        ValueError: If image type is not supported
    """
    mime_type = get_image_mime_type(image_path)
    
    if mime_type is None:
        raise ValueError(
            f"Unsupported image type for {image_path}. "
            f"Supported types: png, jpg, jpeg, gif, webp, bmp"
        )
    
    base64_str = encode_image_to_base64(image_path)
    return f"data:{mime_type};base64,{base64_str}"

