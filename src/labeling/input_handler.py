"""Input handling for labeling pipeline.

This module manages input structure JSON files that define image pairs
for labeling, with comprehensive validation and collision prevention.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Iterator, Optional
from dataclasses import dataclass, field


@dataclass
class ImagePair:
    """Represents a pair of images for labeling.
    
    Attributes:
        id: Unique identifier for this pair
        satellite: Path to satellite image
        street_view: Path to street view image
        output_json: Template path for output JSON (with variables)
        metadata: Additional metadata about the pair
    """
    
    id: str
    satellite: str
    street_view: str
    output_json: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def resolve_output_path(
        self,
        timestamp: Optional[str] = None,
        hash_value: Optional[str] = None
    ) -> Path:
        """Resolve output path with template variables.
        
        Args:
            timestamp: Timestamp string (format: YYYYMMDD_HHMMSS)
            hash_value: Hash value for uniqueness
            
        Returns:
            Resolved output path
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if hash_value is None:
            # Generate hash from input paths
            hash_input = f"{self.satellite}:{self.street_view}:{timestamp}"
            hash_value = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
        
        # Get source name (satellite image filename without extension)
        source_name = Path(self.satellite).stem
        
        # Replace template variables
        output_str = self.output_json
        output_str = output_str.replace('{timestamp}', timestamp)
        output_str = output_str.replace('{hash}', hash_value)
        output_str = output_str.replace('{source_name}', source_name)
        
        return Path(output_str)


class InputHandler:
    """Manages input structure for labeling pipeline.
    
    This class loads and validates input structure JSON files,
    provides access to image pairs, and handles output path resolution
    with collision prevention.
    
    Attributes:
        input_file: Path to input structure JSON
        pairs: Dictionary of image pairs by ID
        project_root: Project root directory for resolving relative paths
    """
    
    def __init__(self, input_file: str, project_root: Optional[Path] = None):
        """Initialize input handler.
        
        Args:
            input_file: Path to input structure JSON file
            project_root: Optional project root for resolving relative paths
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If input structure is invalid
        """
        self.input_file = Path(input_file)
        self.project_root = project_root or Path.cwd()
        self.pairs: Dict[str, ImagePair] = {}
        
        if not self.input_file.exists():
            raise FileNotFoundError(
                f"Input structure file not found: {input_file}"
            )
        
        self._load_and_validate()
    
    def _load_and_validate(self) -> None:
        """Load and validate input structure from JSON file.
        
        Raises:
            ValueError: If structure is invalid or images don't exist
        """
        # Load JSON
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in input file: {e}")
        
        # Validate structure
        if not isinstance(data, dict):
            raise ValueError("Input structure must be a JSON object")
        
        if 'pairs' not in data:
            raise ValueError("Input structure must contain 'pairs' key")
        
        if not isinstance(data['pairs'], list):
            raise ValueError("'pairs' must be a list")
        
        # Load pairs
        for i, pair_data in enumerate(data['pairs']):
            try:
                pair = self._parse_pair(pair_data, i)
                
                # Validate images exist
                self._validate_image_paths(pair)
                
                # Check for duplicate IDs
                if pair.id in self.pairs:
                    raise ValueError(f"Duplicate pair ID: {pair.id}")
                
                self.pairs[pair.id] = pair
                
            except Exception as e:
                raise ValueError(f"Error loading pair {i}: {e}")
        
        if not self.pairs:
            raise ValueError("No valid pairs found in input structure")
    
    def _parse_pair(self, pair_data: Dict[str, Any], index: int) -> ImagePair:
        """Parse a pair from JSON data.
        
        Args:
            pair_data: Pair dictionary from JSON
            index: Index in pairs list (for error messages)
            
        Returns:
            ImagePair instance
            
        Raises:
            ValueError: If pair data is invalid
        """
        # Validate required fields
        required_fields = ['id', 'satellite', 'street_view', 'output_json']
        for field in required_fields:
            if field not in pair_data:
                raise ValueError(f"Pair {index} missing required field: {field}")
        
        # Resolve paths
        satellite = self._resolve_path(pair_data['satellite'])
        street_view = self._resolve_path(pair_data['street_view'])
        
        # Get metadata
        metadata = pair_data.get('metadata', {})
        
        return ImagePair(
            id=pair_data['id'],
            satellite=str(satellite),
            street_view=str(street_view),
            output_json=pair_data['output_json'],
            metadata=metadata
        )
    
    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to project root.
        
        Args:
            path: Path string (absolute or relative)
            
        Returns:
            Absolute Path object
        """
        p = Path(path)
        
        if p.is_absolute():
            return p
        
        return (self.project_root / p).resolve()
    
    def _validate_image_paths(self, pair: ImagePair) -> None:
        """Validate that image paths exist.
        
        Args:
            pair: ImagePair to validate
            
        Raises:
            FileNotFoundError: If images don't exist
        """
        satellite_path = Path(pair.satellite)
        if not satellite_path.exists():
            raise FileNotFoundError(
                f"Satellite image not found: {pair.satellite}"
            )
        
        if not satellite_path.is_file():
            raise ValueError(
                f"Satellite path is not a file: {pair.satellite}"
            )
        
        street_path = Path(pair.street_view)
        if not street_path.exists():
            raise FileNotFoundError(
                f"Street view image not found: {pair.street_view}"
            )
        
        if not street_path.is_file():
            raise ValueError(
                f"Street view path is not a file: {pair.street_view}"
            )
    
    def get_pair(self, pair_id: str) -> ImagePair:
        """Get a specific pair by ID.
        
        Args:
            pair_id: Pair identifier
            
        Returns:
            ImagePair instance
            
        Raises:
            KeyError: If pair ID not found
        """
        if pair_id not in self.pairs:
            raise KeyError(f"Pair not found: {pair_id}")
        
        return self.pairs[pair_id]
    
    def iter_pairs(self) -> Iterator[ImagePair]:
        """Iterate over all pairs.
        
        Yields:
            ImagePair instances
        """
        for pair in self.pairs.values():
            yield pair
    
    def list_pair_ids(self) -> List[str]:
        """Get list of all pair IDs.
        
        Returns:
            List of pair ID strings
        """
        return list(self.pairs.keys())
    
    def get_output_path(
        self,
        pair: ImagePair,
        ensure_unique: bool = True
    ) -> Path:
        """Get resolved output path for a pair with collision prevention.
        
        Args:
            pair: ImagePair instance
            ensure_unique: If True, append version suffix if file exists
            
        Returns:
            Unique output path
        """
        # Generate timestamp and hash
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{pair.satellite}:{pair.street_view}:{timestamp}"
        hash_value = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
        
        # Resolve base path
        output_path = pair.resolve_output_path(timestamp, hash_value)
        
        if not ensure_unique:
            return output_path
        
        # Check for collisions and append version suffix if needed
        if not output_path.exists():
            return output_path
        
        # File exists, find unique version
        base_dir = output_path.parent
        base_name = output_path.stem
        extension = output_path.suffix
        
        version = 1
        while True:
            versioned_path = base_dir / f"{base_name}_v{version}{extension}"
            if not versioned_path.exists():
                return versioned_path
            version += 1
            
            # Safety limit
            if version > 1000:
                raise RuntimeError(
                    f"Could not find unique filename after 1000 attempts for {base_name}"
                )
    
    def validate_output_directory(self, pair: ImagePair) -> bool:
        """Check if output directory is writable.
        
        Args:
            pair: ImagePair to check
            
        Returns:
            True if directory is writable, False otherwise
        """
        output_path = pair.resolve_output_path()
        output_dir = output_path.parent
        
        try:
            # Create directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Test writability with a temporary file
            test_file = output_dir / '.write_test'
            test_file.touch()
            test_file.unlink()
            
            return True
        except Exception:
            return False
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of input structure.
        
        Returns:
            Dictionary with summary information
        """
        return {
            'input_file': str(self.input_file),
            'total_pairs': len(self.pairs),
            'pair_ids': self.list_pair_ids(),
            'project_root': str(self.project_root)
        }
    
    def __len__(self) -> int:
        """Get number of pairs."""
        return len(self.pairs)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"InputHandler(pairs={len(self.pairs)}, file='{self.input_file}')"

