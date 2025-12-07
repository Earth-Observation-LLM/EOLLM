# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

EOLLM (Earth Observation with Large Language Models) is a research pipeline for automated urban scene analysis using multimodal AI. The system processes satellite and street view image pairs to generate detailed annotations and Visual Question Answering (VQA) datasets for urban planning, sustainability assessment, and geospatial analysis.

**Key Features:**
- Multi-provider LLM support (Groq Cloud API and Ollama local)
- Jinja2-based prompt templating system
- Research-grade logging with per-call LLM statistics
- Modular pipeline architecture (labeling and VQA generation)
- Type-safe YAML configuration with Pydantic validation
- Collision-resistant output using SHA256-based filenames
- Zoom tool for detailed image patch extraction

## Development Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment (create .env file)
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
```

### Running Pipelines

**Labeling Pipeline:**
```bash
# Process all pairs in input_structure.json
python run_labeling.py

# Process a specific pair
python run_labeling.py --pair-id aliebykoy_otogar_2015

# Use custom input file
python run_labeling.py --input my_inputs.json

# Use Ollama instead of Groq
python run_labeling.py --provider ollama

# Use custom config
python run_labeling.py --config config/settings.yaml
```

**VQA Generation Pipeline:**
```bash
# Generate VQA from labeling result
python run_vqa.py --from-labeling logs/labeling/results/pair_001_labeling.json

# Batch process all labeling results
python run_vqa.py --batch "logs/labeling/results/*.json"

# Direct mode with image paths
python run_vqa.py \
    --satellite data/sat.png \
    --street data/street.png \
    --sat-annotation "Commercial area with solar panels..." \
    --street-annotation "KOTONTEKS factory outlet..."
```

### Testing Configuration

```bash
# Test configuration loading
python -c "from src.core.config import load_config; config = load_config(); print('Config valid')"

# Test provider registry
python -c "from src.llm.registry import ProviderRegistry; print(ProviderRegistry.list_providers())"

# Verify API key
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('GROQ_API_KEY set:', bool(os.getenv('GROQ_API_KEY')))"

# Test Ollama connection (if using Ollama)
curl http://localhost:11434/api/tags
```

## Architecture

### Core Design Patterns

**Factory Pattern** (`src/llm/registry.py`):
- `ProviderRegistry`: Dynamic LLM provider instantiation
- Use `@register_provider('name')` decorator for automatic registration
- Providers are registered on module import

**Facade Pattern** (`src/llm/client.py`):
- `LLMClient`: Unified interface over multiple LLM providers
- Abstracts provider-specific details
- Integrates logging automatically

**Template Method** (`src/logging/base_logger.py`):
- `BaseLogger`: Base class with common logging infrastructure
- Specialized implementations: `LabelingLogger`, `VQALogger`

**Strategy Pattern** (`src/llm/base.py`):
- `LLMProvider`: Abstract base for swappable LLM implementations
- Each provider (Groq, Ollama) implements the interface

### Module Structure

```
src/
├── core/               # Configuration and utilities
│   ├── config.py       # Pydantic models for YAML config with validation
│   ├── image_utils.py  # Base64 encoding for images
│   └── image_patches.py # Zoom tool patch extraction
│
├── llm/                # Provider abstraction layer
│   ├── base.py         # LLMProvider abstract interface
│   ├── client.py       # LLMClient facade (use this, not providers directly)
│   ├── registry.py     # ProviderRegistry for dynamic instantiation
│   └── providers/      # Concrete provider implementations
│       ├── groq_provider.py
│       └── ollama_provider.py
│
├── prompts/            # Jinja2 templating system
│   ├── renderer.py     # PromptRenderer with custom filters
│   └── templates/      # Jinja2 template files (.j2)
│       ├── labeling/   # Satellite and street view annotation templates
│       └── vqa/        # VQA generation templates
│
├── logging/            # Research-grade logging
│   ├── base_logger.py  # BaseLogger with common infrastructure
│   ├── labeling_logger.py
│   └── vqa_logger.py
│
├── labeling/           # Image annotation pipeline
│   ├── pipeline.py     # LabelingPipeline orchestrator
│   ├── annotator.py    # ImageAnnotator (handles zoom tool)
│   └── input_handler.py # InputHandler for reading input_structure.json
│
├── vqa/                # VQA generation pipeline
│   ├── generator.py    # VQAGenerator
│   └── validator.py    # VQA validation logic
│
└── osm/                # OpenStreetMap API integration
    ├── client.py       # OSMClient for geographic data extraction
    └── __init__.py     # Public API exports
```

### Pipeline Flow

**Labeling Pipeline (2 phases with shared conversation):**
1. Initialize LLMClient with provider from config
2. Load image pairs from `input_structure.json` via InputHandler
3. **Phase 1 - Satellite**:
   - Send thinking prompt → get detailed analysis (800 tokens)
   - Send annotation prompt → get concise annotation (200 tokens)
   - If zoom tool enabled: LLM may request patches via tool calls
4. **Phase 2 - Street View** (continues same conversation):
   - Send thinking prompt with new image → get analysis
   - Send annotation prompt → get annotation
5. Save results with SHA256-based collision prevention
6. Generate comprehensive logs (runs/, llm_calls/, results/)

**VQA Pipeline (fresh conversation):**
1. Load annotations from labeling results or direct input
2. Start new conversation with both images
3. Generate 15-20 VQA examples across 5 categories
4. Validate and save results

### Configuration System

**Main Config** (`config/settings.yaml`):
- LLM provider settings (model, API keys)
- Logging levels and directories
- Research settings (question counts, categories)
- Zoom tool configuration
- Uses Pydantic for validation (see `src/core/config.py`)

**Environment Variables:**
- `GROQ_API_KEY`: Required for Groq provider
- Use `.env` file in project root

**Template Variables** in `input_structure.json`:
- `{timestamp}`: YYYYMMDD_HHMMSS
- `{hash}`: 8-char SHA256 hash (collision prevention)
- `{source_name}`: Satellite image filename without extension

## Adding New LLM Providers

1. **Create provider class** in `src/llm/providers/your_provider.py`:

```python
from ..base import LLMProvider
from ..registry import register_provider

@register_provider('your_provider')
class YourProvider(LLMProvider):
    def __init__(self, config):
        self.model = config.model
        # Initialize your client

    def send_message(self, messages, tools=None):
        # Implementation
        # Return: (response_text, stats_dict, tool_calls_list)
        pass

    def supports_vision(self):
        return True  # or False

    def format_image_message(self, text, image_paths, image_base64_list):
        # Return message dict with images in your provider's format
        pass
```

2. **Add config model** in `src/core/config.py`:

```python
class YourProviderConfig(BaseModel):
    enabled: bool = True
    model: str = "your-default-model"
    # Add provider-specific fields
```

3. **Update settings.yaml**:

```yaml
llm:
  default_provider: "your_provider"
  providers:
    your_provider:
      enabled: true
      model: "model-name"
```

4. **Import in `src/llm/providers/__init__.py`**:

```python
from .your_provider import YourProvider
```

## Zoom Tool Architecture

The zoom tool allows LLMs to request detailed patches from images during annotation.

**Configuration** (`config/settings.yaml`):
```yaml
research:
  labeling:
    zoom_tool:
      enabled: true
      max_zooms_per_image: 5
      save_patches: true
```

**Available Patches:**
- `northwest`, `northeast`, `southwest`, `southeast` (quadrants)
- `center` (center 50%)

**Tool Definition** (sent to LLM in `src/labeling/annotator.py`):
```python
{
    "type": "function",
    "function": {
        "name": "zoom",
        "description": "Extract a detailed patch from the current image",
        "parameters": {
            "type": "object",
            "properties": {
                "patch_name": {
                    "type": "string",
                    "enum": ["northwest", "northeast", "southwest", "southeast", "center"]
                }
            }
        }
    }
}
```

**Implementation Flow:**
1. LLM receives image with zoom tool definition
2. LLM makes tool call: `zoom(patch_name="northwest")`
3. `ImageAnnotator._handle_zoom_requests()` extracts patch
4. Patch added to conversation as new image message
5. LLM continues analysis with zoomed view

## Prompt Engineering

**Template Location:** `src/prompts/templates/`

**Jinja2 Features:**
- Variables: `{{ variable_name }}`
- Filters: `{{ text|truncate_smart(100) }}`, `{{ items|oxford_join }}`
- Conditionals: `{% if condition %} ... {% endif %}`

**Custom Filters** (see `src/prompts/renderer.py`):
- `truncate_smart(length)`: Intelligent truncation
- `oxford_join(conjunction)`: Oxford comma lists
- `wordwrap(width)`: Word wrapping

**Template Modification:**
- Edit `.j2` files directly (no code changes needed)
- Templates are loaded at runtime
- Use `{{ debug(variable) }}` for debugging

## Logging System

**Output Structure:**
```
logs/
├── labeling/
│   ├── runs/              # Human-readable execution logs
│   ├── llm_calls/         # Per-call LLM statistics (JSON)
│   └── results/           # Final labeling outputs
└── vqa/
    ├── runs/
    ├── llm_calls/
    └── results/
```

**LLM Call Logs** (JSON format):
- Request/response content
- Latency and token usage
- Model and provider info
- Tool calls (if any)
- Image paths (not base64 data by default)
- Success/error status

**Log Levels:**
- Console: INFO (configurable in settings.yaml)
- File: DEBUG (more verbose)

## Important Implementation Details

### Conversation Management
- Labeling maintains conversation context between satellite and street view for consistency
- VQA uses fresh conversation to avoid context length issues
- Configurable via `conversation_mode` and `fresh_conversation` settings

### Error Handling
- LLM call failures logged but don't crash pipeline
- Retries configurable via `research.labeling.max_retries`
- Failed pairs tracked separately in pipeline results

### Image Encoding
- Always use `encode_image_to_base64()` from `src/core/image_utils.py`
- Format: `data:image/{ext};base64,{base64_string}`
- Supported formats: png, jpg, jpeg

### Path Resolution
- Use `config.resolve_path()` for relative paths
- Project root set automatically from config file location
- All paths in `input_structure.json` can be absolute or relative

### Collision Prevention
- Output filenames use `{timestamp}_{hash}_{source_name}.json` pattern
- Hash is first 8 chars of SHA256(satellite_path + street_view_path + timestamp)
- Ensures uniqueness even with same image pairs

## Legacy Code

**vqa_pipeline.py** (root directory):
- Original v1.0 monolithic implementation
- Preserved for reference only
- DO NOT modify or use for new features
- All new development uses modular v2.0 architecture

## OpenStreetMap Integration

The project includes an OSM API v0.6 client for extracting geographic context:

**Usage:**
```python
from src.osm import OSMClient

client = OSMClient()
result = client.describe(
    top_left=(lat, lon),      # North-west corner
    bottom_right=(lat, lon)   # South-east corner
)
```

**Features:**
- Landmark extraction (buildings, amenities, shops, etc.)
- Type classification and distribution analysis
- Organization detection (operator, owner, brand, network)
- Automatic bounding box normalization and validation
- Comprehensive error handling

**Configuration:**
- `OSM_API_BASE_URL`: Optional custom endpoint (default: https://api.openstreetmap.org)
- `OSM_API_KEY`: Optional API key for proxied/hosted instances

**Testing:**
```bash
# Unit tests (mocked HTTP)
pytest test_osm_client.py -v -m "not integration"

# Integration test (real API)
pytest test_osm_client.py -v -m integration
```

## Research Context

The pipeline focuses on urban analysis across these domains:
1. **Sustainability**: Solar panels, green infrastructure
2. **Infrastructure**: Road quality, parking organization
3. **Economic**: Commercial activity indicators
4. **Maintenance**: Area cleanliness, upkeep
5. **Development**: Central vs. peripheral zones, urban density

VQA questions are distributed across these categories to ensure comprehensive coverage.
