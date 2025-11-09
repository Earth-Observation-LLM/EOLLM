# EOLLM: Earth Observation with Large Language Models

A professional, modular research pipeline for automated urban scene analysis using multimodal AI. Version 2.0 features a complete refactoring with provider-agnostic LLM infrastructure, Jinja2 templating, and research-grade logging.

## Features

- **Multi-Provider LLM Support**: Groq Cloud API and Ollama (local) with easy extensibility
- **Jinja2 Templating**: Separate prompts from code for easy customization
- **Research-Grade Logging**: Detailed per-call LLM logs with statistics and debugging info
- **Modular Architecture**: Separate pipelines for labeling and VQA generation
- **Type-Safe Configuration**: YAML-based config with Pydantic validation
- **Collision-Resistant Output**: SHA256-based filename generation

## Architecture

```
EOLLM/
├── config/                 # YAML configuration
├── src/                   # Core source code
│   ├── core/             # Config and utilities
│   ├── llm/              # LLM provider abstraction
│   ├── prompts/          # Jinja2 templates
│   ├── logging/          # Research logging
│   ├── labeling/         # Image annotation pipeline
│   └── vqa/              # VQA generation pipeline
├── logs/                  # Structured logs
│   ├── labeling/
│   └── vqa/
├── data/                  # Image datasets
├── run_labeling.py        # Labeling entry point
└── run_vqa.py            # VQA entry point
```

## Installation

1. **Clone and navigate to project:**
   ```bash
   cd /home/ezel/Development/EOLLM
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   # Create .env file
   echo "GROQ_API_KEY=your_groq_api_key_here" > .env
   ```

4. **Configure the pipeline:**
   Edit `config/settings.yaml` to customize:
   - LLM provider and model
   - Logging levels and directories
   - Research settings (question counts, categories, etc.)

## Quick Start

### 1. Image Labeling

Label satellite and street view image pairs:

```bash
# Process all pairs in input_structure.json
python run_labeling.py

# Process a specific pair
python run_labeling.py --pair-id aliebykoy_otogar_2015

# Use custom input file
python run_labeling.py --input my_inputs.json

# Use Ollama instead of Groq
python run_labeling.py --provider ollama
```

### 2. VQA Generation

Generate Visual Question Answering examples:

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

## Configuration

The `config/settings.yaml` file controls all aspects of the pipeline:

### LLM Providers

```yaml
llm:
  default_provider: "groq"  # or "ollama"
  providers:
    groq:
      enabled: true
      model: "meta-llama/llama-4-maverick-17b-128e-instruct"
      api_key_env: "GROQ_API_KEY"
    ollama:
      enabled: true
      model: "llama3.2-vision:11b"
      base_url: "http://localhost:11434"
```

### Research Settings

```yaml
research:
  vqa:
    min_questions: 15
    max_questions: 20
    categories:
      - sustainability
      - infrastructure
      - economic
      - maintenance
      - development
```

## Input Structure

Define image pairs in `input_structure.json`:

```json
{
  "pairs": [
    {
      "id": "example_pair",
      "satellite": "/path/to/satellite.png",
      "street_view": "/path/to/street.png",
      "output_json": "data/outputs/{timestamp}_{hash}_{source_name}.json",
      "metadata": {
        "location": "Example Location",
        "year": 2024
      }
    }
  ]
}
```

**Template Variables:**
- `{timestamp}`: YYYYMMDD_HHMMSS
- `{hash}`: 8-char SHA256 hash (collision prevention)
- `{source_name}`: Satellite image filename without extension

## Logging

The pipeline generates comprehensive logs for research and debugging:

### Labeling Logs
```
logs/labeling/
├── runs/              # Human-readable execution logs
├── llm_calls/         # Per-call LLM statistics (JSON)
└── results/           # Final labeling outputs
```

### VQA Logs
```
logs/vqa/
├── runs/              # Human-readable execution logs
├── llm_calls/         # Per-call LLM statistics (JSON)
└── results/           # Final VQA outputs
```

Each LLM call log includes:
- Request/response content
- Latency and token usage
- Model and provider info
- Image paths (not base64 data)
- Success/error status

## Prompt Templates

Customize prompts without touching code. Located in `src/prompts/templates/`:

```
templates/
├── labeling/
│   ├── satellite_thinking.j2      # Satellite analysis
│   ├── satellite_annotation.j2    # Satellite summary
│   ├── street_thinking.j2         # Street view analysis
│   └── street_annotation.j2       # Street view summary
└── vqa/
    └── generation.j2               # VQA question generation
```

Templates support Jinja2 features including:
- Variables and loops
- Custom filters (`truncate_smart`, `oxford_join`, etc.)
- Template inheritance
- Inline documentation

## Adding New LLM Providers

1. **Create provider class** in `src/llm/providers/`:

```python
from ..base import LLMProvider
from ..registry import register_provider

@register_provider('my_provider')
class MyProvider(LLMProvider):
    def send_message(self, messages):
        # Implementation
        pass
    
    def supports_vision(self):
        return True
```

2. **Add to config** in `config/settings.yaml`:

```yaml
llm:
  providers:
    my_provider:
      enabled: true
      model: "my-model-name"
      api_key_env: "MY_API_KEY"
```

3. **Use it:**

```bash
python run_labeling.py --provider my_provider
```

## Development

### Project Structure

- `src/core/`: Configuration management and utilities
- `src/llm/`: Provider abstraction layer
- `src/prompts/`: Jinja2 templating system
- `src/logging/`: Research-grade logging infrastructure
- `src/labeling/`: Image annotation pipeline
- `src/vqa/`: VQA generation and validation

### Key Design Patterns

- **Factory Pattern**: Provider registry for dynamic instantiation
- **Facade Pattern**: Unified LLM client interface
- **Template Method**: Base logger with specialized implementations
- **Strategy Pattern**: Swappable LLM providers

### Testing

```bash
# Test imports
python -c "from src.core.config import load_config; print('OK')"

# Test configuration
python -c "from src.core.config import load_config; config = load_config(); print(config.llm.default_provider)"

# Test provider creation
python -c "from src.llm.registry import ProviderRegistry; print(ProviderRegistry.list_providers())"
```

## Migration from v1.0

The original `vqa_pipeline.py` is preserved for reference. Key changes:

- **Monolithic → Modular**: Single file → organized packages
- **Hardcoded → Configurable**: YAML-based configuration
- **String prompts → Templates**: Jinja2 templating
- **Single provider → Multi-provider**: Extensible provider system
- **Basic logs → Research logs**: Per-call LLM tracking

## Research Use Cases

1. **Urban Development Monitoring**: Track infrastructure changes over time
2. **Sustainability Assessment**: Identify solar panels, green infrastructure
3. **Economic Activity Analysis**: Assess commercial vs. residential zones
4. **Training Data Generation**: Create VQA datasets for urban AI models
5. **Smart City Planning**: Automated area classification and assessment

## Troubleshooting

### Import Errors

```bash
# Ensure you're in project root
cd /home/ezel/Development/EOLLM

# Test imports
python -c "import sys; sys.path.insert(0, '.'); from src.core import config"
```

### API Errors

```bash
# Check API key
python -c "import os; print(os.getenv('GROQ_API_KEY'))"

# Test Ollama connection
curl http://localhost:11434/api/tags
```

### Configuration Errors

```bash
# Validate config
python -c "from src.core.config import load_config; config = load_config(); print('Config valid')"
```

## Contributing

To add features:

1. Create new provider → `src/llm/providers/`
2. Create new template → `src/prompts/templates/`
3. Add new pipeline → `src/{pipeline_name}/`
4. Update config schema → `src/core/config.py`

## License

Research project - check with project owner for licensing.

## Citation

If you use this pipeline in your research, please cite:

```
EOLLM: Earth Observation with Large Language Models
Version 2.0 - Professional Research Pipeline
2025
```

---

**Version**: 2.0.0  
**Last Updated**: 2025-11-09  
**Python**: 3.8+  
**Status**: Production Ready

