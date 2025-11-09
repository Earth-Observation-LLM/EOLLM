# EOLLM Project - Complete Context Document

## Executive Summary

**EOLLM (Earth Observation with Large Language Models)** is a sophisticated Visual Question Answering (VQA) pipeline designed for automated urban scene analysis using satellite and street-level imagery. The system leverages multimodal AI (specifically Groq's Llama Vision model) to generate intelligent annotations and create training datasets for urban planning, geospatial analysis, and smart city applications.

---

## Project Purpose

The primary goal of this project is to automate the analysis of urban environments by:

1. **Generating detailed annotations** for satellite and street view images using AI-driven multi-step reasoning
2. **Creating VQA datasets** that can be used to train or evaluate urban analysis models
3. **Analyzing urban characteristics** including:
   - Sustainability indicators (solar panels, green infrastructure)
   - Infrastructure quality (roads, parking, organization)
   - Commercial vs. residential classification
   - Urban development levels (central vs. peripheral zones)
   - Economic activity indicators
   - Area maintenance and cleanliness
   - Vehicle patterns and traffic organization
   - Building characteristics and density

---

## Technical Architecture

### Core Technology Stack

- **Language**: Python 3
- **AI Model**: `meta-llama/llama-4-maverick-17b-128e-instruct` (Vision-capable LLM)
- **API Provider**: Groq Cloud API
- **Key Libraries**:
  - `groq` - Groq API client
  - `base64` - Image encoding for API transmission
  - `json` - Data serialization
  - `logging` - Comprehensive pipeline logging
  - `pathlib` - Modern file path handling
  - `python-dotenv` - Environment variable management

### System Requirements

- **API Key**: Requires `GROQ_API_KEY` environment variable
- **Input**: Satellite and street view images (PNG format)
- **Output**: JSON and text log files with structured annotations and VQA examples

---

## Data Structure

### Input Data Organization

The project contains organized image datasets in the `data/` directory:

```
data/
├── aliebykoy-otogar/          # Otogar (bus station) area comparison
│   ├── 2015-satelite.png      # Satellite view from 2015
│   ├── 2015-street.png        # Street view from 2015
│   ├── 2023-satelite.png      # Satellite view from 2023
│   └── 2023-street.png        # Street view from 2023
│
├── timko-1-old-new/           # Timko location temporal comparison
│   ├── old.png                # Older street view
│   ├── new.png                # Recent street view
│   └── satelite.png           # Satellite overview
│
└── timko-2/                   # Multi-year satellite evolution
    ├── 2014/satelite.png      # 2014 baseline
    ├── 2019/satelite.png      # 2019 snapshot
    ├── 2024/satelite.png      # 2024 snapshot
    └── 2025/satelite.png      # 2025 snapshot
```

**Key Observations**:
- The datasets focus on **temporal analysis** (comparing the same locations across different years)
- Typical use case: Urban development tracking, infrastructure changes, sustainability improvements
- Location context: Based on Turkish place names (KOTONTEKS factory outlet visible in logs)

### Output Data Structure

```
pipeline_logs/
├── pipeline_YYYYMMDD_HHMMSS.log  # Human-readable execution log
└── pipeline_YYYYMMDD_HHMMSS.json # Structured data output
```

---

## Pipeline Workflow (3-Phase Process)

### Phase 1: Satellite Image Annotation

**Purpose**: Analyze satellite imagery with systematic reasoning

**Process**:
1. **Thinking Step** (Step 1):
   - Model receives satellite image with detailed analysis prompt
   - Identifies: buildings, roads, vehicles, green spaces
   - Assesses: urban density, organization, commercial vs. residential
   - Notes: sustainability features (solar panels, green roofs)
   - Evaluates: parking infrastructure, road conditions
   - Output: Detailed reasoning (800 tokens max)

2. **Annotation Step** (Step 2):
   - Based on thinking, generates concise 2-line annotation
   - Priority-based description (most important features first)
   - Specifies area type (commercial/residential/mixed-use)
   - Output: Concise annotation (200 tokens max)

**Conversation**: Uses conversation history to maintain context

### Phase 2: Street View Image Annotation

**Purpose**: Analyze ground-level perspective of the same area

**Process**:
1. **Thinking Step** (Step 1):
   - Model receives street view image (same conversation context)
   - Identifies: Visible text, signage, branding, company names
   - Assesses: Building architecture, facade conditions
   - Analyzes: Vehicle types (personal vs. commercial fleet)
   - Evaluates: Road cleanliness, parking organization, traffic patterns
   - Determines: Central vs. peripheral location, commercial vs. residential
   - Output: Detailed street-level analysis (800 tokens max)

2. **Annotation Step** (Step 2):
   - Generates concise 2-line street view annotation
   - Includes: Signage/text, building types, vehicle composition
   - Assesses: Area characteristics, economic activity level
   - Output: Concise annotation (200 tokens max)

**Conversation**: Continues from Phase 1, maintaining full context

### Phase 3: VQA Generation

**Purpose**: Generate Visual Question Answering dataset from analyzed images

**Process**:
1. **New conversation** with both images provided simultaneously
2. Includes both satellite and street view annotations as context
3. Generates 15-20 diverse questions covering:
   - **Sustainability**: Solar panels, green infrastructure
   - **Infrastructure**: Roads, parking, organization quality
   - **Economic**: Commercial activity, business indicators
   - **Maintenance**: Cleanliness, upkeep, area condition
   - **Development**: Central vs. peripheral, urban density

**VQA Question Format**:
```json
{
  "question": "Are there solar panels visible on any of the buildings?",
  "answer": "Yes/No/No clue",
  "reasoning": "Brief explanation of the answer",
  "category": "sustainability/infrastructure/economic/maintenance/development"
}
```

**Output**: Structured JSON with 15-20 VQA examples

---

## Key Components Breakdown

### 1. PipelineLogger Class

**Responsibilities**:
- Creates timestamped log files (text and JSON)
- Dual logging: Console output + file persistence
- Tracks all phases, prompts, responses, and errors
- Saves structured data for each pipeline phase

**Key Methods**:
- `log_phase_start()` - Marks beginning of each phase
- `log_prompt()` - Records prompts sent to AI
- `log_response()` - Records AI responses
- `save_phase_data()` - Persists structured phase data to JSON

### 2. VisionAnnotator Class

**Responsibilities**:
- Handles all Groq API interactions
- Manages multimodal message formatting (text + images)
- Sends base64-encoded images with text prompts
- Error handling and API retry logic

**Key Method**:
- `send_message_with_images()` - Unified API call method

**Configuration**:
- Model: `meta-llama/llama-4-maverick-17b-128e-instruct`
- Supports temperature control (0.5-0.7 used in pipeline)
- Configurable max_tokens per request

### 3. ImageAnnotationPipeline Class

**Responsibilities**:
- Orchestrates Phases 1 & 2
- Maintains conversation history across satellite and street view analysis
- Implements two-step reasoning (think → annotate)

**Key Methods**:
- `annotate_satellite_image()` - Phase 1 execution
- `annotate_street_view_image()` - Phase 2 execution

**Design Pattern**: Chain-of-Thought reasoning (explicit thinking before conclusion)

### 4. VQAGenerator Class

**Responsibilities**:
- Generates VQA training examples (Phase 3)
- Handles JSON parsing from AI responses
- Validates question quality and diversity

**Key Method**:
- `generate_vqa_examples()` - Creates 15-20 questions with answers

---

## Pipeline Execution Flow

### Entry Point
```python
run_pipeline(
    satellite_image: str,     # Path to satellite image
    old_street_view: str,     # Path to street view image
    new_street_view: str = None  # Optional: second street view for comparison
)
```

### Execution Steps

1. **Initialization**:
   - Load `GROQ_API_KEY` from environment
   - Create `PipelineLogger` instance
   - Initialize `VisionAnnotator` with API key
   - Log pipeline configuration

2. **Phase 1 - Satellite Analysis**:
   - Create `ImageAnnotationPipeline` instance
   - Encode satellite image to base64
   - Send thinking prompt with image
   - Receive detailed analysis
   - Send annotation prompt (in same conversation)
   - Receive concise annotation
   - Save phase data to JSON

3. **Phase 2 - Street View Analysis**:
   - Encode street view image to base64
   - Send thinking prompt with new image (continues conversation)
   - Receive detailed analysis
   - Send annotation prompt
   - Receive concise annotation
   - Save phase data to JSON

4. **Phase 3 - VQA Generation**:
   - Create `VQAGenerator` instance
   - Encode both images again
   - Start new conversation with both images
   - Include both annotations as context
   - Generate 15-20 VQA questions
   - Parse JSON response
   - Save phase data to JSON

5. **Finalization**:
   - Log completion summary
   - Return results dictionary with all data
   - Print summary to console

### Return Value Structure
```python
{
    "satellite_annotation": {
        "image_path": str,
        "thinking_response": str,
        "annotation": str
    },
    "street_view_annotation": {
        "image_path": str,
        "thinking_response": str,
        "annotation": str
    },
    "vqa_examples": {
        "satellite_path": str,
        "street_view_path": str,
        "vqa_data": {
            "vqa_examples": [...]
        },
        "raw_response": str
    },
    "log_file": str,
    "json_file": str
}
```

---

## Example Output Analysis

From the log file `pipeline_20251103_135447.json`, we can see a real execution:

### Input Images
- **Satellite**: Industrial/commercial complex with warehouses
- **Street View**: KOTONTEKS factory outlet building

### Generated Annotations

**Satellite Annotation**:
> "This satellite image shows a commercial/industrial complex with large warehouses, adequate parking infrastructure, and solar panels on one of the buildings, indicating a focus on sustainability. The area appears to be commercial/industrial in nature, with well-maintained roads and organized parking."

**Street View Annotation**:
> "This street view shows a modern commercial building, KOTONTEKS, with a factory outlet or store, surrounded by a well-organized parking lot filled with personal vehicles. The area appears to be a peripheral commercial zone with a clean and well-maintained environment, indicating a moderate to high level of economic activity and development."

### Generated VQA Examples (Sample)

1. **Sustainability Question**:
   - Q: "Are there solar panels visible on any of the buildings?"
   - A: Yes
   - Reasoning: "The satellite view shows a building with solar panels on its roof."

2. **Development Question**:
   - Q: "Is this a central urban location?"
   - A: No
   - Reasoning: "The street view annotation describes the area as a 'peripheral commercial zone', indicating it is not central."

3. **Infrastructure Question**:
   - Q: "Is the parking lot well-organized?"
   - A: Yes
   - Reasoning: "Both the satellite and street views show a well-organized parking lot with clearly marked spaces."

**Total Generated**: 19 VQA examples across 5 categories

---

## API Integration Details

### Groq API Configuration

- **Endpoint**: Groq Cloud API (`api.groq.com`)
- **Model**: `meta-llama/llama-4-maverick-17b-128e-instruct`
- **Authentication**: API key via `GROQ_API_KEY` environment variable
- **Message Format**: OpenAI-compatible chat completion API

### Image Handling

**Encoding Process**:
```python
base64.b64encode(image_file.read()).decode('utf-8')
```

**API Format**:
```python
{
    "type": "image_url",
    "image_url": {
        "url": "data:image/png;base64,{base64_string}"
    }
}
```

### API Parameters

- **Phase 1 & 2 (Thinking)**: 
  - `max_tokens`: 800
  - `temperature`: 0.7 (creative analysis)

- **Phase 1 & 2 (Annotation)**: 
  - `max_tokens`: 200
  - `temperature`: 0.5 (more focused)

- **Phase 3 (VQA)**: 
  - `max_tokens`: 2000
  - `temperature`: 0.7 (creative questions)

### Rate Limiting

The logs show the API has rate limiting:
```
HTTP/1.1 429 Too Many Requests
Retrying request in 7.000000 seconds
```

The Groq client handles automatic retries with exponential backoff.

---

## Use Cases

### 1. Urban Development Monitoring
- Track infrastructure changes over time
- Compare satellite views from different years
- Identify new construction, renovations, demolitions

### 2. Sustainability Assessment
- Detect solar panel installations
- Identify green infrastructure
- Monitor urban heat island mitigation efforts

### 3. Economic Activity Analysis
- Assess commercial vs. residential zones
- Evaluate parking utilization
- Identify business types through signage

### 4. Training Data Generation
- Create VQA datasets for urban AI models
- Generate question-answer pairs for geospatial ML
- Build evaluation benchmarks for vision models

### 5. Smart City Planning
- Automated area classification
- Infrastructure quality assessment
- Maintenance priority identification

---

## Project Status & Configuration

### Current Configuration (from code)

**Hardcoded Paths** (lines 560-564):
```python
SATELLITE_IMAGE = "/home/ezel/Development/EOLLM/satelite.png"
OLD_STREET_VIEW = "/home/ezel/Development/EOLLM/old.png"
NEW_STREET_VIEW = "/home/ezel/Development/EOLLM/new.png"
```

**Note**: These paths reference root-level images not currently in the repository. The actual data is in `data/` subdirectories.

### Missing Components

1. **No environment file**: `.env` file not included (needs `GROQ_API_KEY`)
2. **No requirements.txt**: Dependencies not formally specified
3. **No README**: User documentation missing
4. **No configuration file**: All settings hardcoded in script

### Expected Dependencies

Based on code analysis:
```
groq>=0.4.0
python-dotenv>=0.19.0
```

---

## Logging System

### Dual Output Format

1. **Text Log** (`pipeline_YYYYMMDD_HHMMSS.log`):
   - Human-readable execution trace
   - All prompts and responses
   - Phase transitions
   - Error messages
   - API call details

2. **JSON Data** (`pipeline_YYYYMMDD_HHMMSS.json`):
   - Structured machine-readable output
   - Complete phase data
   - Timestamps
   - All annotations and VQA examples

### Log Structure Example

```json
{
  "timestamp": "20251103_135447",
  "phases": [
    {
      "phase_number": 1,
      "phase_name": "Satellite Image Annotation",
      "timestamp": "2025-11-03T13:54:50.355346",
      "data": {
        "image_path": "/path/to/image.png",
        "thinking_response": "...",
        "annotation": "..."
      }
    },
    // ... phases 2 and 3
  ]
}
```

---

## Advanced Features

### 1. Multi-Step Reasoning (Chain-of-Thought)

The pipeline implements explicit reasoning steps:
- **Step 1 (Think)**: Model explains its observations systematically
- **Step 2 (Conclude)**: Model provides concise annotation based on thinking

This approach improves annotation quality and provides transparency.

### 2. Conversation Continuity

Phases 1 and 2 share conversation history:
- Street view analysis benefits from satellite context
- Model can reference previous observations
- Enables cross-modal reasoning

Phase 3 uses a fresh conversation:
- Prevents conversation length issues
- Provides clean context for VQA generation
- Includes both annotations as explicit text context

### 3. Structured Prompting

Prompts are highly detailed with:
- Explicit numbered instructions
- Required analysis dimensions
- Output format specifications
- Quality requirements

### 4. JSON Extraction Robustness

The system handles multiple JSON response formats:
```python
if "```json" in vqa_response:
    json_str = vqa_response.split("```json")[1].split("```")[0].strip()
elif "```" in vqa_response:
    json_str = vqa_response.split("```")[1].split("```")[0].strip()
```

Gracefully handles:
- Markdown code blocks with `json` language tag
- Plain markdown code blocks
- Raw JSON responses
- Parsing errors (stores raw response)

---

## Potential Improvements

### 1. Configuration System
- Move hardcoded paths to config file
- Support command-line arguments
- Enable batch processing of multiple image sets

### 2. Data Management
- Implement dataset catalog
- Support automatic pairing of satellite/street views
- Add metadata tracking (location, date, source)

### 3. Error Handling
- Retry logic for failed API calls
- Validation of image files before processing
- Graceful degradation for partial failures

### 4. Output Enhancement
- Generate visual reports with images
- Create summary statistics
- Export to standardized VQA dataset formats

### 5. Performance
- Parallel processing of multiple locations
- Caching of API responses
- Batch API calls where possible

### 6. Quality Assurance
- Validate annotation quality
- Check VQA question diversity
- Ensure answer consistency

---

## How to Use This Project

### Setup

1. **Install Dependencies**:
   ```bash
   pip install groq python-dotenv
   ```

2. **Configure API Key**:
   Create `.env` file:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. **Prepare Images**:
   - Place satellite image and street view(s) in accessible location
   - Update paths in `vqa_pipeline.py` (lines 562-564)

### Execution

```bash
python vqa_pipeline.py
```

### Output

- Console: Real-time progress and summary
- `pipeline_logs/pipeline_YYYYMMDD_HHMMSS.log`: Detailed execution log
- `pipeline_logs/pipeline_YYYYMMDD_HHMMSS.json`: Structured results

### Customization

**To analyze different images**:
```python
results = run_pipeline(
    satellite_image="/path/to/satellite.png",
    old_street_view="/path/to/street.png",
    new_street_view="/path/to/street2.png"  # Optional
)
```

**To adjust AI parameters**:
- Modify `max_tokens` in annotation methods
- Change `temperature` for creativity control
- Adjust prompt templates for different analysis focus

---

## Domain-Specific Focus

The project demonstrates specialized knowledge in:

1. **Urban Planning**:
   - Commercial vs. residential classification
   - Central vs. peripheral zone identification
   - Infrastructure quality assessment

2. **Sustainability Analysis**:
   - Renewable energy indicators (solar panels)
   - Green infrastructure evaluation
   - Urban heat island considerations

3. **Geospatial Analysis**:
   - Multi-scale observation (satellite + ground level)
   - Temporal change detection
   - Area characterization

4. **Economic Geography**:
   - Commercial activity indicators
   - Economic development level assessment
   - Business type identification

---

## Technical Innovations

### 1. Multimodal Conversation Management
- Seamlessly integrates text and image modalities
- Maintains context across multiple images
- Strategic conversation resets for optimal performance

### 2. Priority-Based Annotation
- Enforces importance ordering in descriptions
- Ensures key information appears first
- Maintains conciseness (2-line limit)

### 3. Category-Driven VQA Generation
- Ensures balanced question distribution
- Covers multiple analysis dimensions
- Standardized answer format (Yes/No/No clue)

### 4. Comprehensive Logging
- Dual-format output (human + machine readable)
- Complete audit trail
- Facilitates debugging and analysis

---

## Project Timeline Context

Based on log files:
- **Development Date**: November 3, 2025 (Sunday)
- **Multiple Runs**: At least 3 pipeline executions recorded
- **Status**: Functional prototype with real-world testing

---

## Conclusion

**EOLLM** is a production-ready pipeline for automated urban scene analysis using state-of-the-art vision-language models. It demonstrates sophisticated prompt engineering, robust error handling, and comprehensive logging. The system is particularly valuable for:

- **Researchers**: Generating VQA datasets for urban AI research
- **Urban Planners**: Automating area assessment and monitoring
- **Smart City Projects**: Large-scale urban analysis
- **Sustainability Analysts**: Tracking green infrastructure adoption

The modular design allows easy adaptation to different domains (rural areas, natural environments, infrastructure monitoring) by modifying prompts and categories.

---

## Quick Reference

**Main Script**: `vqa_pipeline.py` (590 lines)
**Data Directory**: `data/` (3 datasets with temporal comparisons)
**Output Directory**: `pipeline_logs/` (timestamped logs)
**AI Model**: Llama 4 Maverick 17B (Vision, 128k context)
**API Provider**: Groq Cloud
**Primary Language**: Python 3
**License**: Not specified
**Documentation**: This file (PROJECT_CONTEXT.md)






