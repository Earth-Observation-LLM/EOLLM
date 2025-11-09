#!/usr/bin/env python3
"""
Sophisticated VQA Pipeline for Urban Scene Analysis
===================================================
This pipeline processes satellite and street view images to:
1. Generate detailed annotations with priority-based reasoning
2. Create Visual Question Answering (VQA) examples
3. Log all steps comprehensively for inspection
"""

import os
import json
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from dotenv import load_dotenv
from groq import Groq

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

class PipelineLogger:
    """Enhanced logging system for tracking all pipeline steps"""

    def __init__(self, log_dir: str = "pipeline_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"pipeline_{self.timestamp}.log"
        self.json_file = self.log_dir / f"pipeline_{self.timestamp}.json"

        # Setup file and console logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Store structured data
        self.pipeline_data = {
            "timestamp": self.timestamp,
            "phases": []
        }

    def log_phase_start(self, phase_name: str, phase_number: int):
        """Log the start of a pipeline phase"""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"PHASE {phase_number}: {phase_name}")
        self.logger.info(f"{'='*80}\n")

    def log_step(self, step_name: str, details: str = ""):
        """Log a specific step within a phase"""
        self.logger.info(f"STEP: {step_name}")
        if details:
            self.logger.info(f"  {details}")

    def log_prompt(self, prompt_type: str, prompt: str):
        """Log the prompt being sent to the model"""
        self.logger.info(f"\n[{prompt_type} PROMPT]")
        self.logger.info(f"{'-'*80}")
        self.logger.info(prompt)
        self.logger.info(f"{'-'*80}\n")

    def log_response(self, response_type: str, response: str):
        """Log the model's response"""
        self.logger.info(f"\n[{response_type} RESPONSE]")
        self.logger.info(f"{'-'*80}")
        self.logger.info(response)
        self.logger.info(f"{'-'*80}\n")

    def save_phase_data(self, phase_number: int, phase_name: str, data: Dict):
        """Save structured data for a phase"""
        phase_data = {
            "phase_number": phase_number,
            "phase_name": phase_name,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        self.pipeline_data["phases"].append(phase_data)

        # Save to JSON file
        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump(self.pipeline_data, f, indent=2, ensure_ascii=False)

    def log_error(self, error_msg: str):
        """Log an error"""
        self.logger.error(f"ERROR: {error_msg}")


# ============================================================================
# IMAGE PROCESSING UTILITIES
# ============================================================================

def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 string for Groq API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# ============================================================================
# GROQ API INTERACTION
# ============================================================================

class VisionAnnotator:
    """Handles all interactions with Groq Vision API"""

    def __init__(self, api_key: str, logger: PipelineLogger):
        self.client = Groq(api_key=api_key)
        self.logger = logger
        self.model = "meta-llama/llama-4-maverick-17b-128e-instruct"  # Vision-capable model

    def send_message_with_images(
        self,
        messages: List[Dict],
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """Send messages (with images) to Groq and get response"""
        try:
            self.logger.log_step(f"Sending request to Groq", f"Model: {self.model}")

            # Groq uses the same format as OpenAI, no conversion needed
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )

            return response.choices[0].message.content

        except Exception as e:
            self.logger.log_error(f"API call failed: {str(e)}")
            raise


# ============================================================================
# PHASE 1 & 2: IMAGE ANNOTATION WITH MULTI-STEP REASONING
# ============================================================================

class ImageAnnotationPipeline:
    """Handles satellite and street view image annotation"""

    def __init__(self, annotator: VisionAnnotator, logger: PipelineLogger):
        self.annotator = annotator
        self.logger = logger
        self.conversation_history = []

    def annotate_satellite_image(self, image_path: str) -> Dict:
        """Phase 1: Annotate satellite image with thinking step"""
        self.logger.log_phase_start("SATELLITE IMAGE ANNOTATION", 1)

        # Encode image
        self.logger.log_step("Encoding satellite image")
        image_base64 = encode_image_to_base64(image_path)

        # Step 1: Thinking prompt
        thinking_prompt = """You are an expert in urban planning and geospatial analysis. You will be shown a satellite view image.

Your task is to analyze this image systematically. First, think through what you observe:
1. Identify the main elements (buildings, roads, vehicles, green spaces, etc.)
2. Assess the urban density and organization
3. Look for indicators of commercial vs residential areas
4. Note any sustainability features (solar panels, green roofs, etc.)
5. Evaluate parking infrastructure and road conditions

Think step-by-step and be thorough in your observation. What do you see in this satellite image?"""

        self.logger.log_prompt("THINKING", thinking_prompt)

        # Build message with image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": thinking_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]

        # Get thinking response
        thinking_response = self.annotator.send_message_with_images(
            messages, max_tokens=800, temperature=0.7
        )
        self.logger.log_response("THINKING", thinking_response)

        # Add to conversation history
        self.conversation_history.extend([
            messages[0],
            {"role": "assistant", "content": thinking_response}
        ])

        # Step 2: Annotation prompt
        annotation_prompt = """Based on your observations, now provide a concise annotation of this satellite view.

Requirements:
- Maximum 2 lines
- Priority-based description (mention most important features first)
- Include: building types, infrastructure, sustainability indicators, area characteristics
- Specify if this appears to be commercial, residential, or mixed-use
- Note parking organization and road conditions if relevant

Provide the annotation now:"""

        self.logger.log_prompt("ANNOTATION", annotation_prompt)

        self.conversation_history.append({
            "role": "user",
            "content": annotation_prompt
        })

        # Get annotation response
        annotation_response = self.annotator.send_message_with_images(
            self.conversation_history, max_tokens=200, temperature=0.5
        )
        self.logger.log_response("SATELLITE ANNOTATION", annotation_response)

        # Add to conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": annotation_response
        })

        # Save phase data
        phase_data = {
            "image_path": image_path,
            "thinking_response": thinking_response,
            "annotation": annotation_response
        }
        self.logger.save_phase_data(1, "Satellite Image Annotation", phase_data)

        return phase_data

    def annotate_street_view_image(self, image_path: str) -> Dict:
        """Phase 2: Annotate street view image in same conversation"""
        self.logger.log_phase_start("STREET VIEW IMAGE ANNOTATION", 2)

        # Encode image
        self.logger.log_step("Encoding street view image")
        image_base64 = encode_image_to_base64(image_path)

        # Context-setting prompt
        context_prompt = """Now I'm showing you a street-level view of the same area from the satellite image. This is a ground-level perspective that will reveal more details.

First, analyze this street view systematically:
1. Identify visible text, signage, and branding (e.g., store names, company logos)
2. Assess building architecture and facade conditions
3. Look at vehicle types (personal, commercial/fleet vehicles)
4. Evaluate road cleanliness and maintenance
5. Assess parking organization and traffic patterns
6. Determine if this is a central urban area or peripheral zone
7. Identify indicators of commercial vs residential use

Think through what these observations tell you about the area's character and function."""

        self.logger.log_prompt("STREET VIEW THINKING", context_prompt)

        # Add message with new image
        self.conversation_history.append({
            "role": "user",
            "content": [
                {"type": "text", "text": context_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                }
            ]
        })

        # Get thinking response
        thinking_response = self.annotator.send_message_with_images(
            self.conversation_history, max_tokens=800, temperature=0.7
        )
        self.logger.log_response("STREET VIEW THINKING", thinking_response)

        # Add to conversation
        self.conversation_history.append({
            "role": "assistant",
            "content": thinking_response
        })

        # Annotation prompt
        annotation_prompt = """Based on your street-level observations, provide a concise annotation.

Requirements:
- Maximum 2 lines
- Priority-based description
- Include: visible signage/text, building types, vehicle composition (personal vs commercial)
- Assess area characteristics (central vs peripheral, clean vs unkempt, organized vs chaotic)
- Determine commercial vs residential nature
- Note any indicators of economic activity or area development level

Provide the street view annotation now:"""

        self.logger.log_prompt("STREET VIEW ANNOTATION", annotation_prompt)

        self.conversation_history.append({
            "role": "user",
            "content": annotation_prompt
        })

        # Get annotation response
        annotation_response = self.annotator.send_message_with_images(
            self.conversation_history, max_tokens=200, temperature=0.5
        )
        self.logger.log_response("STREET VIEW ANNOTATION", annotation_response)

        # Add to conversation
        self.conversation_history.append({
            "role": "assistant",
            "content": annotation_response
        })

        # Save phase data
        phase_data = {
            "image_path": image_path,
            "thinking_response": thinking_response,
            "annotation": annotation_response
        }
        self.logger.save_phase_data(2, "Street View Image Annotation", phase_data)

        return phase_data


# ============================================================================
# PHASE 3: VQA GENERATION
# ============================================================================

class VQAGenerator:
    """Generates Visual Question Answering examples from annotated images"""

    def __init__(self, annotator: VisionAnnotator, logger: PipelineLogger):
        self.annotator = annotator
        self.logger = logger

    def generate_vqa_examples(
        self,
        satellite_path: str,
        street_view_path: str,
        satellite_annotation: str,
        street_view_annotation: str
    ) -> Dict:
        """Phase 3: Generate VQA examples in new conversation"""
        self.logger.log_phase_start("VQA GENERATION", 3)

        # Encode images
        self.logger.log_step("Encoding images for VQA generation")
        satellite_base64 = encode_image_to_base64(satellite_path)
        street_view_base64 = encode_image_to_base64(street_view_path)

        # VQA generation prompt
        vqa_prompt = f"""You are an expert in creating Visual Question Answering (VQA) datasets for urban analysis.

I'm providing you with:
1. A satellite view image with its annotation
2. A street view image of the same area with its annotation

SATELLITE VIEW ANNOTATION:
{satellite_annotation}

STREET VIEW ANNOTATION:
{street_view_annotation}

Your task is to generate high-quality VQA examples that test understanding of these images.

REQUIREMENTS:
1. Generate 15-20 diverse questions covering:
   - Sustainability indicators (solar panels, green infrastructure)
   - Infrastructure quality (roads, parking, organization)
   - Commercial vs residential characteristics
   - Urban development level (central vs peripheral)
   - Economic activity indicators
   - Area cleanliness and maintenance
   - Vehicle and traffic patterns
   - Building characteristics and density

2. Each question must:
   - Be answerable from the images and annotations
   - Have a clear answer: "Yes", "No", or "No clue"
   - Test meaningful urban planning or geospatial concepts
   - Avoid obvious or trivial questions

3. Question types to include:
   - Sustainability questions (e.g., "Are there signs of renewable energy use in this area?")
   - Infrastructure questions (e.g., "Is parking well-organized in this area?")
   - Economic questions (e.g., "Does this appear to be a commercial zone?")
   - Maintenance questions (e.g., "Is the area well-maintained?")
   - Development questions (e.g., "Is this a central urban location?")

FORMAT YOUR RESPONSE AS JSON:
{{
  "vqa_examples": [
    {{
      "question": "Question text here?",
      "answer": "Yes/No/No clue",
      "reasoning": "Brief explanation of why this is the answer",
      "category": "sustainability/infrastructure/economic/maintenance/development"
    }}
  ]
}}

Generate the VQA examples now:"""

        self.logger.log_prompt("VQA GENERATION", vqa_prompt)

        # Build new conversation with both images
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": vqa_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{satellite_base64}",
                        }
                    },
                    {"type": "text", "text": "[Satellite View ↑]"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{street_view_base64}",
                        }
                    },
                    {"type": "text", "text": "[Street View ↑]"}
                ]
            }
        ]

        # Get VQA response
        vqa_response = self.annotator.send_message_with_images(
            messages, max_tokens=2000, temperature=0.7
        )
        self.logger.log_response("VQA EXAMPLES", vqa_response)

        # Parse JSON response
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_str = vqa_response
            if "```json" in vqa_response:
                json_str = vqa_response.split("```json")[1].split("```")[0].strip()
            elif "```" in vqa_response:
                json_str = vqa_response.split("```")[1].split("```")[0].strip()

            vqa_data = json.loads(json_str)
            self.logger.log_step("Successfully parsed VQA JSON",
                                f"Generated {len(vqa_data.get('vqa_examples', []))} questions")
        except Exception as e:
            self.logger.log_error(f"Failed to parse VQA JSON: {str(e)}")
            vqa_data = {"raw_response": vqa_response, "parse_error": str(e)}

        # Save phase data
        phase_data = {
            "satellite_path": satellite_path,
            "street_view_path": street_view_path,
            "vqa_data": vqa_data,
            "raw_response": vqa_response
        }
        self.logger.save_phase_data(3, "VQA Generation", phase_data)

        return phase_data


# ============================================================================
# MAIN PIPELINE ORCHESTRATOR
# ============================================================================

def run_pipeline(
    satellite_image: str,
    old_street_view: str,
    new_street_view: str = None
):
    """
    Main pipeline orchestrator

    Args:
        satellite_image: Path to satellite view image
        old_street_view: Path to old street view image
        new_street_view: Path to new street view image (optional)
    """
    # Load environment variables
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in .env file")

    # Initialize logger
    logger = PipelineLogger()
    logger.logger.info("="*80)
    logger.logger.info("SOPHISTICATED VQA PIPELINE - STARTING (GROQ)")
    logger.logger.info("="*80)
    logger.logger.info(f"Model: meta-llama/llama-4-maverick-17b-128e-instruct")
    logger.logger.info(f"Satellite Image: {satellite_image}")
    logger.logger.info(f"Street View (Old): {old_street_view}")
    if new_street_view:
        logger.logger.info(f"Street View (New): {new_street_view}")
    logger.logger.info("="*80 + "\n")

    # Initialize annotator
    annotator = VisionAnnotator(groq_api_key, logger)

    # Phase 1 & 2: Image Annotation
    annotation_pipeline = ImageAnnotationPipeline(annotator, logger)

    # Annotate satellite image
    satellite_data = annotation_pipeline.annotate_satellite_image(satellite_image)

    # Annotate street view (old)
    street_view_data = annotation_pipeline.annotate_street_view_image(old_street_view)

    # Phase 3: VQA Generation
    vqa_generator = VQAGenerator(annotator, logger)
    vqa_data = vqa_generator.generate_vqa_examples(
        satellite_image,
        old_street_view,
        satellite_data["annotation"],
        street_view_data["annotation"]
    )

    # Final summary
    logger.logger.info("\n" + "="*80)
    logger.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.logger.info("="*80)
    logger.logger.info(f"Log file: {logger.log_file}")
    logger.logger.info(f"JSON data: {logger.json_file}")
    logger.logger.info("="*80 + "\n")

    return {
        "satellite_annotation": satellite_data,
        "street_view_annotation": street_view_data,
        "vqa_examples": vqa_data,
        "log_file": str(logger.log_file),
        "json_file": str(logger.json_file)
    }


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Define image paths
    SATELLITE_IMAGE = "/home/ezel/Development/EOLLM/satelite.png"
    OLD_STREET_VIEW = "/home/ezel/Development/EOLLM/old.png"
    NEW_STREET_VIEW = "/home/ezel/Development/EOLLM/new.png"

    # Run pipeline
    try:
        results = run_pipeline(
            satellite_image=SATELLITE_IMAGE,
            old_street_view=OLD_STREET_VIEW,
            new_street_view=NEW_STREET_VIEW
        )

        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print(f"\nSatellite Annotation:")
        print(f"  {results['satellite_annotation']['annotation']}")
        print(f"\nStreet View Annotation:")
        print(f"  {results['street_view_annotation']['annotation']}")
        print(f"\nVQA Examples Generated: {len(results['vqa_examples']['vqa_data'].get('vqa_examples', []))}")
        print(f"\nLog files:")
        print(f"  Text log: {results['log_file']}")
        print(f"  JSON data: {results['json_file']}")
        print("="*80 + "\n")

    except Exception as e:
        print(f"Pipeline failed with error: {str(e)}")
        raise
