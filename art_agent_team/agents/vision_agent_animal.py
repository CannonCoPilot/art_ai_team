import logging
import os
import json
import numpy as np
import base64
import io
from PIL import Image, ImageDraw, ImageFont, ImageColor, UnidentifiedImageError
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from openai import OpenAI
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import math # Needed for centroid calculation
from queue import Queue # Import Queue

# Custom Exceptions for Image Processing Errors
class CorruptedImageError(Exception):
    """Custom exception for corrupted image files."""
    pass

class UnsupportedImageFormatError(Exception):
    """Custom exception for unsupported image formats."""
    pass

@dataclass(frozen=True)
class SegmentationMask:
    """Data class for storing segmentation mask information."""
    y0: int  # in [0..height - 1]
    x0: int  # in [0..width - 1]
    y1: int  # in [0..height - 1]
    x1: int  # in [0..width - 1]
    mask: np.ndarray  # [img_height, img_width] with values 0..255
    label: str

class VisionAgentAnimal:
    """Agent for analyzing animal art using Gemini and Grok Vision APIs."""

    def _think_with_grok(self, thought: str) -> str:
        """
        USER REQUIREMENT: Internal 'thinking' step using grok-3-mini-fast-high-beta.
        This method sends the agent's reasoning or validation prompt to the Grok model and returns the response.
        """
        from openai import OpenAI
        api_key = self.config.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logging.warning("[VisionAgentAnimal] No OPENAI_API_KEY found for Grok thinking step.")
            return ""
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="grok-3-mini-fast-high-beta",
                messages=[{"role": "system", "content": "You are an expert vision analysis agent reasoning about your next step."},
                          {"role": "user", "content": thought}],
                max_tokens=256,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"[VisionAgentAnimal] Grok thinking step failed: {e}")
            return ""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the VisionAgentAnimal with configuration."""
        self.config = config if config is not None else {}
        # Always use environment variables for configuration
        self.input_folder = os.environ.get('INPUT_FOLDER', 'input')
        self.output_folder = os.environ.get('OUTPUT_FOLDER', 'output')
        self.workspace_folder = os.environ.get('WORKSPACE_FOLDER', 'workspace')

        # Initialize Gemini models
        google_api_key = config.get('google_api_key') if config and 'google_api_key' in config else os.environ.get('GOOGLE_API_KEY')
        if not google_api_key:
            logging.warning("[VisionAgentAnimal] WARNING: No Google API key found in config or environment variables. Gemini Pro will be unavailable. Analysis will proceed with other available models.")
            self.gemini_pro = None
        else:
            try:
                genai.configure(api_key=google_api_key)
                # USER REQUIREMENT: Use gemini-2.5-pro-exp-03-25 for vision.
                self.gemini_pro = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')
                logging.info("[VisionAgentAnimal] Gemini Pro model initialized.")
            except Exception as e:
                logging.warning(f"[VisionAgentAnimal] Failed to initialize Gemini Pro model due to missing key or other issue: {e}. Analysis will proceed without Gemini Pro.")
                self.gemini_pro = None

        # Initialize Grok Vision model
        grok_api_key = config.get('grok_api_key') if config and 'grok_api_key' in config else os.environ.get('GROK_API_KEY')
        if not grok_api_key:
            logging.warning("[VisionAgentAnimal] WARNING: No Grok API key found in config or environment variables. Grok Vision will be unavailable. Analysis will proceed with other available models.")
            self.grok_client = None
        else:
            self.grok_vision_model = "grok-2-vision-latest"
            self.grok_client = OpenAI(
                api_key=grok_api_key,
                base_url="https://api.x.ai/v1",
            )

        # Colors for visualization
        self.colors = ['red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple',
                       'brown', 'gray', 'beige', 'turquoise', 'cyan', 'magenta',
                       'lime', 'navy', 'maroon', 'teal', 'olive', 'coral', 'lavender',
                       'violet', 'gold', 'silver'] + [c for c in ImageColor.colormap.keys()]

        # Store research data for use in methods like _threshold_important_objects
        self.primary_subject: Optional[str] = None
        self.secondary_subjects: List[str] = []


    def analyze_image(self, image_path: str, research_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Analyzes animal art image using Gemini Pro and Grok Vision,
        incorporating research data.
        """
        self._think_with_grok(f"About to analyze animal artwork image: {os.path.basename(image_path)}. What reasoning steps should I take to ensure a thorough and accurate analysis?")
        try:
            # Store research data
            self.primary_subject = research_data.get("primary_subject")
            self.secondary_subjects = research_data.get("secondary_subjects", [])
            paragraph_description = research_data.get("paragraph_description", "")
            structured_sentence = research_data.get("structured_sentence", "")

            # Load and preprocess image
            try:
                with open(image_path, 'rb') as image_file:
                    content = image_file.read()
                img = Image.open(io.BytesIO(content))
            except FileNotFoundError as e:
                logging.error(f"[VisionAgentAnimal] Input file not found in analyze_image: {image_path}. Error: {e}")
                raise  # Re-raise FileNotFoundError to allow tests to catch it specifically
            except UnidentifiedImageError as e:
                logging.error(f"[VisionAgentAnimal] Cannot identify image file (corrupted or unsupported format) in analyze_image: {image_path}. Error: {e}")
                raise CorruptedImageError(f"Corrupted or unsupported image file: {image_path}") from e
            except IOError as e:
                logging.error(f"[VisionAgentAnimal] IOError opening image in analyze_image: {image_path}. Error: {e}")
                raise UnsupportedImageFormatError(f"IOError, possibly unsupported image format: {image_path}") from e
            # Removed resampling to always use full resolution image

            # Initialize results structure
            results = {
                "objects": [],
                "segmentation_masks": [],
                "image_size": img.size,
                "relationships": [], # Relationships might be added by LLMs
                "style": research_data.get("style", "animal-scene"),
                "primary_subject": self.primary_subject,
                "secondary_subjects": self.secondary_subjects
            }

            # Animal-specific prompt focusing on animals, faces, and interactions
            animal_prompt = f"""
            Analyze this animal painting. Focus on identifying and localizing all animals, especially their faces and distinctive features (beaks, snouts, manes, etc.). Note interactions between animals or with the environment.

            Research Information:
            Paragraph Summary: {paragraph_description}
            Structured Sentence: {structured_sentence}

            Instructions:
            - For each full animal body and each face/head, return a dictionary with "label" and "box_2d" (bounding box coordinates).
            - For each facial feature (including but not limited to: beak, snout, muzzle, nose, mouth, teeth, eye, eyes, ears, mane, crest, neck, gills, horns, antlers), return a dictionary with "label" and "coordinates" (e.g., [x, y] in image pixel space or normalized 0-1000).
            - For other features (e.g., wings, paws, feet, tail, etc.) that do not have a specific location, you may return them as strings.
            - Do not return facial features as strings if you can provide coordinates.

            For each identified object, provide:
            - label: Concise description (e.g., "swan", "swan's head", "ducks swimming")
            - box_2d: [y0, x0, y1, x1]
            - confidence: 0.0-1.0
            - type: Category (e.g., "animal", "animal_face", "group_of_animals", "interaction")
            - importance: Initial estimate 0.0-1.0 (will be refined by agent)
            - features: List of detected features, each as a dictionary with "label" and "coordinates" for facial features, or as a string for other features.

            Format as JSON:
            {{
                "objects": [
                    {{
                        "label": "...",
                        "box_2d": [y0, x0, y1, x1],
                        "confidence": ...,
                        "type": "...",
                        "importance": ...,
                        "features": [
                            {{"label": "beak", "coordinates": [x, y]}},
                            {{"label": "eye", "coordinates": [x, y]}},
                            "wing"
                        ]
                    }}
                ]
            }}
            """

            # --- Dual-Model Analysis ---

            # 1. Gemini Pro Analysis
            gemini_objects = []
            if self.gemini_pro:
                try:
                    gemini_response = self.gemini_pro.generate_content(
                        contents=[animal_prompt, img],
                        generation_config=GenerationConfig(
                            temperature=0.4,
                            top_p=0.8,
                            top_k=40
                        )
                    )
                    gemini_results = self._parse_gemini_response(gemini_response.text)
                    if gemini_results and "objects" in gemini_results:
                        gemini_objects = gemini_results["objects"]
                        logging.info(f"[VisionAgentAnimal] Gemini Pro identified {len(gemini_objects)} objects.")
                except Exception as e:
                    # Use logging.exception to include stack trace for unexpected errors
                    logging.exception(f"[VisionAgentAnimal] Error during Gemini Pro analysis: {e}")
                    # Optionally raise a custom exception or return specific error indicator
                    # For now, logging the error and continuing with potentially empty results

            # 2. Grok Vision Analysis
            grok_objects = []
            if self.grok_client:
                try:
                    def encode_image(image_path):
                        with open(image_path, "rb") as image_file:
                            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                        return encoded_string

                    base64_image = encode_image(image_path)
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                        "detail": "high",
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": animal_prompt,
                                },
                            ],
                        },
                    ]
                    grok_response = self.grok_client.chat.completions.create(
                        model=self.grok_vision_model,
                        messages=messages,
                        temperature=0.4,
                        max_tokens=2048,
                    )
                    grok_results = self._parse_grok_response(grok_response.choices[0].message.content)
                    if grok_results and "objects" in grok_results:
                        grok_objects = grok_results["objects"]
                        logging.info(f"[VisionAgentAnimal] Grok Vision identified {len(grok_objects)} objects.")
                except Exception as e:
                    # Use logging.exception to include stack trace for unexpected errors
                    logging.exception(f"[VisionAgentAnimal] Error during Grok Vision analysis: {e}")
                    # Optionally raise a custom exception or return specific error indicator
                    # For now, logging the error and continuing with potentially empty results


            # --- Result Aggregation ---
            combined_objects = self._aggregate_results(gemini_objects, grok_objects)
            results["objects"] = combined_objects
            logging.info(f"[VisionAgentAnimal] Aggregated {len(results['objects'])} objects from dual models.")

            # --- Post-Analysis Processing ---

            # Process objects and calculate importance scores
            for obj in results["objects"]:
                 base_importance = obj.get("importance", 0.5)
                 obj["importance"] = self._calculate_object_importance(
                     obj.get("box_2d", [0,0,0,0]),
                     obj.get("type", "unknown"),
                     img.size,
                     base_importance,
                     obj.get("label", ""),
                     self.primary_subject,
                     self.secondary_subjects,
                     obj.get("features", []) # Pass features for scoring
                 )

            # Boost primary subject score
            primary_subject_label = self.primary_subject.lower() if self.primary_subject else None
            highest_other_score = 0
            for obj in results["objects"]:
                 if primary_subject_label and obj.get("label", "").lower() != primary_subject_label:
                      highest_other_score = max(highest_other_score, obj.get("importance", 0))

            for obj in results["objects"]:
                 if primary_subject_label and primary_subject_label in obj.get("label", "").lower():
                      obj["importance"] = max(obj.get("importance", 0), 2 * highest_other_score)
                      obj["importance"] = min(1.0, obj["importance"])
                      logging.info(f"[VisionAgentAnimal] Boosted primary subject '{obj['label']}' importance to {obj['importance']:.2f}")


            # Create segmentation masks for important objects
            important_objects = self._threshold_important_objects(results["objects"])
            results["segmentation_masks"] = self._create_segmentation_masks(
                important_objects,
                img.size
            )

            # Note: Cropping logic is in copy_and_crop_image, called by save_analysis_outputs

            return results

        except (CorruptedImageError, UnsupportedImageFormatError): # Re-raise custom exceptions
            raise
        except Exception as e:
            logging.exception(f"[VisionAgentAnimal] Error in analyze_image: {e}")
            return None

    def _aggregate_results(self, gemini_objects: List[Dict], grok_objects: List[Dict]) -> List[Dict]:
        """Aggregates and potentially merges objects from different models."""
        combined = {}
        for obj_list in [gemini_objects, grok_objects]:
            for obj in obj_list:
                box_key = tuple(obj.get("box_2d", [0,0,0,0]))
                if box_key not in combined:
                    combined[box_key] = obj
                else:
                    if obj.get("confidence", 0) > combined[box_key].get("confidence", 0):
                        combined[box_key] = obj

        return list(combined.values())


    def _calculate_object_importance(self, box_2d: List[float], obj_type: str, img_size: Tuple[int, int], base_importance: float, label: str, primary_subject: Optional[str], secondary_subjects: List[str], features: List[Dict]) -> float:
        """Calculate scaled importance score for animal art."""
        width, height = img_size
        y0, x0, y1, x1 = box_2d
        obj_width = (x1 - x0) * width / 1000
        obj_height = (y1 - y0) * height / 1000
        obj_area = obj_width * obj_height
        total_area = width * height
        coverage_ratio = obj_area / total_area

        # Animal-specific multipliers
        type_modifiers = {
            'animal': 3.0,
            'animal_face': 3.5, # Higher importance for faces
            'group_of_animals': 2.5,
            'interaction': 2.8,
            'figure': 1.5, # Humans might be present but animals are focus
            'object': 1.0,
            'terrain': 0.8,
            'water': 1.2,
            'plant': 0.7,
            'sky': 0.5,
            'unknown': 0.5,
        }

        obj_type_lower = obj_type.lower()
        label_lower = label.lower()

        # Base score from type modifier
        style_based_score = base_importance * type_modifiers.get(obj_type_lower, type_modifiers['unknown'])

        # Boost score if it's related to the primary or secondary subject
        primary_subject_lower = primary_subject.lower() if primary_subject else None
        secondary_subject_lowers = [sub.lower() for sub in secondary_subjects]

        if primary_subject_lower and (primary_subject_lower in label_lower or obj_type_lower in ['animal', 'animal_face', 'group_of_animals']):
             style_based_score *= 2.0 # Boost if label matches primary subject or is a key animal type
        elif any(sub_lower in label_lower for sub_lower in secondary_subject_lowers) or obj_type_lower in ['animal', 'animal_face', 'group_of_animals']:
             style_based_score *= 1.3 # Boost if label matches secondary subject or is a key animal type

        # Add bonus for detected facial features
        if features:
             style_based_score *= (1 + len(features) * 0.1) # 10% bonus per feature

        total_score = max(base_importance, style_based_score)

        # Scale by coverage (larger animals/groups are more important)
        coverage_factor = np.sqrt(coverage_ratio)
        final_score = total_score * (0.3 + 0.7 * coverage_factor)

        return min(1.0, final_score)


    def _threshold_important_objects(self, objects: List[Dict[str, Any]], percentile_threshold: float = 80) -> List[Dict[str, Any]]:
        """Filter objects for animal art - prioritize animals and subjects."""
        if not objects:
            return []

        importances = [obj.get('importance', 0) for obj in objects]
        if not importances:
            return objects

        threshold = np.percentile(importances, percentile_threshold)

        primary_subject_label = self.primary_subject.lower() if hasattr(self, 'primary_subject') and self.primary_subject else None
        secondary_subject_labels = [sub.lower() for sub in self.secondary_subjects] if hasattr(self, 'secondary_subjects') and self.secondary_subjects else []

        important = []
        for obj in objects:
            obj_label_lower = obj.get('label', '').lower()
            if obj.get('importance', 0) >= threshold or \
               obj.get('type') in ['animal', 'animal_face', 'group_of_animals'] or \
               (primary_subject_label and primary_subject_label in obj_label_lower) or \
               any(sub_label in obj_label_lower for sub_label in secondary_subject_labels):
                important.append(obj)

        return important


    def save_analysis_outputs(self, image_path: str, analysis_results: Dict[str, Any], output_folder: str) -> None:
        """Save all versions of the analyzed image."""
        try:
            basename = os.path.splitext(os.path.basename(image_path))[0]
            img = Image.open(image_path)

            logging.debug(f"[VisionAgentAnimal] Saving outputs for {basename} to {output_folder}")
            logging.debug(f"[VisionAgentAnimal] Analysis results: {json.dumps(analysis_results, indent=2, default=str)}")

            # 1. Save labeled version (boxes only)
            labeled_path = os.path.join(output_folder, f"{basename}_labeled.jpg")
            self._save_labeled_version(img.copy(), analysis_results, labeled_path)

            # 2. Save masked version (important objects only)
            masked_path = os.path.join(output_folder, f"{basename}_masked.jpg")
            self._save_masked_version(img.copy(), analysis_results, masked_path) # Pass masked_path

            # 3. Save cropped version
            cropped_path = os.path.join(output_folder, f"{basename}_cropped.jpg")
            self.copy_and_crop_image(image_path, cropped_path, analysis_results)

            logging.info(f"[VisionAgentAnimal] Saved all analysis outputs for {basename}")

            # TODO: Handoff cropped_path to UpscaleAgent queue

        except Exception as e:
            logging.exception(f"[VisionAgentAnimal] Error saving analysis outputs: {e}")


    def _save_labeled_version(self, img: Image.Image, analysis_results: Dict[str, Any], output_path: str) -> None:
        """Save version with bounding boxes for objects and labels for facial features."""
        try:
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            draw = ImageDraw.Draw(img)
            width, height = img.size

            # Build a map of feature colors based on unique labels in the data
            feature_labels = set()
            for obj in analysis_results.get("objects", []):
                for feature in obj.get("features", []):
                    if isinstance(feature, dict) and "coordinates" in feature:
                        feature_labels.add(feature.get("label", "").lower())
            feature_colors = {label: self.colors[i % len(self.colors)] 
                            for i, label in enumerate(sorted(feature_labels))}

            # Draw bounding boxes for objects and labels for facial features
            for i, obj in enumerate(analysis_results.get("objects", [])):
                obj_type = obj.get("type", "").lower()
                color = self.colors[i % len(self.colors)]
                
                # Draw object bounding box and label if box_2d exists
                if "box_2d" in obj:
                    box = obj["box_2d"]
                    x0 = int(box[1] * width / 1000)
                    y0 = int(box[0] * height / 1000)
                    x1 = int(box[3] * width / 1000)
                    y1 = int(box[2] * height / 1000)
                    
                    # Draw the box
                    line_width = 3 if 'face' in obj_type or 'head' in obj_type else 2
                    draw.rectangle([x0, y0, x1, y1], outline=color, width=line_width)
                    
                    # Draw simple object label above the box
                    label_text = f"{obj.get('label', '')} ({obj.get('importance', 0):.2f})"
                    draw.text((x0, y0 - 15), label_text, fill=color)

                # Draw facial features with coordinates
                features = obj.get("features", [])
                for j, feature in enumerate(features):
                    if isinstance(feature, dict) and "coordinates" in feature:
                        label = feature.get("label", "").lower()
                        coordinates = feature["coordinates"]
                        
                        # Normalize coordinates to image dimensions
                        if max(coordinates) <= 1.5:  # Assume 0-1 normalized
                            x_coord = int(coordinates[0] * width)
                            y_coord = int(coordinates[1] * height)
                        elif max(coordinates) > 100:  # Assume 0-1000 normalized
                            x_coord = int(coordinates[0] * width / 1000)
                            y_coord = int(coordinates[1] * height / 1000)
                        else:  # Assume raw pixel coordinates
                            x_coord = int(coordinates[0])
                            y_coord = int(coordinates[1])
                        
                        # Draw feature label at coordinates
                        feature_color = feature_colors.get(label, color) # Use object color if label not found
                        draw.text((x_coord, y_coord), label, fill=feature_color)
                    elif isinstance(feature, str):
                        logging.warning(f"[VisionAgentAnimal] Skipping non-facial feature: {feature}")
                    else:
                        # Log warning for other invalid feature types
                        logging.warning(f"[VisionAgentAnimal] Skipping non-facial feature: {feature}")

            # Save the labeled version
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img.save(output_path, quality=95)
            logging.info(f"[VisionAgentAnimal] Saved labeled version to {output_path}")
        except Exception as e:
            logging.exception(f"[VisionAgentAnimal] Error saving labeled version: {e}")


    def _save_masked_version(self, img: Image.Image, analysis_results: Dict[str, Any], output_path: str) -> None: # Changed parameter name
        """Save version with masks for important objects only."""
        try:
            width, height = img.size

            # Get important objects
            important_objects = self._threshold_important_objects(analysis_results.get("objects", []))

            # Create masks for important objects
            masks = self._create_segmentation_masks(important_objects, img.size)

            # Apply masks
            for i, mask in enumerate(masks):
                color = self.colors[i % len(self.colors)]
                img = self._overlay_mask(img, mask.mask, color)

            # Save in RGB mode
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img.save(output_path, format='JPEG', quality=95)

        except Exception as e:
            logging.exception(f"[VisionAgentAnimal] Error saving masked version: {e}")


    def _parse_gemini_response(self, response_text: Optional[str]) -> Optional[Dict]:
        """Parse Gemini API response text into JSON, handling potential errors."""
        tag = "[VisionAgentAnimal][GeminiParse]"
        if not response_text:
            logging.warning(f"{tag} Received empty response text.")
            return None # Indicate failure clearly
        try:
            # Clean up response text (remove markdown code blocks)
            json_str = response_text.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
            json_str = json_str.strip() # Strip again after removing backticks

            # Handle potentially empty string after cleanup
            if not json_str:
                 logging.warning(f"{tag} Response text was empty after cleanup.")
                 return None # Indicate failure clearly

            parsed_json = json.loads(json_str)
            # Basic validation: check if it's a dictionary
            if not isinstance(parsed_json, dict):
                logging.error(f"{tag} Parsed JSON is not a dictionary. Type: {type(parsed_json)}. Response: {response_text[:500]}...")
                return None # Indicate failure clearly
            logging.info(f"{tag} Successfully parsed response.")
            return parsed_json
        except json.JSONDecodeError as e:
            logging.error(f"{tag} Failed to parse JSON: {e}. Response snippet: {response_text[:500]}...")
            # Consider raising a specific exception for critical parsing failures
            # raise APIParsingError(f"{tag} Failed to parse JSON: {e}") from e
            return None # Indicate failure clearly
        except Exception as e:
            # Log unexpected errors with stack trace
            logging.exception(f"{tag} Unexpected error parsing response: {e}")
            return None # Indicate failure clearly

    def _parse_grok_response(self, response_text: Optional[str]) -> Optional[Dict]:
        """Parse Grok API response text into JSON, handling potential errors."""
        tag = "[VisionAgentAnimal][GrokParse]"
        if not response_text:
            logging.warning(f"{tag} Received empty response text.")
            return None # Indicate failure clearly
        try:
            # Clean up response text (remove markdown code blocks)
            json_str = response_text.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
            json_str = json_str.strip() # Strip again after removing backticks

            # Handle potentially empty string after cleanup
            if not json_str:
                 logging.warning(f"{tag} Response text was empty after cleanup.")
                 return None # Indicate failure clearly

            parsed_json = json.loads(json_str)
            # Basic validation: check if it's a dictionary
            if not isinstance(parsed_json, dict):
                logging.error(f"{tag} Parsed JSON is not a dictionary. Type: {type(parsed_json)}. Response: {response_text[:500]}...")
                return None # Indicate failure clearly
            logging.info(f"{tag} Successfully parsed response.")
            return parsed_json
        except json.JSONDecodeError as e:
            logging.error(f"{tag} Failed to parse JSON: {e}. Response snippet: {response_text[:500]}...")
            # Consider raising a specific exception for critical parsing failures
            # raise APIParsingError(f"{tag} Failed to parse JSON: {e}") from e
            return None # Indicate failure clearly
        except Exception as e:
            # Log unexpected errors with stack trace
            logging.exception(f"{tag} Unexpected error parsing response: {e}")
            return None # Indicate failure clearly

    def _create_segmentation_masks(self, detections: List[Dict[str, Any]], image_size: Tuple[int, int]) -> List[SegmentationMask]:
        """Create SegmentationMask objects from detections."""
        masks = []
        width, height = image_size
        for det in detections:
            if "mask" in det and "box_2d" in det:
                try:
                    box = det["box_2d"]
                    y0, x0, y1, x1 = int(box[0] * height / 1000), int(box[1] * width / 1000), \
                                     int(box[2] * height / 1000), int(box[3] * width / 1000)
                    
                    # Assuming mask is base64 encoded PNG or similar format
                    mask_data = base64.b64decode(det["mask"])
                    mask_img = Image.open(io.BytesIO(mask_data)).convert('L') # Convert to grayscale
                    mask_np = np.array(mask_img)
                    
                    # Resize mask to match image dimensions if needed
                    if mask_np.shape != (height, width):
                         mask_img_resized = mask_img.resize((width, height), Image.Resampling.NEAREST)
                         mask_np = np.array(mask_img_resized)

                    masks.append(SegmentationMask(
                        y0=y0, x0=x0, y1=y1, x1=x1,
                        mask=mask_np,
                        label=det.get("label", "unknown")
                    ))
                except Exception as e:
                    logging.error(f"[VisionAgentAnimal] Error processing mask for {det.get('label')}: {e}")
        return masks

    def _overlay_mask(self, img: Image.Image, mask: np.ndarray, color: str, alpha: float = 0.5) -> Image.Image:
        """Overlay a single mask onto the image."""
        try:
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            color_rgb = ImageColor.getrgb(color)
            mask_color = Image.new('RGBA', img.size, color_rgb + (0,)) # Transparent base
            
            # Create a boolean mask where the mask array is > threshold (e.g., 128)
            bool_mask = mask > 128 
            
            # Create an RGBA image from the boolean mask
            mask_rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
            mask_rgba[bool_mask] = color_rgb + (int(alpha * 255),) # Set color and alpha where mask is true
            
            mask_image = Image.fromarray(mask_rgba, 'RGBA')
            
            # Composite the mask onto the image
            img = Image.alpha_composite(img, mask_image)
            return img
        except Exception as e:
            logging.exception(f"[VisionAgentAnimal] Error overlaying mask: {e}")
            return img # Return original image on error


    def copy_and_crop_image(self, input_path: str, output_path: str, analysis_results: Dict[str, Any]) -> str:
        """
        Copy the image and crop it based on analysis results.
        Returns the output_path (str) on success.
        Raises CorruptedImageError or UnsupportedImageFormatError on failure.
        """
        try:
            # Attempt to open the image first to catch format/corruption errors early
            try:
                img = Image.open(input_path)
            except UnidentifiedImageError as e:
                error_msg = f"Cannot identify image file (corrupted or unsupported format): {input_path}. Error: {e}"
                logging.error(f"[VisionAgentAnimal] {error_msg}")
                raise CorruptedImageError(error_msg) from e
            except FileNotFoundError as e: # Keep FileNotFoundError specific
                error_msg = f"Input file not found for cropping: {input_path}. Error: {e}"
                logging.error(f"[VisionAgentAnimal] {error_msg}")
                raise # Re-raise FileNotFoundError as it's a distinct issue
            except IOError as e: # Catch other IOErrors that might indicate unsupported formats
                error_msg = f"IOError opening image for cropping (possibly unsupported format): {input_path}. Error: {e}"
                logging.error(f"[VisionAgentAnimal] {error_msg}")
                raise UnsupportedImageFormatError(error_msg) from e

            width, height = img.size
            target_aspect_ratio = 16 / 9
            target_width = int(min(width, height * target_aspect_ratio))
            target_height = int(target_width / target_aspect_ratio)

            # Ensure target dimensions are valid
            if target_width <= 0 or target_height <= 0:
                error_msg = f"Invalid target dimensions calculated: {target_width}x{target_height}"
                logging.error(f"[VisionAgentAnimal] {error_msg}")
                # This case should ideally not happen if image is valid, but good to keep a check.
                # For consistency, we could raise a generic ValueError or a custom one.
                # However, the main goal is image format errors, so we'll let this be for now
                # or consider if it should also be an exception.
                # For now, let's make it raise an error to be consistent with the new error handling.
                raise ValueError(error_msg)

            # Find the optimal crop box
            best_crop = self._find_optimal_crop(
                img.size,
                (target_width, target_height),
                analysis_results.get("objects", []),
                analysis_results.get("segmentation_masks", []) # Pass masks to cropping logic
            )

            if best_crop:
                cropped_img = img.crop(best_crop)
                # Ensure output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cropped_img.save(output_path, quality=95)
                logging.info(f"[VisionAgentAnimal] Saved cropped image to {output_path}")
                return output_path
            else:
                logging.warning("[VisionAgentAnimal] Could not determine optimal crop. Saving original as fallback.")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                img.save(output_path, quality=95)
                logging.info(f"[VisionAgentAnimal] Saved original image to {output_path} as fallback.")
                return output_path

        except FileNotFoundError: # Already handled by the initial Image.open try-except
            raise # Re-raise to be caught by tests or higher-level handlers
        except (CorruptedImageError, UnsupportedImageFormatError): # Re-raise custom exceptions
            raise
        except IOError as e: # Catch other IOErrors during save, etc.
            error_msg = f"IOError during image cropping/saving (possibly disk issue or format problem): {input_path}. Error: {e}"
            logging.error(f"[VisionAgentAnimal] {error_msg}")
            raise UnsupportedImageFormatError(error_msg) from e
        except Exception as e: # General fallback
            error_msg = f"Unexpected error cropping image {input_path}: {e}"
            logging.exception(f"[VisionAgentAnimal] {error_msg}")
            # Convert generic exceptions to a custom one or re-raise if appropriate
            # For now, let's raise a generic runtime error to signal failure.
            raise RuntimeError(error_msg) from e

    def _find_optimal_crop(self, img_size: Tuple[int, int], target_size: Tuple[int, int],
                           objects: List[Dict[str, Any]], masks: List[SegmentationMask]) -> Optional[Tuple[int, int, int, int]]:
        """Find the best 16:9 crop box."""
        width, height = img_size
        target_width, target_height = target_size
        best_crop = None
        best_score = -1

        # Convert object boxes to pixel coordinates
        pixel_objects = []
        for obj in objects:
            if "box_2d" in obj:
                box = obj["box_2d"]
                pixel_box = [
                    int(box[1] * width / 1000),   # x0
                    int(box[0] * height / 1000),  # y0
                    int(box[3] * width / 1000),   # x1
                    int(box[2] * height / 1000)   # y1
                ]
                pixel_objects.append({
                    "box": pixel_box,
                    "importance": obj.get("importance", 0.5),
                    "type": obj.get("type", "object"),
                    "label": obj.get("label", "") # Include label for cropping logic
                })

        # Add mask regions as objects
        for mask in masks:
            pixel_objects.append({
                "box": [mask.x0, mask.y0, mask.x1, mask.y1],
                "importance": 0.7,  # High importance for masked regions
                "type": "mask",
                "label": mask.label # Include mask label
            })

        # Try different crop positions
        step = min(width, height) // 20  # Adjust step size based on image size
        for x in range(0, width - target_width + 1, step):
            for y in range(0, height - target_height + 1, step):
                score = self._evaluate_crop(
                    (x, y, x + target_width, y + target_height),
                    pixel_objects,
                    self.primary_subject, # Pass primary subject for cropping logic
                    self.secondary_subjects # Pass secondary subjects
                )
                if score > best_score:
                    best_score = score
                    best_crop = (x, y, x + target_width, y + target_height)

        # Abstract-specific cropping refinement (Roadmap Step 7)
        # if best_crop:
        #      best_crop = self._refine_abstract_crop(best_crop, img_size, target_size, pixel_objects)


        return best_crop


    # _refine_abstract_crop removed: not needed for animal art


    def _calculate_box_overlap(self, box1: List[float], box2: List[float]) -> float:
        """Calculate the intersection over union (IoU) between two bounding boxes."""
        # Convert from [y0, x0, y1, x1] to [x0, y0, x1, y1] for calculation
        b1 = [box1[1], box1[0], box1[3], box1[2]]
        b2 = [box2[1], box2[0], box2[3], box2[2]]

        # Calculate intersection
        x_left = max(b1[0], b2[0])
        y_top = max(b1[1], b2[1])
        x_right = min(b1[2], b2[2])
        y_bottom = min(b1[3], b2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)

        # Calculate areas
        b1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
        b2_area = (b2[2] - b2[0]) * (b2[3] - b2[1])

        # Calculate IoU
        union = b1_area + b2_area - intersection
        return intersection / union if union > 0 else 0.0


    def _evaluate_crop(self, crop_box: Tuple[int, int, int, int],
                       objects: List[Dict[str, Any]], primary_subject: Optional[str], secondary_subjects: List[str]) -> float:
        """Calculate score for a potential crop based on included objects."""
        left, top, right, bottom = crop_box
        score = 0.0
        crop_area = (right - left) * (bottom - top)

        primary_subject_lower = primary_subject.lower() if primary_subject else None
        secondary_subject_lowers = [sub.lower() for sub in secondary_subjects]

        for obj in objects:
            obj_box = obj["box"]
            obj_left, obj_top, obj_right, obj_bottom = obj_box
            obj_area = (obj_right - obj_left) * (obj_bottom - obj_top)

            # Calculate intersection area
            inter_left = max(left, obj_left)
            inter_top = max(top, obj_top)
            inter_right = min(right, obj_right)
            inter_bottom = min(bottom, obj_bottom)

            intersection = max(0, inter_right - inter_left) * max(0, inter_bottom - inter_top)

            if intersection > 0:
                overlap_ratio = intersection / obj_area if obj_area > 0 else 0
                importance = obj.get("importance", 0.5)
                obj_label_lower = obj.get("label", "").lower()

                # Base score based on importance and overlap
                obj_score = importance * overlap_ratio

                # Boost score for primary subject
                if primary_subject_lower and primary_subject_lower in obj_label_lower:
                    obj_score *= 3.0 # Strong boost for primary subject

                # Boost score for secondary subjects
                elif any(sub_lower in obj_label_lower for sub_lower in secondary_subject_lowers):
                    obj_score *= 1.5 # Moderate boost for secondary subjects

                # Boost score for faces/heads
                if obj.get("type") in ["animal_face", "face", "head"]:
                    obj_score *= 2.0

                # Penalize crops that cut off important objects significantly
                if overlap_ratio < 0.8 and importance > 0.6:
                    obj_score *= 0.5 # Penalize partial inclusion of important objects

                score += obj_score

        # Normalize score by crop area (optional, can help balance density vs coverage)
        # score /= (crop_area ** 0.5) if crop_area > 0 else 1

        return score


    def handoff_to_upscale(self, cropped_image_path: str, upscale_queue: Queue):
        """Send the cropped image path to the upscale agent's queue."""
        try:
            upscale_queue.put(cropped_image_path)
            logging.info(f"[VisionAgentAnimal] Handed off {os.path.basename(cropped_image_path)} to upscale queue.")
        except Exception as e:
            logging.exception(f"[VisionAgentAnimal] Error handing off to upscale queue: {e}")
