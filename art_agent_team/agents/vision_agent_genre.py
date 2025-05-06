import logging
import os
import json
import numpy as np
import base64
import io
from PIL import Image, ImageDraw, ImageFont, ImageColor, UnidentifiedImageError
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from openai import OpenAI  # xAI Grok API client (OpenAI-compatible)
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from queue import Queue # Import Queue

# Custom Exceptions for Image Processing Errors (consistent with other vision agents)
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

class VisionAgentGenre:
    """Agent for analyzing genre-scene art using Gemini and Grok Vision APIs."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the VisionAgentGenre with configuration."""
        self.config = config if config is not None else {}

        self.output_folder = config.get('output_folder', 'output')

        # Initialize Gemini models
        # Enhanced API key retrieval with detailed error handling and fallback
        google_api_key = config.get('google_api_key') if config and 'google_api_key' in config else os.environ.get('GOOGLE_API_KEY')
        if not google_api_key:
            logging.warning("VisionAgentGenre: WARNING: No GOOGLE_API_KEY found in config or environment variables. Gemini Pro will be unavailable. Analysis will proceed with other available models.")
            self.gemini_pro = None
        else:
            try:
                genai.configure(api_key=google_api_key)
                # USER REQUIREMENT: Use gemini-2.5-pro-exp-03-25 for vision.
                self.gemini_pro = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')
                logging.info("VisionAgentGenre: Gemini Pro model initialized.")
            except Exception as e:
                logging.warning(f"VisionAgentGenre: Failed to initialize Gemini Pro model due to missing key or other issue: {e}. Analysis will proceed without Gemini Pro.")
                self.gemini_pro = None

        grok_api_key = config.get('grok_api_key') if config and 'grok_api_key' in config else os.environ.get('GROK_API_KEY')
        if not grok_api_key:
            logging.warning("VisionAgentGenre: WARNING: No GROK_API_KEY found in config or environment variables. Grok Vision will be unavailable. Analysis will proceed with other available models.")
            self.grok_client = None
        else:
            try:
                from openai import OpenAI
                self.grok_client = OpenAI(
                    api_key=grok_api_key,
                    base_url="https://api.x.ai/v1",
                )
                self.grok_vision_model = "grok-2-vision-1212" # Make sure this model name is correct
                logging.info(f"VisionAgentGenre: Grok Vision client initialized with model {self.grok_vision_model}.")
            except Exception as e:
                logging.warning(f"VisionAgentGenre: Failed to initialize Grok Vision client due to missing key or other issue: {e}. Analysis will proceed without Grok Vision.")
                self.grok_client = None

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
        Analyzes genre-scene image using Gemini Pro and Grok Vision,
        incorporating research data.
        Enhanced: If both Gemini Pro and Grok Vision clients are unavailable, log error and return structured error response.
        """
        self._think_with_grok(f"About to analyze genre-scene artwork image: {os.path.basename(image_path)}. What reasoning steps should I take to ensure a thorough and accurate analysis?")
        if not self.gemini_pro and not self.grok_client:
            logging.error("VisionAgentGenre: Both Gemini Pro and Grok Vision clients are unavailable due to missing API keys.")
            return {
                "error": "Both Gemini Pro and Grok Vision clients are unavailable due to missing API keys.",
                "objects": [],
                "segmentation_masks": [],
                "image_size": None,
                "relationships": [],
                "style": research_data.get("style", "genre-scene") if isinstance(research_data, dict) else "genre-scene",
                "primary_subject": None,
                "secondary_subjects": []
            }
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
            except UnidentifiedImageError as e:
                logging.error(f"VisionAgentGenre: Cannot identify image file (corrupted or unsupported format) in analyze_image: {image_path}. Error: {e}")
                raise CorruptedImageError(f"Corrupted or unsupported image file: {image_path}") from e
            except IOError as e: # Catch other IOErrors that might indicate unsupported formats
                logging.error(f"VisionAgentGenre: IOError opening image in analyze_image: {image_path}. Error: {e}")
                raise UnsupportedImageFormatError(f"IOError, possibly unsupported image format: {image_path}") from e
            
            # Resize for faster processing, maintain aspect ratio
            img.thumbnail([1024, 1024], Image.Resampling.LANCZOS)

            # Initialize results structure
            results = {
                "objects": [],
                "segmentation_masks": [],
                "image_size": img.size,
                "relationships": [], # Relationships might be added by LLMs
                "style": research_data.get("style", "genre-scene"),
                "primary_subject": self.primary_subject,
                "secondary_subjects": self.secondary_subjects
            }

            # Genre-scene specific prompt focusing on people, animals, objects, and context
            genre_prompt = f"""
            Analyze this genre scene painting. Focus on identifying and localizing people, animals, and significant objects within the scene. Pay attention to their actions, expressions, and the overall setting.

            Research Information:
            Paragraph Summary: {paragraph_description}
            Structured Sentence: {structured_sentence}

            Identify objects and their bounding boxes [y0, x0, y1, x1] (normalized 0-1000). Prioritize:
            - All people (full bodies, faces, groups)
            - All animals (full bodies, faces, groups)
            - Important objects related to the scene or figures
            - Elements of the setting (buildings, furniture, tools, etc.)
            - Note if figures/animals are primary/secondary subjects.

            For each identified object, provide:
            - label: Concise description (e.g., "woman sewing", "group of children playing", "dog sleeping", "wooden table")
            - box_2d: [y0, x0, y1, x1]
            - confidence: 0.0-1.0
            - type: Category (e.g., "person", "animal", "object", "group", "setting_element")
            - importance: Initial estimate 0.0-1.0 (will be refined by agent)
            - features: List of detected facial features if applicable

            Format as JSON:
            {{
                "objects": [
                    {{
                        "label": "...",
                        "box_2d": [y0, x0, y1, x1],
                        "confidence": ...,
                        "type": "...",
                        "importance": ...,
                        "features": [...]
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
                        contents=[genre_prompt, img],
                        generation_config=GenerationConfig(
                            temperature=0.4,
                            top_p=0.8,
                            top_k=40
                        )
                    )
                    gemini_results = self._parse_gemini_response(gemini_response.text)
                    if gemini_results and "objects" in gemini_results:
                        gemini_objects = gemini_results["objects"]
                        logging.info(f"VisionAgentGenre: Gemini Pro identified {len(gemini_objects)} objects.")
                except Exception as e:
                    logging.error(f"VisionAgentGenre: Error in Gemini Pro analysis: {e}")

            # 2. Grok Vision Analysis
            grok_objects = []
            if self.grok_client:
                 try:
                    grok_response = self.grok_client.chat.completions.create(
                        messages=[
                            {"role": "user", "content": [{"type": "text", "text": genre_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(content).decode('utf-8')}"}}]}
                        ],
                        model=self.grok_vision_model,
                        temperature=0.4,
                        max_tokens=2048
                    )
                    grok_results = self._parse_grok_response(grok_response.choices[0].message.content)
                    if grok_results and "objects" in grok_results:
                        grok_objects = grok_results["objects"]
                        logging.info(f"VisionAgentGenre: Grok Vision identified {len(grok_objects)} objects.")
                 except Exception as e:
                    logging.error(f"VisionAgentGenre: Error in Grok Vision analysis: {e}")


            # --- Result Aggregation ---
            combined_objects = self._aggregate_results(gemini_objects, grok_objects)
            results["objects"] = combined_objects
            logging.info(f"VisionAgentGenre: Aggregated {len(results['objects'])} objects from dual models.")

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
                      logging.info(f"VisionAgentGenre: Boosted primary subject '{obj['label']}' importance to {obj['importance']:.2f}")


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
            logging.exception(f"VisionAgentGenre: Error in analyze_image: {e}")
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
        """Calculate scaled importance score for genre-scene art."""
        width, height = img_size
        y0, x0, y1, x1 = box_2d
        obj_width = (x1 - x0) * width / 1000
        obj_height = (y1 - y0) * height / 1000
        obj_area = obj_width * obj_height
        total_area = width * height
        coverage_ratio = obj_area / total_area

        # Genre-scene specific multipliers
        type_modifiers = {
            'person': 3.0,
            'human_figure': 2.8,
            'human_face': 3.2, # Faces are important for expressions/identity
            'animal': 2.0,
            'animal_figure': 1.8,
            'animal_face': 2.2,
            'group': 2.5, # Groups of people/animals
            'interaction': 2.8, # Interactions are key to the scene
            'object': 1.5, # Objects relevant to the scene/figures
            'setting_element': 1.0, # Furniture, buildings, etc.
            'background': 0.5, # Discount background elements
            'unknown': 1.0,
        }

        obj_type_lower = obj_type.lower()
        label_lower = label.lower()

        # Base score from type modifier
        style_based_score = base_importance * type_modifiers.get(obj_type_lower, type_modifiers['unknown'])

        # Boost score if it's related to the primary or secondary subject
        primary_subject_lower = primary_subject.lower() if primary_subject else None
        secondary_subject_lowers = [sub.lower() for sub in secondary_subjects]

        if primary_subject_lower and (primary_subject_lower in label_lower or obj_type_lower in ['person', 'human_figure', 'human_face', 'animal', 'animal_figure', 'animal_face']):
             style_based_score *= 2.0 # Boost if label matches primary subject or is a key figure/animal type
        elif any(sub_lower in label_lower for sub_lower in secondary_subject_lowers) or obj_type_lower in ['person', 'human_figure', 'human_face', 'animal', 'animal_figure', 'animal_face']:
             style_based_score *= 1.3 # Boost if label matches secondary subject or is a key figure/animal type

        # Add bonus for detected facial features
        if features:
             style_based_score *= (1 + len(features) * 0.1) # Bonus for facial features

        total_score = max(base_importance, style_based_score)

        # Scale by coverage (larger figures/groups are more important)
        coverage_factor = np.sqrt(coverage_ratio)
        final_score = total_score * (0.4 + 0.6 * coverage_factor)

        return min(1.0, final_score)


    def _threshold_important_objects(self, objects: List[Dict[str, Any]], percentile_threshold: float = 80) -> List[Dict[str, Any]]:
        """Filter objects for genre-scene art - prioritize figures, interactions, and subjects."""
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
               obj.get('type') in ['person', 'human_figure', 'human_face', 'animal', 'animal_figure', 'animal_face', 'group', 'interaction'] or \
               (primary_subject_label and primary_subject_label in obj_label_lower) or \
               any(sub_label in obj_label_lower for sub_label in secondary_subject_labels):
                important.append(obj)

        return important


    def save_analysis_outputs(self, image_path: str, analysis_results: Dict[str, Any], output_folder: str) -> None:
        """Save all versions of the analyzed image."""
        try:
            basename = os.path.splitext(os.path.basename(image_path))[0]
            img = Image.open(image_path)

            # 1. Save labeled version (boxes only)
            labeled_path = os.path.join(output_folder, f"{basename}_labeled.jpg")
            self._save_labeled_version(img.copy(), analysis_results, labeled_path)

            # 2. Save masked version (important objects only)
            masked_path = os.path.join(output_folder, f"{basename}_masked.jpg")
            self._save_masked_version(img.copy(), analysis_results, masked_path)

            # 3. Save cropped version
            cropped_path = os.path.join(output_folder, f"{basename}_cropped.jpg")
            self.copy_and_crop_image(image_path, cropped_path, analysis_results)

            logging.info(f"VisionAgentGenre: Saved all analysis outputs for {basename}")

            # TODO: Handoff cropped_path to UpscaleAgent queue

        except Exception as e:
            logging.exception(f"VisionAgentGenre: Error saving analysis outputs: {e}")


    def _save_labeled_version(self, img: Image.Image, analysis_results: Dict[str, Any], output_path: str) -> None:
        """Save version with bounding boxes and labels."""
        try:
            # Ensure image is in RGBA mode for drawing with transparency
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            draw = ImageDraw.Draw(img)
            width, height = img.size

            # Draw bounding boxes and labels for all objects
            for i, obj in enumerate(analysis_results.get("objects", [])):
                obj_type = obj.get("type", "").lower()
                color = self.colors[i % len(self.colors)]
                box = obj["box_2d"]

                # Get raw coordinates from model
                raw_box = obj["box_2d"]  # [y0, x0, y1, x1]

                # Convert to image coordinates without adjustments
                x0 = int(raw_box[1] * width / 1000)
                y0 = int(raw_box[0] * height / 1000)
                x1 = int(raw_box[3] * width / 1000)
                y1 = int(raw_box[2] * height / 1000)

                # Draw center point marker (Red 'X')
                center_x = (x0 + x1) // 2
                center_y = (y0 + y1) // 2
                marker_size = 10
                draw.line([center_x - marker_size, center_y - marker_size,
                          center_x + marker_size, center_y + marker_size],
                         fill='red', width=2)
                draw.line([center_x - marker_size, center_y + marker_size,
                          center_x + marker_size, center_y - marker_size],
                         fill='red', width=2)

                # Draw original bounding box
                line_width = 4 if 'face' in obj_type or 'head' in obj_type else 2
                draw.rectangle([x0, y0, x1, y1], outline=color, width=line_width)

                # Draw object label with raw coordinates for debugging
                label_parts = [
                    f"{obj['label']} ({obj.get('importance', 0):.2f})",
                    f"Type: {obj.get('type', 'unknown')}",
                    f"Raw coords: [{raw_box[0]:.0f}, {raw_box[1]:.0f}, {raw_box[2]:.0f}, {raw_box[3]:.0f}]"
                ]
                if 'orientation' in obj:
                    label_parts.append(f"View: {obj['orientation']}")

                # Draw multi-line label with background for readability
                for j, text in enumerate(label_parts):
                    text_pos = (x0, y0-20-(j*15))
                    text_bbox = draw.textbbox(text_pos, text)
                    bg_box = [
                        text_bbox[0]-2, text_bbox[1]-2,
                        text_bbox[2]+2, text_bbox[3]+2
                    ]
                    draw.rectangle(bg_box, fill=(0, 0, 0, 180)) # Semi-transparent black background
                    draw.text(text_pos, text, fill=color)

                # Draw features if present
                features = obj.get("features", [])
                if features:
                    for j, feature in enumerate(features):
                        # Color coding for different feature types
                        feature_colors = {
                            'right_eye': 'blue',
                            'left_eye': 'blue',
                            'nose': 'green',
                            'lips': 'red',
                            'mouth': 'red',
                            'teeth': 'white',
                            'right_ear': 'purple',
                            'left_ear': 'purple',
                            'hairline': 'yellow',
                            'neck': 'orange',
                            'beak': 'gold',
                            'snout': 'brown'
                        }
                        feature_color_name = feature_colors.get(feature.get('label', '').lower(),
                                                              self.colors[(i + j + 1) % len(self.colors)])

                        # Adjust color intensity based on status and confidence
                        confidence = feature.get('confidence', 0.5)
                        status = feature.get('status', 'full')
                        base_color_rgb = ImageColor.getrgb(feature_color_name)

                        # For partial/occluded features, use lighter/desaturated colors
                        if status != 'full':
                            fade_factor = 0.6 if status == 'partial' else 0.4
                            r, g, b = base_color_rgb
                            r = int(r + (255 - r) * (1 - fade_factor))
                            g = int(g + (255 - g) * (1 - fade_factor))
                            b = int(b + (255 - b) * (1 - fade_factor))
                            base_color_rgb = (r, g, b)

                        feature_box = feature["box_2d"]

                        # Get raw feature coordinates
                        raw_fx0 = int(feature_box[1] * width / 1000)
                        raw_fy0 = int(feature_box[0] * height / 1000)
                        raw_fx1 = int(feature_box[3] * width / 1000)
                        raw_fy1 = int(feature_box[2] * height / 1000)

                        # Draw feature center point with 'x' marker
                        f_center_x = (raw_fx0 + raw_fx1) // 2
                        f_center_y = (raw_fy0 + raw_fy1) // 2
                        marker_size = 5
                        draw.line([f_center_x - marker_size, f_center_y - marker_size,
                                 f_center_x + marker_size, f_center_y + marker_size],
                                fill=base_color_rgb, width=2)
                        draw.line([f_center_x - marker_size, f_center_y + marker_size,
                                 f_center_x + marker_size, f_center_y - marker_size],
                                fill=base_color_rgb, width=2)

                        # Draw connecting line from feature center to head center
                        head_center_x = (x0 + x1) // 2
                        head_center_y = (y0 + y1) // 2
                        # Use a slightly lighter version for the connecting line
                        faded_color_rgb = tuple(min(255, int(c * 1.3)) for c in base_color_rgb)
                        draw.line([f_center_x, f_center_y, head_center_x, head_center_y],
                                fill=faded_color_rgb, width=1, joint="curve")

                        # Draw feature box with dashed outline
                        box_points = [
                            [(raw_fx0, raw_fy0), (raw_fx1, raw_fy0)],  # top
                            [(raw_fx1, raw_fy0), (raw_fx1, raw_fy1)],  # right
                            [(raw_fx1, raw_fy1), (raw_fx0, raw_fy1)],  # bottom
                            [(raw_fx0, raw_fy1), (raw_fx0, raw_fy0)]   # left
                        ]

                        # Draw dashed box
                        dash_pattern = [4, 4]
                        for start, end in box_points:
                            x0_seg, y0_seg = start
                            x1_seg, y1_seg = end
                            dx = x1_seg - x0_seg
                            dy = y1_seg - y0_seg
                            distance = int(((dx ** 2) + (dy ** 2)) ** 0.5)

                            if distance > 0:
                                num_segments = distance // sum(dash_pattern)
                                for k_seg in range(num_segments + 1):
                                     segment_start = k_seg * sum(dash_pattern)
                                     if segment_start >= distance:
                                         break

                                     segment_end = min(segment_start + dash_pattern[0], distance)
                                     t1 = segment_start / distance
                                     t2 = segment_end / distance

                                     line_start = (int(x0_seg + t1 * dx), int(y0_seg + t1 * dy))
                                     line_end = (int(x0_seg + t2 * dx), int(y0_seg + t2 * dy))

                                     # Adjust line width based on confidence
                                     line_width = max(1, int(confidence * 3))
                                     draw.line([line_start, line_end], fill=base_color_rgb, width=line_width)

                        # Draw feature label with confidence and status
                        feature_label_parts = [
                            f"{feature['label']} ({feature.get('confidence', 0):.2f})",
                            f"Status: {feature.get('status', 'full')}",
                            f"[{feature_box[0]:.0f}, {feature_box[1]:.0f}, {feature_box[2]:.0f}, {feature_box[3]:.0f}]"
                        ]

                        for k_label, text in enumerate(feature_label_parts):
                            text_pos = (raw_fx0, raw_fy0-25-(k_label*15)) # Adjusted offset

                            # Ensure image is RGBA for drawing transparent background
                            img_rgba_copy = img.copy().convert('RGBA') # Draw on a copy
                            draw_rgba = ImageDraw.Draw(img_rgba_copy)

                            text_bbox = draw_rgba.textbbox(text_pos, text)

                            # Draw background with padding
                            bg_box = [
                                text_bbox[0]-3, text_bbox[1]-2,
                                text_bbox[2]+3, text_bbox[3]+2
                            ]

                            # Draw semi-transparent dark background
                            draw_rgba.rectangle(bg_box, fill=(0, 0, 0, 180)) # Use RGBA for transparency

                            # Draw white outline for better contrast
                            draw_rgba.rectangle(bg_box, outline=(255, 255, 255), width=1)

                            # Draw text in feature color (using base RGB color)
                            draw_rgba.text(text_pos, text, fill=base_color_rgb)

                            # Composite back onto original image
                            img = Image.alpha_composite(img, img_rgba_copy)
                            draw = ImageDraw.Draw(img) # Update draw object for original image

            # Save in RGB mode
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            # Ensure output directory exists before saving
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img.save(output_path, format='JPEG', quality=95)

        except Exception as e:
            logging.exception(f"Error saving labeled version: {e}")


    def _save_masked_version(self, img: Image.Image, analysis_results: Dict[str, Any], output_path: str) -> None:
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
            logging.exception(f"Error saving masked version: {e}")


    def _parse_gemini_response(self, response_text: str) -> Optional[Dict]:
        """Parse Gemini API response text into JSON."""
        try:
            json_str = response_text.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            elif json_str.startswith("```"):
                json_str = json_str[3:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
            json_str = json_str.strip()
            if not json_str:
                logging.warning("Gemini response was empty after cleaning markdown/whitespace.")
                return None
            return json.loads(json_str)
        except Exception as e:
            logging.error(f"Failed to parse Gemini response: {e}")
            # logging.debug(f"Raw response text removed for security.")
            return None

    def _parse_grok_response(self, response_text: str) -> Optional[Dict]:
        """Parse Grok API response text into JSON."""
        try:
            json_str = response_text.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            elif json_str.startswith("```"):
                json_str = json_str[3:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
            json_str = json_str.strip()
            if not json_str:
                logging.warning("Grok response was empty after cleaning markdown/whitespace.")
                return None
            return json.loads(json_str)
        except Exception as e:
            logging.error(f"Failed to parse Grok response: {e}")
            # logging.debug(f"Raw response text removed for security.")
            return None


    def _create_segmentation_masks(self, detections: List[Dict[str, Any]], image_size: Tuple[int, int]) -> List[SegmentationMask]:
        """Convert detection results to segmentation masks."""
        width, height = image_size
        masks = []

        for det in detections:
            try:
                box = det["box_2d"]
                y0 = int(box[0] * height / 1000)
                x0 = int(box[1] * width / 1000)
                y1 = int(box[2] * height / 1000)
                x1 = int(box[3] * width / 1000)

                # Ensure coordinates are within bounds
                y0 = max(0, min(y0, height - 1))
                x0 = max(0, min(x0, width - 1))
                y1 = max(0, min(y1, height))
                x1 = max(0, min(x1, width))

                if y1 > y0 and x1 > x0:
                    # Create binary mask
                    mask = np.zeros((height, width), dtype=np.uint8)
                    mask[y0:y1, x0:x1] = 255

                    masks.append(SegmentationMask(
                        y0=y0, x0=x0, y1=y1, x1=x1,
                        mask=mask,
                        label=det["label"]
                    ))
            except Exception as e:
                logging.warning(f"Failed to create mask for {det.get('label', 'unknown')}: {e}")
                continue

        return masks

    def _overlay_mask(self, img: Image.Image, mask: np.ndarray, color: str, alpha: float = 0.5) -> Image.Image:
        """Overlay a segmentation mask on an image."""
        try:
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            # Convert color name to RGB
            rgb = ImageColor.getrgb(color)

            # Ensure mask is 2D and correct size
            if mask.shape != img.size[::-1]:  # PIL uses (width, height), numpy uses (height, width)
                mask = np.resize(mask, img.size[::-1])

            # Convert to uint8 if needed
            if mask.dtype != np.uint8:
                mask = (mask > 0).astype(np.uint8) * 255

            # Create mask overlay
            mask_img = Image.fromarray(mask, mode='L')  # Use 'L' mode for grayscale
            draw.bitmap((0, 0), mask_img, fill=(*rgb, int(255 * alpha)))

            # Composite images
            return Image.alpha_composite(img.convert('RGBA'), overlay)

        except Exception as e:
            logging.warning(f"Failed to overlay mask: {e}", exc_info=True)
            return img


    def copy_and_crop_image(self, input_path: str, output_path: str, analysis_results: Dict[str, Any]) -> str:
        """
        Intelligently crop image based on object detection results.
        Returns the output_path (str) on success.
        Raises CorruptedImageError or UnsupportedImageFormatError on failure.
        """
        try:
            # Attempt to open the image first to catch format/corruption errors early
            try:
                img = Image.open(input_path)
            except UnidentifiedImageError as e:
                error_msg = f"Cannot identify image file (corrupted or unsupported format) for genre: {input_path}. Error: {e}"
                logging.error(f"[VisionAgentGenre] {error_msg}")
                raise CorruptedImageError(error_msg) from e
            except FileNotFoundError as e: # Keep FileNotFoundError specific
                error_msg = f"Input file not found for genre cropping: {input_path}. Error: {e}"
                logging.error(f"[VisionAgentGenre] {error_msg}")
                raise # Re-raise FileNotFoundError
            except IOError as e: # Catch other IOErrors that might indicate unsupported formats
                error_msg = f"IOError opening image for genre cropping (possibly unsupported format): {input_path}. Error: {e}"
                logging.error(f"[VisionAgentGenre] {error_msg}")
                raise UnsupportedImageFormatError(error_msg) from e

            width, height = img.size
            target_ratio = 16/9
            current_ratio = width/height

            # Calculate target dimensions
            if current_ratio > target_ratio:
                new_width = int(height * target_ratio)
                new_height = height
            else:
                new_width = width
                new_height = int(width / target_ratio)

            if new_width <= 0 or new_height <= 0:
                error_msg = f"Invalid target dimensions calculated for genre crop: {new_width}x{new_height}"
                logging.error(f"[VisionAgentGenre] {error_msg}")
                raise ValueError(error_msg) # Consistent error raising

            # Find optimal crop window
            best_crop = self._find_optimal_crop(
                img_size=(width, height),
                target_size=(new_width, new_height),
                objects=analysis_results.get("objects", []),
                masks=analysis_results.get("segmentation_masks", [])
            )
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True) # Ensure output dir exists

            # Apply crop
            if best_crop:
                cropped = img.crop(best_crop)
                if cropped.mode == 'RGBA':
                    cropped = cropped.convert('RGB')
                cropped.save(output_path, format='JPEG', quality=95)
                logging.info(f"[VisionAgentGenre] Saved cropped image to {output_path}")
                return output_path
            else:
                # Fallback to center crop
                logging.warning("[VisionAgentGenre] Could not determine optimal crop. Using center crop as fallback.")
                left = (width - new_width) // 2
                top = (height - new_height) // 2
                right = left + new_width
                bottom = top + new_height

                if left < 0 or top < 0 or right > width or bottom > height or new_width <= 0 or new_height <= 0:
                    error_msg = f"Invalid center crop dimensions for genre: L{left} T{top} R{right} B{bottom} for image {width}x{height}. Saving original."
                    logging.error(f"[VisionAgentGenre] {error_msg}")
                    img_rgb = img.convert('RGB') if img.mode == 'RGBA' else img
                    img_rgb.save(output_path, format='JPEG', quality=95)
                    return output_path # Still return path, but it's the original

                img_rgb = img.convert('RGB') if img.mode == 'RGBA' else img
                cropped_fallback = img_rgb.crop((left, top, right, bottom))
                cropped_fallback.save(output_path, format='JPEG', quality=95)
                logging.info(f"[VisionAgentGenre] Saved center-cropped image to {output_path} as fallback.")
                return output_path

        except FileNotFoundError: # Already handled by the initial Image.open try-except
            raise
        except (CorruptedImageError, UnsupportedImageFormatError): # Re-raise custom exceptions
            raise
        except IOError as e: # Catch other IOErrors during save, etc.
            error_msg = f"IOError during genre image cropping/saving: {input_path}. Error: {e}"
            logging.error(f"[VisionAgentGenre] {error_msg}")
            raise UnsupportedImageFormatError(error_msg) from e
        except Exception as e: # General fallback
            error_msg = f"Unexpected error cropping genre image {input_path}: {e}"
            logging.exception(f"[VisionAgentGenre] {error_msg}")
            raise RuntimeError(error_msg) from e

    def _find_optimal_crop(self, img_size: Tuple[int, int], target_size: Tuple[int, int],
                          objects: List[Dict[str, Any]], masks: List[SegmentationMask]) -> Optional[Tuple[int, int, int, int]]:
        """Find optimal crop position based on detected objects and masks."""
        width, height = img_size
        target_width, target_height = target_size
        best_score = float('-inf')
        best_crop = None

        # Convert normalized coordinates to pixels and combine with masks
        pixel_objects = []
        for obj in objects:
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

        # Genre-scene specific cropping refinement (Roadmap Step 7)
        if best_crop:
             best_crop = self._refine_genre_crop(best_crop, img_size, target_size, pixel_objects, self.primary_subject)


        return best_crop


    def _refine_genre_crop(self, current_crop: Tuple[int, int, int, int], img_size: Tuple[int, int], target_size: Tuple[int, int], objects: List[Dict[str, Any]], primary_subject: Optional[str]) -> Tuple[int, int, int, int]:
        """
        Refines the crop for genre scenes based on preserving contextual action and primary subject.
        (Roadmap Step 7 - Genre/Religious/Historical specific)
        """
        img_width, img_height = img_size
        target_width, target_height = target_size
        crop_left, crop_top, crop_right, crop_bottom = current_crop

        # Find the primary subject object
        primary_subject_obj = None
        if primary_subject:
             primary_subject_lower = primary_subject.lower()
             primary_subject_obj = next((obj for obj in objects if primary_subject_lower in obj.get('label', '').lower()), None)

        # If primary subject found, try to keep it in frame and maintain centrality
        if primary_subject_obj:
             ps_box = primary_subject_obj["box"]
             ps_center_x = (ps_box[0] + ps_box[2]) // 2
             ps_center_y = (ps_box[1] + ps_box[3]) // 2

             # Calculate desired crop center based on primary subject
             desired_crop_center_x = ps_center_x
             desired_crop_center_y = ps_center_y

             # Calculate new top-left corner based on desired center
             new_left = max(0, desired_crop_center_x - target_width // 2)
             new_top = max(0, desired_crop_center_y - target_height // 2)

             # Adjust to stay within image bounds
             if new_left + target_width > img_width:
                  new_left = img_width - target_width
             if new_top + target_height > img_height:
                  new_top = img_height - target_height

             # Ensure left/top are not negative after adjustments
             new_left = max(0, new_left)
             new_top = max(0, new_top)

             adjusted_crop = (new_left, new_top, new_left + target_width, new_top + target_height)
             logging.info(f"VisionAgentGenre: Refined crop to center on primary subject: {adjusted_crop}")
             return adjusted_crop

        # If no primary subject or complex contextual action, fallback to original best crop
        return current_crop


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

        primary_subject_lower = primary_subject.lower() if primary_subject else None
        secondary_subject_lowers = [sub.lower() for sub in secondary_subjects]

        for obj in objects:
            box = obj["box"]
            obj_left, obj_top, obj_right, obj_bottom = box
            importance = float(obj.get("importance", 0.5)) # Use get with default
            obj_type = obj.get("type", "object") # Use get with default
            obj_label_lower = obj.get("label", "").lower() # Use get with default

            # Check if the object is the primary or a secondary subject
            is_primary_subject = primary_subject_lower and primary_subject_lower in obj_label_lower
            is_secondary_subject = any(sub_lower in obj_label_lower for sub_lower in secondary_subject_lowers)


            # Calculate intersection
            intersect_left = max(left, obj_left)
            intersect_right = min(right, obj_right)
            intersect_top = max(top, obj_top)
            intersect_bottom = min(bottom, obj_bottom)

            if intersect_right > intersect_left and intersect_bottom > intersect_top:
                # Calculate coverage
                obj_area = (obj_right - obj_left) * (obj_bottom - obj_top)
                intersect_area = (intersect_right - intersect_left) * (intersect_bottom - intersect_top)
                coverage = intersect_area / obj_area

                # Adjust importance based on object type and context (Genre specific)
                # These multipliers should ideally be defined in the VisionAgentGenre class
                # For now, using simplified logic based on type/subject status
                adjusted_importance = importance

                if obj_type.lower() in ['person', 'human_figure', 'animal', 'animal_figure', 'group', 'interaction']:
                     adjusted_importance *= 2.5 # Boost for figures and interactions
                elif obj_type.lower() in ['object', 'setting_element']:
                     adjusted_importance *= 1.2 # Moderate boost for relevant objects/setting

                if is_primary_subject:
                     adjusted_importance *= 3.0 # High boost for primary subject
                elif is_secondary_subject:
                     adjusted_importance *= 2.0 # Moderate boost for secondary subjects


                # Penalize partial coverage of important objects
                if coverage < 1.0:
                    if is_primary_subject or is_secondary_subject or obj_type.lower() in ['person', 'human_figure', 'animal', 'animal_figure', 'group', 'interaction', 'object']:
                        adjusted_importance *= coverage * 0.6  # Moderate penalty for important elements
                    else:
                        adjusted_importance *= coverage * 0.9  # Less severe penalty for others

                score += adjusted_importance

        return score

    # Placeholder for handoff to UpscaleAgent
    def handoff_to_upscale(self, cropped_image_path: str, upscale_queue: Queue):
        """Places the cropped image path onto the upscale queue."""
        logging.info(f"VisionAgentGenre: Handoff {os.path.basename(cropped_image_path)} to UpscaleAgent queue.")
        upscale_queue.put(cropped_image_path)


# Note: This class is intended to be instantiated and used by the DocentAgent or a Vision Router.

    def _think_with_grok(self, thought: str) -> str:
        """
        Internal 'thinking' step using grok-3-mini-fast-high-beta.
        Sends the agent's reasoning or validation prompt to the Grok model and returns the response.
        """
        try:
            grok_api_key = self.config.get("grok_api_key") or os.environ.get("GROK_API_KEY")
            if not grok_api_key:
                logging.warning("VisionAgentGenre: No GROK_API_KEY found for Grok thinking step.")
                return ""
            client = OpenAI(api_key=grok_api_key, base_url="https://api.x.ai/v1")
            response = client.chat.completions.create(
                model="grok-3-mini-fast-high-beta",
                messages=[
                    {"role": "system", "content": "You are an expert vision agent reasoning about your next step."},
                    {"role": "user", "content": thought}
                ],
                max_tokens=256,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"VisionAgentGenre: Grok thinking step failed: {e}")
            return ""
