import logging
import os
import json
import io
import base64
from PIL import Image, ImageDraw, ImageFont, ImageColor, UnidentifiedImageError
from art_agent_team.agents.vision_agent_abstract import VisionAgentAbstract, CorruptedImageError, UnsupportedImageFormatError
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import math
import numpy as np
from queue import Queue
from vertexai.preview.generative_models import Part, Image as VertexImage, GenerationConfig

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
    label: str
    mask: np.ndarray # 2D boolean array, True where object is present
    confidence: float = 0.0
    score: float = 0.0 # Score or importance for crop planning
    x0: int = 0 # Normalized 0-1000
    y0: int = 0
    x1: int = 0
    y1: int = 0
    type: str = "mask" # For understanding what kind of object this is
    status: str = "full" # full, truncated_top, truncated_bottom, etc.

class VisionAgentReligiousHistorical(VisionAgentAbstract):
    """Agent for analyzing religious/historical art using Vertex AI Gemini model."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the VisionAgentReligiousHistorical with Vertex AI model."""
        super().__init__(config)
        self.genre = "religious/historical"
        self.colors = ['red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple',
                      'brown', 'gray', 'beige', 'turquoise', 'cyan', 'magenta',
                      'lime', 'navy', 'maroon', 'teal', 'olive', 'coral', 'lavender',
                      'violet', 'gold', 'silver'] + [c for c in ImageColor.colormap.keys()]
        logging.info(f"VisionAgentReligiousHistorical initialized for {self.genre} with Vertex AI model.")

    def analyze_image(self, image_path: str, research_data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Analyzes religious/historical image using Vertex AI Gemini model.
        Incorporates research data for better context understanding.
        """
        logging.info(f"[VisionAgentReligiousHistorical] Analyzing image: {image_path} with research data: {research_data is not None}")
        if not image_path or not self.gemini_vertex_model:
            logging.error("[VisionAgentReligiousHistorical] No image path or Vertex AI model not initialized.")
            return None

        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            image_part = Part.from_image(VertexImage.from_bytes(image_bytes))
            generation_config = GenerationConfig(
                temperature=0.2,
                max_output_tokens=2048,
                candidate_count=1
            )

            # Store research data for use in cropping and other methods
            self.primary_subject = research_data.get("primary_subject") if research_data else None
            self.secondary_subjects = research_data.get("secondary_subjects", []) if research_data else []
            paragraph_description = research_data.get("paragraph_description", "") if research_data else ""
            structured_sentence = research_data.get("structured_sentence", "") if research_data else ""

            # Religious/Historical-specific prompt
            prompt_parts = [
                image_part,
                "Analyze this religious/historical painting in detail. Focus on identifying:",
                "- Central religious or historical figures with expressions and gestures",
                "- Religious symbols and artifacts",
                "- Historical elements and cultural context",
                "- Interactions between figures and architectural elements",
                "- Time period and location indicators"
            ]
            if research_data and "artist_name" in research_data and "artwork_title" in research_data:
                prompt_parts.append(f"Consider the artwork '{research_data['artwork_title']}' by {research_data['artist_name']}. ")
            if research_data and "style_period" in research_data:
                prompt_parts.append(f"Consider the {research_data['style_period']} style/period context. ")
            if research_data and "paragraph_description" in research_data:
                prompt_parts.append(f"Additional context: {research_data['paragraph_description']}")

            prompt_parts.append("Provide your analysis as a structured JSON object with keys like 'composition', 'religious_elements', 'historical_context', 'figures', 'symbols', 'relationships', and 'overall_interpretation'.")

            logging.debug(f"[VisionAgentReligiousHistorical] Sending prompt to Vertex AI Gemini: {''.join(prompt_parts[1:])}") 

            # Use the inherited Vertex AI model
            response = self.gemini_vertex_model.generate_content(
                contents=prompt_parts,
                generation_config=generation_config
            )
            
            if response and response.text:
                try:
                    # Parse JSON response
                    analysis_text = response.text
                    try:
                        analysis_result = json.loads(analysis_text)
                    except json.JSONDecodeError:
                        analysis_result = {
                            "analysis": analysis_text,
                            "error": "Failed to parse response as JSON"
                        }
                    
                    analysis_result.update({
                        "image_path": image_path,
                        "genre": self.genre,
                        "primary_subject": self.primary_subject,
                        "secondary_subjects": self.secondary_subjects
                    })
                    
                    logging.info(f"[VisionAgentReligiousHistorical] Successfully analyzed image: {image_path}")
                    return analysis_result

                except (AttributeError, IndexError, TypeError) as e:
                    error_msg = f"Error processing Vertex AI response: {str(e)}"
                    logging.error(f"[VisionAgentReligiousHistorical] {error_msg}. Response: {response.text if hasattr(response, 'text') else response}", exc_info=True)
                    return {
                        "error": error_msg,
                        "image_path": image_path,
                        "genre": self.genre
                    }

            logging.error(f"[VisionAgentReligiousHistorical] No valid analysis content received from Vertex AI Gemini for {image_path}")
            return {
                "error": "No valid analysis content received from Vertex AI Gemini",
                "image_path": image_path,
                "genre": self.genre
            }

        except FileNotFoundError:
            logging.error(f"[VisionAgentReligiousHistorical] Image file not found: {image_path}")
            raise
        except UnidentifiedImageError as e:
            logging.error(f"[VisionAgentReligiousHistorical] Cannot identify image file (corrupted/unsupported): {image_path}. Error: {e}")
            raise CorruptedImageError(f"Corrupted or unsupported image file: {image_path}") from e
        except IOError as e:
            logging.error(f"[VisionAgentReligiousHistorical] IOError opening or reading image: {image_path}. Error: {e}")
            raise UnsupportedImageFormatError(f"IOError, possibly unsupported format or permissions issue: {image_path}") from e
        except Exception as e:
            logging.error(f"[VisionAgentReligiousHistorical] Failed to analyze image with Vertex AI Gemini: {e}", exc_info=True)
            return {
                "error": f"Failed to analyze image with Vertex AI Gemini: {str(e)}",
                "image_path": image_path,
                "genre": self.genre
            }

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


    def _save_labeled_version(self, img: Image.Image, analysis_results: Dict[str, Any], output_path: str) -> None:
        """Save version with bounding boxes and labels."""
        try:
            # Convert to RGBA for transparency support
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            draw = ImageDraw.Draw(img)

            # Process detected objects
            for obj in analysis_results.get("objects", []):
                label = obj.get("label", "unknown")
                importance = obj.get("importance", 0.5)
                bbox = obj.get("box_2d", [0, 0, 0, 0])
                confidence = obj.get("confidence", 0.0)

                # Draw bounding box
                box_color = self.colors[hash(label) % len(self.colors)]
                box_coords = [
                    bbox[1], bbox[0],  # x0, y0
                    bbox[3], bbox[2]   # x1, y1
                ]
                draw.rectangle(box_coords, outline=box_color, width=2)

                # Add label with background
                label_text = f"{label} ({importance:.2f})"
                text_bbox = draw.textbbox((box_coords[0], box_coords[1]-20), label_text)
                draw.rectangle([
                    text_bbox[0]-2, text_bbox[1]-2,
                    text_bbox[2]+2, text_bbox[3]+2
                ], fill=(0, 0, 0, 180))
                draw.text((box_coords[0], box_coords[1]-20), label_text, fill=box_color)

            # Save output
            img = img.convert('RGB')
            img.save(output_path, format='JPEG', quality=95)
            logging.info(f"[VisionAgentReligiousHistorical] Saved labeled image to {output_path}")

        except Exception as e:
            logging.error(f"[VisionAgentReligiousHistorical] Error saving labeled version: {e}", exc_info=True)

    def _save_masked_version(self, img: Image.Image, analysis_results: Dict[str, Any], output_path: str) -> None:
        """Save version with masks for important objects."""
        try:
            # Convert to RGBA for transparency support
            if img.mode != 'RGBA':
                img = img.convert('RGBA')

            # Get important objects based on significance
            important_objects = [obj for obj in analysis_results.get("objects", [])
                               if obj.get("importance", 0) > 0.5]

            # Create and apply masks
            for i, obj in enumerate(important_objects):
                mask_color = self.colors[i % len(self.colors)]
                bbox = obj.get("box_2d", [0, 0, 0, 0])
                
                # Create simple rectangular mask
                mask = Image.new('RGBA', img.size, (0, 0, 0, 0))
                mask_draw = ImageDraw.Draw(mask)
                mask_draw.rectangle([
                    bbox[1], bbox[0],  # x0, y0
                    bbox[3], bbox[2]   # x1, y1
                ], fill=(*ImageColor.getrgb(mask_color), 128))  # Semi-transparent

                # Apply mask
                img = Image.alpha_composite(img, mask)

            # Save output
            img = img.convert('RGB')
            img.save(output_path, format='JPEG', quality=95)
            logging.info(f"[VisionAgentReligiousHistorical] Saved masked image to {output_path}")

        except Exception as e:
            logging.error(f"[VisionAgentReligiousHistorical] Error saving masked version: {e}", exc_info=True)

    def _parse_gemini_response(self, response_text: str) -> Optional[Dict]:
        """Parse Gemini API response text into JSON."""
        try:
            # Clean up response text
            json_str = response_text.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]

            return json.loads(json_str)
        except Exception as e:
            logging.error(f"Failed to parse Gemini response: {e}")
            logging.debug(f"Raw response text: {response_text}")
            return None

    def _parse_grok_response(self, response_text: str) -> Optional[Dict]:
        """Parse Grok API response text into JSON."""
        try:
            # Grok API responses might also be in JSON blocks
            json_str = response_text.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]

            return json.loads(json_str)
        except Exception as e:
            logging.error(f"Failed to parse Grok response: {e}")
            logging.debug(f"Raw response text: {response_text}")
            return None


    def _create_segmentation_masks(self, detections: List[Dict[str, Any]], image_size: Tuple[int, int]) -> List[SegmentationMask]:
        """Create segmentation masks from detection results."""
        width, height = image_size
        masks = []

        for det in detections:
            try:
                box = det["box_2d"]
                y0 = int(box[0] * height / 1000)
                x0 = int(box[1] * width / 1000)
                y1 = int(box[2] * height / 1000)
                x1 = int(box[3] * width / 1000)

                # Create binary mask
                mask = np.zeros((height, width), dtype=np.uint8)
                mask[max(0, y0):min(height, y1), max(0, x0):min(width, x1)] = 255

                masks.append(SegmentationMask(
                    label=det["label"],
                    mask=mask,
                    confidence=det.get("confidence", 0.0),
                    score=det.get("importance", 0.0),
                    x0=x0, y0=y0, x1=x1, y1=y1,
                    type=det.get("type", "mask"),
                    status="full"  # Could be refined based on truncation analysis
                ))

            except Exception as e:
                logging.warning(f"[VisionAgentReligiousHistorical] Failed to create mask for {det.get('label', 'unknown')}: {e}")
                continue

        return masks

    def _calculate_object_importance(self, box: List[float], obj_type: str, img_size: Tuple[int, int], 
                                   base_importance: float, label: str, features: Optional[List[Dict]] = None) -> float:
        """Calculate importance score for an object in religious/historical context."""
        # Convert normalized coordinates to pixels
        width, height = img_size
        y0, x0, y1, x1 = box
        obj_width = (x1 - x0) * width / 1000
        obj_height = (y1 - y0) * height / 1000
        obj_area = obj_width * obj_height
        total_area = width * height

        # Base importance from object type
        type_multipliers = {
            'religious_figure': 2.0,
            'historical_figure': 1.8,
            'symbol': 1.5,
            'artifact': 1.3,
            'group': 1.2,
            'background': 0.5,
            'unknown': 1.0
        }
        
        # Calculate score
        type_multiplier = type_multipliers.get(obj_type.lower(), 1.0)
        size_factor = np.sqrt(obj_area / total_area)  # Sqrt to reduce impact of size
        importance = base_importance * type_multiplier * (0.3 + 0.7 * size_factor)

        # Boost for central objects
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2
        center_dist = np.sqrt((center_x - 500)**2 + (center_y - 500)**2) / 500  # Distance to image center
        centrality_boost = 1.0 + (1.0 - min(1.0, center_dist)) * 0.5

        importance *= centrality_boost

        # Additional boosts for religious/historical elements
        if 'face' in label.lower() or (features and len(features) > 0):
            importance *= 1.2  # Boost for faces/features
        if any(term in label.lower() for term in ['jesus', 'mary', 'saint', 'angel', 'god']):
            importance *= 1.5  # Boost for key religious figures
        if any(term in label.lower() for term in ['cross', 'halo', 'crown', 'altar', 'throne']):
            importance *= 1.3  # Boost for religious symbols

        return min(1.0, importance)  # Cap at 1.0


    def _threshold_important_objects(self, objects: List[Dict[str, Any]], percentile_threshold: float = 85) -> List[Dict[str, Any]]:
        """Filter objects for religious/historical art - prioritize figures, symbols, and subjects."""
        if not objects:
            return []

        importances = [obj.get('importance', 0) for obj in objects]
        if not importances:
            return objects

        # Higher percentile threshold for religious/historical art to focus on key elements
        threshold = np.percentile(importances, percentile_threshold)

        primary_subject_label = self.primary_subject.lower() if hasattr(self, 'primary_subject') and self.primary_subject else None
        secondary_subject_labels = [sub.lower() for sub in self.secondary_subjects] if hasattr(self, 'secondary_subjects') and self.secondary_subjects else []

        important = []
        for obj in objects:
            obj_label_lower = obj.get('label', '').lower()
            if obj.get('importance', 0) >= threshold or \
               obj.get('type') in ['religious_figure', 'historical_figure', 'symbol', 'artifact', 'figure', 'group'] or \
               (primary_subject_label and primary_subject_label in obj_label_lower) or \
               any(sub_label in obj_label_lower for sub_label in secondary_subject_labels):
                important.append(obj)

        return important


    def save_analysis_outputs(self, image_path: str, analysis_results: Dict[str, Any], output_folder: str) -> None:
        """Save all versions of the analyzed image."""
        try:
            basename = os.path.splitext(os.path.basename(image_path))[0]
            if not os.path.exists(image_path):
                logging.warning(f"[VisionAgentReligiousHistorical] Image file not found for saving outputs: {image_path}")
                return

            os.makedirs(output_folder, exist_ok=True)

            # Load image and create copies for different outputs
            with Image.open(image_path) as img:
                # 1. Save labeled version
                labeled_path = os.path.join(output_folder, f"{basename}_labeled.jpg")
                img_labeled = img.copy()
                self._save_labeled_version(img_labeled, analysis_results, labeled_path)

                # 2. Save masked version
                masked_path = os.path.join(output_folder, f"{basename}_masked.jpg")
                img_masked = img.copy()
                self._save_masked_version(img_masked, analysis_results, masked_path)

                # 3. Save cropped version
                cropped_path = os.path.join(output_folder, f"{basename}_cropped.jpg")
                self.copy_and_crop_image(image_path, cropped_path, analysis_results)

            logging.info(f"[VisionAgentReligiousHistorical] Successfully saved all analysis outputs to {output_folder}")

        except Exception as e:
            logging.error(f"[VisionAgentReligiousHistorical] Error saving analysis outputs: {e}", exc_info=True)

    def _save_labeled_version(self, img: Image.Image, analysis_results: Dict[str, Any], output_path: str) -> None:
        """Save version with bounding boxes and labels."""
        try:
            # Convert to RGBA for transparency support
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            draw = ImageDraw.Draw(img)

            # Process detected objects
            for obj in analysis_results.get("objects", []):
                label = obj.get("label", "unknown")
                importance = obj.get("importance", 0.5)
                bbox = obj.get("box_2d", [0, 0, 0, 0])
                confidence = obj.get("confidence", 0.0)

                # Draw bounding box
                box_color = self.colors[hash(label) % len(self.colors)]
                box_coords = [
                    bbox[1], bbox[0],  # x0, y0
                    bbox[3], bbox[2]   # x1, y1
                ]
                draw.rectangle(box_coords, outline=box_color, width=2)

                # Add label with background
                label_text = f"{label} ({importance:.2f})"
                text_bbox = draw.textbbox((box_coords[0], box_coords[1]-20), label_text)
                draw.rectangle([
                    text_bbox[0]-2, text_bbox[1]-2,
                    text_bbox[2]+2, text_bbox[3]+2
                ], fill=(0, 0, 0, 180))
                draw.text((box_coords[0], box_coords[1]-20), label_text, fill=box_color)

            # Save output
            img = img.convert('RGB')
            img.save(output_path, format='JPEG', quality=95)
            logging.info(f"[VisionAgentReligiousHistorical] Saved labeled image to {output_path}")

        except Exception as e:
            logging.error(f"[VisionAgentReligiousHistorical] Error saving labeled version: {e}", exc_info=True)

    def _save_masked_version(self, img: Image.Image, analysis_results: Dict[str, Any], output_path: str) -> None:
        """Save version with masks for important objects."""
        try:
            # Convert to RGBA for transparency support
            if img.mode != 'RGBA':
                img = img.convert('RGBA')

            # Get important objects based on significance
            important_objects = [obj for obj in analysis_results.get("objects", [])
                               if obj.get("importance", 0) > 0.5]

            # Create and apply masks
            for i, obj in enumerate(important_objects):
                mask_color = self.colors[i % len(self.colors)]
                bbox = obj.get("box_2d", [0, 0, 0, 0])
                
                # Create simple rectangular mask
                mask = Image.new('RGBA', img.size, (0, 0, 0, 0))
                mask_draw = ImageDraw.Draw(mask)
                mask_draw.rectangle([
                    bbox[1], bbox[0],  # x0, y0
                    bbox[3], bbox[2]   # x1, y1
                ], fill=(*ImageColor.getrgb(mask_color), 128))  # Semi-transparent

                # Apply mask
                img = Image.alpha_composite(img, mask)

            # Save output
            img = img.convert('RGB')
            img.save(output_path, format='JPEG', quality=95)
            logging.info(f"[VisionAgentReligiousHistorical] Saved masked image to {output_path}")

        except Exception as e:
            logging.error(f"[VisionAgentReligiousHistorical] Error saving masked version: {e}", exc_info=True)

    def _parse_gemini_response(self, response_text: str) -> Optional[Dict]:
        """Parse Gemini API response text into JSON."""
        try:
            # Clean up response text
            json_str = response_text.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]

            return json.loads(json_str)
        except Exception as e:
            logging.error(f"Failed to parse Gemini response: {e}")
            logging.debug(f"Raw response text: {response_text}")
            return None

    def _parse_grok_response(self, response_text: str) -> Optional[Dict]:
        """Parse Grok API response text into JSON."""
        try:
            # Grok API responses might also be in JSON blocks
            json_str = response_text.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]

            return json.loads(json_str)
        except Exception as e:
            logging.error(f"Failed to parse Grok response: {e}")
            logging.debug(f"Raw response text: {response_text}")
            return None


    def _create_segmentation_masks(self, detections: List[Dict[str, Any]], image_size: Tuple[int, int]) -> List[SegmentationMask]:
        """Create segmentation masks from detection results."""
        width, height = image_size
        masks = []

        for det in detections:
            try:
                box = det["box_2d"]
                y0 = int(box[0] * height / 1000)
                x0 = int(box[1] * width / 1000)
                y1 = int(box[2] * height / 1000)
                x1 = int(box[3] * width / 1000)

                # Create binary mask
                mask = np.zeros((height, width), dtype=np.uint8)
                mask[max(0, y0):min(height, y1), max(0, x0):min(width, x1)] = 255

                masks.append(SegmentationMask(
                    label=det["label"],
                    mask=mask,
                    confidence=det.get("confidence", 0.0),
                    score=det.get("importance", 0.0),
                    x0=x0, y0=y0, x1=x1, y1=y1,
                    type=det.get("type", "mask"),
                    status="full"  # Could be refined based on truncation analysis
                ))

            except Exception as e:
                logging.warning(f"[VisionAgentReligiousHistorical] Failed to create mask for {det.get('label', 'unknown')}: {e}")
                continue

        return masks

    def _calculate_object_importance(self, box: List[float], obj_type: str, img_size: Tuple[int, int], 
                                   base_importance: float, label: str, features: Optional[List[Dict]] = None) -> float:
        """Calculate importance score for an object in religious/historical context."""
        # Convert normalized coordinates to pixels
        width, height = img_size
        y0, x0, y1, x1 = box
        obj_width = (x1 - x0) * width / 1000
        obj_height = (y1 - y0) * height / 1000
        obj_area = obj_width * obj_height
        total_area = width * height

        # Base importance from object type
        type_multipliers = {
            'religious_figure': 2.0,
            'historical_figure': 1.8,
            'symbol': 1.5,
            'artifact': 1.3,
            'group': 1.2,
            'background': 0.5,
            'unknown': 1.0
        }
        
        # Calculate score
        type_multiplier = type_multipliers.get(obj_type.lower(), 1.0)
        size_factor = np.sqrt(obj_area / total_area)  # Sqrt to reduce impact of size
        importance = base_importance * type_multiplier * (0.3 + 0.7 * size_factor)

        # Boost for central objects
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2
        center_dist = np.sqrt((center_x - 500)**2 + (center_y - 500)**2) / 500  # Distance to image center
        centrality_boost = 1.0 + (1.0 - min(1.0, center_dist)) * 0.5

        importance *= centrality_boost

        # Additional boosts for religious/historical elements
        if 'face' in label.lower() or (features and len(features) > 0):
            importance *= 1.2  # Boost for faces/features
        if any(term in label.lower() for term in ['jesus', 'mary', 'saint', 'angel', 'god']):
            importance *= 1.5  # Boost for key religious figures
        if any(term in label.lower() for term in ['cross', 'halo', 'crown', 'altar', 'throne']):
            importance *= 1.3  # Boost for religious symbols

        return min(1.0, importance)  # Cap at 1.0


    def _threshold_important_objects(self, objects: List[Dict[str, Any]], percentile_threshold: float = 85) -> List[Dict[str, Any]]:
        """Filter objects for religious/historical art - prioritize figures, symbols, and subjects."""
        if not objects:
            return []

        importances = [obj.get('importance', 0) for obj in objects]
        if not importances:
            return objects

        # Higher percentile threshold for religious/historical art to focus on key elements
        threshold = np.percentile(importances, percentile_threshold)

        primary_subject_label = self.primary_subject.lower() if hasattr(self, 'primary_subject') and self.primary_subject else None
        secondary_subject_labels = [sub.lower() for sub in self.secondary_subjects] if hasattr(self, 'secondary_subjects') and self.secondary_subjects else []

        important = []
        for obj in objects:
            obj_label_lower = obj.get('label', '').lower()
            if obj.get('importance', 0) >= threshold or \
               obj.get('type') in ['religious_figure', 'historical_figure', 'symbol', 'artifact', 'figure', 'group'] or \
               (primary_subject_label and primary_subject_label in obj_label_lower) or \
               any(sub_label in obj_label_lower for sub_label in secondary_subject_labels):
                important.append(obj)

        return important


    def save_analysis_outputs(self, image_path: str, analysis_results: Dict[str, Any], output_folder: str) -> None:
        """Save all versions of the analyzed image."""
        try:
            basename = os.path.splitext(os.path.basename(image_path))[0]
            if not os.path.exists(image_path):
                logging.warning(f"[VisionAgentReligiousHistorical] Image file not found for saving outputs: {image_path}")
                return

            os.makedirs(output_folder, exist_ok=True)

            # Load image and create copies for different outputs
            with Image.open(image_path) as img:
                # 1. Save labeled version
                labeled_path = os.path.join(output_folder, f"{basename}_labeled.jpg")
                img_labeled = img.copy()
                self._save_labeled_version(img_labeled, analysis_results, labeled_path)

                # 2. Save masked version
                masked_path = os.path.join(output_folder, f"{basename}_masked.jpg")
                img_masked = img.copy()
                self._save_masked_version(img_masked, analysis_results, masked_path)

                # 3. Save cropped version
                cropped_path = os.path.join(output_folder, f"{basename}_cropped.jpg")
                self.copy_and_crop_image(image_path, cropped_path, analysis_results)

            logging.info(f"[VisionAgentReligiousHistorical] Successfully saved all analysis outputs to {output_folder}")

        except Exception as e:
            logging.error(f"[VisionAgentReligiousHistorical] Error saving analysis outputs: {e}", exc_info=True)

    def _save_labeled_version(self, img: Image.Image, analysis_results: Dict[str, Any], output_path: str) -> None:
        """Save version with bounding boxes and labels."""
        try:
            # Convert to RGBA for transparency support
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            draw = ImageDraw.Draw(img)

            # Process detected objects
            for obj in analysis_results.get("objects", []):
                label = obj.get("label", "unknown")
                importance = obj.get("importance", 0.5)
                bbox = obj.get("box_2d", [0, 0, 0, 0])
                confidence = obj.get("confidence", 0.0)

                # Draw bounding box
                box_color = self.colors[hash(label) % len(self.colors)]
                box_coords = [
                    bbox[1], bbox[0],  # x0, y0
                    bbox[3], bbox[2]   # x1, y1
                ]
                draw.rectangle(box_coords, outline=box_color, width=2)

                # Add label with background
                label_text = f"{label} ({importance:.2f})"
                text_bbox = draw.textbbox((box_coords[0], box_coords[1]-20), label_text)
                draw.rectangle([
                    text_bbox[0]-2, text_bbox[1]-2,
                    text_bbox[2]+2, text_bbox[3]+2
                ], fill=(0, 0, 0, 180))
                draw.text((box_coords[0], box_coords[1]-20), label_text, fill=box_color)

            # Save output
            img = img.convert('RGB')
            img.save(output_path, format='JPEG', quality=95)
            logging.info(f"[VisionAgentReligiousHistorical] Saved labeled image to {output_path}")

        except Exception as e:
            logging.error(f"[VisionAgentReligiousHistorical] Error saving labeled version: {e}", exc_info=True)

    def _save_masked_version(self, img: Image.Image, analysis_results: Dict[str, Any], output_path: str) -> None:
        """Save version with masks for important objects."""
        try:
            # Convert to RGBA for transparency support
            if img.mode != 'RGBA':
                img = img.convert('RGBA')

            # Get important objects based on significance
            important_objects = [obj for obj in analysis_results.get("objects", [])
                               if obj.get("importance", 0) > 0.5]

            # Create and apply masks
            for i, obj in enumerate(important_objects):
                mask_color = self.colors[i % len(self.colors)]
                bbox = obj.get("box_2d", [0, 0, 0, 0])
                
                # Create simple rectangular mask
                mask = Image.new('RGBA', img.size, (0, 0, 0, 0))
                mask_draw = ImageDraw.Draw(mask)
                mask_draw.rectangle([
                    bbox[1], bbox[0],  # x0, y0
                    bbox[3], bbox[2]   # x1, y1
                ], fill=(*ImageColor.getrgb(mask_color), 128))  # Semi-transparent

                # Apply mask
                img = Image.alpha_composite(img, mask)

            # Save output
            img = img.convert('RGB')
            img.save(output_path, format='JPEG', quality=95)
            logging.info(f"[VisionAgentReligiousHistorical] Saved masked image to {output_path}")

        except Exception as e:
            logging.error(f"[VisionAgentReligiousHistorical] Error saving masked version: {e}", exc_info=True)

    def _parse_gemini_response(self, response_text: str) -> Optional[Dict]:
        """Parse Gemini API response text into JSON."""
        try:
            # Clean up response text
            json_str = response_text.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]

            return json.loads(json_str)
        except Exception as e:
            logging.error(f"Failed to parse Gemini response: {e}")
            logging.debug(f"Raw response text: {response_text}")
            return None

    def _parse_grok_response(self, response_text: str) -> Optional[Dict]:
        """Parse Grok API response text into JSON."""
        try:
            # Grok API responses might also be in JSON blocks
            json_str = response_text.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]

            return json.loads(json_str)
        except Exception as e:
            logging.error(f"Failed to parse Grok response: {e}")
            logging.debug(f"Raw response text: {response_text}")
            return None


    def _create_segmentation_masks(self, detections: List[Dict[str, Any]], image_size: Tuple[int, int]) -> List[SegmentationMask]:
        """Create segmentation masks from detection results."""
        width, height = image_size
        masks = []

        for det in detections:
            try:
                box = det["box_2d"]
                y0 = int(box[0] * height / 1000)
                x0 = int(box[1] * width / 1000)
                y1 = int(box[2] * height / 1000)
                x1 = int(box[3] * width / 1000)

                # Create binary mask
                mask = np.zeros((height, width), dtype=np.uint8)
                mask[max(0, y0):min(height, y1), max(0, x0):min(width, x1)] = 255

                masks.append(SegmentationMask(
                    label=det["label"],
                    mask=mask,
                    confidence=det.get("confidence", 0.0),
                    score=det.get("importance", 0.0),
                    x0=x0, y0=y0, x1=x1, y1=y1,
                    type=det.get("type", "mask"),
                    status="full"  # Could be refined based on truncation analysis
                ))

            except Exception as e:
                logging.warning(f"[VisionAgentReligiousHistorical] Failed to create mask for {det.get('label', 'unknown')}: {e}")
                continue

        return masks

    def _calculate_object_importance(self, box: List[float], obj_type: str, img_size: Tuple[int, int], 
                                   base_importance: float, label: str, features: Optional[List[Dict]] = None) -> float:
        """Calculate importance score for an object in religious/historical context."""
        # Convert normalized coordinates to pixels
        width, height = img_size
        y0, x0, y1, x1 = box
        obj_width = (x1 - x0) * width / 1000
        obj_height = (y1 - y0) * height / 1000
        obj_area = obj_width * obj_height
        total_area = width * height

        # Base importance from object type
        type_multipliers = {
            'religious_figure': 2.0,
            'historical_figure': 1.8,
            'symbol': 1.5,
            'artifact': 1.3,
            'group': 1.2,
            'background': 0.5,
            'unknown': 1.0
        }
        
        # Calculate score
        type_multiplier = type_multipliers.get(obj_type.lower(), 1.0)
        size_factor = np.sqrt(obj_area / total_area)  # Sqrt to reduce impact of size
        importance = base_importance * type_multiplier * (0.3 + 0.7 * size_factor)

        # Boost for central objects
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2
        center_dist = np.sqrt((center_x - 500)**2 + (center_y - 500)**2) / 500  # Distance to image center
        centrality_boost = 1.0 + (1.0 - min(1.0, center_dist)) * 0.5

        importance *= centrality_boost

        # Additional boosts for religious/historical elements
        if 'face' in label.lower() or (features and len(features) > 0):
            importance *= 1.2  # Boost for faces/features
        if any(term in label.lower() for term in ['jesus', 'mary', 'saint', 'angel', 'god']):
            importance *= 1.5  # Boost for key religious figures
        if any(term in label.lower() for term in ['cross', 'halo', 'crown', 'altar', 'throne']):
            importance *= 1.3  # Boost for religious symbols

        return min(1.0, importance)  # Cap at 1.0


    def _threshold_important_objects(self, objects: List[Dict[str, Any]], percentile_threshold: float = 85) -> List[Dict[str, Any]]:
        """Filter objects for religious/historical art - prioritize figures, symbols, and subjects."""
        if not objects:
            return []

        importances = [obj.get('importance', 0) for obj in objects]
        if not importances:
            return objects

        # Higher percentile threshold for religious/historical art to focus on key elements
        threshold = np.percentile(importances, percentile_threshold)

        primary_subject_label = self.primary_subject.lower() if hasattr(self, 'primary_subject') and self.primary_subject else None
        secondary_subject_labels = [sub.lower() for sub in self.secondary_subjects] if hasattr(self, 'secondary_subjects') and self.secondary_subjects else []

        important = []
        for obj in objects:
            obj_label_lower = obj.get('label', '').lower()
            if obj.get('importance', 0) >= threshold or \
               obj.get('type') in ['religious_figure', 'historical_figure', 'symbol', 'artifact', 'figure', 'group'] or \
               (primary_subject_label and primary_subject_label in obj_label_lower) or \
               any(sub_label in obj_label_lower for sub_label in secondary_subject_labels):
                important.append(obj)

        return important


    def save_analysis_outputs(self, image_path: str, analysis_results: Dict[str, Any], output_folder: str) -> None:
        """Save all versions of the analyzed image."""
        try:
            basename = os.path.splitext(os.path.basename(image_path))[0]
            if not os.path.exists(image_path):
                logging.warning(f"[VisionAgentReligiousHistorical] Image file not found for saving outputs: {image_path}")
                return

            os.makedirs(output_folder, exist_ok=True)

            # Load image and create copies for different outputs
            with Image.open(image_path) as img:
                # 1. Save labeled version
                labeled_path = os.path.join(output_folder, f"{basename}_labeled.jpg")
                img_labeled = img.copy()
                self._save_labeled_version(img_labeled, analysis_results, labeled_path)

                # 2. Save masked version
                masked_path = os.path.join(output_folder, f"{basename}_masked.jpg")
                img_masked = img.copy()
                self._save_masked_version(img_masked, analysis_results, masked_path)

                # 3. Save cropped version
                cropped_path = os.path.join(output_folder, f"{basename}_cropped.jpg")
                self.copy_and_crop_image(image_path, cropped_path, analysis_results)

            logging.info(f"[VisionAgentReligiousHistorical] Successfully saved all analysis outputs to {output_folder}")

        except Exception as e:
            logging.error(f"[VisionAgentReligiousHistorical] Error saving analysis outputs: {e}", exc_info=True)

    def _save_labeled_version(self, img: Image.Image, analysis_results: Dict[str, Any], output_path: str) -> None:
        """Save version with bounding boxes and labels."""
        try:
            # Convert to RGBA for transparency support
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            draw = ImageDraw.Draw(img)

            # Process detected objects
            for obj in analysis_results.get("objects", []):
                label = obj.get("label", "unknown")
                importance = obj.get("importance", 0.5)
                bbox = obj.get("box_2d", [0, 0, 0, 0])
                confidence = obj.get("confidence", 0.0)

                # Draw bounding box
                box_color = self.colors[hash(label) % len(self.colors)]
                box_coords = [
                    bbox[1], bbox[0],  # x0, y0
                    bbox[3], bbox[2]   # x1, y1
                ]
                draw.rectangle(box_coords, outline=box_color, width=2)

                # Add label with background
                label_text = f"{label} ({importance:.2f})"
                text_bbox = draw.textbbox((box_coords[0], box_coords[1]-20), label_text)
                draw.rectangle([
                    text_bbox[0]-2, text_bbox[1]-2,
                    text_bbox[2]+2, text_bbox[3]+2
                ], fill=(0, 0, 0, 180))
                draw.text((box_coords[0], box_coords[1]-20), label_text, fill=box_color)

            # Save output
            img = img.convert('RGB')
            img.save(output_path, format='JPEG', quality=95)
            logging.info(f"[VisionAgentReligiousHistorical] Saved labeled image to {output_path}")

        except Exception as e:
            logging.error(f"[VisionAgentReligiousHistorical] Error saving labeled version: {e}", exc_info=True)

    def _save_masked_version(self, img: Image.Image, analysis_results: Dict[str, Any], output_path: str) -> None:
        """Save version with masks for important objects."""
        try:
            # Convert to RGBA for transparency support
            if img.mode != 'RGBA':
                img = img.convert('RGBA')

            # Get important objects based on significance
            important_objects = [obj for obj in analysis_results.get("objects", [])
                               if obj.get("importance", 0) > 0.5]

            # Create and apply masks
            for i, obj in enumerate(important_objects):
                mask_color = self.colors[i % len(self.colors)]
                bbox = obj.get("box_2d", [0, 0, 0, 0])
                
                # Create simple rectangular mask
                mask = Image.new('RGBA', img.size, (0, 0, 0, 0))
                mask_draw = ImageDraw.Draw(mask)
                mask_draw.rectangle([
                    bbox[1], bbox[0],  # x0, y0
                    bbox[3], bbox[2]   # x1, y1
                ], fill=(*ImageColor.getrgb(mask_color), 128))  # Semi-transparent

                # Apply mask
                img = Image.alpha_composite(img, mask)

            # Save output
            img = img.convert('RGB')
            img.save(output_path, format='JPEG', quality=95)
            logging.info(f"[VisionAgentReligiousHistorical] Saved masked image to {output_path}")

        except Exception as e:
            logging.error(f"[VisionAgentReligiousHistorical] Error saving masked version: {e}", exc_info=True)

    def _parse_gemini_response(self, response_text: str) -> Optional[Dict]:
        """Parse Gemini API response text into JSON."""
        try:
            # Clean up response text
            json_str = response_text.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]

            return json.loads(json_str)
        except Exception as e:
            logging.error(f"Failed to parse Gemini response: {e}")
            logging.debug(f"Raw response text: {response_text}")
            return None

    def _parse_grok_response(self, response_text: str) -> Optional[Dict]:
        """Parse Grok API response text into JSON."""
        try:
            # Grok API responses might also be in JSON blocks
            json_str = response_text.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]

            return json.loads(json_str)
        except Exception as e:
            logging.error(f"Failed to parse Grok response: {e}")
            logging.debug(f"Raw response text: {response_text}")
            return None


    def _create_segmentation_masks(self, detections: List[Dict[str, Any]], image_size: Tuple[int, int]) -> List[SegmentationMask]:
        """Create segmentation masks from detection results."""
        width, height = image_size
        masks = []

        for det in detections:
            try:
                box = det["box_2d"]
                y0 = int(box[0] * height / 1000)
                x0 = int(box[1] * width / 1000)
                y1 = int(box[2] * height / 1000)
                x1 = int(box[3] * width / 1000)

                # Create binary mask
                mask = np.zeros((height, width), dtype=np.uint8)
                mask[max(0, y0):min(height, y1), max(0, x0):min(width, x1)] = 255

                masks.append(SegmentationMask(
                    label=det["label"],
                    mask=mask,
                    confidence=det.get("confidence", 0.0),
                    score=det.get("importance", 0.0),
                    x0=x0, y0=y0, x1=x1, y1=y1,
                    type=det.get("type", "mask"),
                    status="full"  # Could be refined based on truncation analysis
                ))

            except Exception as e:
                logging.warning(f"[VisionAgentReligiousHistorical] Failed to create mask for {det.get('label', 'unknown')}: {e}")
                continue

        return masks

    def _calculate_object_importance(self, box: List[float], obj_type: str, img_size: Tuple[int, int], 
                                   base_importance: float, label: str, features: Optional[List[Dict]] = None) -> float:
        """Calculate importance score for an object in religious/historical context."""
        # Convert normalized coordinates to pixels
        width, height = img_size
        y0, x0, y1, x1 = box
        obj_width = (x1 - x0) * width / 1000
        obj_height = (y1 - y0) * height / 1000
        obj_area = obj_width * obj_height
        total_area = width * height

        # Base importance from object type
        type_multipliers = {
            'religious_figure': 2.0,
            'historical_figure': 1.8,
            'symbol': 1.5,
            'artifact': 1.3,
            'group': 1.2,
            'background': 0.5,
            'unknown': 1.0
        }
        
        # Calculate score
        type_multiplier = type_multipliers.get(obj_type.lower(), 1.0)
        size_factor = np.sqrt(obj_area / total_area)  # Sqrt to reduce impact of size
        importance = base_importance * type_multiplier * (0.3 + 0.7 * size_factor)

        # Boost for central objects
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2
        center_dist = np.sqrt((center_x - 500)**2 + (center_y - 500)**2) / 500  # Distance to image center
        centrality_boost = 1.0 + (1.0 - min(1.0, center_dist)) * 0.5

        importance *= centrality_boost

        # Additional boosts for religious/historical elements
        if 'face' in label.lower() or (features and len(features) > 0):
            importance *= 1.2  # Boost for faces/features
        if any(term in label.lower() for term in ['jesus', 'mary', 'saint', 'angel', 'god']):
            importance *= 1.5  # Boost for key religious figures
        if any(term in label.lower() for term in ['cross', 'halo', 'crown', 'altar', 'throne']):
            importance *= 1.3  # Boost for religious symbols

        return min(1.0, importance)  # Cap at 1.0


    def _threshold_important_objects(self, objects: List[Dict[str, Any]], percentile_threshold: float = 85) -> List[Dict[str, Any]]:
        """Filter objects for religious/historical art - prioritize figures, symbols, and subjects."""
        if not objects:
            return []

        importances = [obj.get('importance', 0) for obj in objects]
        if not importances:
            return objects

        # Higher percentile threshold for religious/historical art to focus on key elements
        threshold = np.percentile(importances, percentile_threshold)

        primary_subject_label = self.primary_subject.lower() if hasattr(self, 'primary_subject') and self.primary_subject else None
        secondary_subject_labels = [sub.lower() for sub in self.secondary_subjects] if hasattr(self, 'secondary_subjects') and self.secondary_subjects else []

        important = []
        for obj in objects:
            obj_label_lower = obj.get('label', '').lower()
            if obj.get('importance', 0) >= threshold or \
               obj.get('type') in ['religious_figure', 'historical_figure', 'symbol', 'artifact', 'figure', 'group'] or \
               (primary_subject_label and primary_subject_label in obj_label_lower) or \
               any(sub_label in obj_label_lower for sub_label in secondary_subject_labels):
                important.append(obj)

        return important


    def save_analysis_outputs(self, image_path: str, analysis_results: Dict[str, Any], output_folder: str) -> None:
        """Save all versions of the analyzed image."""
        try:
            basename = os.path.splitext(os.path.basename(image_path))[0]
            if not os.path.exists(image_path):
                logging.warning(f"[VisionAgentReligiousHistorical] Image file not found for saving outputs: {image_path}")
                return

            os.makedirs(output_folder, exist_ok=True)

            # Load image and create copies for different outputs
            with Image.open(image_path) as img:
                # 1. Save labeled version
                labeled_path = os.path.join(output_folder, f"{basename}_labeled.jpg")
                img_labeled = img.copy()
                self._save_labeled_version(img_labeled, analysis_results, labeled_path)

                # 2. Save masked version
                masked_path = os.path.join(output_folder, f"{basename}_masked.jpg")
                img_masked = img.copy()
                self._save_masked_version(img_masked, analysis_results, masked_path)

                # 3. Save cropped version
                cropped_path = os.path.join(output_folder, f"{basename}_cropped.jpg")
                self.copy_and_crop_image(image_path, cropped_path, analysis_results)

            logging.info(f"[VisionAgentReligiousHistorical] Successfully saved all analysis outputs to {output_folder}")

        except Exception as e:
            logging.error(f"[VisionAgentReligiousHistorical] Error saving analysis outputs: {e}", exc_info=True)

    def _save_labeled_version(self, img: Image.Image, analysis_results: Dict[str, Any], output_path: str) -> None:
        """Save version with bounding boxes and labels."""
        try:
            # Convert to RGBA for transparency support
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            draw = ImageDraw.Draw(img)

            # Process detected objects
            for obj in analysis_results.get("objects", []):
                label = obj.get("label", "unknown")
                importance = obj.get("importance", 0.5)
                bbox = obj.get("box_2d", [0, 0, 0, 0])
                confidence = obj.get("confidence", 0.0)

                # Draw bounding box
                box_color = self.colors[hash(label) % len(self.colors)]
                box_coords = [
                    bbox[1], bbox[0],  # x0, y0
                    bbox[3], bbox[2]   # x1, y1
                ]
                draw.rectangle(box_coords, outline=box_color, width=2)

                # Add label with background
                label_text = f"{label} ({importance:.2f})"
                text_bbox = draw.textbbox((box_coords[0], box_coords[1]-20), label_text)
                draw.rectangle([
                    text_bbox[0]-2, text_bbox[1]-2,
                    text_bbox[2]+2, text_bbox[3]+2
                ], fill=(0, 0, 0, 180))
                draw.text((box_coords[0], box_coords[1]-20), label_text, fill=box_color)

            # Save output
            img = img.convert('RGB')
            img.save(output_path, format='JPEG', quality=95)
            logging.info(f"[VisionAgentReligiousHistorical] Saved labeled image to {output_path}")

        except Exception as e:
            logging.error(f"[VisionAgentReligiousHistorical] Error saving labeled version: {e}", exc_info=True)

    def _save_masked_version(self, img: Image.Image, analysis_results: Dict[str, Any], output_path: str) -> None:
        """Save version with masks for important objects."""
        try:
            # Convert to RGBA for transparency support
            if img.mode != 'RGBA':
                img = img.convert('RGBA')

            # Get important objects based on significance
            important_objects = [obj for obj in analysis_results.get("objects", [])
                               if obj.get("importance", 0) > 0.5]

            # Create and apply masks
            for i, obj in enumerate(important_objects):
                mask_color = self.colors[i % len(self.colors)]
                bbox = obj.get("box_2d", [0, 0, 0, 0])
                
                # Create simple rectangular mask
                mask = Image.new('RGBA', img.size, (0, 0, 0, 0))
                mask_draw = ImageDraw.Draw(mask)
                mask_draw.rectangle([
                    bbox[1], bbox[0],  # x0, y0
                    bbox[3], bbox[2]   # x1, y1
                ], fill=(*ImageColor.getrgb(mask_color), 128))  # Semi-transparent

                # Apply mask
                img = Image.alpha_composite(img, mask)

            # Save output
            img = img.convert('RGB')
            img.save(output_path, format='JPEG', quality=95)
            logging.info(f"[VisionAgentReligiousHistorical] Saved masked image to {output_path}")

        except Exception as e:
            logging.error(f"[VisionAgentReligiousHistorical] Error saving masked version: {e}", exc_info=True)

    def _parse_gemini_response(self, response_text: str) -> Optional[Dict]:
        """Parse Gemini API response text into JSON."""
        try:
            # Clean up response text
            json_str = response_text.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]

            return json.loads(json_str)
        except Exception as e:
            logging.error(f"Failed to parse Gemini response: {e}")
            logging.debug(f"Raw response text: {response_text}")
            return None

    def _parse_grok_response(self, response_text: str) -> Optional[Dict]:
        """Parse Grok API response text into JSON."""
        try:
            # Grok API responses might also be in JSON blocks
            json_str = response_text.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]

            return json.loads(json_str)
        except Exception as e:
            logging.error(f"Failed to parse Grok response: {e}")
            logging.debug(f"Raw response text: {response_text}")
            return None


    def _create_segmentation_masks(self, detections: List[Dict[str, Any]], image_size: Tuple[int, int]) -> List[SegmentationMask]:
        """Create segmentation masks from detection results."""
        width, height = image_size
        masks = []

        for det in detections:
            try:
                box = det["box_2d"]
                y0 = int(box[0] * height / 1000)
                x0 = int(box[1] * width / 1000)
                y1 = int(box[2] * height / 1000)
                x1 = int(box[3] * width / 1000)

                # Create binary mask
                mask = np.zeros((height, width), dtype=np.uint8)
                mask[max(0, y0):min(height, y1), max(0, x0):min(width, x1)] = 255

                masks.append(SegmentationMask(
                    label=det["label"],
                    mask=mask,
                    confidence=det.get("confidence", 0.0),
                    score=det.get("importance", 0.0),
                    x0=x0, y0=y0, x1=x1, y1=y1,
                    type=det.get("type", "mask"),
                    status="full"  # Could be refined based on truncation analysis
                ))

            except Exception as e:
                logging.warning(f"[VisionAgentReligiousHistorical] Failed to create mask for {det.get('label', 'unknown')}: {e}")
                continue

        return masks

    def _calculate_object_importance(self, box: List[float], obj_type: str, img_size: Tuple[int, int], 
                                   base_importance: float, label: str, features: Optional[List[Dict]] = None) -> float:
        """Calculate importance score for an object in religious/historical context."""
        # Convert normalized coordinates to pixels
        width, height = img_size
        y0, x0, y1, x1 = box
        obj_width = (x1 - x0) * width / 1000
        obj_height = (y1 - y0) * height / 1000
        obj_area = obj_width * obj_height
        total_area = width * height

        # Base importance from object type
        type_multipliers = {
            'religious_figure': 2.0,
            'historical_figure': 1.8,
            'symbol': 1.5,
            'artifact': 1.3,
            'group': 1.2,
            'background': 0.5,
            'unknown': 1.0
        }
        
        # Calculate score
        type_multiplier = type_multipliers.get(obj_type.lower(), 1.0)
        size_factor = np.sqrt(obj_area / total_area)  # Sqrt to reduce impact of size
        importance = base_importance * type_multiplier * (0.3 + 0.7 * size_factor)

        # Boost for central objects
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2
        center_dist = np.sqrt((center_x - 500)**2 + (center_y - 500)**2) / 500  # Distance to image center
        centrality_boost = 1.0 + (1.0 - min(1.0, center_dist)) * 0.5

        importance *= centrality_boost

        # Additional boosts for religious/historical elements
        if 'face' in label.lower() or (features and len(features) > 0):
            importance *= 1.2  # Boost for faces/features
        if any(term in label.lower() for term in ['jesus', 'mary', 'saint', 'angel', 'god']):
            importance *= 1.5  # Boost for key religious figures
        if any(term in label.lower() for term in ['cross', 'halo', 'crown', 'altar', 'throne']):
            importance *= 1.3  # Boost for religious symbols

        return min(1.0, importance)  # Cap at 1.0


    def _threshold_important_objects(self, objects: List[Dict[str, Any]], percentile_threshold: float = 85) -> List[Dict[str, Any]]:
        """Filter objects for religious/historical art - prioritize figures, symbols, and subjects."""
        if not objects:
            return []

        importances = [obj.get('importance', 0) for obj in objects]
        if not importances:
            return objects

        # Higher percentile threshold for religious/historical art to focus on key elements
        threshold = np.percentile(importances, percentile_threshold)

        primary_subject_label = self.primary_subject.lower() if hasattr(self, 'primary_subject') and self.primary_subject else None
        secondary_subject_labels = [sub.lower() for sub in self.secondary_subjects] if hasattr(self, 'secondary_subjects') and self.secondary_subjects else []

        important = []
        for obj in objects:
            obj_label_lower = obj.get('label', '').lower()
            if obj.get('importance', 0) >= threshold or \
               obj.get('type') in ['religious_figure', 'historical_figure', 'symbol', 'artifact', 'figure', 'group'] or \
               (primary_subject_label and primary_subject_label in obj_label_lower) or \
               any(sub_label in obj_label_lower for sub_label in secondary_subject_labels):
                important.append(obj)

        return important


    def save_analysis_outputs(self, image_path: str, analysis_results: Dict[str, Any], output_folder: str) -> None:
        """Save all versions of the analyzed image."""
        try:
            basename = os.path.splitext(os.path.basename(image_path))[0]
            if not os.path.exists(image_path):
                logging.warning(f"[VisionAgentReligiousHistorical] Image file not found for saving outputs: {image_path}")
                return

            os.makedirs(output_folder, exist_ok=True)

            # Load image and create copies for different outputs
            with Image.open(image_path) as img:
                # 1. Save labeled version
                labeled_path = os.path.join(output_folder, f"{basename}_labeled.jpg")
                img_labeled = img.copy()
                self._save_labeled_version(img_labeled, analysis_results, labeled_path)

                # 2. Save masked version
                masked_path = os.path.join(output_folder, f"{basename}_masked.jpg")
                img_masked = img.copy()
                self._save_masked_version(img_masked, analysis_results, masked_path)

                # 3. Save cropped version
                cropped_path = os.path.join(output_folder, f"{basename}_cropped.jpg")
                self.copy_and_crop_image(image_path, cropped_path, analysis_results)

            logging.info(f"[VisionAgentReligiousHistorical] Successfully saved all analysis outputs to {output_folder}")

        except Exception as e:
            logging.error(f"[VisionAgentReligiousHistorical] Error saving analysis outputs: {e}", exc_info=True)

    def _save_labeled_version(self, img: Image.Image, analysis_results: Dict[str, Any], output_path: str) -> None:
        """Save version with bounding boxes and labels."""
        try:
            # Convert to RGBA for transparency support
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            draw = ImageDraw.Draw(img)

            # Process detected objects
            for obj in analysis_results.get("objects", []):
                label = obj.get("label", "unknown")
                importance = obj.get("importance", 0.5)
                bbox = obj.get("box_2d", [0, 0, 0, 0])
                confidence = obj.get("confidence", 0.0)

                # Draw bounding box
                box_color = self.colors[hash(label) % len(self.colors)]
                box_coords = [
                    bbox[1], bbox[0],  # x0, y0
                    bbox[3], bbox[2]   # x1, y1
                ]
                draw.rectangle(box_coords, outline=box_color, width=2)

                # Add label with background
                label_text = f"{label} ({importance:.2f})"
                text_bbox = draw.textbbox((box_coords[0], box_coords[1]-20), label_text)
                draw.rectangle([
                    text_bbox[0]-2, text_bbox[1]-2,
                    text_bbox[2]+2, text_bbox[3]+2
                ], fill=(0, 0, 0, 180))
                draw.text((box_coords[0], box_coords[1]-20), label_text, fill=box_color)

            # Save output
            img = img.convert('RGB')
            img.save(output_path, format='JPEG', quality=95)
            logging.info(f"[VisionAgentReligiousHistorical] Saved labeled image to {output_path}")

        except Exception as e:
            logging.error(f"[VisionAgentReligiousHistorical] Error saving labeled version: {e}", exc_info=True)

    def _save_masked_version(self, img: Image.Image, analysis_results: Dict[str, Any], output_path: str) -> None:
        """Save version with masks for important objects."""
        try:
            # Convert to RGBA for transparency support
            if img.mode != 'RGBA':
                img = img.convert('RGBA')

            # Get important objects based on significance
            important_objects = [obj for obj in analysis_results.get("objects", [])
                               if obj.get("importance", 0) > 0.5]

            # Create and apply masks
            for i, obj in enumerate(important_objects):
                mask_color = self.colors[i % len(self.colors)]
                bbox = obj.get("box_2d", [0, 0, 0, 0])
                
                # Create simple rectangular mask
                mask = Image.new('RGBA', img.size, (0, 0, 0, 0))
                mask_draw = ImageDraw.Draw(mask)
                mask_draw.rectangle([
                    bbox[1], bbox[0],  # x0, y0
                    bbox[3], bbox[2]   # x1, y1
                ], fill=(*ImageColor.getrgb(mask_color), 128))  # Semi-transparent

                # Apply mask
                img = Image.alpha_composite(img, mask)

            # Save output
            img = img.convert('RGB')
            img.save(output_path, format='JPEG', quality=95)
            logging.info(f"[VisionAgentReligiousHistorical] Saved masked image to {output_path}")

        except Exception as e:
            logging.error(f"[VisionAgentReligiousHistorical] Error saving masked version: {e}", exc_info=True)

    def _parse_gemini_response(self, response_text: str) -> Optional[Dict]:
        """Parse Gemini API response text into JSON."""
        try:
            # Clean up response text
            json_str = response_text.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]

            return json.loads(json_str)
        except Exception as e:
            logging.error(f"Failed to parse Gemini response: {e}")
            logging.debug(f"Raw response text: {response_text}")
            return None

    def _parse_grok_response(self, response_text: str) -> Optional[Dict]:
        """Parse Grok API response text into JSON."""
        try:
            # Grok API responses might also be in JSON blocks
            json_str = response_text.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]

            return json.loads(json_str)
        except Exception as e:
            logging.error(f"Failed to parse Grok response: {e}")
            logging.debug(f"Raw response text: {response_text}")
            return None


    def _create_segmentation_masks(self, detections: List[Dict[str, Any]], image_size: Tuple[int, int]) -> List[SegmentationMask]:
        """Create segmentation masks from detection results."""
        width, height = image_size
        masks = []

        for det in detections:
            try:
                box = det["box_2d"]
                y0 = int(box[0] * height / 1000)
                x0 = int(box[1] * width / 1000)
                y1 = int(box[2] * height / 1000)
                x1 = int(box[3] * width / 1000)

                # Create binary mask
                mask = np.zeros((height, width), dtype=np.uint8)
                mask[max(0, y0):min(height, y1), max(0, x0):min(width, x1)] = 255

                masks.append(SegmentationMask(
                    label=det["label"],
                    mask=mask,
                    confidence=det.get("confidence", 0.0),
                    score=det.get("importance", 0.0),
                    x0=x0, y0=y0, x1=x1, y1=y1,
                    type=det.get("type", "mask"),
                    status="full"  # Could be refined based on truncation analysis
                ))

            except Exception as e:
                logging.warning(f"[VisionAgentReligiousHistorical] Failed to create mask for {det.get('label', 'unknown')}: {e}")
                continue

        return masks

    def _calculate_object_importance(self, box: List[float], obj_type: str, img_size: Tuple[int, int], 
                                   base_importance: float, label: str, features: Optional[List[Dict]] = None) -> float:
        """Calculate importance score for an object in religious/historical context."""
        # Convert normalized coordinates to pixels
        width, height = img_size
        y0, x0, y1, x1 = box
        obj_width = (x1 - x0) * width / 1000
        obj_height = (y1 - y0) * height / 1000
        obj_area = obj_width * obj_height
        total_area = width * height

        # Base importance from object type
        type_multipliers = {
            'religious_figure': 2.0,
            'historical_figure': 1.8,
            'symbol': 1.5,
            'artifact': 1.3,
            'group': 1.2,
            'background': 0.5,
            'unknown': 1.0
        }
        
        # Calculate score
        type_multiplier = type_multipliers.get(obj_type.lower(), 1.0)
        size_factor = np.sqrt(obj_area / total_area)  # Sqrt to reduce impact of size
        importance = base_importance * type_multiplier * (0.3 + 0.7 * size_factor)

        # Boost for central objects
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2
        center_dist = np.sqrt((center_x - 500)**2 + (center_y - 500)**2) / 500  # Distance to image center
        centrality_boost = 1.0 + (1.0 - min(1.0, center_dist)) * 0.5

        importance *= centrality_boost

        # Additional boosts for religious/historical elements
        if 'face' in label.lower() or (features and len(features) > 0):
            importance *= 1.2  # Boost for faces/features
        if any(term in label.lower() for term in ['jesus', 'mary', 'saint', 'angel', 'god']):
            importance *= 1.5  # Boost for key religious figures
        if any(term in label.lower() for term in ['cross', 'halo', 'crown', 'altar', 'throne']):
            importance *= 1.3  # Boost for religious symbols

        return min(1.0, importance)  # Cap at 1.0


    def _threshold_important_objects(self, objects: List[Dict[str, Any]], percentile_threshold: float = 85) -> List[Dict[str, Any]]:
        """Filter objects for religious/historical art - prioritize figures, symbols, and subjects."""
        if not objects:
            return []

        importances = [obj.get('importance', 0) for obj in objects]
        if not importances:
            return objects

        # Higher percentile threshold for religious/historical art to focus on key elements
        threshold = np.percentile(importances, percentile_threshold)

        primary_subject_label = self.primary_subject.lower() if hasattr(self, 'primary_subject') and self.primary_subject else None
        secondary_subject_labels = [sub.lower() for sub in self.secondary_subjects] if hasattr(self, 'secondary_subjects') and self.secondary_subjects else []

        important = []
        for obj in objects:
            obj_label_lower = obj.get('label', '').lower()
            if obj.get('importance', 0) >= threshold or \
               obj.get('type') in ['religious_figure', 'historical_figure', 'symbol', 'artifact', 'figure', 'group'] or \
               (primary_subject_label and primary_subject_label in obj_label_lower) or \
               any(sub_label in obj_label_lower for sub_label in secondary_subject_labels):
                important.append(obj)

        return important


    def save_analysis_outputs(self, image_path: str, analysis_results: Dict[str, Any], output_folder: str) -> None:
        """Save all versions of the analyzed image."""
        try:
            basename = os.path.splitext(os.path.basename(image_path))[0]
            if not os.path.exists(image_path):
                logging.warning(f"[VisionAgentReligiousHistorical] Image file not found for saving outputs: {image_path}")
                return

            os.makedirs(output_folder, exist_ok=True)

            # Load image and create copies for different outputs
            with Image.open(image_path) as img:
                # 1. Save labeled version
                labeled_path = os.path.join(output_folder, f"{basename}_labeled.jpg")
                img_labeled = img.copy()
                self._save_labeled_version(img_labeled, analysis_results, labeled_path)

                # 2. Save masked version
                masked_path = os.path.join(output_folder, f"{basename}_masked.jpg")
                img_masked = img.copy()
                self._save_masked_version(img_masked, analysis_results, masked_path)

                # 3. Save cropped version
                cropped_path = os.path.join(output_folder, f"{basename}_cropped.jpg")
                self.copy_and_crop_image(image_path, cropped_path, analysis_results)

            logging.info(f"[VisionAgentReligiousHistorical] Successfully saved all analysis outputs to {output_folder}")

        except Exception as e:
            logging.error(f"[VisionAgentReligiousHistorical] Error saving analysis outputs: {e}", exc_info=True)