import logging
import json
from typing import Optional, Dict, Any
from art_agent_team.agents.vision_agent_abstract import VisionAgentAbstract, CorruptedImageError, UnsupportedImageFormatError
from PIL import Image, UnidentifiedImageError
from vertexai.preview.generative_models import Part, Image as VertexImage, GenerationConfig

class VisionAgentLandscape(VisionAgentAbstract):
    """Agent for analyzing landscape art using Vertex AI Gemini model."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)  # Call to superclass constructor
        self.genre = "Landscape"
        logging.info(f"VisionAgentLandscape initialized for {self.genre} with Vertex AI model.")

    def analyze_image(self, image_path: str, research_data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Analyzes a landscape image using the Vertex AI Gemini model.
        """
        logging.info(f"[VisionAgentLandscape] Analyzing image: {image_path} with research data: {research_data is not None}")
        if not image_path or not self.gemini_vertex_model:
            logging.error("[VisionAgentLandscape] No image path or Vertex AI model not initialized.")
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

            # Construct the prompt using research_data if available
            prompt_parts = [
                image_part,
                "Analyze this image in detail. Focus on elements typical of landscape art. ",
                "Describe the composition, color palette, light and shadow, and overall mood. ",
                "Identify any notable geographical features or atmospheric conditions. "
            ]
            if research_data and "artist_name" in research_data and "artwork_title" in research_data:
                prompt_parts.append(f"Consider the artwork '{research_data['artwork_title']}' by {research_data['artist_name']}. ")
            if research_data and "style_period" in research_data:
                prompt_parts.append(f"Relate it to the {research_data['style_period']} style/period if applicable. ")

            prompt_parts.append("Provide your analysis as a structured JSON object with keys like 'composition', 'color_palette', 'lighting', 'mood', 'subject_elements', 'style_period_assessment', and 'overall_impression'.")

            logging.debug(f"[VisionAgentLandscape] Sending prompt to Vertex AI Gemini: {''.join(prompt_parts[1:])}") # Log text parts of prompt
            
            # Use the inherited Vertex AI model
            response = self.gemini_vertex_model.generate_content(
                contents=prompt_parts,
                generation_config=generation_config
            )
            
            if response and response.text:
                try:
                    # Parse JSON response - returns a dictionary of analysis results
                    analysis_text = response.text
                    try:
                        analysis_result = json.loads(analysis_text)
                    except json.JSONDecodeError:
                        # If JSON parsing fails, return the raw text as a dict
                        analysis_result = {
                            "analysis": analysis_text,
                            "error": "Failed to parse response as JSON"
                        }
                    
                    analysis_result.update({
                        "image_path": image_path,
                        "genre": "landscape"
                    })
                    
                    logging.info(f"[VisionAgentLandscape] Successfully analyzed image: {image_path}")
                    return analysis_result
                    
                except (AttributeError, IndexError, TypeError) as e:
                    error_msg = f"Error processing Vertex AI response: {str(e)}"
                    logging.error(f"[VisionAgentLandscape] {error_msg}. Response: {response.text if hasattr(response, 'text') else response}", exc_info=True)
                    return {
                        "error": error_msg,
                        "image_path": image_path,
                        "genre": "landscape"
                    }
                except Exception as e:
                    error_msg = f"Unexpected error analyzing image: {str(e)}"
                    logging.error(f"[VisionAgentLandscape] {error_msg}", exc_info=True)
                    return {
                        "error": error_msg,
                        "image_path": image_path,
                        "genre": "landscape"
                    }
                    
            logging.warning(f"[VisionAgentLandscape] No valid analysis content received from Vertex AI Gemini for {image_path}.")
            return {
                "error": "No valid analysis content received from Vertex AI Gemini",
                "image_path": image_path,
                "genre": "landscape"
            }
            
        except FileNotFoundError:
            logging.error(f"[VisionAgentLandscape] Image file not found: {image_path}")
            raise
        except UnidentifiedImageError as e:
            logging.error(f"[VisionAgentLandscape] Cannot identify image file (corrupted/unsupported): {image_path}. Error: {e}")
            raise CorruptedImageError(f"Corrupted or unsupported image file: {image_path}") from e
        except IOError as e: # Catch other IOErrors, e.g. file not readable
            logging.error(f"[VisionAgentLandscape] IOError opening or reading image: {image_path}. Error: {e}")
            raise UnsupportedImageFormatError(f"IOError, possibly unsupported image format or permissions issue: {image_path}") from e
        except Exception as e:
            logging.error(f"[VisionAgentLandscape] Failed to analyze image with Vertex AI Gemini: {e}", exc_info=True)
            return {
                "error": f"Failed to analyze image with Vertex AI Gemini: {str(e)}",
                "image_path": image_path,
                "genre": "landscape"
            }
