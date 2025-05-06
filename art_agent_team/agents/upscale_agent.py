import os
import base64
import io
from PIL import Image
import vertexai # Added
from vertexai.preview.vision_models import ImageGenerationModel # Added
from google.api_core import exceptions as google_exceptions
from google.oauth2 import service_account
from google.auth import exceptions as auth_exceptions

import logging

class UpscaleAgent:
    def __init__(self, config=None):
        """
        Initialize the UpscaleAgent using Google Vertex AI.
        Requires 'vertex_project_id' and 'vertex_location' in the config.
        """
        self.api_ready = False
        self.vertex_model = None
        self.project_id = None
        self.location = None

        if not config:
            logging.error("[UpscaleAgent] ERROR: Configuration dictionary is required for Vertex AI initialization.")
            return

        self.project_id = config.get('vertex_project_id')
        self.location = config.get('vertex_location')

        # credentials_path = config.get('google_credentials_path') # No longer read from config

        if not self.project_id or not self.location:
            logging.error("[UpscaleAgent] ERROR: 'vertex_project_id' and 'vertex_location' must be provided in the config.")
            return # api_ready remains False

        # Check if the environment variable is set (optional, as SDK handles it)
        if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
             logging.warning("[UpscaleAgent] GOOGLE_APPLICATION_CREDENTIALS environment variable not set. Vertex AI will attempt to use Application Default Credentials (ADC).")
        else:
             logging.info("[UpscaleAgent] GOOGLE_APPLICATION_CREDENTIALS found. Vertex AI will attempt to use it.")

        try:
            # Initialize Vertex AI SDK - It will automatically use GOOGLE_APPLICATION_CREDENTIALS
            # or fall back to Application Default Credentials (ADC) if the env var is set.
            logging.info(f"[UpscaleAgent] Initializing Vertex AI for project '{self.project_id}' in location '{self.location}' (using environment credentials)...")
            # No explicit credentials passed here anymore
            vertexai.init(project=self.project_id, location=self.location)

            # Load the model only after successful initialization
            logging.info("[UpscaleAgent] Vertex AI initialized successfully (attempted using environment credentials). Loading ImageGenerationModel...")
            # Using imagegeneration@006 as identified by debugger
            self.vertex_model = ImageGenerationModel.from_pretrained("imagegeneration@006")
            self.api_ready = True
            logging.info("[UpscaleAgent] ImageGenerationModel loaded successfully. API is ready.")

        # FileNotFoundError is less likely now unless GOOGLE_APPLICATION_CREDENTIALS points to a bad path
        except FileNotFoundError:
            logging.error(f"[UpscaleAgent] ERROR: Credentials file specified by GOOGLE_APPLICATION_CREDENTIALS not found.")
            self.api_ready = False
        except auth_exceptions.RefreshError as e:
             logging.error(f"[UpscaleAgent] ERROR: Could not refresh credentials (likely from GOOGLE_APPLICATION_CREDENTIALS or ADC): {e}")
             self.api_ready = False
        except auth_exceptions.DefaultCredentialsError as e:
             logging.error(f"[UpscaleAgent] ERROR: Invalid credentials format or file (likely from GOOGLE_APPLICATION_CREDENTIALS or ADC): {e}")
             self.api_ready = False
        except google_exceptions.GoogleAPIError as e:
            logging.error(f"[UpscaleAgent] ERROR: Vertex AI API Error during initialization or model loading: {e}")
            self.api_ready = False
        except Exception as e:
            # Catch any other unexpected errors during credential loading, init, or model loading
            logging.error(f"[UpscaleAgent] ERROR: An unexpected error occurred during Vertex AI setup: {e}")
            self.api_ready = False

    def upscale_image(self, input_path, output_path):
        """
        Upscale the input image using the Google Vertex AI Image Generation API.
        Saves the upscaled image to output_path.
        """
        if not self.api_ready:
            logging.warning(f"[UpscaleAgent] Skipping upscale for '{input_path}' as Vertex AI API is not ready.")
            return input_path # Return original path if API failed to initialize

        logging.info(f"[UpscaleAgent] Starting Vertex AI upscale for '{input_path}'...")

        try:
            # 1. Read image bytes
            with open(input_path, "rb") as image_file:
                image_bytes = image_file.read()

            # 2. Encode to base64
            base64_string = base64.b64encode(image_bytes).decode("utf-8")

            # 3. Prepare API call parameters
            instances = [
                {
                    "image": {"bytesBase64Encoded": base64_string}
                }
            ]
            parameters = {
                "mode": "upscale",
                "upscaleConfig": {"upscaleFactor": "x4"}, # Assuming x4 upscale factor
                "number_of_images": 1
            }

            # 4. Make the API call
            logging.info(f"[UpscaleAgent] Calling Vertex AI generate_images API...")
            response = self.vertex_model.generate_images(
                instances=instances,
                parameters=parameters,
            )
            logging.info(f"[UpscaleAgent] Vertex AI API call successful.")

            # 5. Process the response
            if not response.images or len(response.images) == 0:
                logging.error(f"[UpscaleAgent] ERROR: No images returned from Vertex AI API for '{input_path}'.")
                return input_path

            # Extract image bytes (assuming the first image is the result)
            upscaled_image_bytes = response.images[0]._image_bytes

            # 6. Save the upscaled image
            img = Image.open(io.BytesIO(upscaled_image_bytes))

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img.save(output_path)
            logging.info(f"[UpscaleAgent] Successfully saved upscaled image to '{output_path}'.")
            return output_path

        except FileNotFoundError:
            logging.error(f"[UpscaleAgent] ERROR: Input file not found at '{input_path}'.")
            return input_path
        except google_exceptions.GoogleAPIError as e:
            logging.error(f"[UpscaleAgent] ERROR: Vertex AI API call failed for '{input_path}': {e}")
            return input_path
        except Exception as e:
            logging.error(f"[UpscaleAgent] ERROR: An unexpected error occurred during upscaling for '{input_path}': {e}")
            return input_path


# # Example usage for testing (Requires credentials and config)
# if __name__ == "__main__":
#     # This requires a config dictionary with valid 'vertex_project_id' and 'vertex_location'
#     # and appropriate Google Cloud authentication configured in the environment.
#     # Example:
#     # test_config = {
#     #     'vertex_project_id': 'your-gcp-project-id',
#     #     'vertex_location': 'us-central1'
#     # }
#     # agent = UpscaleAgent(config=test_config)
#     # if agent.api_ready:
#     #     agent.upscale_image('path/to/your/input.jpg', 'path/to/your/output.png')
#     pass