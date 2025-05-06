import unittest
import os
import sys
import logging
from PIL import Image, UnidentifiedImageError
import numpy as np
import shutil  # For cleaning up test outputs

# Assuming tests are run from the project root or using a test runner that handles paths
from art_agent_team.agents.upscale_agent import UpscaleAgent

# Configure logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [TestUpscaleAgent] %(message)s')

class TestUpscaleAgentReal(unittest.TestCase):
    """Tests for UpscaleAgent using real file I/O and the actual model (if available)."""

    @classmethod
    def setUpClass(cls):
        """Set up paths valid for the entire test class."""
        cls.base_dir = os.path.dirname(__file__) # art_agent_team/tests
        cls.project_root = os.path.abspath(os.path.join(cls.base_dir, '..', '..')) # /Users/nathaniel.cannon/Documents/VScodeWork/Art_AI
        cls.input_dir = os.path.join(cls.base_dir, 'test_data', 'input')
        cls.output_dir = os.path.join(cls.base_dir, 'test_data', 'output', 'upscale_output')
        # cls.model_path = os.path.join(cls.project_root, 'models', 'esrgan_model.h5') # Removed: Agent now uses Vertex AI via config

        # Ensure input directory exists
        if not os.path.isdir(cls.input_dir):
            # Attempt to find it relative to project root if tests dir structure varies
            cls.input_dir = os.path.join(cls.project_root, 'art_agent_team', 'tests', 'test_data', 'input')
            if not os.path.isdir(cls.input_dir):
                 logging.error(f"[TestUpscaleAgent] Input directory not found at primary or secondary path: {cls.input_dir}")
                 raise FileNotFoundError(f"Input directory not found at primary or secondary path: {cls.input_dir}")

        # Create output directory if it doesn't exist
        os.makedirs(cls.output_dir, exist_ok=True)
        logging.info(f"[TestUpscaleAgent] Test output directory: {cls.output_dir}")

        # --- Configuration for Vertex AI UpscaleAgent ---
        # Load required config from environment variables for testing
        # Ensure VERTEX_PROJECT_ID and VERTEX_LOCATION are set in the test environment
        cls.config = {
            'vertex_project_id': os.environ.get('VERTEX_PROJECT_ID'),
            'vertex_location': os.environ.get('VERTEX_LOCATION')
        }
        if not cls.config['vertex_project_id'] or not cls.config['vertex_location']:
             error_msg = "[TestUpscaleAgent] ERROR: Environment variables VERTEX_PROJECT_ID and VERTEX_LOCATION must be set for UpscaleAgent tests."
             logging.error(error_msg)
             cls.fail(error_msg) # Fail class setup if config is missing
        else:
             logging.info("[TestUpscaleAgent] Vertex AI config loaded from environment variables.")
        # --- End Configuration ---

        # Model existence check removed as agent uses Vertex AI now

    def setUp(self):
        """Per-test setup: define file names, ensure clean state, and copy class config."""
        # Log Index: D001 - Copy class config to instance
        self.config = self.__class__.config # Make class config available to test instance
        self.valid_image_name = 'Frank Brangwyn_animal.jpg' # Example valid image from test_data/input
        self.valid_input_path = os.path.join(self.input_dir, self.valid_image_name)
        self.output_image_name = f"upscaled_{self.valid_image_name}"
        self.valid_output_path = os.path.join(self.output_dir, self.output_image_name)

        # Ensure the specific input image exists for each test run
        if not os.path.exists(self.valid_input_path):
             # Try alternative common location if tests run from project root
             alt_input_path = os.path.join('art_agent_team', 'tests', 'test_data', 'input', self.valid_image_name)
             if os.path.exists(alt_input_path):
                 self.valid_input_path = alt_input_path
             else:
                 raise FileNotFoundError(f"Required test input image not found: {self.valid_input_path} or {alt_input_path}")

        # Clean up any previous output file before the test
        if os.path.exists(self.valid_output_path):
            try:
                os.remove(self.valid_output_path)
                logging.debug(f"[TestUpscaleAgent] Removed previous output file: {self.valid_output_path}")
            except OSError as e:
                logging.error(f"[TestUpscaleAgent] Error removing previous output file {self.valid_output_path}: {e}")


    def test_upscale_image_success(self):
        """Test successful upscaling of a valid image using the agent. Assumes model exists due to setUpClass check."""
        # Log Index: D002 - Initialize agent with config
        # Log initialization details (avoid logging sensitive parts of config if necessary)
        logging.info(f"[TestUpscaleAgent] Initializing UpscaleAgent using config (Project: {self.config.get('vertex_project_id')}, Location: {self.config.get('vertex_location')})")
        agent = UpscaleAgent(config=self.config) # Initialize with the config dictionary

        # Run the upscale process
        logging.info(f"[TestUpscaleAgent] Attempting to upscale {self.valid_input_path} to {self.valid_output_path}")
        try:
            result_path = agent.upscale_image(self.valid_input_path, self.valid_output_path)
        except Exception as e:
             # Log the full traceback for debugging
             logging.exception(f"[TestUpscaleAgent] Upscaling failed unexpectedly for {self.valid_input_path}")
             self.fail(f"[TestUpscaleAgent] agent.upscale_image raised an unexpected exception: {e}")


        # --- Verification ---
        self.assertEqual(result_path, self.valid_output_path, "Returned output path does not match expected.")
        self.assertTrue(os.path.exists(self.valid_output_path), f"Output file was not created at {self.valid_output_path}.")
        self.assertGreater(os.path.getsize(self.valid_output_path), 0, "Output file is empty.")

        # Check output image properties (resolution and validity)
        try:
            with Image.open(self.valid_output_path) as img:
                logging.info(f"[TestUpscaleAgent] Output image size: {img.size}, Format: {img.format}")
                # ESRGAN might not hit the exact target, allow some flexibility or check aspect ratio
                # Or check if it was resized to the target resolution if model output differs
                self.assertEqual(img.size, agent.target_resolution, f"[TestUpscaleAgent] Output image resolution {img.size} does not match target {agent.target_resolution}.")
                # Check if file is a valid image by trying to load it
                img.load()
        except UnidentifiedImageError:
             self.fail(f"[TestUpscaleAgent] Output file {self.valid_output_path} is not a valid image file.")
        except Exception as e:
            self.fail(f"[TestUpscaleAgent] Output image validation failed for {self.valid_output_path}: {e}")

        # Log Index: D003 - PSNR/Fidelity Check Note
        # Optional: Check PSNR and fidelity.
        # Note: These metrics and thresholds might need adjustment based on the
        # performance and output characteristics of the Vertex AI upscaling model.
        # Reload original and resize for comparison
        try:
            original_img = Image.open(self.valid_input_path)
            # Resize original to target for fair PSNR comparison
            original_resized = original_img.resize(agent.target_resolution, Image.LANCZOS)
            upscaled_img = Image.open(self.valid_output_path)

            psnr = agent.calculate_psnr(np.array(original_resized), np.array(upscaled_img))
            fidelity = agent.check_fidelity(self.valid_input_path, self.valid_output_path)

            logging.info(f"[TestUpscaleAgent] Test PSNR: {psnr:.2f} dB, Fidelity: {fidelity:.3f} for {self.valid_output_path}")
            # Assertions should run as setUpClass guarantees model existence if test reaches here
            # Example thresholds - adjust as needed based on model performance
            self.assertGreaterEqual(psnr, 25, f"[TestUpscaleAgent] PSNR {psnr} is below expected threshold (25 dB)") # Example threshold
            self.assertGreaterEqual(fidelity, 0.90, f"[TestUpscaleAgent] Fidelity {fidelity} is below expected threshold (0.90)") # Example threshold
        except Exception as e:
             self.fail(f"[TestUpscaleAgent] Error during PSNR/Fidelity calculation or assertion: {e}")


    def test_upscale_invalid_input_path(self):
        """Test upscaling with an invalid input file path."""
        # Log Index: D004 - Initialize agent with config
        agent = UpscaleAgent(config=self.config)
        invalid_input_path = os.path.join(self.input_dir, 'non_existent_image.jpg')
        logging.info(f"[TestUpscaleAgent] Testing with non-existent input: {invalid_input_path}")
        with self.assertRaises(FileNotFoundError, msg=f"[TestUpscaleAgent] Expected FileNotFoundError for missing input {invalid_input_path}"):
            agent.upscale_image(invalid_input_path, self.valid_output_path)

    def test_upscale_corrupted_image(self):
        """Test handling of corrupted image files (should raise PIL.UnidentifiedImageError or similar)."""
        # Log Index: D005 - Initialize agent with config
        agent = UpscaleAgent(config=self.config)
        # Create a dummy corrupted file for testing
        corrupted_image_name = 'corrupted_test.jpg'
        corrupted_input_path = os.path.join(self.input_dir, corrupted_image_name)
        corrupted_output_path = os.path.join(self.output_dir, f"upscaled_{corrupted_image_name}")

        logging.info(f"[TestUpscaleAgent] Creating dummy corrupted file: {corrupted_input_path}")
        try:
            with open(corrupted_input_path, 'w') as f:
                f.write("this is definitely not image data")

            # Expect PIL to raise an error when trying to open the corrupted file
            logging.info(f"[TestUpscaleAgent] Attempting to upscale corrupted file: {corrupted_input_path}")
            # Catch specific PIL error if possible, otherwise broader Exception
            with self.assertRaises((UnidentifiedImageError, ValueError, TypeError, Exception), msg="[TestUpscaleAgent] Expected an error when processing a corrupted image.") as cm:
                 agent.upscale_image(corrupted_input_path, corrupted_output_path)
            logging.info(f"[TestUpscaleAgent] Caught expected exception for corrupted file: {type(cm.exception).__name__} - {cm.exception}")
            # Ensure no output file was created on error
            self.assertFalse(os.path.exists(corrupted_output_path), "[TestUpscaleAgent] Output file should not be created for corrupted input.")

        finally:
            # Clean up the dummy file
            if os.path.exists(corrupted_input_path):
                os.remove(corrupted_input_path)
                logging.debug(f"[TestUpscaleAgent] Removed dummy corrupted file: {corrupted_input_path}")

    # Note: test_upscale_with_mock_model_fallback has been removed as the mock model is no longer used.

    @classmethod
    def tearDownClass(cls):
        """Clean up the output directory after all tests are complete."""
        if os.path.exists(cls.output_dir):
             logging.info(f"[TestUpscaleAgent] Attempting to clean up test output directory: {cls.output_dir}")
             try:
                 # Comment out rmtree for safety during initial runs
                 # shutil.rmtree(cls.output_dir)
                 # logging.info(f"[TestUpscaleAgent] Successfully removed test output directory: {cls.output_dir}")
                 logging.warning(f"[TestUpscaleAgent] Cleanup skipped: Remove directory manually if needed: {cls.output_dir}")
             except OSError as e:
                 logging.error(f"[TestUpscaleAgent] Failed to remove test output directory {cls.output_dir}: {e}")
        else:
             logging.info("[TestUpscaleAgent] Test output directory does not exist, no cleanup needed.")

if __name__ == '__main__':
    # Ensure the script can be run directly without manual path adjustments
    unittest.main(verbosity=2) # Increase verbosity for detailed test output