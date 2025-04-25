import unittest
from unittest.mock import patch, MagicMock, call
import os
import sys
import yaml
import logging

# Add the project root to the Python path to allow importing agent modules
# Adjust the path depth as necessary depending on where tests are run from
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import after adjusting path
try:
    from docent_agent import DocentAgent
    # We will mock the VisionAgent class itself or its methods later
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Current sys.path: {sys.path}")
    # If running tests from the root, the path adjustment might not be needed,
    # but it's generally safer for discoverability.
    raise

# Suppress logging during tests to keep output clean
logging.disable(logging.CRITICAL)

class TestDocentVisionInteraction(unittest.TestCase):

    def setUp(self):
        """Set up test environment before each test."""
        # Create dummy config for testing
        self.test_config_path = 'art_agent_team/tests/test_config.yaml'
        self.test_config = {
            'input_folder': 'art_agent_team/tests/test_data/input',
            'workspace_folder': 'art_agent_team/tests/test_data/workspace',
            'output_folder': 'art_agent_team/tests/test_data/output',
            'docent_llm_model': 'mock-model',
            'grok_api_key': 'fake-key', # Needed for LLM parsing mock setup
            'google_credentials_path': 'art_agent_team/tests/fake_credentials.json' # Path for mock credentials
        }
        # Create dummy directories
        os.makedirs(os.path.dirname(self.test_config_path), exist_ok=True)
        os.makedirs(self.test_config['input_folder'], exist_ok=True)
        os.makedirs(self.test_config['workspace_folder'], exist_ok=True)
        os.makedirs(self.test_config['output_folder'], exist_ok=True)

        # Write dummy config file
        with open(self.test_config_path, 'w') as f:
            yaml.dump(self.test_config, f)

        # Create dummy credentials file
        with open(self.test_config['google_credentials_path'], 'w') as f:
            f.write('{"fake": "credentials"}') # Content doesn't matter for mock

        # Create a dummy input image file
        self.dummy_image_name = 'test_image.jpg'
        self.dummy_image_path = os.path.join(self.test_config['input_folder'], self.dummy_image_name)
        try:
            from PIL import Image
            img = Image.new('RGB', (60, 30), color = 'red')
            img.save(self.dummy_image_path)
        except ImportError:
            # Fallback if Pillow is not installed in the test environment yet
             with open(self.dummy_image_path, 'w') as f:
                f.write('dummy image data')


    def tearDown(self):
        """Clean up test environment after each test."""
        # Remove dummy files and directories (optional, good practice)
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)
        if os.path.exists(self.test_config['google_credentials_path']):
            os.remove(self.test_config['google_credentials_path'])
        if os.path.exists(self.dummy_image_path):
             os.remove(self.dummy_image_path)
        # Use shutil.rmtree for directories if needed, be careful!
        # For simplicity, we might leave the directories if they are empty.
        # import shutil
        # if os.path.exists('art_agent_team/tests/test_data'):
        #     shutil.rmtree('art_agent_team/tests/test_data')


    # --- Test Cases Will Go Here ---

    @patch('docent_agent.requests.post') # Mock the LLM API call
    @patch('docent_agent.VisionAgent')   # Mock the entire VisionAgent class
    def test_docent_calls_vision_agent_analyze_and_crop(self, MockVisionAgent, mock_requests_post):
        """Test that DocentAgent correctly calls VisionAgent's analyze and crop methods."""

        # --- Mock Setup ---
        # Mock LLM response to request analyze and crop
        mock_llm_response = MagicMock()
        mock_llm_response.status_code = 200
        mock_llm_response.json.return_value = {
            "choices": [{"message": {"content": '{"actions": {"analyze": true, "crop": true, "upscale": false, "research": false, "placard": false}, "image_path": null, "input_folder": null}'}}]
        }
        mock_requests_post.return_value = mock_llm_response

        # Mock VisionAgent instance and its methods
        mock_vision_instance = MockVisionAgent.return_value
        mock_analysis_results = {"objects": [{"type": "person", "bbox": [0.1, 0.1, 0.5, 0.5], "confidence": 0.9}]}
        mock_vision_instance.analyze_image.return_value = mock_analysis_results
        mock_cropped_path = os.path.join(self.test_config['workspace_folder'], f"cropped_{self.dummy_image_name}")
        mock_vision_instance.copy_and_crop_image.return_value = mock_cropped_path

        # --- Test Execution ---
        # Initialize DocentAgent (it will load the test config)
        docent = DocentAgent(config_path=self.test_config_path)
        # Process a request for the dummy image
        user_prompt = "Analyze and crop the image."
        docent.process_request(user_prompt, image_path=self.dummy_image_path)

        # --- Assertions ---
        # 1. Assert VisionAgent was initialized with the config
        MockVisionAgent.assert_called_once_with(docent.config) # Check if called with the loaded config

        # 2. Assert analyze_image was called correctly
        mock_vision_instance.analyze_image.assert_called_once_with(self.dummy_image_path)

        # 3. Assert copy_and_crop_image was called correctly
        expected_output_path = os.path.join(docent.workspace_folder, f"cropped_{self.dummy_image_name}")
        mock_vision_instance.copy_and_crop_image.assert_called_once_with(
            self.dummy_image_path,
            expected_output_path,
            mock_analysis_results
        )

        # 4. Assert LLM parsing was called (optional, but good for completeness)
        mock_requests_post.assert_called_once()


    @patch('docent_agent.requests.post') # Mock the LLM API call
    @patch('docent_agent.VisionAgent')   # Mock the entire VisionAgent class
    def test_docent_calls_vision_agent_crop_only(self, MockVisionAgent, mock_requests_post):
        """Test that DocentAgent correctly calls VisionAgent's crop method when only cropping is requested."""

        # --- Mock Setup ---
        # Mock LLM response to request crop only
        mock_llm_response = MagicMock()
        mock_llm_response.status_code = 200
        mock_llm_response.json.return_value = {
            "choices": [{"message": {"content": '{"actions": {"analyze": false, "crop": true, "upscale": false, "research": false, "placard": false}, "image_path": null, "input_folder": null}'}}]
        }
        mock_requests_post.return_value = mock_llm_response

        # Mock VisionAgent instance and its methods
        mock_vision_instance = MockVisionAgent.return_value
        # Ensure analyze_image is NOT called
        mock_vision_instance.analyze_image = MagicMock()
        mock_cropped_path = os.path.join(self.test_config['workspace_folder'], f"cropped_{self.dummy_image_name}")
        mock_vision_instance.copy_and_crop_image.return_value = mock_cropped_path


        # --- Test Execution ---
        # Initialize DocentAgent (it will load the test config)
        docent = DocentAgent(config_path=self.test_config_path)
        # Process a request for the dummy image
        user_prompt = "Crop the image."
        docent.process_request(user_prompt, image_path=self.dummy_image_path)

        # --- Assertions ---
        # 1. Assert VisionAgent was initialized with the config
        MockVisionAgent.assert_called_once_with(docent.config) # Check if called with the loaded config

        # 2. Assert analyze_image was NOT called
        mock_vision_instance.analyze_image.assert_not_called()

        # 3. Assert copy_and_crop_image was called correctly
        expected_output_path = os.path.join(docent.workspace_folder, f"cropped_{self.dummy_image_name}")
        # Note: When only crop is requested, analysis_results will be None
        mock_vision_instance.copy_and_crop_image.assert_called_once_with(
            self.dummy_image_path,
            expected_output_path,
            None # analysis_results should be None
        )

        # 4. Assert LLM parsing was called
        mock_requests_post.assert_called_once()

    @patch('docent_agent.requests.post') # Mock the LLM API call
    @patch('docent_agent.VisionAgent')   # Mock the entire VisionAgent class
    def test_docent_calls_vision_agent_analyze_only(self, MockVisionAgent, mock_requests_post):
        """Test that DocentAgent correctly calls VisionAgent's analyze method when only analysis is requested."""

        # --- Mock Setup ---
        # Mock LLM response to request analyze only
        mock_llm_response = MagicMock()
        mock_llm_response.status_code = 200
        mock_llm_response.json.return_value = {
            "choices": [{"message": {"content": '{"actions": {"analyze": true, "crop": false, "upscale": false, "research": false, "placard": false}, "image_path": null, "input_folder": null}'}}]
        }
        mock_requests_post.return_value = mock_llm_response

        # Mock VisionAgent instance and its methods
        mock_vision_instance = MockVisionAgent.return_value
        mock_analysis_results = {"objects": [{"type": "person", "bbox": [0.1, 0.1, 0.5, 0.5], "confidence": 0.9}]}
        mock_vision_instance.analyze_image.return_value = mock_analysis_results
        # Ensure copy_and_crop_image is NOT called
        mock_vision_instance.copy_and_crop_image = MagicMock()


        # --- Test Execution ---
        # Initialize DocentAgent (it will load the test config)
        docent = DocentAgent(config_path=self.test_config_path)
        # Process a request for the dummy image
        user_prompt = "Analyze the image."
        docent.process_request(user_prompt, image_path=self.dummy_image_path)

        # --- Assertions ---
        # 1. Assert VisionAgent was initialized with the config
        MockVisionAgent.assert_called_once_with(docent.config) # Check if called with the loaded config

        # 2. Assert analyze_image was called correctly
        mock_vision_instance.analyze_image.assert_called_once_with(self.dummy_image_path)

        # 3. Assert copy_and_crop_image was NOT called
        mock_vision_instance.copy_and_crop_image.assert_not_called()

        # 4. Assert LLM parsing was called
        mock_requests_post.assert_called_once()


    @patch('docent_agent.requests.post') # Mock the LLM API call
    @patch('docent_agent.VisionAgent')   # Mock the entire VisionAgent class
    @patch('docent_agent.os.listdir')    # Mock os.listdir to control folder content
    @patch('docent_agent.os.path.isfile') # Mock os.path.isfile
    @patch('docent_agent.os.path.isdir')  # Mock os.path.isdir
    def test_docent_processes_folder_of_images(self, mock_isdir, mock_isfile, mock_listdir, MockVisionAgent, mock_requests_post):
        """Test that DocentAgent processes all supported images in a specified folder."""

        # --- Mock Setup ---
        # Mock LLM response to request analyze and crop for a folder
        mock_llm_response = MagicMock()
        mock_llm_response.status_code = 200
        mock_llm_response.json.return_value = {
            "choices": [{"message": {"content": '{"actions": {"analyze": true, "crop": true, "upscale": false, "research": false, "placard": false}, "image_path": null, "input_folder": "art_agent_team/tests/test_data/input"}'}}]
        }
        mock_requests_post.return_value = mock_llm_response

        # Mock os.listdir to return a list of dummy files
        dummy_files = ['image1.jpg', 'image2.png', 'not_an_image.txt', '.DS_Store']
        mock_listdir.return_value = dummy_files

        # Mock os.path.isfile and os.path.isdir
        mock_isdir.return_value = True # Assume the input folder is a directory
        def isfile_side_effect(path):
            # Simulate which files are actual files
            return os.path.basename(path) in ['image1.jpg', 'image2.png', 'not_an_image.txt']
        mock_isfile.side_effect = isfile_side_effect

        # Mock VisionAgent instance and its methods
        mock_vision_instance = MockVisionAgent.return_value
        mock_analysis_results = {"objects": []} # Simplified analysis result for this test
        mock_vision_instance.analyze_image.return_value = mock_analysis_results
        mock_vision_instance.copy_and_crop_image.return_value = "mock_cropped_path" # Simplified return


        # --- Test Execution ---
        # Initialize DocentAgent (it will load the test config)
        docent = DocentAgent(config_path=self.test_config_path)
        # Process a request for the input folder
        user_prompt = "Analyze and crop images in the input folder."
        docent.process_request(user_prompt) # Let LLM parse the folder path

        # --- Assertions ---
        # 1. Assert VisionAgent was initialized with the config
        MockVisionAgent.assert_called_once_with(docent.config)

        # 2. Assert os.listdir and os.path.isdir/isfile were called correctly
        mock_listdir.assert_called_once_with(self.test_config['input_folder'])
        # Assert that isdir was called with the input folder at least once
        mock_isdir.assert_any_call(self.test_config['input_folder'])
        # isfile will be called for each item in listdir result

        # 3. Assert analyze_image and copy_and_crop_image were called for supported image files
        expected_analyze_calls = [
            call(os.path.join(self.test_config['input_folder'], 'image1.jpg')),
            call(os.path.join(self.test_config['input_folder'], 'image2.png'))
        ]
        mock_vision_instance.analyze_image.assert_has_calls(expected_analyze_calls, any_order=True)
        self.assertEqual(mock_vision_instance.analyze_image.call_count, 2) # Ensure only called for images

        expected_crop_calls = [
            call(os.path.join(self.test_config['input_folder'], 'image1.jpg'), os.path.join(self.test_config['workspace_folder'], 'cropped_image1.jpg'), mock_analysis_results),
            call(os.path.join(self.test_config['input_folder'], 'image2.png'), os.path.join(self.test_config['workspace_folder'], 'cropped_image2.png'), mock_analysis_results)
        ]
        mock_vision_instance.copy_and_crop_image.assert_has_calls(expected_crop_calls, any_order=True)
        self.assertEqual(mock_vision_instance.copy_and_crop_image.call_count, 2) # Ensure only called for images

        # 4. Assert LLM parsing was called
        mock_requests_post.assert_called_once()


    @patch('docent_agent.requests.post') # Mock the LLM API call
    @patch('docent_agent.VisionAgent')   # Mock the entire VisionAgent class
    def test_docent_handles_analyze_failure(self, MockVisionAgent, mock_requests_post):
        """Test that DocentAgent handles VisionAgent's analyze_image returning None."""

        # --- Mock Setup ---
        # Mock LLM response to request analyze and crop
        mock_llm_response = MagicMock()
        mock_llm_response.status_code = 200
        mock_llm_response.json.return_value = {
            "choices": [{"message": {"content": '{"actions": {"analyze": true, "crop": true, "upscale": false, "research": false, "placard": false}, "image_path": null, "input_folder": null}'}}]
        }
        mock_requests_post.return_value = mock_llm_response

        # Mock VisionAgent instance and its methods
        mock_vision_instance = MockVisionAgent.return_value
        # Simulate analyze_image failing
        mock_vision_instance.analyze_image.return_value = None
        # Mock copy_and_crop_image to check if it's still called (it should be, with None analysis_results)
        mock_vision_instance.copy_and_crop_image = MagicMock()


        # --- Test Execution ---
        docent = DocentAgent(config_path=self.test_config_path)
        user_prompt = "Analyze and crop the image."
        # We expect the workflow to continue even if analysis fails
        docent.process_request(user_prompt, image_path=self.dummy_image_path)

        # --- Assertions ---
        # 1. Assert analyze_image was called
        mock_vision_instance.analyze_image.assert_called_once_with(self.dummy_image_path)

        # 2. Assert copy_and_crop_image was still called, with None analysis_results
        expected_output_path = os.path.join(docent.workspace_folder, f"cropped_{self.dummy_image_name}")
        mock_vision_instance.copy_and_crop_image.assert_called_once_with(
            self.dummy_image_path,
            expected_output_path,
            None # analysis_results should be None
        )

        # 3. Assert LLM parsing was called
        mock_requests_post.assert_called_once()


    @patch('docent_agent.requests.post') # Mock the LLM API call
    @patch('docent_agent.VisionAgent')   # Mock the entire VisionAgent class
    def test_docent_handles_crop_failure(self, MockVisionAgent, mock_requests_post):
        """Test that DocentAgent handles VisionAgent's copy_and_crop_image returning None."""

        # --- Mock Setup ---
        # Mock LLM response to request analyze and crop
        mock_llm_response = MagicMock()
        mock_llm_response.status_code = 200
        mock_llm_response.json.return_value = {
            "choices": [{"message": {"content": '{"actions": {"analyze": true, "crop": true, "upscale": false, "research": false, "placard": false}, "image_path": null, "input_folder": null}'}}]
        }
        mock_requests_post.return_value = mock_llm_response

        # Mock VisionAgent instance and its methods
        mock_vision_instance = MockVisionAgent.return_value
        mock_analysis_results = {"objects": [{"type": "person", "bbox": [0.1, 0.1, 0.5, 0.5], "confidence": 0.9}]}
        mock_vision_instance.analyze_image.return_value = mock_analysis_results
        # Simulate copy_and_crop_image failing
        mock_vision_instance.copy_and_crop_image.return_value = None

        # Mock subsequent agent methods to ensure they are NOT called
        mock_vision_instance.upscale_image = MagicMock()
        mock_vision_instance.research_artwork = MagicMock()
        mock_vision_instance.add_placard = MagicMock()


        # --- Test Execution ---
        docent = DocentAgent(config_path=self.test_config_path)
        user_prompt = "Analyze and crop the image."
        # --- Test Execution ---
        docent = DocentAgent(config_path=self.test_config_path)
        user_prompt = "Analyze and crop the image."
        # We expect the workflow to stop for this image if cropping fails
        docent.process_request(user_prompt, image_path=self.dummy_image_path)

        # --- Assertions ---
        # 1. Assert analyze_image was called
        mock_vision_instance.analyze_image.assert_called_once_with(self.dummy_image_path)

        # 2. Assert copy_and_crop_image was called
        expected_output_path = os.path.join(docent.workspace_folder, f"cropped_{self.dummy_image_name}")
        mock_vision_instance.copy_and_crop_image.assert_called_once_with(
            self.dummy_image_path,
            expected_output_path,
            mock_analysis_results
        )

        # 3. Assert LLM parsing was called
        mock_requests_post.assert_called_once()

        # 4. Assert subsequent agent methods were NOT called
        mock_vision_instance.upscale_image.assert_not_called()
        mock_vision_instance.research_artwork.assert_not_called()
        mock_vision_instance.add_placard.assert_not_called()


if __name__ == '__main__':
    unittest.main()