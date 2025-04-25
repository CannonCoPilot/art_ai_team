import unittest
import os
import logging
import json
import numpy as np
from PIL import Image
# Import specialized VisionAgent classes
from art_agent_team.agents.vision_agent_abstract import VisionAgentAbstract
from art_agent_team.agents.vision_agent_animal import VisionAgentAnimal
from art_agent_team.agents.vision_agent_figurative import VisionAgentFigurative
from art_agent_team.agents.vision_agent_genre import VisionAgentGenre
from art_agent_team.agents.vision_agent_landscape import VisionAgentLandscape
from art_agent_team.agents.vision_agent_portrait import VisionAgentPortrait
from art_agent_team.agents.vision_agent_religious_historical import VisionAgentReligiousHistorical
from art_agent_team.agents.vision_agent_still_life import VisionAgentStillLife


class TestVisionWorkflowIntegration(unittest.TestCase):
    """Integration tests for the complete vision workflow."""

    def setUp(self):
        """Set up test environment with real paths and credentials."""
        # Project base directory
        self.base_dir = '/Users/nathaniel.cannon/Documents/VScodeWork/Art_AI'

        # Use actual production paths
        self.input_folder = os.path.join(self.base_dir, 'art_agent_team/tests/test_data/input')
        self.output_folder = os.path.join(self.base_dir, 'art_agent_team/tests/test_data/output')
        # Use the specified test input image
        self.test_image_path = os.path.join(self.input_folder, 'Frank Brangwyn, Swans, c.1921.jpg')

        # API credentials
        self.google_credentials_path = os.path.join(self.base_dir, 'API_keys/geministudioapi-b5da91c0cd01.json')
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.google_credentials_path

        # Get API key from environment
        google_api_key = os.environ.get('GOOGLE_API_KEY')
        if not google_api_key:
            # Fallback to reading from the credentials file if not in environment
            try:
                with open(self.google_credentials_path, 'r') as f:
                    creds = json.load(f)
                    google_api_key = creds.get('api_key')
            except Exception as e:
                logging.warning(f"Could not read API key from credentials file: {e}")

        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable or 'api_key' in credentials file not set")

        # Load Grok API key from API_keys/keys file
        grok_api_key = None
        api_keys_path = os.path.join(self.base_dir, 'API_keys', 'keys')
        try:
            with open(api_keys_path, 'r') as f:
                for line in f:
                    if line.startswith('GrokAPI:'):
                        grok_api_key = line.split(':', 1)[1].strip()
                        break
        except Exception as e:
            logging.warning(f"Could not read Grok API key from {api_keys_path}: {e}")
            # Fallback to environment variable
            grok_api_key = os.environ.get('GROK_API_KEY')
            if not grok_api_key:
                logging.warning("Grok API key not found in file or environment variables")

        # Test configuration with both API keys
        self.test_config = {
            'input_folder': self.input_folder,
            'output_folder': self.output_folder,
            'google_credentials_path': self.google_credentials_path,
            'google_api_key': google_api_key,
            'grok_api_key': grok_api_key # Include Grok API key in config
        }

        # Create directories if needed
        os.makedirs(self.output_folder, exist_ok=True)

    def test_api_key_initialization(self):
        """Test that API keys are properly passed to and initialized in the agent."""
        vision_agent = VisionAgentAnimal(self.test_config)
        
        # Verify Gemini client initialization
        self.assertIsNotNone(vision_agent.gemini_pro, "Gemini Pro model not initialized")
        
        # Verify Grok client initialization
        self.assertIsNotNone(vision_agent.grok_client, "Grok Vision client not initialized")
        self.assertEqual(vision_agent.grok_vision_model, "grok-2-vision-1212", "Incorrect Grok model specified")

    def test_feature_handling(self):
        """Test handling of mixed feature types (strings and dictionaries)."""
        vision_agent = VisionAgentAnimal(self.test_config)
        
        # Create test data with mixed feature types
        test_objects = [{
            "label": "swan",
            "box_2d": [100, 100, 200, 200],
            "features": [
                "beak",  # String feature
                {"label": "eye", "box_2d": [120, 120, 130, 130], "confidence": 0.9},  # Dict feature
                {"label": "neck", "box_2d": [150, 150, 160, 180]},  # Dict without confidence
                "wing"  # Another string feature
            ]
        }]
        
        # Create minimal analysis results
        test_results = {
            "objects": test_objects,
            "image_size": (400, 300)
        }
        
        # Test save_labeled_version with mixed features
        test_img = Image.new('RGB', (400, 300))
        output_path = os.path.join(self.output_folder, "test_features.jpg")
        
        # This should not raise any exceptions
        vision_agent._save_labeled_version(test_img, test_results, output_path)
        self.assertTrue(os.path.exists(output_path), "Feature test output not created")

    def test_vision_workflow_animal(self):
        """Test the animal vision workflow with output verification."""
        # Initialize the specialized animal vision agent
        vision_agent = VisionAgentAnimal(self.test_config)

        # Define research data specific to the test image (Frank Brangwyn, Swans, c.1921.jpg)
        research_data = {
            "primary_subject": "swans",
            "secondary_subjects": ["water", "trees"],
            "paragraph_description": "A painting of swans on water near trees.",
            "structured_sentence": "The painting depicts swans on a body of water, with trees in the background."
        }

        # Analyze image using the specialized agent
        analysis_results = vision_agent.analyze_image(
            self.test_image_path,
            research_data
        )

        # Verify analysis results
        self.assertIsNotNone(analysis_results, "Analysis failed")
        self.assertIn("objects", analysis_results, "No objects detected")
        self.assertIn("segmentation_masks", analysis_results, "No segmentation masks created")

        # Verify objects have importance scores
        for obj in analysis_results["objects"]:
            self.assertIn("importance", obj, "Object missing importance score")
            self.assertGreaterEqual(obj["importance"], 0.0)
            self.assertLessEqual(obj["importance"], 1.0)

        # Verify that "animal" or "swan" objects are detected and have high importance
        animal_objects = [obj for obj in analysis_results["objects"]
                          if obj.get("type", "").lower() in ["animal", "group_of_animals"] or "swan" in obj.get("label", "").lower()]
        self.assertGreater(len(animal_objects), 0, "No animal objects detected")

        # Check if at least one animal object has a high importance score (e.g., > 0.7)
        self.assertTrue(any(obj.get("importance", 0) > 0.7 for obj in animal_objects), "No important animal objects detected")


        # Save all versions
        basename = os.path.splitext(os.path.basename(self.test_image_path))[0]
        labeled_path = os.path.join(self.output_folder, f"{basename}_labeled.jpg")
        masked_path = os.path.join(self.output_folder, f"{basename}_masked.jpg")
        cropped_path = os.path.join(self.output_folder, f"{basename}_cropped.jpg")

        vision_agent.save_analysis_outputs(self.test_image_path, analysis_results, self.output_folder)

        # Verify all output files exist
        self.assertTrue(os.path.exists(labeled_path), "Labeled image not created")
        self.assertTrue(os.path.exists(masked_path), "Masked image not created")
        self.assertTrue(os.path.exists(cropped_path), "Cropped image not created")

        # Verify cropped image aspect ratio (assuming 16:9 is the target)
        with Image.open(cropped_path) as img:
            width, height = img.size
            actual_ratio = width / height
            self.assertAlmostEqual(actual_ratio, 16/9, places=2, msg="Cropped image aspect ratio is not 16:9")

        # Verify important objects have masks (threshold based on percentile)
        important_objects_for_masking = [obj for obj in analysis_results["objects"]
                                         if obj.get("importance", 0) >= np.percentile([o.get("importance", 0)
                                         for o in analysis_results["objects"]], 80)]
        self.assertLessEqual(len(analysis_results["segmentation_masks"]), len(important_objects_for_masking),
                           "Too many segmentation masks created")

        # Verify object importance scaling (subjects should generally be more important than background elements)
        subjects = [obj for obj in analysis_results["objects"]
                   if obj.get("type", "").lower() in ["animal", "animal_face", "group_of_animals"]]
        background_elements = [obj for obj in analysis_results["objects"]
                              if obj.get("type", "").lower() in ["terrain", "water", "plant", "sky", "background"]]

        if subjects and background_elements:
            max_subject_importance = max(obj.get("importance", 0) for obj in subjects)
            max_background_importance = max(obj.get("importance", 0) for obj in background_elements)
            # This assertion might need adjustment based on specific image content and expected results
            # For a general test, we expect subjects to be more important than typical background.
            if max_subject_importance > 0 and max_background_importance > 0:
                self.assertGreater(max_subject_importance, max_background_importance,
                                 "Subject importance not properly scaled above background elements")

    def test_masked_version_output(self):
        """Test the masked version output path handling and functionality."""
        vision_agent = VisionAgentAnimal(self.test_config)
        
        # Create test data
        test_img = Image.new('RGB', (400, 300))
        test_objects = [{
            "label": "swan",
            "box_2d": [100, 100, 200, 200],
            "importance": 1.0,
            "type": "animal"
        }]
        test_results = {
            "objects": test_objects,
            "image_size": (400, 300)
        }
        
        # Test with different output paths
        test_paths = [
            os.path.join(self.output_folder, "test_masked.jpg"),
            os.path.join(self.output_folder, "subdir", "test_masked.jpg"),
            os.path.join(self.output_folder, "test_masked_with spaces.jpg")
        ]
        
        for path in test_paths:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Test _save_masked_version
            vision_agent._save_masked_version(test_img.copy(), test_results, path)
            
            # Verify output
            self.assertTrue(os.path.exists(path), f"Masked version not created at {path}")
            with Image.open(path) as img:
                self.assertEqual(img.mode, 'RGB', f"Image at {path} not in RGB mode")
                self.assertEqual(img.size, (400, 300), f"Image at {path} has incorrect size")


if __name__ == '__main__':
    unittest.main()
