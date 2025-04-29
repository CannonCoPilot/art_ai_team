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
        """Set up test environment with real paths and credentials, loading API keys only from config.yaml."""
        import yaml

        # Project base directory
        self.base_dir = '/Users/nathaniel.cannon/Documents/VScodeWork/Art_AI'

        # Use actual production paths
        self.input_folder = os.path.join(self.base_dir, 'art_agent_team/tests/test_data/input')
        self.output_folder = os.path.join(self.base_dir, 'art_agent_team/tests/test_data/output')
        # Use the specified test input image
        self.test_image_path = os.path.join(self.input_folder, 'Frank Brangwyn, Swans, c.1921.jpg')

        # Load API keys from config.yaml only
        config_path = os.path.join(self.base_dir, 'art_agent_team/config/config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        google_api_key = config.get('google_api_key')
        grok_api_key = config.get('grok_api_key')

        if not google_api_key:
            raise ValueError("google_api_key not set in config.yaml")
        if not grok_api_key:
            raise ValueError("grok_api_key not set in config.yaml")

        # Test configuration with both API keys
        self.test_config = {
            'input_folder': self.input_folder,
            'output_folder': self.output_folder,
            'google_api_key': google_api_key,
            'grok_api_key': grok_api_key
        }

        # Create directories if needed
        os.makedirs(self.output_folder, exist_ok=True)

    def test_api_key_initialization(self):
        """Test that API keys are properly passed to and initialized in the agent."""
        vision_agent = VisionAgentAnimal(self.test_config)
        
        # Verify Gemini client initialization
        self.assertIsNotNone(vision_agent.gemini_pro, "Gemini Pro model not initialized")
        
        # Verify Grok API key is present and model name is set
        self.assertIsNotNone(getattr(vision_agent, "grok_api_key", None), "Grok API key not set in VisionAgentAnimal")
        self.assertEqual(vision_agent.grok_vision_model, "grok-2-vision-latest", "Incorrect Grok model specified")

    # test_feature_handling removed as it used dummy images

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

    # test_masked_version_output removed as it used dummy images


if __name__ == '__main__':
    unittest.main()
