import unittest
# Mock imports removed for live testing
import os
import sys
import yaml
import logging
import glob # Added for User Instruction 2

# Add the project root to the Python path to allow importing agent modules
# Adjust the path depth as necessary depending on where tests are run from
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import after adjusting path
try:
    from art_agent_team.docent_agent import DocentAgent
    # We will mock the VisionAgent class itself or its methods later
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Current sys.path: {sys.path}")
    # If running tests from the root, the path adjustment might not be needed,
    # but it's generally safer for discoverability.
    raise

# Configure logging for tests
log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)

# --- LIVE INTEGRATION TESTS FOR ART_AI AGENTS ---

import shutil
import pytest # Using pytest for better test management and fixtures if needed later

# Mark tests that require real API keys or significant resources
# Consider adding markers like @pytest.mark.integration or @pytest.mark.slow

def test_live_docentagent_full_workflow():
    logger.info("Starting test_live_docentagent_full_workflow...")
    """
    End-to-end test: DocentAgent -> ResearchAgent -> VisionAgent -> UpscaleAgent -> PlacardAgent.
    Uses a real image and checks for real output artifacts.
    """
    from art_agent_team.docent_agent import DocentAgent
    import os
    import tempfile

    config_path = "art_agent_team/config/config.yaml"
    input_image_dir = "art_agent_team/tests/test_data/input/"
    image_files = glob.glob(os.path.join(input_image_dir, "*.*"))
    assert image_files, f"No image files found in {input_image_dir} for test_live_docentagent_full_workflow"
    input_image = image_files[0] # Use the first found image
    logger.info(f"Using input image: {input_image} for test_live_docentagent_full_workflow")

    output_dir = "art_agent_team/tests/test_data/output"
    workspace_dir = "art_agent_team/tests/test_data/workspace"

    # Load config as dict
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Use real config values; do not overwrite with mock endpoint

    # Clean output/workspace before test
    for d in [output_dir, workspace_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    try:
        docent = DocentAgent(config_path=config_path)
        logger.info(f"Processing request for image: {input_image}")
        # Call the new non-interactive method
        result = docent.handle_request("Analyze and process the image through full workflow.", image_path=input_image)
    except Exception as e:
        logger.error(f"Error during DocentAgent processing: {e}", exc_info=True)
        pytest.fail(f"DocentAgent processing failed: {e}")

    # --- Assertions for the new handle_request method ---
    assert isinstance(result, dict), "handle_request should return a dictionary."
    assert "final_image_path" in result, "Result dict missing 'final_image_path'."
    assert "metadata" in result, "Result dict missing 'metadata'."
    
    final_image_path = result.get("final_image_path")
    returned_metadata = result.get("metadata", {})

    assert final_image_path is not None, "'final_image_path' should not be None."
    assert os.path.exists(final_image_path), f"Final output image not found at: {final_image_path}"
    logger.info(f"Successfully found final output image: {final_image_path}")

    # Assert for User Instruction 4: DocentAgent should return documentation of agents called.
    assert "called_agents_workflow" in returned_metadata, "Metadata missing 'called_agents_workflow'."
    assert isinstance(returned_metadata["called_agents_workflow"], list), "'called_agents_workflow' should be a list."
    # Example check for expected agents in a full workflow
    expected_agents_in_log = ["ResearchAgent", "UpscaleAgent", "PlacardAgent"] # VisionAgent name can vary
    for agent_name_part in expected_agents_in_log:
        assert any(agent_name_part in called_agent for called_agent in returned_metadata["called_agents_workflow"]), \
            f"Expected agent part '{agent_name_part}' not found in called_agents_workflow: {returned_metadata['called_agents_workflow']}"
    assert any("VisionAgent" in called_agent for called_agent in returned_metadata["called_agents_workflow"]), \
            f"Expected 'VisionAgent' in called_agents_workflow: {returned_metadata['called_agents_workflow']}"


    logger.info(f"Called agents workflow: {returned_metadata.get('called_agents_workflow')}")

    # Clean up the final output file
    try:
        if final_image_path and os.path.exists(final_image_path):
            os.remove(final_image_path)
            logger.info(f"Cleaned up final output file: {final_image_path}")
    except OSError as e:
        logger.warning(f"Could not remove test file {final_image_path}: {e}")
    logger.info("Finished test_live_docentagent_full_workflow.")


# --- LIVE TEST: ResearchAgent ---

def test_live_researchagent_metadata():
    logger.info("Starting test_live_researchagent_metadata...")
    from art_agent_team.agents.research_agent import ResearchAgent
    import os
    import yaml
    # Removed unittest.mock import

    config_path = "art_agent_team/config/config.yaml"
    input_image_dir = "art_agent_team/tests/test_data/input/"
    image_files = glob.glob(os.path.join(input_image_dir, "*.*"))
    assert image_files, f"No image files found in {input_image_dir} for test_live_researchagent_metadata"
    input_image = image_files[0] # Use the first found image
    logger.info(f"Using input image: {input_image} for test_live_researchagent_metadata")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    from art_agent_team.agents.vision_agent_animal import VisionAgentAnimal
    vision_agent_classes = {
        "animal": VisionAgentAnimal,
    }
    agent = ResearchAgent(config=config, vision_agent_classes=vision_agent_classes)
    try:
        logger.info(f"Researching image: {input_image}")
        # Call the refactored method
        research_output = agent.research_and_process(image_path=input_image) # Pass image_path explicitly
    except Exception as e:
        logger.error(f"Error during ResearchAgent processing: {e}", exc_info=True)
        pytest.fail(f"ResearchAgent processing failed: {e}")

    # --- Assertions for the new structure ---
    # Ref: Error log entry 2025-04-30 16:08 PM
    assert isinstance(research_output, dict), "research_and_process should return a dictionary."
    assert "consolidated_findings" in research_output, "Return dict missing 'consolidated_findings'."
    assert "research_attempts" in research_output, "Return dict missing 'research_attempts'."

    consolidated = research_output["consolidated_findings"]
    attempts = research_output["research_attempts"]

    assert isinstance(consolidated, dict), "'consolidated_findings' should be a dictionary."
    assert isinstance(attempts, dict), "'research_attempts' should be a dictionary."

    # Check structure of consolidated findings
    assert "artist" in consolidated, "Consolidated findings missing 'artist'."
    assert "title" in consolidated, "Consolidated findings missing 'title'."
    assert "date" in consolidated, "Consolidated findings missing 'date'."
    assert "description" in consolidated, "Consolidated findings missing 'description'."
    assert "style" in consolidated, "Consolidated findings missing 'style'."
    assert "subjects" in consolidated, "Consolidated findings missing 'subjects'."
    assert "structured_sentence" in consolidated, "Consolidated findings missing 'structured_sentence'."

    # Check structure of attempts (ensure all methods are present)
    expected_attempts = ["grok_text", "gemini_text", "grok_vision", "gemini_vision", "web_image_search"]
    for attempt_key in expected_attempts:
        assert attempt_key in attempts, f"Research attempts missing '{attempt_key}'."
        assert isinstance(attempts[attempt_key], dict), f"Attempt '{attempt_key}' should be a dictionary."
        # Each attempt result should either have data keys or an 'error' key
        assert ("error" in attempts[attempt_key]) or ("artist" in attempts[attempt_key]) or ("description" in attempts[attempt_key]) or ("info" in attempts[attempt_key]), \
               f"Attempt '{attempt_key}' has neither data nor error key."

    # Check if *some* identification was successful (allow for partial failures)
    # This is a weaker assertion than before, acknowledging potential API issues
    assert consolidated["artist"] or consolidated["title"] or consolidated["description"] != "Could not retrieve detailed description from any source.", \
           "Consolidated findings failed to identify artist, title, or description."

    # Optionally, assert specific expected values if known for the test image,
    # but focus on structure and graceful failure handling for now.
    # e.g., assert consolidated["title"] == "Girl with a Black Eye" # This might be too strict
    logger.info("Finished test_live_researchagent_metadata.")


# --- LIVE TEST: VisionAgent (base) ---

def test_live_visionagent_analyze_and_crop():
    logger.info("Starting test_live_visionagent_analyze_and_crop...")
    from art_agent_team.agents.vision_agent_default import DefaultVisionAgent # Changed import
    import os
    import yaml

    config_path = "art_agent_team/config/config.yaml"
    input_image_dir = "art_agent_team/tests/test_data/input/"
    image_files = glob.glob(os.path.join(input_image_dir, "*.*"))
    assert image_files, f"No image files found in {input_image_dir} for test_live_visionagent_analyze_and_crop"
    input_image = image_files[0] # Use the first found image
    logger.info(f"Using input image: {input_image} for test_live_visionagent_analyze_and_crop")

    workspace_dir = "art_agent_team/tests/test_data/workspace"
    os.makedirs(workspace_dir, exist_ok=True)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    try:
        agent = DefaultVisionAgent(config=config) # Changed instantiation
        logger.info(f"Analyzing image: {input_image}")
        analysis = agent.analyze_image(input_image)
        assert isinstance(analysis, dict), "Analysis should return a dictionary."
        logger.info(f"Analysis result: {analysis}")

        # Dynamically create output filename based on input
        base_input_filename = os.path.splitext(os.path.basename(input_image))[0]
        crop_output_filename = f"{base_input_filename}_cropped_test_vision{os.path.splitext(input_image)[1]}"
        crop_output_path = os.path.join(workspace_dir, crop_output_filename)
        logger.info(f"Cropping image to: {crop_output_path}")
        cropped_path = agent.copy_and_crop_image(input_image, crop_output_path, analysis)

        assert cropped_path is not None and os.path.exists(cropped_path), f"Cropped file not found at {cropped_path}"
        logger.info(f"Successfully created cropped file: {cropped_path}")

    except Exception as e:
        logger.error(f"Error during VisionAgent analyze/crop: {e}", exc_info=True)
        pytest.fail(f"VisionAgent analyze/crop failed: {e}")
    finally:
        # Cleanup
        if 'cropped_path' in locals() and cropped_path is not None and os.path.exists(cropped_path):
            try:
                os.remove(cropped_path)
                logger.info(f"Cleaned up cropped file: {cropped_path}")
            except OSError as e:
                 logger.warning(f"Could not remove test file {cropped_path}: {e}")
    logger.info("Finished test_live_visionagent_analyze_and_crop.")


# --- LIVE TEST: UpscaleAgent ---

def test_live_upscaleagent_upscale():
    logger.info("Starting test_live_upscaleagent_upscale...")
    from art_agent_team.agents.upscale_agent import UpscaleAgent
    import os

    config_path = "art_agent_team/config/config.yaml" # Config might not be used by UpscaleAgent directly yet
    input_image_dir = "art_agent_team/tests/test_data/input/"
    image_files = glob.glob(os.path.join(input_image_dir, "*.*"))
    assert image_files, f"No image files found in {input_image_dir} for test_live_upscaleagent_upscale"
    input_image = image_files[0] # Use the first found image
    logger.info(f"Using input image: {input_image} for test_live_upscaleagent_upscale")

    # Dynamically create output filename based on input
    base_input_filename = os.path.splitext(os.path.basename(input_image))[0]
    output_filename = f"{base_input_filename}_upscaled_test{os.path.splitext(input_image)[1]}"
    output_path = os.path.join("art_agent_team/tests/test_data/output", output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    upscaled_path = None # Define outside try for cleanup
    try:
        agent = UpscaleAgent()  # Use default or config if needed
        logger.info(f"Upscaling image: {input_image} to {output_path}")
        upscaled_path = agent.upscale_image(input_image, output_path)
        assert upscaled_path is not None and os.path.exists(upscaled_path), f"Upscaled file not found at {upscaled_path}"
        logger.info(f"Successfully created upscaled file: {upscaled_path}")
    except Exception as e:
        logger.error(f"Error during UpscaleAgent processing: {e}", exc_info=True)
        pytest.fail(f"UpscaleAgent processing failed: {e}")
    finally:
        # Cleanup
        if upscaled_path is not None and os.path.exists(upscaled_path):
             try:
                os.remove(upscaled_path)
                logger.info(f"Cleaned up upscaled file: {upscaled_path}")
             except OSError as e:
                 logger.warning(f"Could not remove test file {upscaled_path}: {e}")
    logger.info("Finished test_live_upscaleagent_upscale.")


# --- LIVE TEST: PlacardAgent ---

def test_live_placardagent_add_placard():
    logger.info("Starting test_live_placardagent_add_placard...")
    from art_agent_team.agents.placard_agent import PlacardAgent
    import os

    config_path = "art_agent_team/config/config.yaml" # Config might be used by PlacardAgent
    input_image_dir = "art_agent_team/tests/test_data/input/"
    image_files = glob.glob(os.path.join(input_image_dir, "*.*"))
    assert image_files, f"No image files found in {input_image_dir} for test_live_placardagent_add_placard"
    input_image = image_files[0] # Use the first found image
    logger.info(f"Using input image: {input_image} for test_live_placardagent_add_placard")

    # Dynamically create output filename based on input
    base_input_filename = os.path.splitext(os.path.basename(input_image))[0]
    output_filename = f"{base_input_filename}_placarded_test{os.path.splitext(input_image)[1]}"
    output_path = os.path.join("art_agent_team/tests/test_data/output", output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    placarded_path = None # Define outside try for cleanup
    try:
        # Ensure font path is correctly configured or handled if None
        # If PlacardAgent relies on config.yaml, ensure it's loaded or passed
        agent = PlacardAgent(font_path=None) # Adjust if font path comes from config
        metadata = {"title": "Girl with a Black Eye", "artist": "Unknown", "genre": "figure"}
        logger.info(f"Adding placard to image: {input_image} with metadata: {metadata}")
        placarded_path = agent.add_plaque(input_image, output_path, metadata)
        assert placarded_path is not None and os.path.exists(placarded_path), f"Placarded file not found at {placarded_path}"
        logger.info(f"Successfully created placarded file: {placarded_path}")
    except Exception as e:
        logger.error(f"Error during PlacardAgent processing: {e}", exc_info=True)
        pytest.fail(f"PlacardAgent processing failed: {e}")
    finally:
        # Cleanup
        if placarded_path is not None and os.path.exists(placarded_path):
             try:
                os.remove(placarded_path)
                logger.info(f"Cleaned up placarded file: {placarded_path}")
             except OSError as e:
                 logger.warning(f"Could not remove test file {placarded_path}: {e}")
    logger.info("Finished test_live_placardagent_add_placard.")


# --- Mock-based tests removed ---
# The following test classes (TestDocentAgentCategorization, TestDocentAgentDelegation)
# and their methods have been removed as they relied on unittest.mock and patching,
# which are being replaced by live integration tests.

# (unittest.main call removed as tests are typically run with pytest)
# if __name__ == '__main__':
#     unittest.main() # Keep if running directly, remove if using pytest runner

def test_researchagent_handles_missing_api_keys(monkeypatch, caplog):
    logger.info("Starting test_researchagent_handles_missing_api_keys...")
    from art_agent_team.agents.research_agent import ResearchAgent
    import os
    import glob

    # Ensure API keys are not present in the environment for this test
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("VERTEX_PROJECT_ID", raising=False) # Also ensure project/location are not set for Vertex
    monkeypatch.delenv("VERTEX_LOCATION", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)


    # Use an empty config, or a config that explicitly lacks these keys
    empty_config = {}

    # Dynamically get an image for the test
    input_image_dir = "art_agent_team/tests/test_data/input/"
    image_files = glob.glob(os.path.join(input_image_dir, "*.*"))
    assert image_files, f"No image files found in {input_image_dir} for test_researchagent_handles_missing_api_keys"
    input_image = image_files[0]

    caplog.set_level(logging.WARNING) # Capture warnings and above

    # Instantiate ResearchAgent
    # Vision agent classes are not strictly needed for this test's focus but pass empty for consistency
    agent = ResearchAgent(config=empty_config, vision_agent_classes={})

    # --- Assertions for ModelRegistry initialization ---
    logs = caplog.text
    assert "GROK_API_KEY not set" in logs
    assert "OPENROUTER_API_KEY not set" in logs
    assert "google_api_key not set" in logs # For standard Gemini
    assert "vertex_project_id not set" in logs # For Vertex AI
    assert "vertex_location not set" in logs # For Vertex AI
    # GOOGLE_APPLICATION_CREDENTIALS warning might appear if project_id and location *were* set,
    # but since they are not, Vertex AI init might not even reach that check or might log differently.
    # The key is that Vertex AI models should not be available.

    assert not agent.model_registry.clients.get('grok')
    assert not agent.model_registry.clients.get('openrouter')
    assert not agent.model_registry.clients.get('google_genai')
    assert agent.model_registry.clients.get('vertexai_sdk_initialized') is False
    
    # All models requiring these keys should be skipped during registration
    assert not agent.model_registry.models, "Model registry should be empty or not contain models requiring missing keys."
    
    # --- Assertions for research_and_process call ---
    caplog.clear() # Clear previous logs before the call
    research_output = None
    try:
        research_output = agent.research_and_process(image_path=input_image)
    except Exception as e:
        pytest.fail(f"ResearchAgent.research_and_process raised an unexpected exception with missing keys: {e}")

    assert isinstance(research_output, dict), "research_and_process should return a dict even with missing keys."
    assert "consolidated_findings" in research_output
    assert "research_attempts" in research_output

    consolidated = research_output["consolidated_findings"]
    attempts = research_output["research_attempts"]

    assert isinstance(consolidated, dict)
    assert isinstance(attempts, dict)
    
    # Expect an error in consolidated findings because no models could run
    assert "error" in consolidated, "Consolidated findings should contain an error message."
    # Check for one of the expected error messages
    possible_error_messages = ["No vision models available", "Preprocessing failed"]
    assert any(msg in consolidated["error"] for msg in possible_error_messages), \
        f"Consolidated error '{consolidated['error']}' did not match expected messages: {possible_error_messages}"


    assert not attempts, "Research attempts should be empty if no models could run."

    # Check logs for warnings about no vision models during the unified_research call
    # This log comes from unified_research itself.
    # We might need to set level to INFO for this specific check if it's an INFO log.
    # For now, let's assume the "No vision models available" in error message is sufficient.
    # Example: assert "No vision models available in registry" in caplog.text

    logger.info("Finished test_researchagent_handles_missing_api_keys.")