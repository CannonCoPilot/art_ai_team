import logging
import os
import yaml
import json
import time
import importlib
import requests  # Required for test mocking
from typing import Dict, Any, Optional, List, Tuple

from art_agent_team.agents.research_agent import ResearchAgent
# Removed OpenAI import as it's not directly used by Docent anymore

# Dynamically load VisionAgent subclasses
vision_agent_classes = {}
agent_folder = os.path.join(os.path.dirname(__file__), 'agents')
for filename in os.listdir(agent_folder):
    # Ensure we only process python files that seem like vision agents and aren't abstract/base
    if filename.startswith('vision_agent_') and filename.endswith('.py') and filename != 'vision_agent_abstract.py' and filename != 'vision_agent.py':
        module_name = f"art_agent_team.agents.{filename[:-3]}"
        class_name_parts = filename[:-3].split('_') # e.g., ['vision', 'agent', 'landscape']
        # Construct class name like VisionAgentLandscape
        class_name = "".join(part.capitalize() for part in class_name_parts)
        try:
            module = importlib.import_module(module_name)
            agent_class = getattr(module, class_name, None)
            if agent_class and isinstance(agent_class, type): # Check if it's a class
                 # Use a key derived from the class name, e.g., 'Landscape' from 'VisionAgentLandscape'
                genre_key = class_name.replace('VisionAgent', '')
                if genre_key: # Ensure we have a key
                    vision_agent_classes[genre_key] = agent_class
                    logging.debug(f"Dynamically imported VisionAgent class: {class_name} for genre '{genre_key}'")
                else: # Fallback for base VisionAgent if needed, though typically abstract
                     vision_agent_classes['Default'] = agent_class # Or handle as needed
                     logging.debug(f"Dynamically imported VisionAgent class: {class_name} as Default")

        except ImportError as e:
            logging.error(f"Failed to import module {module_name}: {e}")
        except AttributeError as e:
             logging.error(f"Failed to find class {class_name} in module {module_name}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error loading vision agent {module_name}: {e}")
# --- Ensure DefaultVisionAgent is always registered as 'Default' ---
try:
    from art_agent_team.agents.vision_agent_default import DefaultVisionAgent
    if 'Default' not in vision_agent_classes:
        vision_agent_classes['Default'] = DefaultVisionAgent
        logging.info("Registered DefaultVisionAgent as fallback for VisionAgent.")
except Exception as e:
    logging.error(f"Failed to import or register DefaultVisionAgent: {e}")



# Add a check to ensure ResearchAgent is available
try:
    from art_agent_team.agents.research_agent import ResearchAgent
except ImportError:
    logging.error("ResearchAgent class not found. Ensure the module is correctly imported and available.")
    raise

# Configure basic logging
# Use a more detailed format including thread name
log_format = '%(asctime)s - %(levelname)s - [%(threadName)s] - %(name)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
# Get a logger specific to this module
logger = logging.getLogger(__name__)


class DocentAgent:
    """
    Orchestrates the art image processing workflow for the AI Art Team project.
    This class coordinates a series of AI agents to process artwork images, ensuring modifications respect artistic sensibilities such as composition, color balance, historical context, and emotional impact. The workflow includes intelligent cropping, upscaling with ESRGAN for detail preservation, and plaque overlay for metadata addition. Threading is used for concurrent processing to handle batches efficiently, with queues facilitating safe data handoff between stages. Configuration is loaded from YAML to allow easy parameter tuning, and logging is implemented for traceability and debugging.
    """

    def __init__(self, config_path='art_agent_team/config/config.yaml'):
        """
        Initialize the DocentAgent, setting up all necessary components for the AI Art Team's image processing pipeline.
        This constructor ensures that the agent is configured for secure and efficient operation, loading settings from a YAML file and preparing the environment for multi-threaded processing.
        - Config handling: The method loads configuration and sets API keys as environment variables, enhancing security by avoiding hard-coded secrets.
        - Directory creation: Automatically creates required directories to organize files, reducing the risk of runtime errors and improving code reliability.
        - Queue initialization: Establishes communication channels between workflow stages, supporting concurrent execution and preventing bottlenecks in the processing chain.
        - Agent setup: Instantiates or references agents (Research, Upscale, Placard) based on best practices for object-oriented design, allowing for easy extension or modification of the system.
        Design decision: Agents are instantiated here for simplicity, but research agents are created per image to handle potential state changes, balancing performance and memory usage.
        Artistic integrity: Configuration allows for tuning parameters that affect how images are processed, such as upscaling models or placard styles, to accommodate different artistic requirements and genres.
        Best practice adherence: Follows SOLID principles with single responsibility for each agent, and uses type hints and logging for better code maintainability and debugging.
        """
        # Load configuration (empty dict is valid for testing)
        self.config = self._load_config(config_path) or {}

        # Set up file system directories to maintain a clean and organized workflow structure
        self.input_folder = self.config.get('input_folder', 'input')  # Directory for user-provided input images
        self.workspace_folder = self.config.get('workspace_folder', 'workspace')  # Area for temporary or intermediate files during processing
        self.output_folder = self.config.get('output_folder', 'output')  # Location to save final processed images and reports
        os.makedirs(self.input_folder, exist_ok=True)  # Ensure directories exist to handle file I/O gracefully
        os.makedirs(self.workspace_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)

        # Queues removed for sequential processing

        # Set up agents with configuration-driven parameters for modularity and adaptability
        # Import agents needed for the workflow stages
        from art_agent_team.agents.upscale_agent import UpscaleAgent
        from art_agent_team.agents.placard_agent import PlacardAgent
        self.agents = {
            # ResearchAgent is instantiated per image in the processing thread
            'upscale': UpscaleAgent(config=self.config),
            'placard': PlacardAgent(config=self.config),
        }
        # Store vision agent classes loaded at the start
        self.vision_agent_classes = vision_agent_classes
        logger.info(f"Loaded Vision Agent classes: {list(self.vision_agent_classes.keys())}")

        # LLM client initialization removed - DocentAgent no longer interprets prompts directly
        # self.llm_client = None
        # self.llm_model_name = "grok-3-mini-fast-beta"
        # self._init_llm_client() # Removed

        logger.info("DocentAgent initialized. Ready to orchestrate user-selected workflow stages.")

    def _load_config(self, config_path='art_agent_team/config/config.yaml'): # Provide default here
        """Loads configuration from YAML, extracts API keys, and sets them as environment variables."""
        config = {}
        
        if config_path is None:
            logger.warning("No config path provided. Using empty configuration.")
            return config

        # Assume config_path is relative to project root if not absolute
        if not os.path.isabs(config_path):
             project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
             full_config_path = os.path.join(project_root, config_path)
        else:
             full_config_path = config_path


        # --- Load Config from YAML ---
        try:
            with open(full_config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                if loaded_config: # Ensure loaded config is not None
                    config.update(loaded_config)
            logger.info(f"Configuration loaded from {full_config_path}")
        except FileNotFoundError:
            logger.error(f"YAML configuration file not found at {full_config_path}. Cannot proceed without configuration.")
            raise # Re-raise as this is critical
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration file {full_config_path}: {e}")
            raise # Re-raise as this is critical
        except Exception as e:
            logger.error(f"Unexpected error loading YAML config file {full_config_path}: {e}")
            raise # Re-raise as this is critical

        # --- Extract API Keys from Config and Set Environment Variables ---
        # This ensures agents can access keys if needed via os.environ
        google_api_key = config.get('google_api_key')
        grok_api_key   = config.get('grok_api_key')
        openai_key = config.get('openai_api_key') # Explicit OpenAI key

        if google_api_key:
            os.environ['GOOGLE_API_KEY'] = google_api_key
            logger.info("Google API Key set in environment.")
        else:
            logger.warning("google_api_key not found in config.yaml.")
            if 'GOOGLE_API_KEY' in os.environ: del os.environ['GOOGLE_API_KEY']

        if grok_api_key:
            os.environ['GROK_API_KEY'] = grok_api_key
            logger.info("Grok API Key set in environment.")
        else:
            logger.warning("grok_api_key not found in config.yaml.")
            if 'GROK_API_KEY' in os.environ: del os.environ['GROK_API_KEY']

        # Set OPENAI_API_KEY - prioritize explicit openai_key, fallback to grok_key if needed by some libraries
        final_openai_key = openai_key or grok_api_key
        if final_openai_key:
            os.environ['OPENAI_API_KEY'] = final_openai_key
            logger.info("OpenAI API Key set in environment (using explicit key if available, else Grok key).")
        else:
            logger.warning("Neither openai_api_key nor grok_api_key found in config; OPENAI_API_KEY unset.")
            if 'OPENAI_API_KEY' in os.environ: del os.environ['OPENAI_API_KEY']

        return config

    # Removed _init_llm_client and _call_llm as DocentAgent no longer directly uses LLM for prompts

    def start_workflow(self):
        """
        Orchestrates the image processing workflow based on user selection.
        Allows users to choose which stages (Research, Vision, Upscale, Placard)
        to run sequentially on images found in the input folder.
        """
        logger.info("Starting user-selectable workflow.")

        # Step 0: Display workflow options as a numbered list (reverted to original, non-conversational)
        # No persona prompt or conversational placeholder.

        # Step 1: Get list of images to process
        image_paths = self._get_image_paths(self.input_folder) or []
        num_images = len(image_paths)

        if num_images == 0:
            logger.warning(f"No supported image files found in {self.input_folder}. Workflow ending.")
            print("No images found to process.")
            return

        logger.info(f"Found {num_images} images for potential processing.")
        print(f"Found {num_images} images in '{self.input_folder}'.")

        # Step 2: Present numbered workflow options to user
        print("\nSelect the workflow stages to run on the image(s):")
        print("  1. Research (analyze and extract metadata)")
        print("  2. Vision (intelligent cropping)")
        print("  3. Upscale (enhance image quality)")
        print("  4. Placard (add museum-style label)")
        print("  5. Full Workflow (all stages in order)")
        print("Enter the numbers of the stages you want to run, separated by commas (e.g., 1,3,4 or 5 for full workflow):")
        user_input = input("Your selection: ").strip()
        selected = set(s.strip() for s in user_input.split(",") if s.strip())

        run_research = "1" in selected or "5" in selected
        run_vision = "2" in selected or "5" in selected
        run_upscale = "3" in selected or "5" in selected
        run_placard = "4" in selected or "5" in selected

        selected_stages = []
        if run_research: selected_stages.append("Research")
        if run_vision: selected_stages.append("Vision")
        if run_upscale: selected_stages.append("Upscale")
        if run_placard: selected_stages.append("Placard")
        logger.info(f"User selected workflow: {' -> '.join(selected_stages)}")
        print(f"Executing workflow: {' -> '.join(selected_stages)}")
        print(f"Processing {num_images} image(s)...")

        # Step 4: Process each image sequentially based on selected stages
        processed_count = 0
        skipped_count = 0
        for initial_image_path in image_paths:
            base_filename = os.path.basename(initial_image_path)
            logger.info(f"--- Processing image: {base_filename} ---")
            print(f"\nProcessing: {base_filename}")

            # Initialize state for this image
            current_image_path = initial_image_path # Start with the original image
            metadata: Dict[str, Any] = {}
            analysis_results: Optional[Dict[str, Any]] = None # Store vision analysis if run
            genre = 'Default' # Default genre if research is skipped or fails
            image_processed_successfully = True # Flag to track if processing should continue
            cropped_intermediate_path = None # Track cropped path for potential cleanup
            upscaled_intermediate_path = None # Track upscaled path for potential cleanup

            # --- Research Stage ---
            if run_research:
                logger.info(f"[{base_filename}] Running Research stage...")
                try:
                    # Instantiate ResearchAgent for each image
                    research_agent = ResearchAgent(self.config, self.vision_agent_classes)
                    research_output = research_agent.research_and_process(current_image_path)

                    # --- Display Individual Research Attempts ---
                    print(f"  --- Individual Research Attempts for {base_filename} ---")
                    if research_output and isinstance(research_output, dict) and "individual_results" in research_output:
                        individual_results = research_output.get("individual_results", {})
                        display_order = ["grok_text", "gemini_text", "grok_image", "gemini_image", "openrouter_text", "openrouter_qwen_image", "vertex_vision_subjects"]

                        for source in display_order:
                            if source not in individual_results:
                                print(f"    - {source.replace('_', ' ').title()}: Not Run/Found")
                                continue

                            attempt = individual_results[source]
                            container_success = attempt.get("success", False)
                            result_data = attempt.get("result", {}) # Should always be a dict now

                            # Helper function to format confidence
                            def format_confidence(score):
                                if score is None: return "N/A"
                                try:
                                    return f"{float(score)*100:.1f}%"
                                except (ValueError, TypeError):
                                    return "Invalid"

                            # Helper function to display common fields
                            def display_common_fields(res_dict):
                                if not isinstance(res_dict, dict): return
                                conf = format_confidence(res_dict.get('confidence_score'))
                                ground = res_dict.get('grounding_used', 'N/A')
                                primary_subj = res_dict.get('primary_subjects', [])
                                secondary_subj = res_dict.get('secondary_subjects', [])
                                print(f"        Confidence: {conf}, Grounding: {ground}")
                                if primary_subj: print(f"        Primary Subjects: {primary_subj}")
                                if secondary_subj: print(f"        Secondary Subjects: {secondary_subj}")

                            # Special handling for openrouter_text nested results
                            if source == "openrouter_text":
                                print(f"    - OpenRouter Text Search:")
                                if not container_success:
                                     print(f"      Status: Failure (Container)")
                                     if isinstance(result_data, dict) and "error" in result_data:
                                          print(f"      Error: {result_data.get('error', 'Unknown container error')}")
                                     continue

                                llama_mav = result_data.get("llama_maverick", {"error": "Not found"})
                                llama_sco = result_data.get("llama_scout", {"error": "Not found"})

                                for llama_key, llama_res in [("Llama Maverick", llama_mav), ("Llama Scout", llama_sco)]:
                                    if isinstance(llama_res, dict):
                                        llama_success = "error" not in llama_res
                                        llama_success_str = "Success" if llama_success else "Failure"
                                        print(f"      - {llama_key}: {llama_success_str}")
                                        if not llama_success:
                                            print(f"        Error: {llama_res.get('error', 'Unknown error')}")
                                        # Display common fields even on failure (might have defaults)
                                        display_common_fields(llama_res)
                                        if llama_success:
                                            # Display other details only on success
                                            # Safely print other details, truncating long strings
                                            details_list = []
                                            max_detail_len = 200 # Max length for printing unknown values
                                            for k, v in llama_res.items():
                                                if k not in ['error', 'confidence_score', 'grounding_used', 'primary_subjects', 'secondary_subjects']:
                                                    v_str = str(v) # Convert to string
                                                    if len(v_str) > max_detail_len:
                                                        v_display = v_str[:max_detail_len] + "... <truncated>"
                                                    else:
                                                        v_display = v_str
                                                    details_list.append(f"{k}: {v_display}")
                                            if details_list: print(f"        Other Details: {{{', '.join(details_list)}}}")
                                    else:
                                         print(f"      - {llama_key}: Invalid Format ({type(llama_res)})")

                            else: # Handle other sources (grok, gemini, qwen, subjects)
                                success_str = "Success" if container_success else "Failure"
                                print(f"    - {source.replace('_', ' ').title()}: {success_str}")

                                # result_data should always be a dict due to _make_llm_call changes
                                if not isinstance(result_data, dict):
                                     print(f"      Error: Unexpected result format ({type(result_data)})")
                                     continue # Skip if format is wrong despite safeguards

                                error_msg = result_data.get("error")

                                if error_msg: # Check for error key first
                                     print(f"      Error: {error_msg}")
                                     # Still display common fields if they exist (might have defaults)
                                     display_common_fields(result_data)
                                elif not container_success:
                                     # Should ideally have an error message, but handle just in case
                                     print(f"      Error: Unknown container failure (no error message in result)")
                                     display_common_fields(result_data)
                                else: # Successful container and no error key in result_data
                                    # Display common fields
                                    display_common_fields(result_data)
                                    # Display other details
                                    if source == "vertex_vision_subjects":
                                        # Subjects already printed by display_common_fields
                                        pass
                                    else: # Grok/Gemini/Qwen results
                                        # Safely print other details, truncating long strings
                                        details_list = []
                                        max_detail_len = 200 # Max length for printing unknown values
                                        for k, v in result_data.items():
                                            if k not in ['error', 'confidence_score', 'grounding_used', 'primary_subjects', 'secondary_subjects']:
                                                v_str = str(v) # Convert to string
                                                if len(v_str) > max_detail_len:
                                                    v_display = v_str[:max_detail_len] + "... <truncated>"
                                                else:
                                                    v_display = v_str
                                                details_list.append(f"{k}: {v_display}")
                                        if details_list: print(f"      Other Details: {{{', '.join(details_list)}}}")

                    else:
                        print("    No individual results data found.")
                    print(f"  -------------------------------------------------")

                    # --- Process and Display Consolidated Research Results ---
                    metadata = {} # Initialize metadata as empty
                    genre = 'Default' # Initialize genre as default

                    if research_output and isinstance(research_output, dict) and "consolidated_results" in research_output:
                        consolidated_metadata = research_output.get("consolidated_results")

                        # consolidated_metadata should always be a dict now
                        if consolidated_metadata and isinstance(consolidated_metadata, dict):
                            if 'error' not in consolidated_metadata:
                                metadata = consolidated_metadata # Assign valid consolidated data
                                print(f"  --- Consolidated Research Results for {base_filename} ---")
                                # Display standard fields
                                keys_to_display = ['author', 'title', 'date', 'nationality', 'style', 'genre', 'brief_description']
                                for key in keys_to_display:
                                    display_key = key.replace('_', ' ').capitalize()
                                    value = metadata.get(key) # Use .get() which returns None if missing
                                    print(f"    {display_key}: {value if value is not None else 'N/A'}") # Display N/A for None

                                # Display new fields
                                conf = format_confidence(metadata.get('confidence_score'))
                                ground = metadata.get('grounding_used', 'N/A')
                                primary_subj = metadata.get('primary_subjects', [])
                                secondary_subj = metadata.get('secondary_subjects', [])
                                print(f"    Confidence score: {conf}")
                                print(f"    Grounding used: {ground}")
                                print(f"    Primary subjects: {primary_subj if primary_subj else 'N/A'}")
                                print(f"    Secondary subjects: {secondary_subj if secondary_subj else 'N/A'}")

                                print(f"  -------------------------------------------------")
                                genre = metadata.get('genre') or 'Default' # Update genre
                                logger.info(f"[{base_filename}] Research consolidation complete. Genre: {genre}. Confidence: {conf}. Grounding: {ground}. Metadata keys: {list(metadata.keys())}")
                                print(f"  - Research: OK (Consolidated Genre: {genre}, Confidence: {conf})")
                            else: # Error key exists in consolidated_metadata
                                error_msg = consolidated_metadata.get('error', 'Unknown consolidation error')
                                logger.warning(f"[{base_filename}] Research consolidation completed with an error message: {error_msg}")
                                print(f"  - Research: CONSOLIDATION ERROR ({error_msg})")
                                print("  - Consolidated Research Results: Not displayed due to error.")
                                # Keep metadata empty and genre default
                        else: # Handle None or unexpected type (less likely now)
                            logger.warning(f"[{base_filename}] Research consolidation did not return a valid dictionary.")
                            print("  - Research: FAILED (Invalid consolidation output type)")
                            print("  - Consolidated Research Results: No data to display.")
                            # Keep metadata empty and genre default
                    else: # Handle case where research_output structure is missing keys
                         # Log keys instead of the full potentially large/problematic dictionary
                         output_keys = research_output.keys() if isinstance(research_output, dict) else type(research_output)
                         logger.warning(f"[{base_filename}] Research output structure invalid or missing keys. Output keys/type: {output_keys}")
                         print("  - Research: FAILED (Invalid output structure)")
                         # Keep metadata empty and genre default

                except Exception as research_err:
                    logger.error(f"[{base_filename}] Research stage failed unexpectedly: {research_err}", exc_info=True)
                    print(f"  - Research: FAILED ({research_err})")
                    # Ensure metadata is empty and genre is default on unexpected failure
                    metadata = {}
                    genre = 'Default'

            # --- Vision Stage ---
            if run_vision:
                logger.info(f"[{base_filename}] Running Vision stage (Genre: {genre})...")
                # Determine VisionAgent class
                vision_agent_class = self.vision_agent_classes.get(genre)
                if not vision_agent_class:
                    logger.warning(f"[{base_filename}] No specific VisionAgent for genre '{genre}'. Using Default.")
                    vision_agent_class = self.vision_agent_classes.get('Default')

                if not vision_agent_class:
                    logger.error(f"[{base_filename}] Default VisionAgent not found. Skipping Vision stage.")
                    print(f"  - Vision: SKIPPED (Default agent not found)")
                else:
                    try:
                        vision_agent = vision_agent_class(self.config)
                        logger.debug(f"[{base_filename}] Instantiated Vision Agent: {vision_agent.__class__.__name__}")

                        # Optional: Analyze image first
                        if hasattr(vision_agent, 'analyze_image'):
                            try:
                                # Pass metadata if available from research stage
                                analysis_results = vision_agent.analyze_image(current_image_path, metadata)
                                logger.info(f"[{base_filename}] Vision analysis complete.")
                                print(f"  - Vision Analysis: OK")
                            except Exception as analyze_err:
                                logger.error(f"[{base_filename}] Vision analysis failed: {analyze_err}", exc_info=True)
                                print(f"  - Vision Analysis: FAILED ({analyze_err})")
                                analysis_results = None # Ensure it's None on failure

                        # Define cropped output path
                        cropped_filename = f"{os.path.splitext(base_filename)[0]}_cropped{os.path.splitext(base_filename)[1]}"
                        cropped_output_path = os.path.join(self.workspace_folder, cropped_filename)

                        # Attempt cropping
                        cropped_path = None
                        if hasattr(vision_agent, 'copy_and_crop_image'):
                            logger.debug(f"[{base_filename}] Attempting crop with copy_and_crop_image...")
                            cropped_path = vision_agent.copy_and_crop_image(
                                current_image_path,
                                cropped_output_path,
                                analysis_results # Pass analysis results (can be None)
                            )
                        # Add fallback if needed, e.g., crop_to_aspect_ratio
                        # elif hasattr(vision_agent, 'crop_to_aspect_ratio'): ...
                        else:
                            logger.warning(f"[{base_filename}] Vision agent {vision_agent.__class__.__name__} has no 'copy_and_crop_image' method. Skipping crop.")
                            print(f"  - Vision Crop: SKIPPED (No suitable method)")

                        # Update current_image_path if cropping was successful
                        if cropped_path and os.path.exists(cropped_path):
                            if cropped_path != current_image_path:
                                logger.info(f"[{base_filename}] Vision crop successful. New path: {cropped_path}")
                                print(f"  - Vision Crop: OK -> {os.path.basename(cropped_path)}")
                                current_image_path = cropped_path
                                cropped_intermediate_path = cropped_path # Store for potential cleanup
                            else:
                                logger.info(f"[{base_filename}] Vision agent did not perform crop. Using previous path.")
                                print(f"  - Vision Crop: OK (No change)")
                        elif cropped_path is None: # Case where cropping method was skipped
                            pass # Path remains unchanged, already logged/printed
                        else: # Case where cropping method ran but failed (returned invalid path)
                            logger.error(f"[{base_filename}] Vision crop failed or output path invalid: {cropped_path}. Using previous path for subsequent stages.")
                            print(f"  - Vision Crop: FAILED (Output invalid). Continuing with previous image.")
                            # Keep current_image_path as it was before this stage

                    except Exception as vision_err:
                        logger.error(f"[{base_filename}] Vision stage failed: {vision_err}", exc_info=True)
                        print(f"  - Vision: FAILED ({vision_err})")
                        # Keep current_image_path as it was before this stage

            # --- Upscale Stage ---
            if run_upscale:
                logger.info(f"[{base_filename}] Running Upscale stage...")
                # Check if input path exists before attempting upscale
                if not os.path.exists(current_image_path):
                     logger.error(f"[{base_filename}] Input image for Upscale not found: {current_image_path}. Skipping Upscale.")
                     print(f"  - Upscale: SKIPPED (Input not found)")
                else:
                    try:
                        upscale_agent = self.agents['upscale']
                        # Define upscale output path
                        name, ext = os.path.splitext(os.path.basename(current_image_path))
                        # Avoid double suffixes if input was already _cropped
                        base_name_for_upscale = name.replace('_cropped', '')
                        upscaled_filename = f"{base_name_for_upscale}_upscaled{ext}"
                        upscale_output_path = os.path.join(self.workspace_folder, upscaled_filename)

                        upscaled_path = upscale_agent.upscale_image(current_image_path, upscale_output_path)

                        if upscaled_path and os.path.exists(upscaled_path):
                            logger.info(f"[{base_filename}] Upscaling successful. New path: {upscaled_path}")
                            print(f"  - Upscale: OK -> {os.path.basename(upscaled_path)}")
                            current_image_path = upscaled_path
                            upscaled_intermediate_path = upscaled_path # Store for potential cleanup
                        else:
                            logger.error(f"[{base_filename}] Upscaling failed or output path invalid. Using previous path for subsequent stages.")
                            print(f"  - Upscale: FAILED (Output invalid). Continuing with previous image.")
                            # Keep current_image_path as it was

                    except Exception as upscale_err:
                        logger.error(f"[{base_filename}] Upscale stage failed: {upscale_err}", exc_info=True)
                        print(f"  - Upscale: FAILED ({upscale_err})")
                        # Keep current_image_path as it was

            # --- Placard Stage ---
            if run_placard:
                logger.info(f"[{base_filename}] Running Placard stage...")
                # Check if input path exists before attempting placard
                if not os.path.exists(current_image_path):
                     logger.error(f"[{base_filename}] Input image for Placard not found: {current_image_path}. Skipping Placard.")
                     print(f"  - Placard: SKIPPED (Input not found)")
                else:
                    try:
                        placard_agent = self.agents['placard']
                        # Define final output path
                        name, ext = os.path.splitext(os.path.basename(current_image_path))
                        # Clean up intermediate suffixes for final name
                        original_base_name = name.replace('_upscaled', '').replace('_cropped', '')
                        final_filename = f"{original_base_name}_final{ext}"
                        final_output_path = os.path.join(self.output_folder, final_filename)

                        # Pass metadata (might be empty if research failed or was skipped)
                        placarded_path = placard_agent.add_plaque(current_image_path, final_output_path, metadata)

                        if placarded_path and os.path.exists(placarded_path):
                            logger.info(f"[{base_filename}] Placard addition successful. Final output: {placarded_path}")
                            print(f"  - Placard: OK -> {os.path.basename(placarded_path)}")
                            current_image_path = placarded_path # Update path to final output

                            # --- Optional Cleanup ---
                            cleanup_enabled = self.config.get('cleanup_workspace', True)
                            if cleanup_enabled:
                                files_to_remove = []
                                # Add intermediate files if they exist and are different from final output
                                if cropped_intermediate_path and os.path.exists(cropped_intermediate_path) and cropped_intermediate_path != final_output_path:
                                    files_to_remove.append(cropped_intermediate_path)
                                if upscaled_intermediate_path and os.path.exists(upscaled_intermediate_path) and upscaled_intermediate_path != final_output_path:
                                     files_to_remove.append(upscaled_intermediate_path)
                                # Also remove the direct input to placard if it wasn't the final output (e.g. if upscale ran but placard saved elsewhere)
                                input_to_placard = upscaled_intermediate_path or cropped_intermediate_path or initial_image_path
                                if input_to_placard != final_output_path and input_to_placard not in files_to_remove and os.path.exists(input_to_placard) and self.workspace_folder in input_to_placard:
                                     files_to_remove.append(input_to_placard)


                                for file_to_remove in set(files_to_remove): # Use set to avoid duplicates
                                    try:
                                        os.remove(file_to_remove)
                                        logger.debug(f"[{base_filename}] Removed intermediate file: {file_to_remove}")
                                    except OSError as remove_err:
                                        logger.warning(f"[{base_filename}] Failed to remove intermediate file '{file_to_remove}': {remove_err}")
                            else:
                                logger.debug(f"[{base_filename}] Workspace cleanup disabled.")

                        else:
                            logger.error(f"[{base_filename}] Placard addition failed or output path invalid.")
                            print(f"  - Placard: FAILED (Output invalid)")
                            # current_image_path remains the input to this failed stage

                    except Exception as placard_err:
                        logger.error(f"[{base_filename}] Placard stage failed: {placard_err}", exc_info=True)
                        print(f"  - Placard: FAILED ({placard_err})")
                        # current_image_path remains the input to this failed stage

            # --- Image Processing Summary ---
            if image_processed_successfully: # Check overall success flag if implemented, otherwise assume processed
                processed_count += 1
                logger.info(f"--- Finished processing image: {base_filename} ---")
            else:
                skipped_count += 1
                logger.warning(f"--- Skipped or failed processing image: {base_filename} ---")

        # Step 5: Final Workflow Summary
        print("\n--- Workflow Summary ---")
        print(f"Selected stages: {' -> '.join(selected_stages)}")
        print(f"Total images found: {num_images}")
        print(f"Successfully processed: {processed_count}")
        print(f"Skipped/Failed: {skipped_count}")
        print(f"Final outputs (if Placard ran) are in: '{self.output_folder}'")
        print(f"Intermediate files (if not cleaned) are in: '{self.workspace_folder}'")
        print("--------------------------")
        logger.info(f"DocentAgent workflow finished. Processed: {processed_count}, Skipped/Failed: {skipped_count}.")


    def _get_image_paths(self, folder_path) -> List[str]:
        """Finds supported image files in the folder and returns a list of their paths."""
        # Ensure logger is accessible, assuming it's defined as self.logger or globally as logger
        log = getattr(self, 'logger', logging.getLogger(__name__)) # Use self.logger if available, else module logger

        log.info(f"Scanning folder for images: {folder_path}")
        image_paths = []
        supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')

        try:
            if not os.path.isdir(folder_path):
                log.error(f"Input folder does not exist or is not a directory: {folder_path}")
                print(f"Error: Input folder '{folder_path}' not found.")
                return [] # Return empty list on critical error

            for filename in os.listdir(folder_path):
                log.debug(f"Processing entry: {filename}") # Log every entry

                if filename.startswith('.'):
                    log.debug(f"Skipping hidden entry: {filename}")
                    continue

                file_path = os.path.join(folder_path, filename)

                # Check if it's a file *before* checking extension
                if not os.path.isfile(file_path):
                    log.debug(f"Skipping non-file entry: {filename}")
                    continue

                # Check if extension is supported
                if not filename.lower().endswith(supported_extensions):
                    log.debug(f"Skipping unsupported file extension: {filename}")
                    continue

                # If all checks pass, add the file path
                image_paths.append(file_path)
                log.debug(f"Added supported image file: {filename}")

            # Report final count after loop
            if image_paths:
                log.info(f"Found {len(image_paths)} supported images in {folder_path}.")
            else:
                log.warning(f"No supported images found in {folder_path}.")
                # Keep the print statement for user visibility if desired
                print(f"Warning: No supported images (.png, .jpg, .jpeg, .bmp, .gif, .tiff, .webp) found in '{folder_path}'.")

        except PermissionError:
            log.error(f"Permission denied when trying to scan folder: {folder_path}")
            print(f"Error: Permission denied for folder '{folder_path}'.")
            return [] # Return empty list on permission error
        except Exception as e:
            log.exception(f"An unexpected error occurred scanning input folder {folder_path}: {e}") # Use log.exception to include traceback
            print(f"Error scanning input folder: {e}")
            return [] # Return empty list on other exceptions

        # Return the list (potentially empty) after successful scan
        return image_paths

    def _think_with_grok(self, prompt: str):
        """
        Internal reasoning using Grok LLM (grok-3-mini-fast-high-beta).
        """
        import os
        grok_api_key = os.environ.get("GROK_API_KEY")
        if not grok_api_key:
            logger.warning("[DocentAgent] Grok API key not found. Skipping internal reasoning.")
            return None
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=grok_api_key,
                base_url="https://api.x.ai/v1",
            )
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="grok-3-mini-fast-high-beta",
                temperature=0.2,
                max_tokens=256
            )
            reasoning = response.choices[0].message.content
            logger.info(f"[DocentAgent] Grok thinking output: {reasoning}")
            return reasoning
        except Exception as e:
            logger.error(f"[DocentAgent] Grok thinking failed: {e}")
            return None

# --- Integrated Thinking Step for Agent Refinement ---
def _refine_agent_input_with_grok(
    self,
    target_agent_name: str,
    current_image_path: str,
    previous_stage_results: Optional[dict],
    user_context_input: str
) -> Optional[dict]:
    """
    Calls _think_with_grok to refine parameters/instructions for the next agent stage.
    Returns a validated dictionary of parameters for the target agent, or empty dict if none.
    """
    # Define expected keys for each agent
    expected_keys = {
        "Vision": ["focus_areas", "crop_preference"],
        "Upscale": ["upscale_model", "preserve_texture_level"],
        "Placard": ["placard_style", "additional_notes"]
    }
    keys = expected_keys.get(target_agent_name, [])

    # Summarize previous results for prompt
    prev_summary = ""
    if previous_stage_results:
        try:
            prev_summary = json.dumps(previous_stage_results, ensure_ascii=False)
        except Exception:
            prev_summary = str(previous_stage_results)
    else:
        prev_summary = "None"

    prompt = (
        f"Generate refined parameters/instructions for the upcoming {target_agent_name} stage.\n"
        f"Previous stage results: {prev_summary}\n"
        f"User preference: '{user_context_input}'.\n"
        f"The target image is {current_image_path}.\n"
        f"Respond ONLY with a JSON dictionary containing keys relevant for the {target_agent_name} agent, "
        f"such as {keys}. If no specific refinement is needed, return an empty JSON object {{}}."
    )

    response = self._think_with_grok(prompt)
    if not response:
        logger.warning(f"[DocentAgent] No response from Grok for {target_agent_name} thinking step.")
        return {}

    # Try to extract JSON from response
    import re
    json_str = None
    # Try to find a JSON object in the response
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        json_str = match.group(0)
    else:
        json_str = response.strip()

    try:
        parsed = json.loads(json_str)
        if not isinstance(parsed, dict):
            logger.warning(f"[DocentAgent] Grok output for {target_agent_name} is not a dict: {parsed}")
            return {}
    except Exception as e:
        logger.warning(f"[DocentAgent] Failed to parse Grok output for {target_agent_name}: {e}. Raw output: {response}")
        return {}

    # Validate keys
    if keys:
        valid = all((k in keys for k in parsed.keys()))
        if not valid:
            logger.warning(f"[DocentAgent] Grok output for {target_agent_name} contains unexpected keys: {parsed.keys()}")
            # Optionally filter to only expected keys
            parsed = {k: v for k, v in parsed.items() if k in keys}
    return parsed

    def handle_request(self, task_description: str, image_path: str) -> Dict[str, Any]:
        """
        Handles a specific request to process an image through the full workflow.
        This is a non-interactive method intended for programmatic calls (e.g., from tests or other agents).

        Args:
            task_description (str): A description of the task (e.g., "Analyze and crop the image.").
                                    Currently used for logging.
            image_path (str): The path to the input image.

        Returns:
            Dict[str, Any]: A dictionary containing the path to the final processed image
                            and any collected metadata.
                            Example: {"final_image_path": "path/to/output.jpg", "metadata": {...}}
                            Returns an error structure if processing fails.
        """
        base_filename = os.path.basename(image_path)
        logger.info(f"--- Handling request for image: {base_filename} ---")
        logger.info(f"Task description: {task_description}")

        if not os.path.exists(image_path):
            logger.error(f"Input image not found: {image_path}")
            return {"error": f"Input image not found: {image_path}", "final_image_path": None, "metadata": {}}

        # Simulate full workflow selection
        run_research = True
        run_vision = True
        run_upscale = True
        run_placard = True

        current_image_path = image_path
        metadata: Dict[str, Any] = {}
        analysis_results: Optional[Dict[str, Any]] = None
        genre = 'Default'
        cropped_intermediate_path = None
        upscaled_intermediate_path = None
        
        # Track called agents for User Instruction 4
        called_agents_log = []


        # --- Research Stage ---
        if run_research:
            called_agents_log.append("ResearchAgent")
            logger.info(f"[{base_filename}] Running Research stage...")
            try:
                research_agent = ResearchAgent(self.config, self.vision_agent_classes)
                research_output = research_agent.research_and_process(current_image_path)
                if research_output and isinstance(research_output, dict) and "consolidated_results" in research_output:
                    consolidated_metadata = research_output.get("consolidated_results")
                    if consolidated_metadata and isinstance(consolidated_metadata, dict) and 'error' not in consolidated_metadata:
                        metadata = consolidated_metadata
                        genre = metadata.get('genre') or 'Default'
                        logger.info(f"[{base_filename}] Research complete. Genre: {genre}. Metadata collected.")
                    else:
                        logger.warning(f"[{base_filename}] Research consolidation error: {consolidated_metadata.get('error') if isinstance(consolidated_metadata, dict) else 'Unknown'}")
                else:
                    logger.warning(f"[{base_filename}] Research output invalid or missing consolidated_results.")
            except Exception as research_err:
                logger.error(f"[{base_filename}] Research stage failed: {research_err}", exc_info=True)
                metadata = {"error": f"Research stage failed: {research_err}"}


        # --- Vision Stage ---
        if run_vision:
            vision_agent_to_log = "DefaultVisionAgent" # Start with default
            logger.info(f"[{base_filename}] Running Vision stage (Genre: {genre})...")
            vision_agent_class = self.vision_agent_classes.get(genre)
            if not vision_agent_class:
                logger.warning(f"[{base_filename}] No specific VisionAgent for genre '{genre}'. Using Default.")
                vision_agent_class = self.vision_agent_classes.get('Default')

            if vision_agent_class:
                try:
                    vision_agent = vision_agent_class(self.config)
                    vision_agent_to_log = vision_agent.__class__.__name__
                    called_agents_log.append(vision_agent_to_log)
                    logger.debug(f"[{base_filename}] Instantiated Vision Agent: {vision_agent_to_log}")
                    if hasattr(vision_agent, 'analyze_image'):
                        analysis_results = vision_agent.analyze_image(current_image_path, metadata)
                    
                    cropped_filename = f"{os.path.splitext(base_filename)[0]}_cropped{os.path.splitext(base_filename)[1]}"
                    cropped_output_path = os.path.join(self.workspace_folder, cropped_filename)
                    
                    cropped_path_result = None
                    if hasattr(vision_agent, 'copy_and_crop_image'):
                        cropped_path_result = vision_agent.copy_and_crop_image(current_image_path, cropped_output_path, analysis_results)
                    
                    if cropped_path_result and os.path.exists(cropped_path_result) and cropped_path_result != current_image_path:
                        current_image_path = cropped_path_result
                        cropped_intermediate_path = cropped_path_result
                        logger.info(f"[{base_filename}] Vision crop successful. New path: {current_image_path}")
                    elif cropped_path_result == current_image_path:
                         logger.info(f"[{base_filename}] Vision agent did not perform crop. Using previous path.")
                    else:
                        logger.warning(f"[{base_filename}] Vision crop failed or output path invalid. Using previous path.")
                except Exception as vision_err:
                    logger.error(f"[{base_filename}] Vision stage failed for {vision_agent_to_log}: {vision_err}", exc_info=True)
            else:
                logger.error(f"[{base_filename}] Default VisionAgent not found. Skipping Vision stage.")
                called_agents_log.append(f"VisionAgent (Error: Default not found for genre {genre})")


        # --- Upscale Stage ---
        if run_upscale:
            called_agents_log.append("UpscaleAgent")
            logger.info(f"[{base_filename}] Running Upscale stage...")
            if not os.path.exists(current_image_path):
                logger.error(f"[{base_filename}] Input image for Upscale not found: {current_image_path}. Skipping.")
            else:
                try:
                    upscale_agent = self.agents['upscale']
                    name, ext = os.path.splitext(os.path.basename(current_image_path))
                    base_name_for_upscale = name.replace('_cropped', '')
                    upscaled_filename = f"{base_name_for_upscale}_upscaled{ext}"
                    upscale_output_path = os.path.join(self.workspace_folder, upscaled_filename)
                    upscaled_path_result = upscale_agent.upscale_image(current_image_path, upscale_output_path)
                    if upscaled_path_result and os.path.exists(upscaled_path_result):
                        current_image_path = upscaled_path_result
                        upscaled_intermediate_path = upscaled_path_result
                        logger.info(f"[{base_filename}] Upscaling successful. New path: {current_image_path}")
                    else:
                        logger.warning(f"[{base_filename}] Upscaling failed or output path invalid. Using previous path.")
                except Exception as upscale_err:
                    logger.error(f"[{base_filename}] Upscale stage failed: {upscale_err}", exc_info=True)

        # --- Placard Stage ---
        if run_placard:
            called_agents_log.append("PlacardAgent")
            logger.info(f"[{base_filename}] Running Placard stage...")
            if not os.path.exists(current_image_path):
                logger.error(f"[{base_filename}] Input image for Placard not found: {current_image_path}. Skipping.")
            else:
                try:
                    placard_agent = self.agents['placard']
                    name, ext = os.path.splitext(os.path.basename(current_image_path))
                    original_base_name = name.replace('_upscaled', '').replace('_cropped', '')
                    final_filename = f"{original_base_name}_final{ext}"
                    final_output_path = os.path.join(self.output_folder, final_filename)
                    placarded_path_result = placard_agent.add_plaque(current_image_path, final_output_path, metadata)
                    if placarded_path_result and os.path.exists(placarded_path_result):
                        current_image_path = placarded_path_result # This is the final image
                        logger.info(f"[{base_filename}] Placard addition successful. Final output: {current_image_path}")
                        
                        cleanup_enabled = self.config.get('cleanup_workspace', True)
                        if cleanup_enabled:
                            files_to_remove = []
                            if cropped_intermediate_path and os.path.exists(cropped_intermediate_path) and cropped_intermediate_path != current_image_path:
                                files_to_remove.append(cropped_intermediate_path)
                            if upscaled_intermediate_path and os.path.exists(upscaled_intermediate_path) and upscaled_intermediate_path != current_image_path:
                                files_to_remove.append(upscaled_intermediate_path)
                            
                            input_to_placard_stages = [initial_image_path]
                            if cropped_intermediate_path: input_to_placard_stages.append(cropped_intermediate_path)
                            if upscaled_intermediate_path: input_to_placard_stages.append(upscaled_intermediate_path)

                            for file_to_remove in set(files_to_remove):
                                if file_to_remove != current_image_path and self.workspace_folder in file_to_remove : # only remove from workspace
                                    try:
                                        os.remove(file_to_remove)
                                        logger.debug(f"[{base_filename}] Removed intermediate file: {file_to_remove}")
                                    except OSError as remove_err:
                                        logger.warning(f"[{base_filename}] Failed to remove intermediate file '{file_to_remove}': {remove_err}")
                    else:
                        logger.error(f"[{base_filename}] Placard addition failed or output path invalid.")
                except Exception as placard_err:
                    logger.error(f"[{base_filename}] Placard stage failed: {placard_err}", exc_info=True)
        
        logger.info(f"--- Finished handling request for image: {base_filename} ---")
        
        # Add called agents to metadata for User Instruction 4
        metadata["called_agents_workflow"] = called_agents_log

        return {"final_image_path": current_image_path, "metadata": metadata}

# --- End of Class ---
# Worker methods (_initial_image_processor, _upscale_worker, _placard_worker) are removed.