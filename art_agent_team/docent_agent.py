import logging
import os
import yaml
import json
import time
import importlib
from queue import Queue, Empty
from threading import Thread
from typing import Dict, Any, Optional

from art_agent_team.agents.research_agent import ResearchAgent  # Direct import for ResearchAgent

# Dynamically import VisionAgent classes only
vision_agent_classes = {}
agent_folder = os.path.join(os.path.dirname(__file__), 'agents')
for filename in os.listdir(agent_folder):
    if filename.endswith('.py') and filename != '__init__.py' and filename != 'research_agent.py':
        module_name = f"art_agent_team.agents.{filename[:-3]}"
        try:
            module = importlib.import_module(module_name)
            for name, obj in module.__dict__.items():
                if isinstance(obj, type) and 'VisionAgent' in name:  # Only import VisionAgent subclasses
                    vision_agent_classes[name] = obj
                    logging.debug(f"Dynamically imported VisionAgent class: {name}")
        except ImportError as e:
            logging.error(f"Failed to import module {module_name}: {e}")

# Add a check to ensure ResearchAgent is available
try:
    ResearchAgent  # Test if ResearchAgent is defined
except NameError:
    logging.error("ResearchAgent class not found. Ensure the module is correctly imported and available.")
    raise

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')

class DocentAgent:
    """Orchestrates the art image processing workflow."""

    def __init__(self, config_path='art_agent_team/config/config.yaml'):
        self.config = self._load_config(config_path)
        if not self.config:
            raise ValueError("Failed to load configuration.")

        self.input_folder = self.config.get('input_folder', 'input')
        self.workspace_folder = self.config.get('workspace_folder', 'workspace')
        self.output_folder = self.config.get('output_folder', 'output')
        self.num_research_agents = 8 # Hardcoded to 8 as requested

        # Ensure necessary directories exist
        os.makedirs(self.input_folder, exist_ok=True)
        os.makedirs(self.workspace_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)

        # Queues for managing workflow
        self.image_queue = Queue()
        self.upscale_input_queue = Queue() # VisionAgent -> UpscaleAgent Router (if needed, but may be removed later)

        # No fixed research agent pool; will create instances per image in start_workflow
        logging.info("DocentAgent initialized.")

    def _load_config(self, config_path):
        """Loads configuration from YAML, extracts API keys, and sets them as environment variables."""
        config = {}
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        full_config_path = os.path.join(project_root, config_path)

        # --- Load Config from YAML ---
        try:
            with open(full_config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                if loaded_config: # Ensure loaded config is not None
                    config.update(loaded_config)
            logging.info(f"Configuration loaded from {full_config_path}")
        except FileNotFoundError:
            logging.error(f"YAML configuration file not found at {full_config_path}. Cannot proceed without configuration.")
            return None # Return None if config file is essential
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML configuration file {full_config_path}: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error loading YAML config file {full_config_path}: {e}")
            return None

        # --- Extract API Keys from Config and Set Environment Variables ---
        google_api_key = config.get('google_api_key')
        grok_api_key = config.get('grok_api_key')

        if google_api_key:
            os.environ['GOOGLE_API_KEY'] = google_api_key
            logging.info("Google API Key loaded from config and set as environment variable.")
        else:
            logging.warning("google_api_key not found in config.yaml.")
            # Optionally remove from env vars if previously set
            if 'GOOGLE_API_KEY' in os.environ: del os.environ['GOOGLE_API_KEY']


        if grok_api_key:
            os.environ['GROK_API_KEY'] = grok_api_key
            logging.info("Grok API Key loaded from config and set as environment variable.")
        else:
            logging.warning("grok_api_key not found in config.yaml.")
            if 'GROK_API_KEY' in os.environ: del os.environ['GROK_API_KEY']

        # No need to read API_keys/keys file anymore

        # Return the loaded config dictionary (which includes the keys)
        return config

    def start_workflow(self):
        """Starts the main processing workflow."""
        print("\n--- Art Agent Team Workflow ---")
        print("1. Input: I will ask for the folder containing images to process.")
        print("2. Research: Each image will be researched by one of 8 Research Agents.")
        print("3. Analysis: Based on research, style is determined & passed to a specialized Vision Agent.")
        print("4. Vision Processing: Dual-model analysis, scoring, masking.")
        print("5. Cropping: Style-specific intelligent 16:9 crop.")
        print("6. Output: Labeled, masked, and cropped versions saved.")
        print("7. Handoff: Cropped images queued for Upscale Agent.")
        print("---------------------------------")

        # 1. Get input folder from user
        folder_path = input(f"Enter the path to the image folder (default: {self.input_folder}): ")
        if not folder_path:
            folder_path = self.input_folder
        
        if not os.path.isdir(folder_path):
            logging.error(f"Invalid folder path: {folder_path}")
            print(f"Error: Folder not found at '{folder_path}'. Exiting.")
            return

        # 2. Fetch images and populate queue
        self._populate_image_queue(folder_path)
        if self.image_queue.empty():
            logging.warning(f"No images found in {folder_path}.")
            print("No images found to process.")
            return
            
        num_images = self.image_queue.qsize()
        print(f"Found {num_images} images to process.")

        # 3. Start a new ResearchAgent thread for each image
        print("Starting Research for each image...")
        for _ in range(num_images):
            thread = Thread(target=self._research_worker_new, name=f"ResearchThread", daemon=True)
            thread.start()
            # Note: _research_worker_new will be defined below to handle per-image ResearchAgent instantiation

        # No vision router needed; ResearchAgent handles VisionAgent directly
        
        # 5. Start Upscale Agent Router/Worker thread (placeholder)
        # upscale_router_thread = Thread(target=self._upscale_worker, args=(num_images,), name="UpscaleWorker", daemon=True)
        # upscale_router_thread.start()

        # 6. Wait for all images to be processed
        self.processed_queue.join() # Wait until task_done() called for all items

        print(f"--- Workflow Complete: {num_images} images processed ---")
        logging.info("DocentAgent workflow finished.")
        
        # Optional: Wait for threads to finish cleanly (though daemon=True means they exit with main)
        # for t in self.research_threads:
        #     t.join(timeout=10)
        # vision_router_thread.join(timeout=10)

    def _populate_image_queue(self, folder_path):
        """Finds images in the folder and adds them to the queue."""
        logging.info(f"Scanning folder for images: {folder_path}")
        count = 0
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_path = os.path.join(folder_path, filename)
                self.image_queue.put(image_path)
                count += 1
        logging.info(f"Added {count} images to the processing queue.")

    def _research_worker(self, research_agent: Any): # Use Any type hint due to dynamic loading
        """Worker function for ResearchAgent threads."""
        agent_id = research_agent.agent_id
        logging.info(f"ResearchAgent {agent_id} started.")
        while True:
            try:
                image_path = self.image_queue.get(block=False) # Non-blocking get
                logging.info(f"ResearchAgent {agent_id} processing: {os.path.basename(image_path)}")
                
                # Perform research
                artwork_info = research_agent.research_and_describe(image_path)
                
                if artwork_info:
                    self.vision_input_queue.put(artwork_info) # Put result onto the vision queue
                else:
                    logging.error(f"ResearchAgent {agent_id} failed for {os.path.basename(image_path)}")
                    # Mark task as done even on failure to prevent deadlock
                    self.processed_queue.put(f"FAILED_RESEARCH: {os.path.basename(image_path)}") 
                    
                self.image_queue.task_done()
                logging.info(f"ResearchAgent {agent_id} finished: {os.path.basename(image_path)}")
                
            except Empty:
                logging.info(f"ResearchAgent {agent_id}: No more images in queue. Exiting.")
                break # Exit thread when queue is empty
            except Exception as e:
                logging.exception(f"ResearchAgent {agent_id} encountered an error: {e}")
                # Try to mark task done if possible, otherwise log and exit
                try:
                    self.image_queue.task_done()
                except ValueError: # If task already marked done or item not acquired
                    pass
                self.processed_queue.put(f"ERROR_RESEARCH: {os.path.basename(image_path if 'image_path' in locals() else 'unknown')}")
                break # Exit thread on error

    def _research_worker_new(self, image_path: str):
        """New worker function that creates a new ResearchAgent instance for each image and handles the workflow."""
        try:
            # Create a new ResearchAgent instance for this image
            # Pass the vision_agent_classes dictionary to the ResearchAgent constructor
            research_agent = ResearchAgent(self.config, vision_agent_classes)
            # Call the correct method which now handles vision processing internally
            research_agent.research_and_process(image_path)
            # No need to check artwork_info or handle vision agent here, it's done in research_and_process
        except Exception as e:
            logging.exception(f"Error in new research worker for image {os.path.basename(image_path)}: {e}")

    # Remove unused queues and related logic
    # Remove _research_worker (old version)
    # Remove processed_queue reference in start_workflow
    # Remove _vision_router method entirely

    # Cleaned __init__
    def __init__(self, config_path='art_agent_team/config/config.yaml'):
        self.config = self._load_config(config_path)
        if not self.config:
            raise ValueError("Failed to load configuration.")
        self.input_folder = self.config.get('input_folder', 'input')
        self.workspace_folder = self.config.get('workspace_folder', 'workspace')
        self.output_folder = self.config.get('output_folder', 'output')
        os.makedirs(self.input_folder, exist_ok=True)
        os.makedirs(self.workspace_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)
        self.image_queue = Queue()
        # self.upscale_input_queue = Queue() # Keep if upscale is still needed (commented out for now)
        logging.info("DocentAgent initialized.")

    def start_workflow(self):
        print("\n--- Art Agent Team Workflow ---")
        print("1. Input: I will ask for the folder containing images to process.")
        print("2. Research and Analysis: Each image is researched and analyzed by a new ResearchAgent instance, which directly handles VisionAgent processing.")
        print("3. Output: Labeled, masked, and cropped versions saved.")
        print("4. Handoff: Cropped images queued for Upscale Agent if applicable.")
        print("---------------------------------")
        folder_path = input(f"Enter the path to the image folder (default: {self.input_folder}): ")
        if not folder_path:
            folder_path = self.input_folder
        if not os.path.isdir(folder_path):
            logging.error(f"Invalid folder path: {folder_path}")
            print(f"Error: Folder not found at '{folder_path}'. Exiting.")
            return
        self._populate_image_queue(folder_path)
        if self.image_queue.empty():
            logging.warning(f"No images found in {folder_path}.")
            print("No images found to process.")
            return
        num_images = self.image_queue.qsize()
        print(f"Found {num_images} images to process.")
        print("Starting processing for each image...")
        for _ in range(num_images):
            thread = Thread(target=self._research_worker_new, args=(self.image_queue.get(),), name=f"ImageProcessorThread", daemon=True)
            thread.start()
        # Wait for threads to complete (need a mechanism to track threads)
        # Simple join on the queue might not be sufficient if threads exit early on error
        # Placeholder: Add proper thread joining/management if needed
        # For now, assume threads complete or handle errors internally
        # self.image_queue.join() # This might hang if task_done() isn't called reliably
        # Instead, let's manage threads directly
        active_threads = []
        for _ in range(num_images):
            image_path = self.image_queue.get()
            thread = Thread(target=self._research_worker_new, args=(image_path,), name=f"ImageProcessorThread-{os.path.basename(image_path)}", daemon=True)
            thread.start()
            active_threads.append(thread)
            self.image_queue.task_done() # Mark task done after starting thread

        # Wait for all threads to finish
        for thread in active_threads:
            thread.join()
        print(f"--- Workflow Complete: {num_images} images processed ---")
        logging.info("DocentAgent workflow finished.")

    def _populate_image_queue(self, folder_path):
        logging.info(f"Scanning folder for images: {folder_path}")
        count = 0
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_path = os.path.join(folder_path, filename)
                self.image_queue.put(image_path)
                count += 1
        logging.info(f"Added {count} images to the processing queue.")

    # Remove old _research_worker method
    # Remove _vision_router method (already removed conceptually)

    # Placeholder for Upscale Worker (if needed)
    # def _upscale_worker(self, total_images: int):
    #     logging.info("UpscaleWorker started.")
    #     processed_upscale_count = 0
    #     while processed_upscale_count < total_images:
    #         try:
    #             cropped_path = self.upscale_input_queue.get(timeout=5)
    #             if self.upscale_agent:
    #                 # Call upscale agent
    #                 # self.upscale_agent.process(cropped_path)
    #                 logging.info(f"UpscaleWorker: (Simulated) Processing {os.path.basename(cropped_path)}")
    #                 time.sleep(0.2) # Simulate work
    #             else:
    #                 logging.warning("UpscaleAgent not initialized, skipping upscale.")
    #
    #             self.upscale_input_queue.task_done()
    #             self.processed_queue.put(f"COMPLETED: {os.path.basename(cropped_path)}") # Signal final completion
    #             processed_upscale_count += 1
    #
    #         except Empty:
    #             # Check if vision router is done
    #             # Need a more robust way to know when upstream is finished
    #             if not vision_router_thread.is_alive() and self.upscale_input_queue.empty():
    #                  logging.info("UpscaleWorker: Vision router finished and queue empty. Exiting.")
    #                  break
    #             continue
    #         except Exception as e:
    #             logging.exception(f"UpscaleWorker encountered an error: {e}")
    #             self.upscale_input_queue.task_done()
    #             self.processed_queue.put(f"ERROR_UPSCALE: {os.path.basename(cropped_path if 'cropped_path' in locals() else 'unknown')}")
    #             # Decide if loop should continue on error

    #     logging.info("UpscaleWorker finished.")