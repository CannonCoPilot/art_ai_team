import logging
import os
import json
import time
import random # Import random for ID generation
# Removed duplicate import from line 6
try:
    import google.generativeai as genai
    # We will import specific types later, closer to usage
    logging.info("Successfully imported google.generativeai base package.")
except ImportError:
    logging.warning("google-generativeai package not found or import failed. Image search functionality will be limited.")
    genai = None # Set to None if import fails
# Removed duplicate import from line 14
from groq import Groq # Import Groq client
from typing import List, Dict, Optional, Any

# Placeholder for search API clients
# from googleapiclient.discovery import build

class ResearchAgent:
    """Researches artwork using web search and LLMs."""

    def __init__(self, config, vision_agent_classes):
        self.config = config
        self.agent_id = random.randint(10000, 99999) # Generate unique 5-digit ID
        self.vision_agent_classes = vision_agent_classes # Store the vision agent classes
        self.search_api_key = config.get('google_search_api_key')
        self.search_engine_id = config.get('google_search_engine_id')
        
        # Configure Grok (OpenAI‑compatible) key
        grok_key = config.get('grok_api_key')
        if grok_key:
            try:
                import openai
                openai.api_key = grok_key
            except ImportError:
                logging.warning("openai not installed; cannot set openai.api_key.")
        else:
            logging.warning("grok_api_key missing in config for ResearchAgent.")

        # Configure Gemini (Google) key
        gemini_key = config.get('google_api_key')
        if gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
            except ImportError:
                logging.warning("google.generativeai not installed; cannot configure Gemini.")
        else:
            logging.warning("google_api_key missing in config for ResearchAgent.")

        # Initialize Grok LLM client
        grok_api_key = self.config.get('grok_api_key')
        logging.debug(f"ResearchAgent {self.agent_id}: Loaded Groq API key: {grok_api_key}")  # Add debugging log for Groq API key
        if grok_api_key:
            try:
                self.llm_client = Groq(api_key=grok_api_key)
                self.llm_model_name = "grok-3-mini-fast-beta" # As requested
                logging.info(f"ResearchAgent {self.agent_id}: Groq client initialized with model {self.llm_model_name}.") # Re-add agent_id
            except Exception as e:
                logging.error(f"ResearchAgent {self.agent_id}: Failed to initialize Groq client: {e}") # Re-add agent_id
                self.llm_client = None
        else:
            logging.error(f"ResearchAgent {self.agent_id}: No Groq API key found.") # Re-add agent_id
            self.llm_client = None
            
        # Placeholder for search service client
        self.search_service = None
        # TODO: Initialize search service (e.g., Google Search API) if keys are available
        # if self.search_api_key and self.search_engine_id:
        #     try:
        #         self.search_service = build("customsearch", "v1", developerKey=self.search_api_key)
        #         logging.info(f"ResearchAgent {self.agent_id}: Google Search client initialized.") # Ensure agent_id is present
        #     except Exception as e:
        #         logging.error(f"ResearchAgent {self.agent_id}: Failed to initialize Google Search client: {e}") # Ensure agent_id is present
        # else:
        #     logging.warning(f"ResearchAgent {self.agent_id}: Google Search API key or Engine ID missing. Web search disabled.") # Ensure agent_id is present

    def _call_llm(self, prompt: str, is_json_output: bool = False) -> Optional[str]:
        """Helper function to call the configured LLM."""
        if not self.llm_client:
            logging.error(f"ResearchAgent {self.agent_id}: LLM client not available.") # Re-add agent_id
            return None
        
        messages = [{"role": "user", "content": prompt}]
        try:
            response_format_config = {"type": "json_object"} if is_json_output else None
            chat_completion = self.llm_client.chat.completions.create(
                messages=messages,
                model=self.llm_model_name,
                temperature=0.3, # Lower temp for more deterministic research tasks
                max_tokens=1024,
                response_format=response_format_config
            )
            result = chat_completion.choices[0].message.content
            # Simple clean up for potential markdown ```json ... ```
            if is_json_output:
                result = result.strip().removeprefix("```json").removesuffix("```").strip()
            return result
        except Exception as e:
            logging.error(f"ResearchAgent {self.agent_id}: LLM call failed: {e}") # Re-add agent_id
            return None

    def _perform_text_search(self, query: str) -> List[Dict]:
        """Performs text-based web search using Grok 3 API."""
        logging.info(f"ResearchAgent {self.agent_id}: Performing text search for query: '{query}' (Grok 3)") # Re-add agent_id
        try:
            from openai import OpenAI
        except ImportError:
            logging.error("openai package is required for Grok 3 API search.")
            return []

        grok_api_key = self.config.get('grok_api_key')
        if not grok_api_key:
            logging.error("No Grok API key found for text search.")
            return []

        client = OpenAI(
            api_key=grok_api_key,
            base_url="https://api.x.ai/v1",
        )

        prompt = (
            f"Perform a web search for the following artwork or artist query. "
            f"Return a JSON array of 3-5 relevant search results, each with 'title', 'link', and 'snippet'.\n"
            f"Query: {query}\n"
            f"Respond ONLY with the JSON array."
        )

        try:
            completion = client.chat.completions.create(
                model="grok-3-beta",
                messages=[{"role": "user", "content": prompt}],
            )
            import json as _json
            # Try to parse the response as JSON
            content = completion.choices[0].message.content
            # Remove markdown if present
            if content.strip().startswith("```json"):
                content = content.strip()[7:]
            if content.strip().endswith("```"):
                content = content.strip()[:-3]
            results = _json.loads(content)
            if isinstance(results, list):
                logging.info(f"ResearchAgent {self.agent_id}: Grok search returned {len(results)} results.") # Add missing agent_id
                return results
            else:
                logging.warning(f"ResearchAgent {self.agent_id}: Grok search did not return a list. Raw content: {content}") # Re-add agent_id
                return []
        except Exception as e:
            logging.error(f"ResearchAgent {self.agent_id}: Grok 3 API search failed: {e}") # Re-add agent_id
            return []

    def _perform_image_search(self, image_path: str) -> List[Dict]:
        """Performs image-based search using Gemini + Google Search API tools."""
        logging.info(f"ResearchAgent {self.agent_id}: Performing image search for: {os.path.basename(image_path)} (Gemini + Google Search)") # Re-add agent_id
        # Assuming google.generativeai (genai) is imported at the top level now
        if not genai: # Check if top-level import failed
             logging.error("google.generativeai package failed to import earlier. Cannot perform image search.")
             return []
        try:
            # Import specific types here, closer to usage
            from google.generativeai.types import Tool, GenerateContentConfig, GoogleSearch
            # Check if Tool was successfully imported (it might be None if top-level import failed partially)
            if Tool is None:
                 raise ImportError("Tool type not available from google.generativeai.types")
        except ImportError:
            logging.error("google.generativeai types (Tool, GenerateContentConfig, GoogleSearch) could not be imported. Ensure the package installation is complete and correct.")
            return []

        google_api_key = self.config.get('google_api_key')
        if not google_api_key:
            logging.error("No Google API key found for Gemini search.")
            return []

        genai.configure(api_key=google_api_key)
        client = genai.Client()
        model_id = "gemini-2.0-flash"

        google_search_tool = Tool(
            google_search=GoogleSearch()
        )

        # Use the image filename as the search query for now
        query = f"Find web information about the artwork or artist depicted in the image file: {os.path.basename(image_path)}. Return a JSON array of 3-5 relevant search results, each with 'title', 'link', and 'snippet'. Respond ONLY with the JSON array."

        try:
            response = client.models.generate_content(
                model=model_id,
                contents=query,
                config=GenerateContentConfig(
                    tools=[google_search_tool],
                    response_modalities=["TEXT"],
                )
            )
            # Try to extract JSON from the response
            content = ""
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text"):
                    content += part.text
            import json as _json
            # Remove markdown if present
            if content.strip().startswith("```json"):
                content = content.strip()[7:]
            if content.strip().endswith("```"):
                content = content.strip()[:-3]
            results = _json.loads(content)
            if isinstance(results, list):
                logging.info(f"ResearchAgent {self.agent_id}: Gemini search returned {len(results)} results.") # Re-add agent_id
                return results
            else:
                logging.warning(f"ResearchAgent {self.agent_id}: Gemini search did not return a list. Raw content: {content}") # Re-add agent_id
                return []
        except Exception as e:
            logging.error(f"ResearchAgent {self.agent_id}: Gemini image search failed: {e}") # Re-add agent_id
            return []

    def _summarize_results(self, search_results: List[Dict], query: str) -> str:
        """Uses LLM to summarize search results."""
        if not search_results:
            return "No search results to summarize."

        context = f"Search query: '{query}'\n\nSearch Results:\n"
        for i, item in enumerate(search_results[:5]): # Summarize top 5
            context += f"{i+1}. Title: {item.get('title', 'N/A')}\n   Snippet: {item.get('snippet', 'N/A')}\n\n"

        prompt = f"Based ONLY on the following search results, provide a concise summary (2-3 sentences) identifying the artwork, artist, and potential date or style:\n\n{context}"
        summary = self._call_llm(prompt)
        return summary if summary else "Summary generation failed."

    def _compare_summaries(self, text_summary: str, image_summary: str) -> bool:
        """Uses LLM to compare summaries and confirm identity match."""
        prompt = f"""
        Compare the following two summaries. Do they likely refer to the same artwork?
        Respond with only "YES" or "NO".

        Text Search Summary: {text_summary}
        Image Search Summary: {image_summary}
        """
        confirmation = self._call_llm(prompt)
        match = confirmation == "YES" if confirmation else False
        logging.info(f"ResearchAgent {self.agent_id}: Comparison result: {'Match' if match else 'No Match'}") # Re-add agent_id
        return match

    def _generate_paragraph_description(self, combined_summary: str) -> str:
        """Uses LLM to generate a paragraph-length description."""
        prompt = f"Expand the following summary into a descriptive paragraph (4-6 sentences) suitable for an art enthusiast, highlighting key aspects of the artwork:\n\n{combined_summary}"
        description = self._call_llm(prompt)
        if description:
             logging.info(f"ResearchAgent {self.agent_id}: Generated paragraph description.") # Re-add agent_id
        return description if description else "Paragraph description generation failed."

    def _classify_style(self, description: str) -> str:
        """Uses LLM to classify the artwork style."""
        styles = ["abstract", "landscape", "still life", "surrealist", "genre-scene", "animal-scene", "portrait", "figurative", "religious/historical"]
        prompt = f"""
        Based on the following description, classify the artwork's style into ONE of these categories:
        {', '.join(styles)}

        Description: {description}

        Respond with ONLY the category name (e.g., landscape, portrait).
        """
        style = self._call_llm(prompt)
        if style and style.lower() in styles:
            logging.info(f"ResearchAgent {self.agent_id}: Classified style as: {style.lower()}") # Re-add agent_id
            return style.lower()
        else:
            logging.warning(f"ResearchAgent {self.agent_id}: LLM returned invalid style '{style}'. Defaulting to 'unknown'.") # Re-add agent_id
            return "unknown"

    def _identify_subjects(self, description: str, style: str) -> Dict[str, Any]:
        """Uses LLM to identify primary and secondary subjects based on style rules."""
        style_guidance = {
            "landscape": "Identify the main landscape feature (e.g., valley, mountain, river) as the primary subject.",
            "abstract": "Describe the most prominent abstract shapes/forms as the primary subject.",
            "surrealist": "Describe the central theme or most striking element of the scene as the primary subject.",
            "portrait": "Identify the person depicted as the primary subject.",
        }.get(style, "Identify the main figure, animal, or central object cluster as the primary subject.") # Default

        prompt = f"""
        Analyze the artwork description to identify subjects:

        Description: {description}
        Style: {style}

        Instructions:
        1. {style_guidance}
        2. Identify 1-3 secondary subjects based on visual prominence or contextual importance. List them.

        Respond ONLY in JSON format:
        {{
            "primary_subject": "...",
            "secondary_subjects": ["...", "..."]
        }}
        """
        subjects_json = self._call_llm(prompt, is_json_output=True)
        try:
            subjects = json.loads(subjects_json) if subjects_json else {}
            logging.info(f"ResearchAgent {self.agent_id}: Identified subjects: {subjects}") # Re-add agent_id
            return {
                "primary": subjects.get("primary_subject", "unknown"),
                "secondary": subjects.get("secondary_subjects", [])
            }
        except json.JSONDecodeError as e:
             logging.error(f"ResearchAgent {self.agent_id}: Failed to parse subjects JSON: {e}. Response: {subjects_json}") # Re-add agent_id
             return {"primary": "unknown", "secondary": []}
        except Exception as e:
            logging.error(f"ResearchAgent {self.agent_id}: LLM subject identification failed: {e}") # Re-add agent_id
            return {"primary": "unknown", "secondary": []}

    def _generate_structured_sentence(self, style: str, subjects: Dict[str, Any], description: str) -> str:
        """Uses LLM to generate the structured one-sentence description."""
        primary_subject = subjects['primary']
        secondary_subjects_str = ", ".join(subjects['secondary']) if subjects['secondary'] else "no prominent secondary subjects"

        prompt = f"""
        Generate a one-sentence description following this exact format:
        "This is a(n) {{style}} painting of a(n) {{primary_subject}} {{description_of_action}} set in a {{description_of_scene}}, together with {{secondary_subjects}}."

        Use the provided information:
        - Style: {style}
        - Primary Subject: {primary_subject}
        - Secondary Subjects: {secondary_subjects_str}
        - Full Description: {description}

        Infer the {{description_of_action}} (what the primary subject is doing or its state) and {{description_of_scene}} (the setting/background) based on the full description. Keep these phrases concise (3-7 words each).

        Respond with ONLY the completed sentence.
        """
        sentence = self._call_llm(prompt)
        if sentence and sentence.startswith("This is a(n)") and primary_subject in sentence:
             logging.info(f"ResearchAgent {self.agent_id}: Generated structured sentence.") # Re-add agent_id
             return sentence
        else:
             logging.warning(f"ResearchAgent {self.agent_id}: LLM generated invalid sentence format: {sentence}") # Re-add agent_id
             # Fallback construction
             return f"This is a(n) {style} painting of a(n) {primary_subject} set in a scene, together with {secondary_subjects_str}."

    def _determine_target_vision_agent(self, style: str) -> str:
        """Determines the appropriate VisionAgent class name based on style."""
        # Map styles to their corresponding VisionAgent script names (without .py)
        style_map = {
            "abstract": "vision_agent_abstract",
            "landscape": "vision_agent_landscape",
            "still life": "vision_agent_still_life",
            "surrealist": "vision_agent_surrealist",
            "genre-scene": "vision_agent_genre",
            "animal-scene": "vision_agent_animal",
            "portrait": "vision_agent_portrait",
            "figurative": "vision_agent_figurative",
            "religious/historical": "vision_agent_genre", # Group with Genre for now
            "unknown": "vision_agent" # Default fallback to the base agent
        }
        agent_script_name = style_map.get(style, "vision_agent")
        logging.info(f"ResearchAgent {self.agent_id}: Determined target VisionAgent script: {agent_script_name}.py for style: {style}") # Re-add agent_id
        return agent_script_name # Return the script name prefix

    def research_and_process(self, image_path: str) -> None:
        """
        Orchestrates the research and vision processing for a single image.
        Directly instantiates and runs the appropriate VisionAgent.
        """
        original_filename = os.path.basename(image_path)
        logging.info(f"ResearchAgent {self.agent_id}: Starting research and processing for {original_filename}") # Re-add agent_id

        # 1. Perform Searches
        search_query = os.path.splitext(original_filename)[0].replace('_', ' ').replace('-', ' ')
        text_results = self._perform_text_search(search_query)
        image_results = self._perform_image_search(image_path)

        # 2. Summarize Results
        text_summary = self._summarize_results(text_results, search_query)
        image_summary = self._summarize_results(image_results, f"Image search for {original_filename}")

        # 3. Combine and Generate Descriptions
        combined_summary = f"Text Search Summary:\n{text_summary}\n\nImage Search Summary:\n{image_summary}"
        paragraph_description = self._generate_paragraph_description(combined_summary)

        # 4. Classify Style and Identify Subjects
        style = self._classify_style(paragraph_description)
        subjects = self._identify_subjects(paragraph_description, style)

        # 5. Generate Structured Sentence
        structured_sentence = self._generate_structured_sentence(style, subjects, paragraph_description)

        # 6. Determine and Instantiate Target Vision Agent
        target_vision_agent_script = self._determine_target_vision_agent(style)
        target_class_name = "".join(part.capitalize() for part in target_vision_agent_script.split('_'))

        # Need to reference the dynamically loaded vision agent classes from docent_agent
        # This requires passing the dictionary or importing it here.
        # For now, assuming it's globally accessible or passed somehow (needs fix in DocentAgent)
        if target_class_name in self.vision_agent_classes: # Use the stored dictionary
            vision_agent_instance = self.vision_agent_classes[target_class_name](self.config) # Use the stored dictionary
            vision_output = vision_agent_instance.analyze_image(image_path, custom_prompt=paragraph_description)
            if vision_output:
                vision_agent_instance.save_analysis_outputs(image_path, vision_output, os.path.join(os.getcwd(), 'output'))  # Use absolute or relative path as needed
                logging.info(f"ResearchAgent {self.agent_id}: Vision processing complete for {original_filename}") # Re-add agent_id
            else:
                logging.error(f"ResearchAgent {self.agent_id}: Vision analysis failed for {original_filename}") # Re-add agent_id
        else:
            logging.error(f"ResearchAgent {self.agent_id}: Could not find VisionAgent class '{target_class_name}' for {original_filename}") # Re-add agent_id

        # No return value; processing is handled within this method

# Note: This class is intended to be run as part of the DocentAgent workflow.