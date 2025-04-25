# Art Agent Team Plan

This document outlines the plan for developing a team of AI agents to process and enhance art images.

## Phase 1: Initial Setup & Vision Agent (Completed)

*   **Goal:** Create a basic workflow where a single agent analyzes an image, identifies key features, and performs an intelligent crop to a 16:9 aspect ratio.
*   **Agents:**
    *   `VisionAgent`: Analyzes image using Google Vision API, identifies objects/landmarks/labels, calculates feature importance, performs intelligent cropping, saves labeled and cropped outputs.
*   **Status:** Completed. Basic analysis and cropping implemented and tested. Enhanced with Gemini API integration, improved scoring, and detailed facial feature detection.

## Phase 2: Workflow Redesign & Specialization (In Progress)

*   **Goal:** Redesign the workflow for a multi-agent system with specialized roles for research, style-specific analysis, and upscaling. Implement asynchronous processing and dual-model analysis.
*   **Agents & Workflow:**
    1.  **`DocentAgent` (Orchestrator):**
        *   Interacts with the user: Prompts for image input path, explains processing steps.
        *   Manages Image Stack: Fetches images from the input folder (`/Users/nathaniel.cannon/Documents/VScodeWork/Art_AI/art_agent_team/tests/test_data/input`) and places them in a processing queue (`image_queue`).
        *   Manages Research Agents: Initializes and manages a pool of 8 `ResearchAgent` worker threads.
        *   Manages Vision Handoff: Receives results (including style) from `ResearchAgents` via `vision_input_queue`. Routes tasks to the appropriate specialized `VisionAgent` instance (implementation TBD - could be direct call or another queue/worker pool).
        *   Manages Upscale Handoff: Receives cropped image paths from `VisionAgents` via `upscale_queue`. Routes tasks to `UpscaleAgent` (implementation TBD).
        *   Uses LLM (`grok-3-mini-fast-beta`) for coordination if needed.
    2.  **`ResearchAgent` (8 Instances):**
        *   Pulls image path from `DocentAgent`'s `image_queue`.
        *   Performs Text Search: Uses image filename tokens for web search (using Google Search API or similar), summarizes findings using LLM (`grok-3-mini-fast-beta`).
        *   Performs Image Search: Uses the image for reverse image search (using appropriate API/tool), summarizes findings using LLM.
        *   Confirms Artwork Identity: Compares text and image search summaries using LLM.
        *   Generates Descriptions (using LLM):
            *   Creates a paragraph-length summary description.
            *   Creates a structured one-sentence description: "This is a(n) {style} painting of a(n) {primary_subject} {description_of_action} set in a {description_of_scene}, together with {secondary_subjects}."
        *   Determines Style (using LLM): Assigns one style from {abstract, landscape, still life, surrealist, genre-scene, animal-scene, portrait, figurative, religious/historical}.
        *   Identifies Subjects (using LLM): Determines primary and secondary subjects based on style rules.
        *   Handoff to Docent: Puts results (image path, descriptions, style, subjects) onto `DocentAgent`'s `vision_input_queue`.
    3.  **`VisionAgent` (8 Style-Specific Scripts):**
        *   Receives data (image path, descriptions, style, subjects) from `DocentAgent` (or router).
        *   Fetches the image file.
        *   Dual-Model Analysis:
            *   Sends customized prompts (incorporating `ResearchAgent` descriptions and tailored to the specific artwork style) to `grok-2-vision-1212` and `gemini-2.5-pro-preview-03-25`.
        *   Aggregates Results: Combines object identification results from both models into a unified set (handling potential conflicts/duplicates).
        *   Calculates Importance Scores (Style-Specific): Uses the refined scoring system with style-specific adjustments. Boosts primary subject score.
        *   Identifies Important Objects: Uses percentile thresholding.
        *   Generates Segmentation Masks: Creates masks only for the most important objects.
        *   Performs Style-Specific Cropping: Implements cropping logic tailored to the artwork style (Abstract, Landscape, Still Life, etc.) as defined in the plan.
        *   Saves Outputs: Generates labeled (boxes), masked (important objects), and cropped versions.
        *   Handoff to Docent: Places the cropped image path onto `DocentAgent`'s `upscale_queue`.
    4.  **`UpscaleAgent`:**
        *   Receives cropped image path from `DocentAgent`'s `upscale_queue`.
        *   Performs image upscaling (specific method TBD).
        *   Saves final upscaled image.
*   **Status:** Refactoring in progress. Specialized VisionAgent files created. ResearchAgent and DocentAgent refactoring pending. Style-specific logic implementation pending.

## Phase 3: Placard Generation & Final Output

*   **Goal:** Add placard generation and finalize the output format.
*   **Agents:**
    *   `PlacardAgent`: Receives analysis data, generates a museum-style placard.
*   **Status:** Not started.

## Functional Review (Pre-Refactor)

*   **`DocentAgent`:** Needs significant refactoring for async worker management (Research & Vision agents), queue-based handoffs, removal of outdated logic (LLM parsing, direct agent calls), and integration of 8 ResearchAgents. Needs model update to `grok-3-mini-fast-beta`.
*   **`ResearchAgent`:** Requires complete implementation of core logic: web/image search integration (API placeholders needed), LLM integration (`grok-3-mini-fast-beta`) for summarization/comparison/generation/classification, style/subject identification logic, and proper handoff via queue.
*   **`VisionAgent` (Template):** Needs Grok Vision API integration (replacing Vision API client), dual-model analysis (Gemini Pro + Grok Vision), result aggregation, style-based routing logic (or be replaced by specialized agents), primary subject score boosting, and handoff mechanism. Input method needs changing.
*   **Specialized `VisionAgents`:** Created but contain template code. Require customization of prompts, scoring, and cropping logic per style.

## Implementation Roadmap

1.  **Update Plan:** Add functional review and roadmap (Completed).
2.  **Dependencies:** Update `requirements.txt` (add `google-api-python-client`, `groq`). Run `pip install`.
3.  **`ResearchAgent` Refactor:**
    *   Implement core structure (init with ID, worker loop).
    *   Add LLM (`grok-3-mini-fast-beta`) init.
    *   Implement placeholder search functions.
    *   Implement LLM calls for all generation/classification tasks.
    *   Implement style determination logic.
    *   Implement handoff to `vision_input_queue`.
4.  **`DocentAgent` Refactor:**
    *   Update `__init__` for 8 `ResearchAgent` instances and new queues.
    *   Refine worker management loop.
    *   Implement basic routing logic (log target VisionAgent type).
    *   Update model to `grok-3-mini-fast-beta`.
5.  **`VisionAgent` Template Refactor:**
    *   Update `__init__` for Grok Vision (`grok-2-vision-1212`) & Gemini Pro (`gemini-2.5-pro-preview-03-25`).
    *   Modify `analyze_image` signature.
    *   Implement dual-model calling & basic aggregation.
    *   Implement primary subject score boosting.
    *   Add handoff logic (`upscale_queue`).
    *   Remove hardcoded input folder.
6.  **Update Specialized `VisionAgent` Files:** Copy refactored template code into the 8 specialized files, ensuring class names match filenames.
7.  **Customize Specialized `VisionAgents` (Iterative - One style at a time):**
    *   **Landscape:** Customize prompts, scoring (discount small objects), cropping (mask avoidance, bottom-up crop).
    *   **Abstract:** Customize prompts, scoring (shape-based), cropping (centroid-based).
    *   **Still Life:** Customize prompts, scoring (central cluster focus), cropping (preserve cluster centrality).
    *   **Surrealist:** Customize prompts, scoring (context focus), cropping (follow Landscape).
    *   **Portrait:** Customize prompts (facial features), scoring (discount background), cropping (prioritize face).
    *   **Figurative:** Customize prompts (faces, figures), scoring, cropping (prioritize faces then figures).
    *   **Genre/Religious/Historical:** Customize prompts, scoring (discount crowds/background), cropping (preserve action centrality).
    *   **Animal-Scene:** Customize prompts (animal features), scoring (discount background), cropping (prioritize subjects).
8.  **Implement `VisionAgent` Routing:** Update `DocentAgent` to instantiate/call the correct specialized `VisionAgent`.
9.  **Implement `UpscaleAgent`:** Basic upscale logic, pull from `upscale_queue`.
10. **Update `main.py`:** Orchestrate the full flow.
11. **Testing:** Write/update tests.

## Key Technologies

*   Python 3.12+
*   Google Gemini API (`google-generativeai`) - Models: `gemini-2.5-flash-preview-04-17`, `gemini-2.5-pro-preview-03-25`
*   Grok API (`groq`) - Models: `grok-2-vision-1212`, `grok-3-mini-fast-beta` (Requires integration)
*   Pillow (PIL Fork) for image manipulation
*   NumPy for numerical operations
*   PyYAML for configuration
*   Web search library (e.g., `google-api-python-client`, `requests`, `BeautifulSoup`, or `serpapi`)
*   Asynchronous programming library (e.g., `threading`, `queue`, potentially `asyncio`)


# To add: Focusing and Future Steps for the Roadmap
* A main aim of the project is to customize and optimize each VisionAgent variant to the nuanced differences between the genres of art that they analyze.  Focus on customizing, optimizing, and troubleshooting the object detection, object localization, object scoring, object bounding and masking, and the cropping logic.
* Focus on designing the VisionAgents' functions, prompts, and workflow algorithm to identify sets of the major high-ranking features of an image, and then decide how to minimize loss when cropping to the 16:9 aspect ratio.
* Focus on customizing the algorithms to identify the set of high ranking objects in an image, and then decide how to best preserve them when cropping to the 16:9 aspect ratio cropping boundary.
* We need to ensure VisionAgents use the newest and best LLMs for image analysis.  For now, always use 'gemini-2.5-pro-preview-03-25' and 'grok-2-vision-1212' for image analysis and artistic reasoning.
* Allow the "thinking" steps done by the VisionAgents, DocentAgent, ResearchAgent and other agents to be done by 'grok-3-mini-fast-beta' and should include 'reasoning: { effort: "high" }' in the prompt.

* We need to ensure that we are calling the correct tools for Gemini to identify objects in the image and retun bounded boxes on those objects. Import the necessary generative, reasoning, and image processing modules.
* The vision agents explicitly instruct the LLMs through prompting in order to identify things like animals, people, major landscape features, furniture, and so on.  
* Explicitly instruct the models to get bounding boxes, and then the agent calls functions to plot them onto the original image, label them
* VisionAgents will verify that all detected objects are given their own scores, that high-importance objects have bounding boxes, that high-importance objects have sophisticated high-polygon segmental masks
* VisionAgents will implement a cropping algorithm that leverages all of that object metadata (boxes, masks, scores, and flags like "primary_subject" or "prominent_object" to calculate the a cropping boundary that avoids cropping important features, maximizes the cropping boundary, and avoids leaving partially cropped mid-to-high scoring (aka "important") objects.
* When a single object takes up the majority of an image and needs to be bisected by the cropping boundary, the algorithm should identify faces and attempt to retain as many faces of high ranking objects within the cropping boundary.
* For optimizing the cropping logic, first implement a region-based algorithm that evaluates minimal bounding regions around the top N features (expanding them to the 16:9 frame)
* Then modify the cropping logic by integrating face-detection so that when a single or few large promary or secondary subjects dominate the image, the cropping boundary will include as many detected faces as possible.
* Create logic that can identify faces as their own scored objects.  These should be the highest scoring features.  The portrait and figurative vision agents should identify human faces and the animal-scene agent should identify animal faces.  When possible the cropping algorithm should attempt to keep face objects within the cropping boundary.
* Create tests that will output the number and types of objects identified by the vision model.  Have the test script present all of the information from the returned output of the model in a table format, and also save as a csv file, to compare those results with a visual inspection of the image.

* Explore and adopt sophisticated high-polygon segmental masking algorithms and techniques to minimize unnecessary penalties when cropping around objects.

* API calls sent to grok and gemini models by the vision agents will be essentially and functionally equivalent, allowing for the minor structural differences in the APIs.  This is so that 

Make all necessary changes.

# To add details to the agents and workflow