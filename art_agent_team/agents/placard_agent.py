import logging
import os
from PIL import Image, ImageDraw, ImageFont

class PlacardAgent:
    def __init__(self, config):
        self.config = config
        self.output_folder = config.get('output_folder', 'output') # Use .get for safer config access
        # Placeholder for tool analysis findings and selected libraries/APIs
        # Pillow is expected to be sufficient for image manipulation and text overlay.

    def add_placard(self, image_path, artwork_info):
        """
        Adds a museum-style placard to the lower-right corner of the image.
        artwork_info is expected to be a dictionary with 'title', 'artist', 'date'.
        """
        logging.info(f"Adding placard to image: {os.path.basename(image_path)}")
        output_filename = f"placarded_{os.path.basename(image_path)}"
        output_path = os.path.join(self.output_folder, output_filename)

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder, exist_ok=True) # Use exist_ok=True

        try:
            img = Image.open(image_path).convert("RGBA") # Ensure alpha channel for transparency
            img_width, img_height = img.size

            # Placard dimensions and position (needs refinement based on desired look)
            placard_width = int(img_width * 0.3) # Example: 30% of image width
            placard_height = int(img_height * 0.15) # Example: 15% of image height
            padding = int(img_width * 0.02) # Example: 2% of image width for padding

            placard_x = img_width - placard_width - padding
            placard_y = img_height - placard_height - padding

            # Create a transparent image for the placard
            placard_img = Image.new('RGBA', (placard_width, placard_height), (0, 0, 0, 128)) # Semi-transparent black
            draw = ImageDraw.Draw(placard_img)

            # Define text and font (needs refinement)
            title = f"Title: {artwork_info.get('title', 'N/A')}"
            artist = f"Artist: {artwork_info.get('artist', 'N/A')}"
            date = f"Date: {artwork_info.get('date', 'N/A')}"
            text_lines = [title, artist, date]

            # TODO: Choose a suitable font and size. May need to dynamically adjust size.
            try:
                font = ImageFont.truetype("arial.ttf", int(placard_height / 6)) # Example font and size
            except IOError:
                font = ImageFont.load_default()
                logging.warning("arial.ttf not found, using default font.") # Use logging.warning

            text_color = (255, 255, 255, 255) # White text

            # Draw text on the placard (needs refinement for positioning and wrapping)
            text_y = padding
            for line in text_lines:
                # TODO: Implement text wrapping if necessary
                draw.text((padding, text_y), line, fill=text_color, font=font)
                text_y += font.getbbox(line)[3] + 5 # Move down for next line (simple spacing)


            # Paste the placard onto the main image
            img.paste(placard_img, (placard_x, placard_y), placard_img)

            # Save the final image
            img.save(output_path)
            logging.info(f"Image with placard saved to: {output_path}")
            return output_path

        except Exception as e:
            logging.exception(f"Error adding placard to image {image_path}: {e}") # Use logging.exception
            return None


if __name__ == "__main__":
    # Example Usage (for testing the agent individually)
    # In the final application, this will be orchestrated by main.py
    sample_config = {
        'output_folder': 'output' # Relative path
    }
    # Ensure output folder exists for standalone testing
    os.makedirs(sample_config['output_folder'], exist_ok=True)

    # Use the provided test image path for testing
    test_image_path = '/Users/nathaniel.cannon/Documents/VScodeWork/Art_AI/test_image/Tomioka Soichiro - Trees (1961).jpeg'

    # Simulate artwork info that would come from the ResearchAgent
    test_artwork_info = {
        "title": "Trees",
        "artist": "Tomioka Soichiro",
        "date": "1961"
    }

    # Ensure the test image exists
    if not os.path.exists(test_image_path):
        logging.error(f"Test image not found at: {test_image_path}")
    else:
        agent = PlacardAgent(sample_config)
        final_image_path = agent.add_placard(test_image_path, test_artwork_info)

        if final_image_path:
            logging.info(f"Placard added and saved to: {final_image_path}")
        else:
            logging.error(f"Failed to add placard to: {test_image_path}")