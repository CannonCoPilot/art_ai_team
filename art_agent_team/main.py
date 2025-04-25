import sys
import os
import logging
from art_agent_team.docent_agent import DocentAgent

# Configure basic logging (can be further configured in config.yaml later)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the Art Agent Team workflow.")
    parser.add_argument("--input_folder", type=str, default="input", help="Path to the input folder containing images.")
    args = parser.parse_args()

    docent = DocentAgent(config_path='art_agent_team/config/config.yaml')
    docent.input_folder = args.input_folder  # Set the input folder from arguments
    docent.start_workflow()