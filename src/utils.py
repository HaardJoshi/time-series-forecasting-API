# src/utils.py

import yaml
from pathlib import Path

def read_config(config_path: Path = Path("config.yaml")) -> dict:
    """
    Reads a YAML configuration file and returns its contents as a dictionary.

    Args:
        config_path (Path): The path to the YAML configuration file.

    Returns:
        dict: The configuration parameters.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print("Configuration file loaded successfully.")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return None
    except Exception as e:
        print(f"Error reading the configuration file: {e}")
        return None