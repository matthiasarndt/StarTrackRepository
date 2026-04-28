import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

# TODO: In future this will NOT return a dictionary, but rather a dataclass which can be read in directly as an object.

class Config:
    def __init__(self, output_directory="."):

        self.config_path = Path(output_directory) / "project_config.json"

        # Tracked Parameters
        self.stack_shape = None
        self.frame_shape = None
        self.stack_path = None
        self.stacked_img_path = None
        self.reference_frame = None
        self.threshold = None
        self.ref_star_detect_pixels = None
        self.add_star_detect_pixels = None
        self.add_tuning_multiplier = None

    def write_from_object(self, source):

        # Generate list with all stored parameters (every attribute except self.config_path)
        parameter_filter = [key for key in vars(self) if key != 'config_path']

        # Filter attributes from source object to those being tracked in the config
        for key in parameter_filter:
            if hasattr(source, key):
                new_value = getattr(source, key)
                setattr(self, key, new_value)

        # Convert inputs dataclass to dict
        project_parameters = {
            key: asdict(getattr(self, key)) if is_dataclass(getattr(self, key))
            else getattr(self, key)
            for key in parameter_filter
        }

        with open(self.config_path, "w") as json_file:
            json.dump(project_parameters, json_file, indent=4, default=str)

    def read_parameters(self):

        # Find json file and raise error if not found
        json_path = self.config_path
        if not json_path.is_file():
            raise FileNotFoundError(f"Tuned parameters not found at: {json_path}")
        with open(json_path, "r") as f:
            data = json.load(f)

        return data
