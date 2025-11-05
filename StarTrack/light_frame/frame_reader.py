
# dependencies
import numpy as np
import rawpy
from PIL import Image, ImageFilter
from pathlib import Path

class FrameReader:
    def __init__(self, state):
        # separate out inputs (immutable) from state (mutable, and will be used later on in processing)
        self.state = state
        self.inputs = self.state.inputs

    def read_rgb(self, *args):

        # print update
        if self.inputs.verbosity > 0: print(f"Loading {self.inputs.frame_directory}\\{self.inputs.frame_name}")

        # find image
        read_path = Path(self.inputs.frame_directory) / self.inputs.frame_name

        # decide on path and then import
        if read_path.suffix.lower() == ".nef":
            with rawpy.imread(str(read_path)) as raw:
                rgb_array_raw = raw.postprocess(use_camera_wb=True, output_bps=8)
            rgb_frame = Image.fromarray(rgb_array_raw)
        elif read_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            rgb_frame = Image.open(read_path)
        else:
            # raise erroer
            raise TypeError("Unsupported file type")

        return rgb_frame

    def pre_process(self,*args):

        # read data and extract mono:
        rgb_frame = self.read_rgb()
        mono_frame = rgb_frame.convert('L')
        self.state.mono_array = np.array(mono_frame)

        # threshold and blur the frame:
        process_frame = mono_frame.point(lambda p: 255 if p > self.inputs.threshold else 0)
        process_frame.filter(ImageFilter.GaussianBlur(radius=self.inputs.blur_radius))
        process_frame.point(lambda p: 255 if p > 100 else 0)

        # calculate the pixels to be cropped:
        # cropping is done to remove distortions effects on the edge of a frame from impacting the alignment process!
        masked_array = np.zeros_like(self.state.mono_array)
        crop_x = (1 - self.inputs.crop_factor) * np.shape(self.state.mono_array)[0]
        crop_x = int(crop_x)
        crop_y = (1 - self.inputs.crop_factor) * np.shape(self.state.mono_array)[1]
        crop_y = int(crop_y)

        # crop the image and mutate threshold array:
        masked_array[crop_x:-crop_x, crop_y:-crop_y] = np.array(process_frame)[crop_x:-crop_x, crop_y:-crop_y]
        self.state.threshold_array = masked_array

        # print completion confirmation
        if self.inputs.verbosity > 0: print(f"{self.inputs.frame_name} Processing complete: monochrome, threshold and crop applied")

        # debugging information
        if self.inputs.verbosity > 1:
            process_frame.show()
            mono_frame.show()

        return self



