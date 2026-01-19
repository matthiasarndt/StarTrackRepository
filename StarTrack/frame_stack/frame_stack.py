
# Dependencies
import os
import math
import gc
import traceback
from os import mkdir

import numpy as np
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from collections import namedtuple
from pathlib import Path
from matplotlib import pyplot as plt
from tifffile import tifffile
from StarTrack import LightFrame
from StarTrack.frame_stack.coupled_frames import CoupledFrames
from StarTrack.light_frame.frame_reader import FrameReader

# Define named tuples required for parallelisation
frameTuningArgs = namedtuple("frameTuningArgs", ["frame", "n_desired_stars"])
additionalFrameAlignmentArgs = namedtuple("additionalFrameAlignmentArgs", ["i_layer", "image_name", "data_dir", "detect_threshold", "light_ref", "threshold"])

class FrameStack:
    """
    Manages the end-to-end image stacking pipeline for astrophotography data.

    This class handles directory indexing, multiprocessor frame alignment, and
    memory-efficient image stacking using disk-based memory mapping.

    Operations:
    - Indexes input directories and identifies the reference frame
    - Manages a high-performance memory map for large-scale image processing
    - Orchestrates parallel alignment and 16-bit image integration

    Arguments:
        **kwargs: Configuration settings for StackInputs, including data_directory,
        stack_directory, and processing constraints.

    Attributes:
        inputs (StackInputs): Dataclass containing user-defined processing parameters.
        output_directory (Path): The file path where results and metadata are stored.
        frame_list (list): List of all image filenames found in the source directory.
        frame_shape (tuple): The pixel dimensions of the images being processed.
    """

    # TODO: Freeze for immutability in a future release.
    @dataclass(frozen=False)
    class StackInputs:
        data_directory: Path
        stack_directory: Path = Path('C:\StarTrack')
        ref_frame_name: str = None
        verbosity: int = 0
        n_aligning_stars: int = 5
        threshold: int = 240
        stacking_method: str = 'mean' # TODO: Implement the following - 'sigma_clipping', 'mean_pixel_rejection', 'median', 'SNR weighted mean average'
        max_cores: int = 5

    def __init__(self, **kwargs):

        self.inputs = self.StackInputs(**kwargs)
        self.frame_list = None
        self.ref_star_detect_pixels = None
        self.add_star_detect_pixels = None
        self.aligned_stack_read = None
        self.stacked_array = None
        self.stacked_frame_mono = None
        self.stacked_frame_rgb = None
        self.chunk_load_size_GB: int = 3

        # Determine output directory:
        self.output_directory = Path(self.inputs.data_directory) / "outputs"
        if not self.inputs.stack_directory.is_dir():
            mkdir(self.inputs.stack_directory)

        # Determine the reference image and therefore the dimensions of the reference image
        self.frame_list = _list_data_in_directory(self.inputs.data_directory)
        print(f"Indexing image data directory: {self.inputs.data_directory}")

        # Determine index of reference frame:
        if self.inputs.ref_frame_name is None:
            self.i_ref_frame = 0
        else:
            self.i_ref_frame = self.frame_list.index(self.inputs.ref_frame_name)

        # Load reference frame, find data dimensions, delete reference frame from memory. NOTE: this is a temporary solution, will be refactored for efficiency
        light_temp = LightFrame(frame_directory=self.inputs.data_directory, frame_name=self.frame_list[self.i_ref_frame], verbosity=0)
        self.frame_shape = light_temp.get_frame_shape()
        del light_temp

    def compute_stack(self):
        """
        Executes the frame alignment pipeline and initializes the memory-mapped data stack.

        This method synchronizes the input dataset by aligning all secondary frames to a
        selected reference frame using parallel processing.

        Operations:
        - Prepares the output directory and removes previous results
        - Creates a persistent disk-based memory map to handle large image volumes
        - Loads cached parameters or prompts the user for a new calibration
        - Inserts the primary reference frame as the base layer of the stack
        - Distributes frame alignment across multiple CPU cores for faster execution

        Arguments:
            self: The class instance containing metadata and processing settings.

        Returns:
            None: Aligned data is saved directly to the stack_aligned_array.dat file.
        """

        self.output_directory.mkdir(parents=True, exist_ok=True)
        _delete_previous_output(directory=self.output_directory, output_list=["reference_frame_mono.jpg", "reference_frame_rgb.jpg", "stacked_frames_mono.jpg", "stacked_frames_rgb.jpg"])

        # Create memory mapped arrays
        aligned_stack_write = np.memmap(filename=self.inputs.stack_directory / "stack_aligned_array.dat",
                                             dtype=np.uint8,
                                             mode='w+',
                                             shape=(len(self.frame_list), 4, self.frame_shape[0], self.frame_shape[1]))

        # By default, tuning is set to 1:
        flag_tuning = True

        # Check if tuned parameters already exist - check if the user doesn't want to tune, and set to 0 if true
        if (Path(self.output_directory) / "tuned_parameters.json").is_file():
            while True:
                response = input("Previously tuned parameters found for this dataset! Continue? (Y), or retune (N)?: ").strip().lower()
                if response in ("y", "n"):
                    if response == "y":
                        flag_tuning = False
                        break
                    break
                print("Please enter 'Y' or 'N'.")

        if flag_tuning:
            light_ref = self._tune_parameters()
        else:
            light_ref = self._load_parameters()

        # Store light_ref mono & r/g/b data
        ref_rgb_frame = FrameReader.read_rgb(light_ref)
        aligned_stack_write[0, :, :, :] = np.stack([
                                                    light_ref.mono_array,
                                                    np.array(ref_rgb_frame.getchannel('R')),
                                                    np.array(ref_rgb_frame.getchannel('G')),
                                                    np.array(ref_rgb_frame.getchannel('B'))
                                                    ])

        aligned_stack_write.flush()

        gc.collect()

        # Remove the reference image from the image_list
        align_frame_list = self.frame_list[:self.i_ref_frame] + self.frame_list[self.i_ref_frame + 1:]

        # Define inputs for the remainder of frames to be processed, with a named tuple as input structure and list comprehension
        args_tuple = [additionalFrameAlignmentArgs(i_layer, align_frame_list[i_layer], self.inputs.data_directory, self.add_star_detect_pixels, light_ref, self.inputs.threshold)
                      for i_layer in range(0, len(align_frame_list))]

        with ProcessPoolExecutor(max_workers=self.inputs.max_cores) as executor:

            # Wrap each process in try/except so that errors won't crash the pool. this has been implemented with multiprocessing futures
            futures = [executor.submit(_worker_align_additional_frame_wrapper, args) for args in args_tuple]

            for future in as_completed(futures):
                result = future.result()
                if "error" in result:
                    print(f"Failed to align {result['image_name']}")
                    continue  # skip this frame

                aligned_stack_write[result["i_layer"] + 1, :, :, :] = np.stack([result["mono"], result["r"], result["g"], result["b"]])
                aligned_stack_write.flush()

        del aligned_stack_write
        gc.collect()

    def convert_to_stacked_image(self):
        """
        Integrates the aligned image stack into high-bit-depth master images.

        This method performs the final stacking calculations using chunked processing
        to manage memory while collapsing the stack into 16-bit TIFFs.

        Operations:
        - Opens the aligned memory map in read-only mode to protect source data
        - Displays an interactive plot that updates as each image segment is processed
        - Divides the image into vertical slices to prevent system memory exhaustion
        - Upsamples data to 16-bit and removes failed alignment layers
        - Applies a mean stacking algorithm to reduce noise and maximize signal
        - Exports the final Mono and RGB results as high-detail TIFF files

        Arguments:
            self: The class instance containing the memory map and stacking settings.

        Returns:
            self: The class instance to allow for method chaining.
        """

        print("Stacking data ...")
        _delete_previous_output(self.output_directory,
                                output_list=["stacked_frame_mono.tiff", "stacked_frame_rgb.tiff"])

        # create memory mapped array to read data. a new variable is created to ensure this data is read only!
        aligned_stack_read = np.memmap(filename=self.inputs.stack_directory / "stack_aligned_array.dat",
                                       dtype=np.uint8,
                                       mode='r',
                                       shape=(len(self.frame_list), 4, self.frame_shape[0], self.frame_shape[1]))

        # pre define stacked array - 16 bit:
        stacked_array = np.empty((4, self.frame_shape[0], self.frame_shape[1]), dtype=np.float32)

        # set up interactive stacked image plot - allowing results to be shown "live":
        plt.ion()
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 8)
        image = ax.imshow(stacked_array[0], cmap='gray', vmin=0, vmax=255)
        plt.show()

        # calculate the number of chunks required:
        stack_size_bytes = aligned_stack_read.nbytes * 2 # multiplied by 2 as 16 bit uses double the storage of 8 bit
        stack_size_gb = stack_size_bytes / (1000 ** self.chunk_load_size_GB)
        n_chunks = math.ceil(stack_size_gb / self.chunk_load_size_GB)
        print(f"Aligned stack array requires {round(stack_size_gb, 2)}GB. Dividing stacking process into {n_chunks} {'chunk' if n_chunks == 1 else 'chunks'}, with a size of {self.chunk_load_size_GB}GB{' each' if n_chunks > 1 else ''}")

        # determine chunk shape:
        chunk_width = int(self.frame_shape[0] / n_chunks)
        chunk_start = 0
        chunk_end = chunk_width

        # loop through each chunk, read data, and stack:
        for i_chunk in range(1, n_chunks + 1):

            # read the chunk required from the memory map, ensuring it is 16 bit:
            chunk_array = aligned_stack_read[:, :, chunk_start:chunk_end, :].astype(np.float32)

            # remove any layers that have only zeros in them (indicating an unsuccessful frame alignment):
            chunk_array = chunk_array[~np.all(chunk_array == 0, axis=(1, 2, 3))]

            # apply stacking method (currently only "mean" implemented):
            if self.inputs.stacking_method == 'mean':
                stacked_array[:, chunk_start:chunk_end, :] = np.mean(chunk_array, axis=0)

            # update the chunk parameters:
            chunk_start = chunk_start + chunk_width
            chunk_end = chunk_end + chunk_width
            if chunk_end > self.frame_shape[0]: chunk_end = self.frame_shape[0] + 1

            # update image:
            image.set_data(stacked_array[0])
            fig.canvas.draw()
            fig.canvas.flush_events()

        # keep figure open until it is manually closed:
        plt.ioff()
        plt.show()

        # extract mono & rgb data, then save as tiff:
        stacked_array_mono = (stacked_array[0]*257).astype(np.uint16)
        stacked_array_rgb = np.stack([stacked_array[1]*257, stacked_array[2]*257, stacked_array[3]*257], axis=-1).astype(np.uint16)
        del stacked_array
        tifffile.imwrite(Path(self.output_directory) / "stacked_frame_mono.tiff", stacked_array_mono)
        tifffile.imwrite(Path(self.output_directory) / "processing_stacked_frame_mono.tiff", stacked_array_mono)
        del stacked_array_mono
        tifffile.imwrite(Path(self.output_directory) / "stacked_frame_rgb.tiff", stacked_array_rgb)
        tifffile.imwrite(Path(self.output_directory) / "processing_stacked_frame_rgb.tiff", stacked_array_rgb)
        del stacked_array_rgb
        print("... process complete!")
        print(f"Stacked frames saved as stacked_frame_mono.tiff and stacked_frame_rgb.tiff")

        return self

    def _tune_parameters(self):
        """
        Calibrates alignment settings and saves them for future use.

        Operations:
        - Calculates the optimal brightness threshold and star detection parameters
        - Saves the calibrated settings to a JSON file and exports reference images

        Arguments:
            self: The class instance.

        Returns:
            LightFrame: The fully tuned reference frame object.
        """

        _delete_previous_output(directory=self.output_directory, output_list=["tuned_parameters.json"])

        # tune threshold with light_tuning instance of LightFrame. delete once generated, to reduce memory usage:
        if self.inputs.threshold == -1:
            print(f"Threshold set to -1 -tuning threshold parameter...")
            tuned_threshold = LightFrame(frame_directory=self.inputs.data_directory,
                                      frame_name=self.frame_list[self.i_ref_frame],
                                      verbosity=0).tune_threshold()
            print(f"... process complete")
            self.inputs = replace(self.inputs, threshold=tuned_threshold)

        # create reference light frame instance:
        print(f"Tuning star detect pixels parameters...")
        light_tuning = LightFrame(frame_directory=self.inputs.data_directory,
                                  frame_name=self.frame_list[self.i_ref_frame],
                                  verbosity=self.inputs.verbosity,
                                  threshold=self.inputs.threshold)

        # tune star_detect_parameter:
        add_tuning_multiplier = 6
        ref_frame_args = frameTuningArgs(light_tuning, self.inputs.n_aligning_stars)
        add_frame_args = frameTuningArgs(light_tuning, self.inputs.n_aligning_stars * add_tuning_multiplier)
        with ProcessPoolExecutor(max_workers=2) as tuning_executor:
            light_ref_tuned, light_ref_tuned_for_additional = tuning_executor.map(_worker_process_with_tuning, [ref_frame_args, add_frame_args])

        # store tuned parameters:
        self.ref_star_detect_pixels = light_ref_tuned.inputs.star_detect_pixels
        self.add_star_detect_pixels = light_ref_tuned_for_additional.inputs.star_detect_pixels

        # create json file to store tuning parameters:
        tuned_parameters = {"reference_frame": self.frame_list[self.i_ref_frame],
                            "threshold": self.inputs.threshold,
                            "ref_star_detect_pixels": self.ref_star_detect_pixels,
                            "add_star_detect_pixels": self.add_star_detect_pixels}
        with open(self.output_directory / "tuned_parameters.json", "w") as json_file: json.dump(tuned_parameters,json_file, indent=4)

        # save solving parameters, reference frame image, & print to terminal:
        light_ref_tuned.mono_frame.save(Path(self.output_directory) / "reference_frame_mono.jpg")
        light_ref_tuned.rgb_frame.save(Path(self.output_directory) / "reference_frame_rgb.jpg")
        print(f"... star detection pixels tuned to {round(self.ref_star_detect_pixels,2)} for reference frame")
        print(f"... star detection pixels tuned to {round(self.add_star_detect_pixels,2)} for additional frames")

        return light_ref_tuned

    def _load_parameters(self):
        """
        Loads saved tuning settings from a JSON file to skip calibration.

        Operations:
        - Reads parameters from disk and updates instance settings
        - Returns a processed reference LightFrame using the loaded data

        Arguments:
            self: The class instance.

        Returns:
            LightFrame: The processed reference frame object.

        Raises:
            FileNotFoundError: If the settings file is not found.
        """

        # find json file and raise error if not found
        json_path = Path(self.output_directory) / "tuned_parameters.json"
        if not json_path.is_file():
            raise FileNotFoundError(f"Tuned parameters not found at: {json_path}")
        with open(json_path, "r") as f:
            data = json.load(f)

        # assign parameters from JSON to self
        self.frame_list[self.i_ref_frame] = data["reference_frame"]
        self.inputs.threshold = data["threshold"]
        self.ref_star_detect_pixels = data["ref_star_detect_pixels"]
        self.add_star_detect_pixels = data["add_star_detect_pixels"]

        # create light_ref with pre tuned variables and process
        light_ref_pre_tuned = LightFrame(frame_directory=self.inputs.data_directory,
                                            frame_name=self.frame_list[self.i_ref_frame],
                                            verbosity=self.inputs.verbosity,
                                            threshold=self.inputs.threshold,
                                            star_detect_pixels=self.ref_star_detect_pixels).process()

        return light_ref_pre_tuned


def _delete_previous_output(directory, output_list):

    # loop through each image
    for output_image in output_list:

        # if the image already exists, delete the previous output
        if output_image in os.listdir(directory): os.remove(os.path.join(directory, output_image))

def _list_data_in_directory(directory):

    image_files = []
    all_files = os.listdir(directory)
    for file in all_files:
        if os.path.isfile(Path(directory, file)):
            image_files.append(file)

    return image_files

def _worker_process_with_tuning(solver_initialisation_args):

    solve_frame = solver_initialisation_args.frame
    solve_frame.process_tuning_star_detect(n_desired_clusters=solver_initialisation_args.n_desired_stars)

    return solve_frame

def _worker_align_additional_frame_wrapper(arg_tuple):

    # try/catch to keep the pool running if there are errors with any individual frames
    try:

        # define and process addition frame
        frame_add = LightFrame(frame_directory=arg_tuple.data_dir,frame_name=arg_tuple.image_name,star_detect_pixels=arg_tuple.detect_threshold,threshold=arg_tuple.threshold)
        frame_add.process()

        # memory crashes occur if a memory cleanup is not performed as each frame is processed
        gc.collect()

        # align the frame
        i_layer, aligned_array, aligned_array_r, aligned_array_g, aligned_array_b = _align_single_frame(arg_tuple.i_layer, arg_tuple.light_ref, frame_add)

        # create a dict to return a successful results:
        return {"i_layer": i_layer,"mono": aligned_array, "r": aligned_array_r, "g": aligned_array_g, "b": aligned_array_b}

    except Exception as e:

        # generate diagnostics error dict
        return {"error": str(e),"traceback": traceback.format_exc(),"i_layer": getattr(arg_tuple, "i_layer", None),"image_name": getattr(arg_tuple, "image_name", None)}

def _align_single_frame(i_layer, light_ref, light_addition):

    # create object which couples the reference and additional light
    lights_paired = CoupledFrames(light_ref, light_addition)

    # align the additional frame with the reference frame
    lights_paired.compute_alignment_and_transform()
    print(f"Aligned image {light_addition.inputs.frame_name} with {light_ref.inputs.frame_name}")

    # store the resulting aligned image into the output array below
    return i_layer, lights_paired.addition_aligned_array_mono, lights_paired.addition_aligned_array_r, lights_paired.addition_aligned_array_g, lights_paired.addition_aligned_array_b