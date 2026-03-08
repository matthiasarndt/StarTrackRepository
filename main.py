
import os
import psutil
from pathlib import Path
from StarTrack import FrameStack, Info
from multiprocessing import freeze_support

# verbosity = 0: minimal information, tracking overall status
# verbosity = 1: figures of identified stars
# verbosity = 2: debugging information [not all information is printed, some debugging is found by individually running relevant scripts]

if __name__ == "__main__":

    # Reload modules for each parallel spawn
    freeze_support()

    # Print StarTrack logo!
    Info.print_logo()

    # Inputs
    run_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = Path(run_dir) / "raw_data_horsehead"
    n_aligning_stars = 5  # recommended to be 5, maximum value is 9
    t_value = 240  # manually set this based on the brightness of the starfield, setting it to -1 will make StarTrack tune the value

    # Determine how many cores to use, in the example below n-1 available physical cores are used.
    n_physical_cores = psutil.cpu_count(logical=False)
    n_cores = max(1, n_physical_cores - 1)

    # Create image object from astrophoto
    astro_photo = FrameStack(data_directory=data_dir, n_aligning_stars=n_aligning_stars, verbosity=1, ref_frame_name="20210212210734 [ISO400] [60.2s] [f4.7] [288mm].NEF", threshold=t_value, max_cores=4)
    # astro_photo.compute_stack()
    astro_photo.convert_to_stacked_image()
