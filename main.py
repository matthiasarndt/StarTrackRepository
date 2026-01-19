
# dependencies
import os
import psutil
from pathlib import Path
from StarTrack import FrameStack, Info
from multiprocessing import freeze_support

# verbosity = 0: minimal information, tracking overall status
# verbosity = 1: figures of identified stars
# verbosity = 2: debugging information [not all information is printed, some debugging is found by individually running relevant scripts]

if __name__ == "__main__":

    # reload modules for each parallel spawn
    freeze_support()

    # print StarTrack logo!
    Info.print_logo()

    # determine inputs
    run_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = Path(run_dir) / "raw_data_horsehead"
    n_aligning_stars = 5  # recommended to be 5, maximum value is 9
    t_value = 240  # manually set this based on the brightness of the starfield, setting it to -1 will make StarTrack tune the value

    # determine how many cores to use, in the example below n-1 available physical cores are used.
    n_physical_cores = psutil.cpu_count(logical=False)
    n_cores = max(1, n_physical_cores - 1)

    # create image object from astrophoto
    m82_data = FrameStack(data_directory=data_dir, n_aligning_stars=n_aligning_stars, verbosity=0, ref_frame_name="L_0099.NEF", threshold=t_value, max_cores=4)
    m82_data.compute_stack()
    m82_data.convert_to_stacked_image()
