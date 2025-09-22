
# dependencies
import os
import psutil
from pathlib import Path
from StarTrack import AstroPhoto
from multiprocessing import freeze_support

# determine inputs
run_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = Path(run_dir) / "raw_data_m82"
n_aligning_stars = 5 # recommended to be 5, maximum value is 9
t_value = 240 # manually set this based on the brightness of the starfield
# verbosity = 0: minimal information, tracking overall status
# verbosity = 1: figures of identified stars
# verbosity = 2: debugging information [not all information is printed, some debugging is found by individually running relevant scripts]

# determine how many cores to use, in the example below n-1 available physical cores are used.
n_physical_cores = psutil.cpu_count(logical=False)
n_cores = max(1, n_physical_cores - 1)

# create image object from astro photo
image = AstroPhoto(data_directory=data_dir, n_aligning_stars=n_aligning_stars,verbosity=0,ref_frame_name="L_0168.jpg",threshold_value=t_value,max_cores=n_cores)

if __name__ == "__main__":

    freeze_support()
    image.align_frames()
    image.stack_aligned_frames()
