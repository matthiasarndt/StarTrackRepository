
# TODO: Add a plotting feature, to show a light_frame with stars identified.

# Dependencies
from dataclasses import dataclass
from pathlib import Path
import tracemalloc
from StarTrack.light_frame.star_alignment_vectors import StarAlignmentVectors
from StarTrack.light_frame.star_filter import StarFilter
from StarTrack.light_frame.frame_reader import FrameReader
from StarTrack.light_frame.star_field import Starfield


class LightFrame:
    """Manages the analysis of a single astronomical image (light frame).

    This class acts as a central hub. It stores the image data and settings,
    then uses specialized tools to find stars, catalog them, and calculate
    alignment vectors used for stacking.

    Attributes:
        inputs (FrameInputs): A group of settings like file paths, star
            detection limits, and image thresholds.
        mono_array (np.array): The grayscale version of the image.
        n_clusters (int): The total number of stars found in the image.
        centroid_list (list): The (x, y) pixel locations of found stars.
        ref_vectors (list): Math vectors used to align this frame with others.

    Args:
        **kwargs: Settings passed to the FrameInputs dataclass (e.g.,
            frame_name, frame_directory, threshold).
    """

    @dataclass(frozen=True) # Frozen for immutability
    class FrameInputs:
        frame_name: str
        frame_directory: Path
        threshold: int = 254
        verbosity: int = 0
        star_detect_radius: int = 20
        star_detect_pixels: int = 50
        crop_factor: float = 0.85
        blur_radius: float = 3

    def __init__(self, **kwargs):

        # Inputs
        self.inputs = self.FrameInputs(**kwargs)
        self.threshold_array = None
        self.mono_array = None
        self.cluster_array = None

        # Outputs
        self.pixels_in_clusters = None
        self.n_clusters = None
        self.centroid_list = None
        self.magnitude_list = None
        self.ref_vectors = None
        self.ref_angles = None
        self.ref_star = None
        self.i_ref_star = None
        self.non_ref_stars = None

    def process(self):
        """Runs the full analysis pipeline on the image.

        Reads the image, filters for stars, counts them,
        catalogues star locations, and calculates alignment vectors.

        Returns:
            LightFrame: The current object with all star data filled in.
        """

        FrameReader(self).pre_process()
        StarFilter(self).local_density()
        Starfield(self).count_stars()
        Starfield(self).register_stars()
        StarAlignmentVectors(self).compute_from_biggest_star()

        return self

    def register_stars_and_alignment_vectors(self):

        Starfield(self).register_stars()
        StarAlignmentVectors(self).compute_from_biggest_star()

        return self

    def get_frame_shape(self):
        """Gets the pixel height and width of the frame.

        Runs the image reader to prepare the data before checking the size.

        Returns:
            tuple: The (height, width) of the image in pixels.
        """

        FrameReader(self).pre_process()

        return self.mono_array.shape


if __name__ == "__main__":

    tracemalloc.start()

    data_dir = Path(r"D:\_Local\OneDrive\Astronomy\StarTrack\dev\raw_data_m82")
    verbosity = 1
    light = LightFrame(frame_directory=data_dir, frame_name="L_0173.jpg", verbosity=verbosity, star_detect_pixels=500.5, threshold=231)
    # light.tune_threshold() - Tuning functionality no longer included in light_frame object, so ths command is now defunct!

    # Memory profiling
    snapshot = tracemalloc.take_snapshot()
    for stat in snapshot.statistics('lineno')[:10]:
        print(stat)
    top_stats = snapshot.statistics('lineno')
    total = sum(stat.size for stat in top_stats)
    print(f"\nTotal memory allocated (tracemalloc): {total / (1024 ** 2):.2f} MiB")


