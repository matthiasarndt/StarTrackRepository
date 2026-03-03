
# dependencies
from scipy.optimize import least_squares, fsolve, bisect
from StarTrack.light_frame.star_alignment_vectors import StarAlignmentVectors
from StarTrack.light_frame.star_filter import StarFilter
from StarTrack.light_frame.frame_reader import FrameReader
from StarTrack.light_frame.star_field import Starfield
from dataclasses import dataclass
from pathlib import Path
from dataclasses import replace
import tracemalloc

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

    @dataclass(frozen=True) # frozen for immutability
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

        # Outputs:
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
        Starfield(self).catalogue_detected_stars()
        StarAlignmentVectors(self).compute_from_biggest_star()

        return self

    def process_tuning_star_detect(self, n_desired_clusters):
        """Finds the best star detection settings and analyses the frame.

        Adjusts 'star_detect_pixels' until the target number of stars is found,
        then catalogues the stars and calculates alignment vectors.

        Args:
            n_desired_clusters (int): How many stars you want to find.

        Returns:
            float: The setting value that found the desired number of stars.
        """

        tuned_star_detect_pixels = self.tune_star_detect_pixels(n_desired_clusters) # Calls FilterImage and CatalogueClusters to do this
        Starfield(self).catalogue_detected_stars()
        StarAlignmentVectors(self).compute_from_biggest_star()

        return tuned_star_detect_pixels

    def get_frame_shape(self):
        """Gets the pixel height and width of the frame.

        Runs the image reader to prepare the data before checking the size.

        Returns:
            tuple: The (height, width) of the image in pixels.
        """

        FrameReader(self).pre_process()

        return self.mono_array.shape

    def tune_threshold(self):
        """
        function: tunes to provide a good thresholding value:
        * implementation of a line solver, reducing threshold from 254 at increments of 4
        * stop when an acceptable spread of stars is found between min_star_radius 1000  & 1
        returns: appropriate threshold
        """

        # initialise
        iterate = True
        threshold_iteration = 254
        threshold_reduction = 4
        residual = self._get_threshold_residual(threshold_iteration)
        print(residual)
        if residual == 0: iterate = False

        # continue to calculate values until the residual drops to 0
        while iterate:

            # increase the value of the star detection radius:
            threshold_reduction = threshold_reduction * 1.05
            threshold_iteration = threshold_iteration - threshold_reduction

            residual = self._get_threshold_residual(threshold_iteration)
            print(residual)

            if residual == 0:
                iterate = False

        return int(threshold_iteration)

    def _get_threshold_residual(self, x):
        """Private helper to evaluate star detection quality at a specific threshold."""
        # evaluate star detections & resulting residuals at extreme limits
        min_radius_residual = self._evaluate_threshold(star_detect_pixels=10, count=50, case="upper_bound", x=x)
        max_radius_residual = self._evaluate_threshold(star_detect_pixels=1000, count=5, case="lower_bound", x=x)

        total_residual = min_radius_residual + max_radius_residual

        return total_residual

    def _evaluate_threshold(self, star_detect_pixels, count, case, x):
        """Private helper to process the image and calculate the error (residual)."""
        # try a new input, with an updated star_detect_pixels
        self.inputs = replace(self.inputs, star_detect_pixels=star_detect_pixels, threshold=x)

        # run processing and find residual
        if self.inputs.verbosity > 0: print(f"Assessing threshold: {x}")
        FrameReader(self).pre_process()
        StarFilter(self).local_density()

        # assume high residual
        z = 1000

        # only run this if pixels are detected
        if len(self.pixels_in_clusters) > 0:
            # count stars
            Starfield(self).count_stars()

            # assess results
            if case == "upper_bound":
                if self.n_clusters > count:
                    z = 0
                else:
                    z = abs(self.n_clusters - count)

            if case == "lower_bound":
                if self.n_clusters < count:
                    z = 0
                else:
                    z = abs(self.n_clusters - count)

        return z

    def tune_star_detect_pixels(self, n_desired_clusters):

        def fitness_function(x):

            # run processing and find residual:
            self.inputs = replace(self.inputs, star_detect_pixels=x)
            StarFilter(self).local_density()
            if len(self.pixels_in_clusters) > 0:
                Starfield(self).count_stars()
                residual = n_desired_clusters - self.n_clusters
            else:
                residual = 100

            # debugging information:
            if __name__ == "__main__":
                print(f"Residual = {residual}, from star_detect_pixels = {x}")

            return residual

        # read in the image:
        FrameReader(self).pre_process()

        # bisection solving to determine ideal star detection threshold inside search radius:
        result = bisect(fitness_function, 1, 1000)

        return result

if __name__ == "__main__":

    tracemalloc.start()

    # inputs
    data_dir = Path(r"D:\_Local\OneDrive\Astronomy\StarTrack\dev\raw_data_m82")
    verbosity = 1
    light = LightFrame(frame_directory=data_dir, frame_name="L_0173.jpg", verbosity=verbosity, star_detect_pixels=500.5, threshold=231)
    light.tune_threshold()

    # memory profiling
    snapshot = tracemalloc.take_snapshot()
    for stat in snapshot.statistics('lineno')[:10]:
        print(stat)
    top_stats = snapshot.statistics('lineno')
    total = sum(stat.size for stat in top_stats)
    print(f"\nTotal memory allocated (tracemalloc): {total / (1024 ** 2):.2f} MiB")


