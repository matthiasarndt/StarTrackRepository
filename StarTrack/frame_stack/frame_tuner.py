
from dataclasses import replace
from pathlib import Path
from scipy.optimize import least_squares, fsolve, bisect
from StarTrack.light_frame.frame_reader import FrameReader
from StarTrack.light_frame.light_frame import LightFrame
from StarTrack.light_frame.star_field import Starfield
from StarTrack.light_frame.star_filter import StarFilter


class FrameTuner:
    """Handles optimisation of image processing parameters for astronomical star detection.

    This class provides tools to automatically tune parameters like brightness
    thresholds and pixel cluster sizes within a LightFrame. It aims to find
    the "sweet spot" where star detection is most accurate by minimizing
    the residual difference between detected stars and a target count.

    Attributes:
        frame (LightFrame)
    """

    def __init__(self, light_frame: LightFrame, **kwargs):
        self.frame = light_frame

    def tune_threshold(self):
        """Iteratively tunes the image threshold to achieve an optimal star spread.

        Decreases the threshold from an initial value of 254 using an exponentially
        increasing step size. The tuning stops when the calculated threshold
        residual drops to zero, indicating an acceptable distribution of stars.

        Returns:
            int: The calculated optimal threshold value.
        """

        # Initialise
        iterate = True
        threshold_iteration = 254
        threshold_reduction = 4
        residual = self._get_threshold_residual(threshold_iteration)
        print(residual)
        if residual == 0: iterate = False

        # Continue to calculate values until the residual drops to 0
        while iterate:

            # Increase the value of the star detection radius:
            threshold_reduction = threshold_reduction * 1.05
            threshold_iteration = threshold_iteration - threshold_reduction

            residual = self._get_threshold_residual(threshold_iteration)
            print(residual)

            if residual == 0:
                iterate = False

        return int(threshold_iteration)

    def _get_threshold_residual(self, x):
        """Private helper to evaluate star detection quality at a specific threshold."""

        # Evaluate star detections & resulting residuals at extreme limits
        min_radius_residual = self._evaluate_threshold(star_detect_pixels=10, count=50, case="upper_bound", x=x)
        max_radius_residual = self._evaluate_threshold(star_detect_pixels=1000, count=5, case="lower_bound", x=x)

        total_residual = min_radius_residual + max_radius_residual

        return total_residual

    def _evaluate_threshold(self, star_detect_pixels, count, case, x):
        """Private helper to process the image and calculate the error (residual)."""

        # Try a new input, with an updated star_detect_pixels
        self.frame.inputs = replace(self.frame.inputs, star_detect_pixels=star_detect_pixels, threshold=x)

        # Run processing and find residual
        if self.frame.inputs.verbosity > 0: print(f"Assessing threshold: {x}")
        FrameReader(self).pre_process()
        StarFilter(self).local_density()

        # Assume high residual
        z = 1000

        # Only run this if pixels are detected
        if len(self.frame.pixels_in_clusters) > 0:

            Starfield(self).count_stars()

            if case == "upper_bound":
                if self.frame.n_clusters > count:
                    z = 0
                else:
                    z = abs(self.frame.n_clusters - count)

            if case == "lower_bound":
                if self.frame.n_clusters < count:
                    z = 0
                else:
                    z = abs(self.frame.n_clusters - count)

        return z

    def tune_star_detect_pixels(self, n_desired_clusters):
        """Finds an optimal pixel detection radius using a bisection method.

        Adjusts the `star_detect_pixels` input within a specified range [1, 1000]
        to match a target number of star clusters. This method requires that the
        target star count is bracketed by the search limits.

        Args:
            n_desired_clusters (int): The target number of stars to be
                identified in the frame.

        Returns:
            float: The optimized star detection pixel value.

        Raises:
            ValueError: If the residual at the lower bound (1) and upper
                bound (1000) have the same sign.
        """

        FrameReader(self.frame).pre_process()

        bisect(self._evaluate_star_detect_pixels, 1, 1000, args=n_desired_clusters, xtol=0.1)

        return self.frame

    def _evaluate_star_detect_pixels(self, x_evaluate, n_desired_clusters):
        """Private helper to process the image and calculate the error (residual)."""

        self.frame.inputs = replace(self.frame.inputs, star_detect_pixels=x_evaluate)

        StarFilter(self.frame).local_density()

        if len(self.frame.pixels_in_clusters) > 0:
            Starfield(self.frame).count_stars()
            residual = n_desired_clusters - self.frame.n_clusters
        else:
            residual = 100

        if __name__ == "__main__":
            print(f"Residual = {residual}, from star_detect_pixels = {x_evaluate}")

        return residual

if __name__ == "__main__":

    data_dir = Path(r"D:\_Local\OneDrive\Astronomy\StarTrack\dev\raw_data_m82")
    verbosity = 1
    light = LightFrame(frame_directory=data_dir, frame_name="L_0173.jpg", verbosity=verbosity, star_detect_pixels=500.5, threshold=254)
    print(light.inputs)
    auto_tuner = FrameTuner(light_frame=light, verbosity=verbosity)
    auto_tuner.tune_star_detect_pixels(n_desired_clusters=5)

