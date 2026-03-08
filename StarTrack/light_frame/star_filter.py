
import numpy as np
from PIL import Image

class StarFilter:

    def __init__(self, state):
        self.state = state
        self.inputs = self.state.inputs

    def local_density(self, *args):
        """Identifies star clusters based on the local density of thresholded pixels.

        Checks each star pixel against a defined detection radius. If the number of
        surrounding star pixels exceeds the minimum detection count, the pixel is
        flagged as part of a cluster.

        Args:
            *args: Variable length argument list (currently unused).

        Returns:
            State: The updated state object containing the `pixels_in_clusters` array.
        """

        if self.inputs.verbosity > 0: print(f"Filtering with detection radius: {self.inputs.star_detect_radius} and minimum detection count: {self.inputs.star_detect_pixels}")

        self.state.pixels_in_clusters = []

        clustered_pixels_array = np.zeros(self.state.threshold_array.shape)

        star_pixels_yx = np.array(np.where(self.state.threshold_array == 255))
        star_pixels_y, star_pixels_x = star_pixels_yx

        # Check every point, and log which pixels exceed the star detection radius
        for i_pixel in range(star_pixels_y.shape[0]):

            # Define the bounds to be searched
            ref_y, ref_x = star_pixels_y[i_pixel], star_pixels_x[i_pixel]
            lower_x = ref_x - self.inputs.star_detect_radius
            upper_x = ref_x + self.inputs.star_detect_radius
            lower_y = ref_y - self.inputs.star_detect_radius
            upper_y = ref_y + self.inputs.star_detect_radius

            # Check where mask overlaps and combine output into a single array
            mask_x = (star_pixels_x > lower_x) & (star_pixels_x < upper_x)
            mask_y = (star_pixels_y > lower_y) & (star_pixels_y < upper_y)
            mask_yx = mask_x & mask_y
            adjacent_star_pixels = np.vstack((star_pixels_y[mask_yx], star_pixels_x[mask_yx]))

            # 1 is added because filtered_stars will always have a minimum length of 1 (as the reference star will be in it!)
            if adjacent_star_pixels.shape[1] > (self.inputs.star_detect_pixels + 1):

                self.state.pixels_in_clusters.append([ref_x, ref_y])
                clustered_pixels_array[ref_y][ref_x] = 255

                if self.inputs.verbosity > 1: print(f"Multipixel object detected at ({ref_x},{ref_y}) with {(adjacent_star_pixels.shape[1] - 1)} adjacent pixels!")

        if self.inputs.verbosity > 0:print(f"Filtering complete! {len(self.state.pixels_in_clusters)} pixels in clusters detected")

        if self.inputs.verbosity > 2:
            clustered_pixels_frame = Image.fromarray(clustered_pixels_array)
            clustered_pixels_frame.show()

        self.state.pixels_in_clusters = np.array(self.state.pixels_in_clusters)

        return self.state