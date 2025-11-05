
# dependencies:
import psutil
import os
import tracemalloc
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from skimage.transform import estimate_transform, warp
from StarTrack import LightFrame
from StarTrack.light_frame.star_alignment_vectors import StarAlignmentVectors
from StarTrack.light_frame.frame_reader import FrameReader

class FrameAligner:
    def __init__(self, ref_frame, align_frame):
        self.ref_frame = ref_frame
        self.addition_frame = align_frame
        self.coords_ref = None
        self.angles_ref = None
        self.coords_addition = None
        self.angles_addition = None
        self.addition_aligned_array_mono = None
        self.addition_aligned_array_r = None
        self.addition_aligned_array_g = None
        self.addition_aligned_array_b = None

    def align(self):

        # run alignment algorithms
        self._compute_ref_frame_aligning_stars() 
        self._compute_additional_frame_aligning_stars()
        self._align_and_transform_additional_frame()

    def _compute_ref_frame_aligning_stars(self):

        # calculate star alignment vectors from the biggest star:
        StarAlignmentVectors(self.ref_frame).compute_from_biggest_star()
        centroids = self.ref_frame.centroid_list
        i_ref = self.ref_frame.i_ref_star

        # move the reference star to the end:
        coords_ref_primary_star_last = np.concatenate([centroids[:i_ref],centroids[i_ref + 1:],centroids[i_ref][np.newaxis, :]], axis=0)
        angles_ref_primary_star_last = np.concatenate([self.ref_frame.ref_angles, np.array([0])], axis=0)

        # sort both arrays so that they match, in the order of angle from reference star:
        sort_indices = np.argsort(angles_ref_primary_star_last)[::-1]
        self.coords_ref = np.array(coords_ref_primary_star_last)[sort_indices]
        self.angles_ref = np.array(angles_ref_primary_star_last)[sort_indices]

        return self

    def _compute_additional_frame_aligning_stars(self):
        """
        steps:
        - rank all stars from largest to smallest, assuming the largest is most likely the primary reference star used as a reference point for alignment
        - compute alignment vectors for each star to the primary star and compare them to the alignment vectors from the primary star.
        -
        identify alignment stars once a match is found within specified angular and radial tolerances.

        returns:
        - co-ordinates of reference points for additional frame
        """

        # rank stars from largest to smallest. the larger the star, the more likely it is to be the PRIMARY ALIGNMENT STAR:
        ranked_indices = np.argsort(self.addition_frame.magnitude_list)[::-1]

        # debugging information:
        if __name__ == '__main__':
            print(f"Potential primary alignment stars, ranked in order of size: {ranked_indices}")

        # analyse each star in turn to find the PRIMARY ALIGNMENT STAR:
        for i_star in ranked_indices:

            # debugging information:
            if __name__ == '__main__':
                print(f"Attempting to align with primary alignment star index: {i_star}")
                print(self.addition_frame.centroid_list)

            # calculate star alignment vectors from the index star:
            StarAlignmentVectors(self.addition_frame).from_index_star(i_star)

            # debugging information:
            if __name__ == '__main__':
                print(f"Star aligment angles from primary alignment star index {i_star} are:")
                print(self.addition_frame.ref_angles)

            # try to determine addition co-ordinates and angles based on the reference star guess above. If it doesn't work, it means the guess star is not the reference star:
            try:
                vector_matcher = AlignmentVectorMatcher(self.ref_frame, self.addition_frame)
                self.coords_addition, self.angles_addition = vector_matcher.check_if_primary_star_is_correct()

                # debugging information:
                if __name__ == '__main__':
                    print("Successful star alignment, with the following stars:")
                    print(len(self.coords_addition))
                break

            # if the above code doesn't work, it will return an IndexError, as it will be searching an empty array (see the static method below):
            except IndexError:

                # debugging information:
                if __name__ == '__main__':
                    print("Alignment failed, retrying...")

        return self

    def _align_and_transform_additional_frame(self):

        def _warp_array(array,transform):

            # function wrapping warping code:
            warped_array = warp(array, inverse_map=transform.inverse,output_shape=array.shape, preserve_range=True)

            return warped_array.astype(np.uint8) # ensure this is 8 bit to avoid memory usage

        # estimate transform matrices for "affine" image distortion assumption, using the coordinates calculated:
        alignment_transform = estimate_transform('affine', self.coords_addition, self.coords_ref)

        # apply warp with inverse transform:
        rgb_frame = FrameReader.read_rgb(self.addition_frame)
        self.addition_aligned_array_mono = _warp_array(array=np.array(rgb_frame.convert('L')),transform=alignment_transform)
        self.addition_aligned_array_r = _warp_array(array=np.array(rgb_frame.getchannel('R')),transform=alignment_transform)
        self.addition_aligned_array_g = _warp_array(array=np.array(rgb_frame.getchannel('G')),transform=alignment_transform)
        self.addition_aligned_array_b = _warp_array(array=np.array(rgb_frame.getchannel('B')),transform=alignment_transform)

        # print update:
        if __name__ == '__main__':
            print("Reference alignment co-ordinates: ")
            print(self.coords_ref)
            print("Reference alignment angles: ")
            print(self.angles_ref)
            print("Addition alignment co-ordinates: ")
            print(self.coords_addition)
            print("Addition alignment angles: ")
            print(self.angles_addition)

        return self

class AlignmentVectorMatcher:
    def __init__(self, frame_main, frame_addition, angle_tol=0.01, radius_scale=1000):
        self.frame_main = frame_main
        self.frame_addition = frame_addition
        self.angle_tol = angle_tol
        self.radius_scale = radius_scale

    def check_if_primary_star_is_correct(self):

        # calculate how many alignment stars are required and create empty output lists:
        n_alignments_stars = len(self.frame_main.centroid_list)
        add_align_coords_list = []
        add_align_angles_list = []

        # filter star co-ordinates and store properties of identified aligning stars in the reference frame:
        # run through stars up until n_alignment - 1, because the reference alignment star is already known, and the rest need to be processed
        for i_star in range(n_alignments_stars - 1):
            add_align_star_coord, add_align_star_angle = self._is_alignment_vector_in_tolerance(i_star)
            add_align_coords_list.append(add_align_star_coord)
            add_align_angles_list.append(add_align_star_angle)

        # add reference star details:
        add_align_coords_list.append(self.frame_addition.centroid_list[self.frame_addition.i_ref_star])
        add_align_angles_list.append(int(0))

        # convert add_align_angles_array to numpy array:
        add_align_angles_array = np.array(add_align_angles_list)

        # sort both arrays so that they match, in the order of angle from reference star:
        sort_indices = np.argsort(add_align_angles_array)[::-1]
        sorted_add_align_angles_array = np.array(add_align_angles_list)[sort_indices]
        sorted_add_align_coords_array = np.array(add_align_coords_list)[sort_indices]

        return sorted_add_align_coords_array, sorted_add_align_angles_array

    def _is_alignment_vector_in_tolerance(self, i_primary_star_candidate):
        # determine tolerance for determining the aligning stars:
        tol = self.angle_tol
        filter_angle = self.frame_main.ref_angles[i_primary_star_candidate]
        i_ref_angle_list = np.where(np.abs(self.frame_addition.ref_angles - filter_angle) < tol)[0]

        if len(i_ref_angle_list) == 1:
            i_ref_angle = i_ref_angle_list[0]

            # debugging information:
            if __name__ == '__main__':
                print(f"Angle check completed:")
                print(f"Successfully identified alignment stars, with reference angle index: {i_ref_angle}")

        # multiple stars are within the search angle. to isolate the correct alignment star, a further search is done on radius:
        # tolerance scaled to account for order of magnitude difference between radiance and pixel distance
        else:
            filter_distance = self.frame_main.ref_vectors[i_primary_star_candidate]
            i_ref_distance_list = np.where(np.abs(self.frame_addition.ref_vectors - filter_distance) < self.radius_scale * tol)[0]
            i_ref_angle = np.intersect1d(i_ref_angle_list, i_ref_distance_list)[0]

            # debugging information:
            if __name__ == '__main__':
                print("Angle & distance check completed:")
                print(f"Successfully identified alignment star, with reference angle index: {i_ref_angle}")
                print("Details:")
                print(f"Target distance: {filter_distance}")
                print(f"Reference vector distance: {self.frame_addition.ref_vectors}")

        # co-ordinates of non-reference stars flagged by the indices above are extracted:
        # the indices in the main array of star centroids is found here, based on the exact star co-ordinates provided above
        non_ref_stars = self.frame_addition.non_ref_stars[i_ref_angle, :]
        i_alignment_star = np.where(self.frame_addition.centroid_list == non_ref_stars)[0][0]

        # extract the exact co-ordinate and angle of this alignment star
        coord = self.frame_addition.centroid_list[i_alignment_star]
        angle = self.frame_addition.ref_angles[i_ref_angle]

        return coord, angle

def process_frame(frame):
    """
    local function for parallel processing, used below for testing purposes.
    """

    print(f"Processing frame: {frame.inputs.frame_name}")
    frame.process()

    return frame

if __name__ == '__main__':

    tracemalloc.start()

    # define data path:
    run_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    data_dir = run_dir.parent.parent / "raw_data_m45"
    verbosity = 0
    t_val = 254

    # create reference frame objects:
    frame_ref = LightFrame(frame_directory=data_dir, frame_name="L_0786_ISO800_90s__NA.NEF", verbosity=verbosity, star_detect_pixels=750.25, threshold=t_val)
    frame_add = LightFrame(frame_directory=data_dir, frame_name="L_0787_ISO800_90s__NA.NEF", verbosity=verbosity, star_detect_pixels=235.140625, threshold=t_val)
    frame_ref.process()
    frame_add.process()

    # profile:
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Total memory: {mem_info.rss / (1024 ** 2):.2f} MB")

    # couple frames and find aligning stars for reference frame:
    coupled_frames = FrameAligner(frame_ref, frame_add)
    coupled_frames.align()

    # profile:
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Total memory: {mem_info.rss / (1024 ** 2):.2f} MB")

    # show alignment of additional frame to reference frame:
    plt.imshow(coupled_frames.addition_aligned_array_mono, cmap='gray')
    plt.show()

    # memory profiling;
    snapshot = tracemalloc.take_snapshot()
    for stat in snapshot.statistics('lineno')[:10]:
        print(stat)
    top_stats = snapshot.statistics('lineno')
    total = sum(stat.size for stat in top_stats)
    print(f"\nTotal memory allocated (tracemalloc): {total / (1024 ** 2):.2f} MB")
