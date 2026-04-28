
import numpy as np


class StackingAlgorithm:

    def __init__(self, chunk_array, method, n_iterations = 5, sigma_clip = 5):

        self.chunk_array: np.array = chunk_array
        self.method: str = method
        self.n_iterations: int = n_iterations
        self.sigma_clip: int = sigma_clip

    def run(self):

        if self.method.lower() == "mean":
            self._compute_with_mean()
        elif self.method.lower() == 'median':
            self._compute_with_median()
        elif self.method.lower() == 'sigma_gamma_clipping':
            self._compute_with_sigma_gamma_clipping()
        elif self.method.lower() == 'weighted_mean_average':
            self._compute_with_weighted_mean_average()

    def _compute_with_mean(self):

        return np.mean(self.chunk_array, axis=0).dtype(np.float64)

    def _compute_with_median(self):

        return np.median(self.chunk_array, axis=0).dtype(np.float64)

    def _compute_with_sigma_gamma_clipping(self):

        return self

    def _compute_with_weighted_mean_average(self):

        chunk_shape = self.chunk_array.shape()

        baseline_stack = self._compute_with_mean()
        #baseline_chunk = baseline*


        # 1. compute the average value for each pixel?
        # 2. compute a weighting factor for each pixel, getting the distance of each pixel to the mean
        # 3. compute a new mean, this time using a weighting factor




# TODO: Implement code in this file. It has been kept empty as a placeholder.



