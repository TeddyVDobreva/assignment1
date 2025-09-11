from collections import Counter

import numpy as np


class KNNeighbours:
    def __init__(self, k: int = 3) -> None:
        """
        Constructor
        """
        if self.__validate_k(k):
            self._k = k
        self._parameters = {}

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit function for the k-nearest neighbours algorithm
        """
        # check dimentions of observations and ground truth
        if self.__validate_observations_ground_truth(observations, ground_truth):
            self._parameters["observations"] = observations
            self._parameters["ground_truth"] = ground_truth

    def __predict_single(self, x: np.ndarray) -> int:
        """
        Private function to determine the ground truth for a singular new observation
        """
        distances = np.linalg.norm(self._parameters["observations"] - x, axis=1)
        nn_indices = np.argsort(distances)[: self._k]
        nn_ground_truths = self._parameters["ground_truth"][nn_indices]
        most_common = Counter(nn_ground_truths).most_common(1)
        return most_common[0][0]

    def predict(self, new_observation: np.ndarray) -> np.ndarray:
        """
        A function that predicts the ground truth for some new observations
        based on the k-nearest neighbours algorithm
        """
        pred = [self.__predict_single(x) for x in new_observation]
        return np.array(pred)

    def __validate_k(self, k: int) -> bool:
        """
        Validator for the k value, since it should be greater than 0
        """
        return k > 0

    def __validate_observations_ground_truth(
        self, observations: np.ndarray, ground_truth: np.ndarray
    ) -> bool:
        """
        Validator for the observations and ground truth, since they have to have the same number of rows (data points)
        """
        return observations.shape[0] == ground_truth.shape[0]
