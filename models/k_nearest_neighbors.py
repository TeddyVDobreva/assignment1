from collections import Counter
import math
import numpy as np


class KNearestNeighbors:
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

        Params:
        observations : np.ndarray (num of samples, variables)
        ground_truth : np.ndarray

        Return:
        None
        """
        # Check dimentions of observations and ground truth
        if self.__validate_observations_ground_truth(observations, ground_truth):
            self._parameters["observations"] = observations
            self._parameters["ground_truth"] = ground_truth

    def __predict_single(self, x: np.ndarray) -> int:
        """
        Private function to determine the ground truth for a singular new observation

        Params:
        x : np.ndarray (variables)

        Return:
        int : The predicted class for the new observation
        """
        # Distances = np.linalg.norm(self._parameters["observations"] - x, axis=1)
        distances = self.__find_distances(self._parameters["observations"] - x)
        nn_indices = np.argsort(distances)[: self._k]
        nn_ground_truths = self._parameters["ground_truth"][nn_indices]
        most_common = Counter(nn_ground_truths).most_common(1)
        return most_common[0][0]

    def predict(self, new_observation: np.ndarray) -> np.ndarray:
        """
        A function that predicts the ground truth for some new observations
        based on the k-nearest neighbours algorithm

        Params:
        new_observation : np.ndarray (num of samples, variables)

        Return:
        np.ndarray : The predicted values for the new observations.
        """
        pred = [self.__predict_single(x) for x in new_observation]
        return np.array(pred)

    def __validate_k(self, k: int) -> bool:
        """
        Validator for the k value, since it should be greater than 0

        Params:
        k : int (the number of nearest neighbours to consider)

        Return:
        bool : True if k > 0, False otherwise.
        """
        return k > 0

    def __validate_observations_ground_truth(
        self, observations: np.ndarray, ground_truth: np.ndarray
    ) -> bool:
        """
        Validator for the observations and ground truth, since they have to have the same number of rows (data points)

        Params:
        observations : np.ndarray (num of samples, variables)
        ground_truth : np.ndarray

        Return:
        bool : True if the number of rows are equal, False otherwise.
        """
        return observations.shape[0] == ground_truth.shape[0]

    def __find_single_dist(self, x: np.ndarray) -> float:
        """
        A private function to calculate the distance of one data point to the new data
        
        Params:
        x : np.ndarray (variables)

        Return:
        float : The euclidean distance between the two points.
        """
        dist_squared = 0
        for val in x:
            dist_squared = val**2
        return math.sqrt(dist_squared)

    def __find_distances(self, vectors_array: np.ndarray) -> np.ndarray:
        """
        A private function to calculte the distances between the points in the observations and the new data

        Params:
        vectors_array : np.ndarray (num of samples, variables)

        Return:
        np.ndarray : The euclidean distances between the points.
        """
        dist = [self.__find_single_dist(vector) for vector in vectors_array]
        return np.array(dist)
