import numpy as np
import math


class KNNeighbours:
    def __init__(
        self, obseravtions: np.ndarray = 0, ground_truth: np.ndarray = 0, k: int = 3
    ) -> None:
        """
        Constructor
        """
        self._observations = obseravtions
        self._ground_truth = ground_truth
        self._k = k

    def predict(self, new_observation: np.ndarray) -> str:
        """
        A function that predicts the ground thruth for some new observations
        based on the k-nearest neighbours algorithm
        """
        # init the dictionary to store the nearest k neighbours
        k_dict = {x: float("inf") for x in range(self._k)}
        # go through all points to find the closest neighbour
        for row in self._observations:
            # find the distance
            distance = 0
            for i in range(row.size()):
                distance += (row[i] - new_observation[i]) ** 2
            distance = math.sqrt(distance)
            # find the var with the largest distance in the k_dict
            largest_dist_var = max(k_dict, key=k_dict.get)
            # check if its bigger thant the new distance
            if distance < k_dict[largest_dist_var]:
                # if it is not, replace it
                del k_dict[largest_dist_var]
                k_dict[row] = distance

        # find the number each ground thruth occurs in the knns
        ground_truth_occcurance = {x: 0 for x in self._ground_truth}
        for key in k_dict:
            key_ground_truth = self._ground_truth[key]
            ground_truth_occcurance[key_ground_truth] += 1
        # return the most common ground thruth
        return max(ground_truth_occcurance, key=ground_truth_occcurance.get)
