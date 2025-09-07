import numpy as np
import math
import heapq

class KNNeighbours:
    def __init__(self, obseravtions: np.ndarray = 0, ground_truth: np.ndarray = 0, k: int = 3) -> None:
        """
        Constructor
        """
        self._observations = obseravtions
        self._ground_truth = ground_truth
        self._k = k

    def predict(self, new_observation: np.ndarray) -> str:
        #it should be a dict



        #initialize heap with k infinities
        init_list = [float('inf') for _ in range (self._k)]
        k_heap = heapq.heapify(init_list)
        #go through all points
        for row in self._observations:
            #find the distance
            distance = 0
            for i in range(row.size()):
                distance += (row[i] - new_observation[i])**2
            distance = math.sqrt(distance)
            #check if its smaller than the largest distance in the heap
            if distance < k_heap[-1]:
            #if it is, replace it
                k_heap[-1] = distance
            

            


