import numpy as np


class MultipleLinearRegressor:
    def __init__(self, default_parameters: np.ndarray = 0):
        """
        Constructor

        is the constructor with params and b or with observations and training data or is it empty
        """
        self._parameters = dict(parameters=default_parameters)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        A function that calculates the parameters as from Equation based on a given training dataset
        composed of observations and ground truth.

        formula: params = (X_trans * X) ^ (-1) * X_trans * y

        Params:
        obseravtions : np.ndarray (num of samples, variables)
        ground_truth : np.ndarray

        Returns:
        None
        """
        # calculate if observations and ground truth have equal nums of rows
        if self.__validate_observations_ground_truth(observations, ground_truth):
            num_samples = observations.shape[0]
            # add colums of 1s to the observations as observations_prime
            addition = np.ones((num_samples, 1))
            observations_prime = np.column_stack((observations, addition))
            # find the transpose observations_prime
            trans_observations = observations_prime.transpose()
            # implement formula
            optimal_parameters = np.dot(
                np.dot(
                    (np.dot(trans_observations, observations_prime) ** (-1)),
                    trans_observations,
                ),
                ground_truth,
            )
            self._parameters["parameters"] = optimal_parameters

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Return the prediction from observations based on the parameters
        """
        # add colums of 1s to the observations as observations_prime
        addition = np.ones((observations.shape[0], 1))
        observations_prime = np.column_stack((observations, addition))
        return np.dot(observations_prime, self._parameters["parameters"])

    @property
    def parameters(self) -> np.ndarray:
        """
        Getter: Provides a read only view of the parameters
        """
        return self._parameters["parameters"]

    def __validate_observations_ground_truth(
        self, observations: np.ndarray, ground_truth: np.ndarray
    ) -> bool:
        # check if observations and ground thruth have the same amount of rows (data points)
        if observations.shape[0] != ground_truth.shape[0]:
            return False
