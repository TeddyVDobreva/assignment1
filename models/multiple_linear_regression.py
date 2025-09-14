import numpy as np


class MultipleLinearRegression:
    """
    Class for multiple linear regression
    """

    def __init__(self) -> None:
        """
        Constructor
        """
        self._parameters = {}

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Finds the parameters from given observations and ground truth.

        Formula: params = (X_trans * X) ^ (-1) * X_trans * y

        Params:
        observations : np.ndarray (num of samples, variables)
        ground_truth : np.ndarray

        Returns:
        None
        """
        # Calculate if observations and ground truth have equal nums of rows
        if self.__validate_observations_ground_truth(observations, ground_truth):
            num_samples = observations.shape[0]
            # Add colums of 1s to the observations as observations_prime
            addition = np.ones((num_samples, 1))
            observations_prime = np.column_stack((observations, addition))
            # Find the transpose observations_prime
            trans_observations = observations_prime.transpose()
            # Implement formula
            # calculate the first part in the formula (X_trans * X)
            xt_x = np.dot(trans_observations, observations_prime)
            # calculate the next part of the formula (X_trans * X) ^ (-1)
            xt_x_inverse = np.linalg.inv(xt_x)
            # calculate the next part of the formula (X_trans * X) ^ (-1) * X_trans
            xt_x_inverse_xt = np.dot(xt_x_inverse, trans_observations)
            # Parameters are calculated with formula (X_trans * X) ^ (-1) * X_trans * y
            optimal_parameters = np.dot(
                xt_x_inverse_xt,
                ground_truth,
            )
            self._parameters["parameters"] = optimal_parameters

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Returns the prediction from observations based on the parameters

        Params:
        observations : np.ndarray (num of samples, variables)

        Return:
        np.ndarray : The predicted values for the new observations.
        """
        # Add colums of 1s to the observations as observations_prime
        addition = np.ones((observations.shape[0], 1))
        observations_prime = np.column_stack((observations, addition))
        return np.dot(observations_prime, self._parameters["parameters"])

    @property
    def parameters(self) -> np.ndarray:
        """
        Getter: Provides a read only view of the parameters.

        Returns:
        np.ndarray : The parameters of the model.
        """
        return self._parameters["parameters"]

    def __validate_observations_ground_truth(
        self,
        observations: np.ndarray,
        ground_truth: np.ndarray,
    ) -> bool:
        """
        Checks if the observations and ground truth have the same number of rows.

        Params:
        observations : np.ndarray (num of samples, variables)
        ground_truth : np.ndarray

        Return:
        bool : True if the number of rows are equal, False otherwise.
        """
        return observations.shape[0] == ground_truth.shape[0]
