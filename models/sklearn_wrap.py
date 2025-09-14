import numpy as np
from sklearn.linear_model import Lasso


class SklearnLassoRegressor:
    """
    Lasso scikit-learn wrapper class
    """

    def __init__(self) -> None:
        """
        Constructor: Initializes a Lasso regression model from sklearn
        and stores the paremeters in a dictionary.
        """
        self.model = Lasso()
        self._parameters = {}

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the Lasso regression model to the provided observations and ground truth.

        Params:
        observations : np.ndarray (num of samples, variables)
        ground_truth : np.ndarray

        Returns:
        None
        """
        self.model.fit(observations, ground_truth)
        self._parameters["parameters"] = self.model.coef_

    def predict(self, new_observations: np.ndarray) -> np.ndarray:
        """
        Predict the ground truth for new observations using
        the trained Lasso regression model.

        Params:
        new_observations : np.ndarray (num of samples, variables)

        Returns:
        np.ndarray : Predicted values for the new observations.
        """
        return self.model.predict(new_observations)
