import numpy as np

class MultipleLinearRegressor:

    def __init__(self, default_parameters: np.ndarray = 0): 
        """
        Constructor

        is the constructor with params and b or with observations and training data or is it empty
        """
        self._parameters = dict(parameters= default_parameters)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) ->  None:
        """
        DO  WE CHECK

        The observations and ground truth should be np.ndarrays. Make
        sure the number of samples is in the row dimension, while the
        variables are in the column dimension.
        """




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
        num_samples = observations.shape[0]
        #add colums of 1s to the observations as observations_prime
        addition = np.ones((num_samples, 1))
        observations_prime = np.column_stack((observations, addition))
        #find the transpose observations_prime
        trans_observations = observations.transpose()
        #implement formula
        optimal_parameters = ((trans_observations * observations_prime) ** (-1)) * trans_observations * ground_truth
        self._parameters['parameters'] = optimal_parameters
        #turn it into a dict

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Return the prediction from observations based on the parameters
        """
        return [[key * self._parameters[col] for col, key in enumerate[row]] for row in observations]
    
    def get_parameters(self) -> np.ndarray:
        return self._parameters['parameters']







