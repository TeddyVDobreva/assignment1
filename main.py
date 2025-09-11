import pandas as pd
from models import k_nearest_neighbors, multiple_linear_regression

if __name__ == "__main__":
    # For Multiple Linear Regression
    # Loading the data set for regression
    wine_df = pd.read_csv("data/winequality-red[1].csv", delimiter=";")
    wine_dataset = wine_df.values

    # Getting the ground truth and observations
    mlr_ground_truth = wine_dataset[:1200, -1]
    mlr_observations = wine_dataset[:1200, :-1]

    # Separating some data for predictions
    mlr_test_observations = wine_dataset[1200:, :-1]

    # Instantation of the Multiple Linear Regressor
    regression_model = multiple_linear_regression.MultipleLinearRegressor()

    # Training the model
    regression_model.fit(mlr_observations, mlr_ground_truth)
    # Prediction based on found parameters
    mlr_prediction = regression_model.predict(mlr_test_observations)

    # Printing of the parameters
    print(mlr_prediction)

    # For K-Nearest Neighbors
    # Loading the data set for classification
    iris_df = pd.read_csv("data/iris.csv", delimiter=",")
    iris_dataset = iris_df.values

    # Getting the observations and ground truth
    knn_observations = iris_dataset[:113, :-1]
    knn_ground_truth = iris_dataset[:113, -1]

    # Separating some data for predictions
    knn_test_observations = iris_dataset[113:, :-1]

    # Instantation of the K-Nearest Neighbors model
    knn = k_nearest_neighbors.KNNeighbours()

    # Train the model
    knn.fit(knn_observations, knn_ground_truth)

    # Prediction of new observations
    knn_prediction = knn.predict(knn_test_observations)

    # Printing the ground truth
    print(knn_prediction)
