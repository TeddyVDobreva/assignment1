import pandas as pd
from models import k_nearest_neighbors, multiple_linear_regression, sklearn_wrap

if __name__ == "__main__":
    # For Multiple Linear Regression
    # Loading the data set for regression
    wine_df = pd.read_csv("data/winequality-red[1].csv", delimiter=";")
    wine_dataset = wine_df.values

    # Getting the ground truth and observations
    regression_ground_truth = wine_dataset[:1200, -1]
    regression_observations = wine_dataset[:1200, :-1]

    # Separating some data for predictions
    regression_test_observations = wine_dataset[1200:, :-1]

    # Instantation of the Multiple Linear Regressor
    mlr_model = multiple_linear_regression.MultipleLinearRegression()

    # Training the model
    mlr_model.fit(regression_observations, regression_ground_truth)

    # Prediction based on found parameters
    mlr_prediction = mlr_model.predict(regression_test_observations)

    # Printing of the parameters
    print(mlr_prediction)

    # For K-Nearest Neighbors
    # Loading the data set for classification
    iris_df = pd.read_csv("data/iris.csv", delimiter=",")
    # Encoding the labels
    label_dict = {"Setosa": 0, "Versicolor": 1, "Virginica": 2}
    # Mapping the labels to numbers
    iris_df.iloc[:, -1] = iris_df.iloc[:, -1].map(label_dict)
    iris_dataset = iris_df.values

    # Getting the observations and ground truth
    knn_observations = iris_dataset[:113, :-1]
    knn_ground_truth = iris_dataset[:113, -1]

    # Separating some data for predictions
    knn_test_observations = iris_dataset[113:, :-1]

    # Instantation of the K-Nearest Neighbors model
    knn = k_nearest_neighbors.KNearestNeighbors()

    # Training the model
    knn.fit(knn_observations, knn_ground_truth)

    # Prediction of new observations
    knn_prediction = knn.predict(knn_test_observations)

    # Printing the ground truth
    print(knn_prediction)

    # For Sklearn Lasso Regressor
    # Instatiation of the Sklearn Lasso Regressor
    lasso_model = sklearn_wrap.SklearnLassoRegressor()

    # Training the model
    lasso_model.fit(regression_observations, regression_ground_truth)

    # Prediction based on found parameters
    lasso_prediction = lasso_model.predict(regression_test_observations)

    # Printing of the parameters
    print(lasso_prediction)
