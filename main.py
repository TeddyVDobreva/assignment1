import pandas as pd
from models import k_nearest_neighbors, multiple_linear_regression
# from k_nearest_neighbors import KNNeighbors
# from multiple_linear_regression import MultipleLinearRegressor

# For Multiple Linear Regression
if __name__ == "__main__":
    wine_df = pd.read_csv("winequality-red[1]", delimiter=';')
    wine_dataset = wine_df.values
    print(wine_dataset)

    ground_truth = wine_dataset[:9, -1]
    observations = wine_dataset[:9, :-1]
    test_observations = wine_dataset[9:, :-1]

    regression_model = multiple_linear_regression.MultipleLinearRegressor()

    regression_model.fit(observations, ground_truth)
    prediction = regression_model.predict(test_observations)
    # print()

# For K Nearest Neighbors
if __name__ == "__main__":
    df = pd.read_csv("iris")
    dataset = df.values

    observations = dataset[:, :-1]

    neighbors = k_nearest_neighbors.KNNeighbours()

    prediction = regression_model.predict(observations)
