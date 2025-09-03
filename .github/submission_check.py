import os
import sys
from typing import Union

import pydantic

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to
# the sys.path.
sys.path.append(parent)


class TestResult(pydantic.BaseModel):
    status: str
    message: str

    def __str__(self):
        return f"{self.status}: {self.message}"


TESTS = {
    "python_version": None,
    "integration.k_nearest_neighbors.import": None,
    "integration.k_nearest_neighbors.fit": None,
    "integration.k_nearest_neighbors.predict": None,
    "integration.k_nearest_neighbors.k": None,
    "integration.k_nearest_neighbors.parameters": None,
    "integration.multiple_linear_regression.import": None,
    "integration.multiple_linear_regression.fit": None,
    "integration.multiple_linear_regression.predict": None,
    "integration.sklearn_wrap.import": None,
    "integration.sklearn_wrap.fit": None,
    "integration.sklearn_wrap.predict": None,
    "dependencies": None,
}


def print_report():
    # Add green color for pass and red color for fail
    for test_name, result in TESTS.items():
        if result is not None:
            if result.status == "Pass":
                print(f"\033[92m{test_name}: {result}\033[0m")
            else:
                print(f"\033[91m{test_name}: {result}\033[0m")
        else:
            print(f"{test_name}: None")
    import os

    if os.path.exists("test_results.json"):
        os.remove("test_results.json")
    else:
        os.system("touch test_results.json")
    # Save results
    data = []
    for test_name, result in TESTS.items():
        if result is not None:
            data.append(result.dict())
    import json

    with open("test_results.json", "w") as f:
        json.dump(data, f)


def check_python_version(target="3.10"):
    import sys

    a = sys.version_info
    if a.major != 3 or a.minor < 10:
        TESTS["python_version"] = TestResult(
            status="Fail",
            message=f"Python version is {a.major}.{a.minor} but should be {target}",
        )
    else:
        TESTS["python_version"] = TestResult(
            status="Pass", message=f"Python version is {a.major}.{a.minor}"
        )


def run_dependency_tests():
    try:
        import numpy
        import sklearn

        TESTS["dependencies"] = TestResult(
            status="Pass", message="All dependencies are installed"
        )
    except ImportError as e:
        TESTS["dependencies"] = TestResult(
            status="Fail", message=f"Missing dependency {e}"
        )


def check_model_class_signature(model):
    from inspect import signature

    import numpy as np

    results = {"fit": None, "predict": None}
    # Check the fit function
    try:
        fit_sig = signature(model.fit)
        if len(fit_sig.parameters) != 3:
            results["fit"] = TestResult(
                status="Fail", message="Model.fit should have 3 parameters"
            )
        else:
            for i, param in enumerate(fit_sig.parameters):
                # skip over self
                if param == "self":
                    continue
                if fit_sig.parameters[param].annotation != np.ndarray:
                    results["fit"] = TestResult(
                        status="Fail",
                        message=f"{param} should be annotated with np.ndarray",
                    )
                else:
                    results["fit"] = TestResult(
                        status="Pass", message="Model.fit has correct signature"
                    )
    except Exception as e:
        results["fit"] = TestResult(status="Fail", message=f"Something went wrong {e}")

    # Check the predict function
    try:
        predict_sig = signature(model.predict)
        if len(predict_sig.parameters) != 2:
            results["predict"] = TestResult(
                status="Fail", message="Model.predict should have 2 parameters"
            )
        else:
            for i, param in enumerate(predict_sig.parameters):
                # skip over self
                if param == "self":
                    continue
                if predict_sig.parameters[param].annotation != np.ndarray:
                    results["predict"] = TestResult(
                        status="Fail",
                        message=f"{param} should be annotated with np.ndarray",
                    )
                else:
                    results["predict"] = TestResult(
                        status="Pass", message="Model.predict has correct signature"
                    )
    except Exception as e:
        results["predict"] = TestResult(
            status="Fail", message=f"Something went wrong {e}"
        )
    return results


def run_integration_tests():
    # Test multiple linear regression
    try:
        from models.multiple_linear_regression import MultipleLinearRegression

        TESTS["integration.multiple_linear_regression.import"] = TestResult(
            status="Pass", message="MultipleLinearRegression imported"
        )
        mlr_results = check_model_class_signature(MultipleLinearRegression)
        TESTS["integration.multiple_linear_regression.fit"] = mlr_results["fit"]
        TESTS["integration.multiple_linear_regression.predict"] = mlr_results["predict"]
    except ImportError:
        TESTS["integration.multiple_linear_regression.import"] = TestResult(
            status="Fail", message="Cannot import MultipleLinearRegression"
        )
        TESTS["integration.multiple_linear_regression.fit"] = TestResult(
            status="Fail", message="Cannot import MultipleLinearRegression"
        )
        TESTS["integration.multiple_linear_regression.predict"] = TestResult(
            status="Fail", message="Cannot import MultipleLinearRegression"
        )

    # Test k nearest neighbors
    try:
        from models.k_nearest_neighbors import KNearestNeighbors

        TESTS["integration.k_nearest_neighbors.import"] = TestResult(
            status="Pass", message="KNearestNeighbors imported"
        )
        knn_results = check_model_class_signature(KNearestNeighbors)
        TESTS["integration.k_nearest_neighbors.fit"] = knn_results["fit"]
        TESTS["integration.k_nearest_neighbors.predict"] = knn_results["predict"]
        model = KNearestNeighbors()
        if model.k < 0:
            TESTS["integration.k_nearest_neighbors.k"] = TestResult(
                status="Fail", message="k should be greater than 0"
            )
        else:
            TESTS["integration.k_nearest_neighbors.k"] = TestResult(
                status="Pass", message="k is greater than 0"
            )
        # Check if
        params = model.parameters
        if id(params) == id(model.parameters):
            TESTS["integration.k_nearest_neighbors.parameters"] = TestResult(
                status="Fail", message="parameters is a reference"
            )
        else:
            TESTS["integration.k_nearest_neighbors.parameters"] = TestResult(
                status="Pass", message="parameters is a copy"
            )

    except ImportError:
        TESTS["integration.k_nearest_neighbors.import"] = TestResult(
            status="Fail", message="Cannot import KNearestNeighbors"
        )
        TESTS["integration.k_nearest_neighbors.fit"] = TestResult(
            status="Fail", message="Cannot import KNearestNeighbors"
        )
        TESTS["integration.k_nearest_neighbors.predict"] = TestResult(
            status="Fail", message="Cannot import KNearestNeighbors"
        )
        TESTS["integration.k_nearest_neighbors.k"] = TestResult(
            status="Fail", message="Cannot import KNearestNeighbors"
        )
        TESTS["integration.k_nearest_neighbors.parameters"] = TestResult(
            status="Fail", message="Cannot import KNearestNeighbors"
        )

    # Test sklearn wrap
    try:
        from models.sklearn_wrap import Lasso

        TESTS["integration.sklearn_wrap.import"] = TestResult(
            status="Pass", message="Lasso imported"
        )
        lasso_results = check_model_class_signature(Lasso)
        TESTS["integration.sklearn_wrap.fit"] = lasso_results["fit"]
        TESTS["integration.sklearn_wrap.predict"] = lasso_results["predict"]
    except ImportError:
        TESTS["integration.sklearn_wrap.import"] = TestResult(
            status="Fail", message="Cannot import Lasso"
        )
        TESTS["integration.sklearn_wrap.fit"] = TestResult(
            status="Fail", message="Cannot import Lasso"
        )
        TESTS["integration.sklearn_wrap.predict"] = TestResult(
            status="Fail", message="Cannot import Lasso"
        )


def program_exit():
    import sys

    # If there are any failing tests, exit with status code 1 or None
    for test_name, result in TESTS.items():
        if result is None or result.status == "Fail":
            print("A test has failed. Exiting with status code 1")
            sys.exit(1)
    sys.exit(0)


# Test for students
def run_tests():
    check_python_version()
    run_dependency_tests()
    run_integration_tests()
    print_report()
    program_exit()


if __name__ == "__main__":
    run_tests()
