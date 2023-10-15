from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn import linear_model
from sklearn.base import RegressorMixin


DATAPATH = Path("data/")


def get_train_data(
    data_file: Path = DATAPATH / "data.csv",
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Load and preprocess the data from the CSV file.
    Return the data as a tuple of numpy arrays (X, y),
    where X is the run number and y is the birthdate in decimal years.
    """

    df = pd.read_csv(data_file, sep=";")

    # Transform dates into decimal years
    df["birthdate"] = pd.to_datetime(df["birthdate"])
    df["birthyear"] = df["birthdate"].dt.year + (df["birthdate"].dt.month - 1.0) / 12.0

    # Remove the last two characters ("-X") of the run number
    df["run"] = df["run"].astype(str).str[:-2].astype(np.int64)

    X = df["run"].values.reshape(-1, 1)
    y = df["birthyear"].values

    return X, y


def fit_linear_model(
    X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> RegressorMixin:
    """
    Fit a linear model to the data.
    Return the fitted model.
    """

    model = linear_model.LinearRegression()
    model.fit(X, y)

    print(
        "Fitting results:",
        f"Coef: {model.coef_[0]:.6f}, " f"Intercept: {model.intercept_:.6f}",
        f"R2: {model.score(X, y):.4f}, ",
    )

    return model


def predict_birthyear(model: RegressorMixin, x: float) -> float:
    """
    Predict the birthyear of a person born in the year of the given run number.
    Return the predicted birthyear as a decimal year.
    """

    return model.predict(np.array(x).reshape(-1, 1))[0]


def parse_decimal_year(decimal_year: float) -> str:
    """
    Parse a decimal year into a string of the form "YYYY-MM".
    """

    year = int(decimal_year)
    month = int((decimal_year - year) * 12) + 1
    return f"{year}-{month:02d}"
