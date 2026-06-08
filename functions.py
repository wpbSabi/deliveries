from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
import numpy as np
import pandas as pd


def one_hot_encode(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Encode categorical features with one-hot encoding for specified columns in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the categorical columns to be encoded.
        columns (list): A list of column names in the DataFrame that should be one-hot encoded.

    Returns:
        pd.DataFrame: A new DataFrame with the specified columns one-hot encoded and the original columns dropped.
    """
    for col in columns:
        dummies = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummies], axis=1)
        # Drop the column that was encoded
        df.drop(col, axis=1, inplace=True)
    return df


def summarize_results(
    df: pd.DataFrame,
    run_date: pd.Timestamp,
    y_test: pd.Series,
    y_pred: pd.Series,
    model_name: str,
    window_days: int,
    print_stats: bool = True,
) -> pd.DataFrame:
    """
    Compute metrics and return a results DataFrame for the predictions.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data for evaluation.
            This df includes the dimensions such as 'origin_country', 'us_destination_state',
              'order_date', and 'order_number'.
        run_date (pd.Timestamp): The date to filter the test data.
        y_test (pd.Series): The actual values for the test set.
        y_pred (pd.Series): The predicted values for the test set.
        model_name (str): The name of the model for which to summarize results.
        window_days (int): The number of days to include in the testing window.
        print_stats (bool): Whether to print the evaluation metrics to the console.

    Returns:
        pd.DataFrame: A new DataFrame with the evaluation metrics and predictions.
    """
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    # If print_stats is True, print the metrics to the console
    if print_stats:
        print(f"Predicted rows: {len(y_pred)}")
        print(
            f" MAE: {mae:.3f} \n RMSE: {rmse:.3f} \n R2: {r2:.3f} \n MAPE: {mape:.3f}"
        )

    # Create a results DataFrame that includes the dimensions and the actual vs predicted metrics
    results = (
        df.loc[
            (df["order_date"] >= run_date)
            & (df["order_date"] <= run_date + pd.Timedelta(days=window_days)),
            ["origin_country", "us_destination_state", "order_date", "order_number"],
        ]
        .reset_index(drop=True)
        .copy()
    )
    results["actual"] = y_test.reset_index(drop=True)
    results["pred"] = np.round(y_pred, 2)
    results["abs_err"] = (results["actual"] - results["pred"]).abs()
    results["model"] = model_name

    return results, mape


def train_test_split(
    df: pd.DataFrame, run_date: pd.Timestamp, window_days: int
) -> tuple:
    """
    Split the DataFrame into training and testing sets based on the order_date.
    The testing data will include one week of data starting from the run_date,
        and the training data will include all data up to the run_date.

    Args:
        df (pd.DataFrame): The input DataFrame containing the features data to be split.
        run_date (pd.Timestamp): The date to use as the cutoff for splitting the data.
        window_days (int): The number of days to include in the testing window.

    Returns:
        tuple: A tuple containing the training features (X_train), training target (y_train),
               testing features (X_test), and testing target (y_test).
    """
    # Training data to include all data up to the run_date,
    # and testing data to include all data from the run_date to one week after the run_date
    features_train = df[df["order_date"] < run_date]
    features_test1 = df[df["order_date"] >= run_date]
    features_test = features_test1[
        features_test1["order_date"] <= run_date + pd.Timedelta(days=window_days)
    ]

    # Remove order date from features, maybe add back for future features, such as a rolling average or seasonality
    features_train = features_train.drop(
        columns=["delivery_date", "order_date", "order_number"]
    )
    features_test = features_test.drop(
        columns=["delivery_date", "order_date", "order_number"]
    )

    features_train.reset_index(drop=True, inplace=True)
    features_test.reset_index(drop=True, inplace=True)

    # drop non-feature columns, and move the target actual_days to the last columns
    feature_cols = [col for col in features_train.columns if col != "actual_days"]
    features_train = features_train[feature_cols + ["actual_days"]]
    features_train.reset_index(drop=True, inplace=True)

    # Create training and testing variables
    X_train = features_train[feature_cols]
    y_train = features_train["actual_days"]
    X_test = features_test[feature_cols]
    y_test = features_test["actual_days"]

    return X_train, y_train, X_test, y_test, feature_cols


def return_regressor_results(
    model: pd.DataFrame,
    df: pd.DataFrame,
    run_date: pd.Timestamp,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Runs a regression model and returns the results and feature importances

    Arguments:
                model: a regressor model, e.g. AdaBoostRegressor()
                df (pd.DataFrame): the original dataframe, including dimensions and target variable
                run_date (pd.Timestamp): the date to use as the cutoff for splitting the data into
                X_train (pd.DataFrame): training data features
                y_train (pd.Series): training data target variable
                X_test (pd.DataFrame): testing data features
                y_test (pd.Series): testing data target variable

    Returns:
                results (pd.DataFrame): a dataframe summarizing the results of the model, including MAPE and RMSE
                feature_importances (numpy.ndarray): a series of the feature importances from the model
    """

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Testing data dimension columns
    data = (
        df.loc[
            (df["order_date"] >= run_date)
            & (df["order_date"] <= run_date + pd.Timedelta(days=7)),
            ["origin_country", "us_destination_state", "order_date", "order_number"],
        ]
        .reset_index(drop=True)
        .copy()
    )

    results, mape = summarize_results(
        data, run_date, y_test, y_pred, model_name=str(model), window_days=7
    )

    return results, model.feature_importances_
