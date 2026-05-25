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

    Returns:
        pd.DataFrame: A new DataFrame with the evaluation metrics and predictions.
    """
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print(f"Predicted rows: {len(y_pred)}")
    print(f" MAE: {mae:.3f} \n RMSE: {rmse:.3f} \n R2: {r2:.3f} \n MAPE: {mape:.3f}")

    # Create a results DataFrame that includes the dimensions and the actual vs predicted metrics
    results = (
        df.loc[
            (df["order_date"] >= run_date)
            & (df["order_date"] <= run_date + pd.Timedelta(days=7)),
            ["origin_country", "us_destination_state", "order_date", "order_number"],
        ]
        .reset_index(drop=True)
        .copy()
    )
    results["actual"] = y_test.reset_index(drop=True)
    results["pred"] = np.round(y_pred, 2)
    results["abs_err"] = (results["actual"] - results["pred"]).abs()
    results["model"] = model_name

    return results
