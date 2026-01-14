"""
Base forecaster module containing simple forecasting methods.

This module provides basic forecasting functions that can be used
for time series prediction.
"""


def make_forecast(df, target, forecast_method):
    """
    Generate a one-step ahead forecast using simple forecasting methods.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing historical time series data.
    target : str
        Name of the column to forecast.
    forecast_method : str
        Forecasting method to use. Options:
        - "running_average": Simple mean of past observations
        - "least_squares": Recursive least squares forecast
    
    Returns
    -------
    float
        One-step ahead forecast value.
    
    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({'y': [1, 2, 3, 4, 5]})
    >>> make_forecast(data, 'y', 'running_average')
    3.0
    
    Notes
    -----
    The least squares method uses recursive computation but could be
    made more efficient using online algorithms.
    """
    if forecast_method == "running_average":
        # simple method: running average of past observations
        return df[target].mean()
    if forecast_method == "least_squares":
        # recursive least squares
        # note: online least squares more computationally efficient
        numerator = 0
        denominator = 0
        for k in range(1, len(df)):
            numerator += df[target].iloc[k] * df[target].iloc[k-1]
        for k in range(len(df)):
            denominator += df[target].iloc[k] ** 2
        theta_hat = numerator / denominator
        return df[target].iloc[len(df)-1] * theta_hat
