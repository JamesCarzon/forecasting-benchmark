"""Forecast module containing various forecasting methods."""

from .kalman_filter import create_kalman_forecaster, evaluate_forecasts

__all__ = [
    'create_kalman_forecaster',
    'evaluate_forecasts'
]
