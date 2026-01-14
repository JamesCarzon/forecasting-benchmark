"""Forecast module containing various forecasting methods."""

from .kalman_filter import create_kalman_forecaster
from .evaluation import evaluate_forecasts
from .base_forecaster import make_forecast

__all__ = [
    'create_kalman_forecaster',
    'evaluate_forecasts',
    'make_forecast'
]
