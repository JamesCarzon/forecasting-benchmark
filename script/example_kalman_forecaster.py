#!/usr/bin/env python3
"""
Example script demonstrating the Kalman filter forecaster.

This script shows how to use the create_kalman_forecaster function
with both univariate and multivariate time series data.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from forecast import create_kalman_forecaster


def example_univariate():
    """Example with univariate time series."""
    print("=" * 60)
    print("Example 1: Univariate Time Series (Sine Wave)")
    print("=" * 60)
    
    # Generate synthetic univariate data
    t = np.linspace(0, 4 * np.pi, 100)
    data = np.sin(t) + 0.1 * np.random.randn(100)
    
    print(f"Data shape: {data.shape}")
    print(f"First 5 values: {data[:5]}")
    
    # Create forecaster using last 3 observations
    observation_lag = 3
    forecaster = create_kalman_forecaster(data, observation_lag=observation_lag)
    
    print(f"\nUsing observation_lag={observation_lag}")
    print("Making forecasts using the last few observations...\n")
    
    # Make forecast using last 3 observations
    y_hat_1 = forecaster(data[-1], data[-2], data[-3])
    print(f"Forecast (individual args): {y_hat_1:.4f}")
    
    # Alternative: pass as a list
    y_hat_2 = forecaster([data[-1], data[-2], data[-3]])
    print(f"Forecast (list arg): {y_hat_2:.4f}")
    
    # Make multiple forecasts
    print("\nMultiple sequential forecasts:")
    recent_obs = data[-observation_lag:].tolist()
    for i in range(5):
        y_hat = forecaster(*recent_obs[::-1])  # Reverse order: most recent first
        print(f"  Forecast {i+1}: {y_hat:.4f}")
        # Update recent observations (rolling window)
        recent_obs = recent_obs[1:] + [y_hat]
    
    print()


def example_multivariate():
    """Example with multivariate time series."""
    print("=" * 60)
    print("Example 2: Multivariate Time Series (2D Random Walk)")
    print("=" * 60)
    
    # Generate synthetic multivariate data (2D random walk)
    np.random.seed(42)
    n_samples = 100
    n_features = 2
    
    data = np.cumsum(np.random.randn(n_samples, n_features) * 0.1, axis=0)
    
    print(f"Data shape: {data.shape}")
    print(f"First 3 values:\n{data[:3]}")
    
    # Create forecaster using last 2 observations
    observation_lag = 2
    forecaster = create_kalman_forecaster(data, observation_lag=observation_lag)
    
    print(f"\nUsing observation_lag={observation_lag}")
    print("Making forecasts using the last few observations...\n")
    
    # Make forecast using last 2 observations
    y_hat_1 = forecaster(data[-1], data[-2])
    print(f"Forecast (individual args): {y_hat_1}")
    
    # Alternative: pass as a list
    y_hat_2 = forecaster([data[-1], data[-2]])
    print(f"Forecast (list arg): {y_hat_2}")
    
    # Make multiple forecasts
    print("\nMultiple sequential forecasts:")
    recent_obs = list(data[-observation_lag:])
    for i in range(5):
        y_hat = forecaster(*recent_obs[::-1])  # Reverse order: most recent first
        print(f"  Forecast {i+1}: {y_hat}")
        # Update recent observations (rolling window)
        recent_obs = recent_obs[1:] + [y_hat]
    
    print()


def example_custom_parameters():
    """Example with custom Kalman filter parameters."""
    print("=" * 60)
    print("Example 3: Custom Kalman Filter Parameters")
    print("=" * 60)
    
    # Generate noisy data
    t = np.linspace(0, 2 * np.pi, 50)
    data = np.sin(t) + 0.3 * np.random.randn(50)
    
    print(f"Data shape: {data.shape}")
    
    # Create forecaster with custom noise parameters
    forecaster = create_kalman_forecaster(
        data,
        observation_lag=4,
        process_noise_cov=1e-4,  # Low process noise (smooth dynamics)
        measurement_noise_cov=0.1,  # Higher measurement noise (noisy observations)
        initial_state_cov=0.5
    )
    
    print("Created forecaster with custom parameters:")
    print("  observation_lag=4")
    print("  process_noise_cov=1e-4")
    print("  measurement_noise_cov=0.1")
    print("  initial_state_cov=0.5")
    
    # Make forecast
    y_hat = forecaster(data[-1], data[-2], data[-3], data[-4])
    print(f"\nForecast: {y_hat:.4f}")
    print(f"Last observation: {data[-1]:.4f}")
    print()


if __name__ == "__main__":
    print("\nKalman Filter Forecaster Examples")
    print("=" * 60)
    print()
    
    example_univariate()
    example_multivariate()
    example_custom_parameters()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
