#!/usr/bin/env python3
"""
Simple tests for the Kalman filter forecaster.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from forecast import create_kalman_forecaster


def test_univariate_basic():
    """Test basic univariate forecasting."""
    print("Test 1: Basic univariate forecasting...")
    
    # Simple linear trend
    data = np.arange(20, dtype=float)
    forecaster = create_kalman_forecaster(data, observation_lag=2)
    
    # Make forecast
    y_hat = forecaster(data[-1], data[-2])
    
    # Check that forecast is numeric
    assert isinstance(y_hat, (int, float)), "Forecast should be a scalar"
    assert not np.isnan(y_hat), "Forecast should not be NaN"
    
    print(f"  ✓ Forecast: {y_hat:.4f} (expected ~20)")
    print()


def test_multivariate_basic():
    """Test basic multivariate forecasting."""
    print("Test 2: Basic multivariate forecasting...")
    
    # 2D linear trend
    data = np.column_stack([np.arange(20), np.arange(20, 40)])
    forecaster = create_kalman_forecaster(data, observation_lag=2)
    
    # Make forecast
    y_hat = forecaster(data[-1], data[-2])
    
    # Check that forecast is array-like
    assert isinstance(y_hat, np.ndarray), "Forecast should be an array"
    assert y_hat.shape == (2,), f"Forecast shape should be (2,), got {y_hat.shape}"
    assert not np.any(np.isnan(y_hat)), "Forecast should not contain NaN"
    
    print(f"  ✓ Forecast: {y_hat} (expected ~[20, 40])")
    print()


def test_list_input():
    """Test forecaster with list input."""
    print("Test 3: List input format...")
    
    data = np.arange(20, dtype=float)
    forecaster = create_kalman_forecaster(data, observation_lag=3)
    
    # Test with list input
    y_hat = forecaster([data[-1], data[-2], data[-3]])
    
    assert isinstance(y_hat, (int, float)), "Forecast should be a scalar"
    assert not np.isnan(y_hat), "Forecast should not be NaN"
    
    print(f"  ✓ Forecast with list input: {y_hat:.4f}")
    print()


def test_custom_parameters():
    """Test with custom Kalman filter parameters."""
    print("Test 4: Custom parameters...")
    
    data = np.random.randn(50)
    
    # Should not raise any errors
    forecaster = create_kalman_forecaster(
        data,
        observation_lag=4,
        process_noise_cov=1e-4,
        measurement_noise_cov=0.1,
        initial_state_cov=0.5
    )
    
    y_hat = forecaster(data[-1], data[-2], data[-3], data[-4])
    assert not np.isnan(y_hat), "Forecast should not be NaN"
    
    print(f"  ✓ Custom parameters work correctly")
    print()


def test_small_dataset():
    """Test with small dataset."""
    print("Test 5: Small dataset (edge case)...")
    
    data = np.array([1.0, 2.0, 3.0])
    forecaster = create_kalman_forecaster(data, observation_lag=2)
    
    y_hat = forecaster(data[-1], data[-2])
    assert not np.isnan(y_hat), "Forecast should not be NaN"
    
    print(f"  ✓ Small dataset handled correctly")
    print()


def test_sequential_forecasts():
    """Test making multiple sequential forecasts."""
    print("Test 6: Sequential forecasts...")
    
    data = np.arange(30, dtype=float)
    forecaster = create_kalman_forecaster(data, observation_lag=2)
    
    # Make several forecasts
    recent = [data[-1], data[-2]]
    forecasts = []
    
    for _ in range(5):
        y_hat = forecaster(*recent)
        forecasts.append(y_hat)
        recent = [y_hat, recent[0]]
    
    assert len(forecasts) == 5, "Should have 5 forecasts"
    assert all(not np.isnan(f) for f in forecasts), "All forecasts should be valid"
    
    print(f"  ✓ Made {len(forecasts)} sequential forecasts")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Running Kalman Filter Forecaster Tests")
    print("=" * 60)
    print()
    
    try:
        test_univariate_basic()
        test_multivariate_basic()
        test_list_input()
        test_custom_parameters()
        test_small_dataset()
        test_sequential_forecasts()
        
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
