# Kalman Filter Forecaster

A generic Python implementation of a Kalman filter-based forecaster for time series prediction.

## Overview

The `create_kalman_forecaster` function takes a dataset (y₁, ..., yₜ), which can be vectors, and creates a forecaster function that yields one-step ahead forecasts. The forecaster uses the previous `observation_lag` time points to create predictions.

## Features

- **Univariate and Multivariate Support**: Works with both scalar and vector-valued time series
- **Flexible Configuration**: Customizable parameters for process noise, measurement noise, and initial state covariance
- **Simple Interface**: Easy-to-use function that returns a callable forecaster
- **Kalman Filtering**: Uses optimal recursive estimation for linear dynamical systems

## Usage

### Basic Example (Univariate)

```python
import numpy as np
from forecast import create_kalman_forecaster

# Generate or load your time series data
data = np.sin(np.linspace(0, 4*np.pi, 100))

# Create a forecaster that uses the last 3 observations
forecaster = create_kalman_forecaster(data, observation_lag=3)

# Make a one-step ahead forecast
y_hat = forecaster(data[-1], data[-2], data[-3])
print(f"Forecast: {y_hat}")
```

### Multivariate Example

```python
import numpy as np
from forecast import create_kalman_forecaster

# Multivariate time series (100 time points, 2 features)
data = np.random.randn(100, 2)

# Create forecaster using last 2 observations
forecaster = create_kalman_forecaster(data, observation_lag=2)

# Make forecast
y_hat = forecaster(data[-1], data[-2])
print(f"Forecast: {y_hat}")
```

### Custom Parameters

```python
forecaster = create_kalman_forecaster(
    data,
    observation_lag=4,
    process_noise_cov=1e-4,      # Low process noise (smooth dynamics)
    measurement_noise_cov=0.1,    # Higher measurement noise (noisy observations)
    initial_state_cov=0.5
)
```

## Function Signature

```python
def create_kalman_forecaster(
    data: Union[np.ndarray, List],
    observation_lag: int = 1,
    state_dim: int = None,
    process_noise_cov: float = 1e-5,
    measurement_noise_cov: float = 1e-3,
    initial_state_cov: float = 1.0
) -> Callable
```

### Parameters

- **data**: Historical time series data (shape: (T,) for univariate or (T, d) for multivariate)
- **observation_lag**: Number of previous time points used for forecasting (default: 1)
- **state_dim**: Dimension of the state space (default: observation_lag × feature_dim)
- **process_noise_cov**: Process noise covariance Q (default: 1e-5)
- **measurement_noise_cov**: Measurement noise covariance R (default: 1e-3)
- **initial_state_cov**: Initial state covariance P₀ (default: 1.0)

### Returns

A callable forecaster function with signature:
```python
forecaster(y_t, y_{t-1}, ..., y_{t-l+1}) -> y_hat_{t+1}
```

The forecaster can be called with:
- Individual arguments: `forecaster(y_t, y_{t-1}, ...)`
- A single array/list: `forecaster([y_t, y_{t-1}, ...])`

## Mathematical Background

The Kalman filter assumes a linear dynamical system:

```
x_{t+1} = A × x_t + w_t,  w_t ~ N(0, Q)
y_t = C × x_t + v_t,      v_t ~ N(0, R)
```

where:
- x_t is the hidden state
- y_t is the observation
- A is the state transition matrix
- C is the observation matrix
- Q is the process noise covariance
- R is the measurement noise covariance

The filter recursively estimates the state using:
1. **Prediction step**: Forecast the next state
2. **Update step**: Correct the prediction using new observations

## Examples

See `script/example_kalman_forecaster.py` for complete working examples including:
- Univariate time series forecasting
- Multivariate time series forecasting
- Custom parameter configurations

Run the examples:
```bash
python script/example_kalman_forecaster.py
```
