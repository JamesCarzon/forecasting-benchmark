# forecasting-benchmark

A Python library for time series forecasting using Kalman filters.

## Overview

This repository provides a Kalman filter-based forecaster that can be used for univariate and multivariate time series prediction. The forecaster learns from historical data and provides one-step ahead forecasts.

## Features

- **Kalman Filter Forecasting**: State-of-the-art recursive estimation for time series
- **Univariate and Multivariate Support**: Works with both scalar and vector-valued time series
- **Flexible Configuration**: Customizable parameters for process noise, measurement noise, and state covariance
- **Simple API**: Easy-to-use interface that returns a callable forecaster function

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import numpy as np
from src.forecast import create_kalman_forecaster

# Generate or load your time series data
data = np.sin(np.linspace(0, 4*np.pi, 100))

# Create a forecaster that uses the last 3 observations
forecaster = create_kalman_forecaster(data, observation_lag=3)

# Make a one-step ahead forecast
y_hat = forecaster(data[-1], data[-2], data[-3])
print(f"Forecast: {y_hat}")
```

## Structure

```
forecasting-benchmark/
├── src/
│   └── forecast/
│       ├── __init__.py
│       ├── kalman_filter.py  # Main implementation
│       └── README.md          # Detailed documentation
├── script/
│   ├── example_kalman_forecaster.py  # Usage examples
│   └── test_kalman_forecaster.py     # Test suite
├── requirements.txt
└── README.md
```

## Examples

See `script/example_kalman_forecaster.py` for comprehensive examples including:
- Univariate time series forecasting
- Multivariate time series forecasting
- Custom parameter configurations

Run the examples:
```bash
python script/example_kalman_forecaster.py
```

## Testing

Run the test suite:
```bash
python script/test_kalman_forecaster.py
```

## Documentation

For detailed API documentation and mathematical background, see [src/forecast/README.md](src/forecast/README.md).

## License

See [LICENSE](LICENSE) file for details.
