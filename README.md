# forecasting-benchmark

A Python library for time-series forecasting using Kalman filters, with multi-step forecasts, model persistence, batch prediction, and improved configurability.

Quick installation
- For development:
```bash
pip install -e .
```
- For production:
```bash
pip install .
```

Quick start
```python
import numpy as np
from src.forecast import create_kalman_forecaster

data = np.sin(np.linspace(0, 4*np.pi, 100))

# Create a forecaster that uses the last 3 observations and predicts 5 steps ahead
forecaster = create_kalman_forecaster(data, observation_lag=3, horizon=5)

# One-series, multi-step prediction
y_hat = forecaster.predict(data[-3:])   # returns array of length `horizon`

# Batch prediction (N series x T lags)
batch = np.vstack([data[-3:], data[-3:] + 0.1])  # example batch of 2 series
batch_preds = forecaster.predict_batch(batch)

# Persist the model
forecaster.save("models/kalman_forecaster.pkl")
# later...
forecaster = create_kalman_forecaster.load("models/kalman_forecaster.pkl")
```

## License

See [LICENSE](LICENSE) file for details.
