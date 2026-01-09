"""
Kalman Filter based forecaster.

This module provides a generic Kalman filter implementation for time series forecasting.
The Kalman filter is a recursive algorithm that estimates the state of a linear dynamic
system from a series of noisy measurements.
"""

import numpy as np
from typing import Callable, Union, List


def create_kalman_forecaster(
    data: Union[np.ndarray, List],
    observation_lag: int = 1,
    state_dim: int = None,
    process_noise_cov: float = 1e-5,
    measurement_noise_cov: float = 1e-3,
    initial_state_cov: float = 1.0
) -> Callable:
    """
    Create a Kalman filter-based forecaster from historical data.
    
    This function takes a dataset (y_1, ..., y_T) which can be vectors, and creates
    a forecaster function that yields one-step ahead forecasts. The forecaster uses
    the previous `observation_lag` time points to create a forecast.
    
    Parameters
    ----------
    data : array-like of shape (T,) or (T, d)
        Historical time series data where T is the number of time points.
        Can be univariate (1D array) or multivariate (2D array with d features).
    observation_lag : int, default=1
        Number of previous time points (l) used for forecasting.
        The forecaster will use y_t, y_{t-1}, ..., y_{t-l+1} to predict y_{t+1}.
    state_dim : int, optional
        Dimension of the state space. If None, defaults to observation_lag.
    process_noise_cov : float, default=1e-5
        Process noise covariance (Q). Controls the expected variance of the
        system dynamics.
    measurement_noise_cov : float, default=1e-3
        Measurement noise covariance (R). Controls the expected variance of
        the observations.
    initial_state_cov : float, default=1.0
        Initial state covariance (P_0). Represents initial uncertainty in the state.
    
    Returns
    -------
    forecaster : callable
        A function that takes previous observations and returns a one-step ahead forecast.
        Signature: forecaster(y_t, y_{t-1}, ..., y_{t-l+1}) -> y_hat_{t+1}
        
        The forecaster can be called with:
        - Individual arguments: forecaster(y_t, y_{t-1}, ...)
        - A single array/list: forecaster([y_t, y_{t-1}, ...])
    
    Examples
    --------
    >>> import numpy as np
    >>> # Univariate time series
    >>> data = np.sin(np.linspace(0, 4*np.pi, 100))
    >>> forecaster = create_kalman_forecaster(data, observation_lag=3)
    >>> # Forecast using the last 3 observations
    >>> y_hat = forecaster(data[-1], data[-2], data[-3])
    
    >>> # Multivariate time series
    >>> data = np.random.randn(100, 2)  # 100 time points, 2 features
    >>> forecaster = create_kalman_forecaster(data, observation_lag=2)
    >>> y_hat = forecaster(data[-1], data[-2])
    
    Notes
    -----
    The Kalman filter assumes a linear dynamical system:
        x_{t+1} = A * x_t + w_t,  w_t ~ N(0, Q)
        y_t = C * x_t + v_t,      v_t ~ N(0, R)
    
    where x_t is the hidden state and y_t is the observation.
    """
    # Convert data to numpy array
    data = np.asarray(data)
    
    # Determine if data is univariate or multivariate
    if data.ndim == 1:
        data = data.reshape(-1, 1)  # Convert to 2D array
        is_univariate = True
    else:
        is_univariate = False
    
    n_samples, obs_dim = data.shape
    
    # Set state dimension
    if state_dim is None:
        state_dim = observation_lag * obs_dim
    
    # Initialize Kalman filter parameters by learning from data
    # State transition matrix A (assumes AR-like dynamics)
    A = np.eye(state_dim)
    
    # If we have enough data, estimate transition dynamics
    if n_samples > observation_lag + 1:
        # Create lagged features for learning
        X = []
        Y = []
        for i in range(observation_lag, n_samples - 1):
            # Stack previous observations as state
            state_features = []
            for lag in range(observation_lag):
                state_features.append(data[i - lag])
            X.append(np.concatenate(state_features))
            
            # Next state (shifted by one time step)
            next_state_features = []
            for lag in range(observation_lag):
                next_state_features.append(data[i - lag + 1])
            Y.append(np.concatenate(next_state_features))
        
        if len(X) > 0:
            X = np.array(X)
            Y = np.array(Y)
            
            # Pad or truncate to match state_dim
            if X.shape[1] < state_dim:
                X = np.pad(X, ((0, 0), (0, state_dim - X.shape[1])), mode='constant')
                Y = np.pad(Y, ((0, 0), (0, state_dim - Y.shape[1])), mode='constant')
            else:
                X = X[:, :state_dim]
                Y = Y[:, :state_dim]
            
            # Learn state transition matrix A using least squares
            # Solve Y = X @ A for A, where x_{t+1} = A @ x_t
            # This gives A.T such that Y = X @ A.T, so A = (lstsq result).T
            # But we want x_{t+1} = A @ x_t, which in batch form is Y.T = A @ X.T
            # So A = Y.T @ X.T^+ = (X @ Y.T^+).T
            # Simpler: solve X @ A.T = Y, then A = (solution).T
            try:
                A_T = np.linalg.lstsq(X, Y, rcond=None)[0]
                A = A_T.T
            except np.linalg.LinAlgError:
                # If learning fails, keep identity matrix
                A = np.eye(state_dim)
    
    # Observation matrix C (extracts the predicted observation from state)
    C = np.zeros((obs_dim, state_dim))
    C[:, :obs_dim] = np.eye(obs_dim)
    
    # Covariance matrices
    Q = np.eye(state_dim) * process_noise_cov  # Process noise covariance
    R = np.eye(obs_dim) * measurement_noise_cov  # Measurement noise covariance
    P = np.eye(state_dim) * initial_state_cov  # Initial state covariance
    
    # Initialize state estimate
    if n_samples >= observation_lag:
        # Use last observations to initialize state
        initial_obs = []
        for i in range(min(observation_lag, n_samples)):
            initial_obs.append(data[-(i+1)])
        state_estimate = np.concatenate(initial_obs[::-1])
        
        # Pad or truncate to match state_dim
        if len(state_estimate) < state_dim:
            state_estimate = np.pad(state_estimate, (0, state_dim - len(state_estimate)), mode='constant')
        else:
            state_estimate = state_estimate[:state_dim]
    else:
        state_estimate = np.zeros(state_dim)
    
    # Store Kalman filter state
    kalman_state = {
        'A': A,
        'C': C,
        'Q': Q,
        'R': R,
        'P': P.copy(),
        'x': state_estimate.copy(),
        'obs_dim': obs_dim,
        'state_dim': state_dim,
        'is_univariate': is_univariate
    }
    
    def forecaster(*observations):
        """
        Generate one-step ahead forecast using Kalman filter.
        
        Parameters
        ----------
        *observations : array-like
            Previous observations in order y_t, y_{t-1}, ..., y_{t-l+1}.
            Can also be passed as a single list/array of observations.
        
        Returns
        -------
        y_hat : array-like
            One-step ahead forecast y_hat_{t+1}.
            Returns scalar for univariate series, array for multivariate.
        """
        # Handle different input formats
        if len(observations) == 1 and isinstance(observations[0], (list, np.ndarray)):
            # Single argument that is a list/array of observations
            obs_array = np.asarray(observations[0])
        else:
            # Multiple arguments
            obs_array = np.array(observations)
        
        # Ensure observations are 2D
        if obs_array.ndim == 1:
            if kalman_state['is_univariate']:
                obs_array = obs_array.reshape(-1, 1)
            else:
                # Multivariate case: each observation should be a vector
                obs_array = obs_array.reshape(1, -1)
        
        # Get current Kalman filter state
        A = kalman_state['A']
        C = kalman_state['C']
        Q = kalman_state['Q']
        R = kalman_state['R']
        P = kalman_state['P']
        x = kalman_state['x']
        
        # Update state based on observations (Kalman filter update step)
        # Use most recent observation for update
        if len(obs_array) > 0:
            y_current = obs_array[0].flatten()[:kalman_state['obs_dim']]
            
            # Kalman update
            # Innovation
            y_pred = C @ x
            innovation = y_current - y_pred
            
            # Innovation covariance
            S = C @ P @ C.T + R
            
            # Kalman gain
            K = P @ C.T @ np.linalg.inv(S)
            
            # Update state estimate
            x = x + K @ innovation
            
            # Update state covariance (Joseph form for numerical stability)
            I_KC = np.eye(kalman_state['state_dim']) - K @ C
            P = I_KC @ P @ I_KC.T + K @ R @ K.T
            
            # Store updated state
            kalman_state['x'] = x
            kalman_state['P'] = P
        
        # Prediction step (forecast next time step)
        x_pred = A @ x
        P_pred = A @ P @ A.T + Q
        
        # Update stored state for next iteration
        kalman_state['x'] = x_pred
        kalman_state['P'] = P_pred
        
        # Predicted observation
        y_hat = C @ x_pred
        
        # Return scalar for univariate, array for multivariate
        if kalman_state['is_univariate']:
            return float(y_hat[0])
        else:
            return y_hat
    
    return forecaster
