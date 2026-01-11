"""
Kalman Filter based forecaster.

This module provides a generic Kalman filter implementation for time series forecasting.
The Kalman filter is a recursive algorithm that estimates the state of a linear dynamic
system from a series of noisy measurements.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Union, List
from math import pi, gamma
from typing import Tuple, Dict, Any


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
            if X.shape[1] != state_dim:
                if X.shape[1] < state_dim:
                    # Pad with zeros if state dimension is larger
                    X = np.pad(X, ((0, 0), (0, state_dim - X.shape[1])), mode='constant')
                    Y = np.pad(Y, ((0, 0), (0, state_dim - Y.shape[1])), mode='constant')
                else:
                    # Truncate if state dimension is smaller
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
            y_current = obs_array[0].flatten()
            
            # Validate observation dimensions
            if len(y_current) < kalman_state['obs_dim']:
                raise ValueError(
                    f"Observation dimension {len(y_current)} is less than expected "
                    f"dimension {kalman_state['obs_dim']}"
                )
            
            # Use only the expected dimensions
            y_current = y_current[:kalman_state['obs_dim']]
            
            # Kalman update
            # Innovation
            y_pred = C @ x
            innovation = y_current - y_pred
            
            # Innovation covariance
            S = C @ P @ C.T + R
            
            # Kalman gain (using solve for numerical stability)
            # K = P @ C.T @ inv(S), solve: S.T @ K.T = C @ P.T = C @ P
            K = np.linalg.solve(S, C @ P).T
            
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


def evaluate_forecasts(
    forecaster: callable,
    train_data: np.ndarray,
    test_data: np.ndarray,
    observation_lag: int = 1,
    alpha: float = 0.32,
    rolling_window: int = 15,
    show: bool = True
) -> Dict[str, Any]:
    """
    Apply a forecaster to a test dataset and produce diagnostic plots and metrics.

    Parameters
    ----------
    forecaster : callable
        One-step ahead forecaster produced by create_kalman_forecaster.
        It will be called sequentially with the true previous `observation_lag`
        observations to produce each yhat_{T+i}.
    train_data : array-like, shape (T,) or (T, d)
        Historical training data y_1..y_T used to create the forecaster (and for
        calibration of conformal intervals).
    test_data : array-like, shape (k,) or (k, d)
        Test sequence y_{T+1}..y_{T+k} to evaluate.
    observation_lag : int
        Number of previous observations forecaster expects.
    alpha : float
        Miscoverage level for conformal intervals (e.g. 0.05 for 95% intervals).
    rolling_window : int
        Window length for rolling MAE/RMSE/Coverage/Width (default 15).
    show : bool
        If True, displays matplotlib figures.

    Returns
    -------
    results : dict
        Contains keys:
            - 'yhat' : array of shape (k, d) predicted values
            - 'intervals' : dict with 'lower' and 'upper' (for univariate) or 'radius' (for multivariate)
            - 'rolling_mae', 'rolling_rmse' : arrays of length k - (rolling_window-1)
            - 'rolling_cov', 'rolling_width' : arrays of same length (coverage in [0,1], width = volume)
            - 'times_rolling' : global time indices for rolling arrays (T+rolling_window .. T+k)
            - 'times' : global time indices for test points (T+1 .. T+k)
    """
    # Ensure numpy arrays and 2D shape (n, d)
    train = np.asarray(train_data)
    test = np.asarray(test_data)

    if train.ndim == 1:
        train = train.reshape(-1, 1)
        is_univariate = True
    else:
        is_univariate = train.shape[1] == 1 and test.ndim == 1

    if test.ndim == 1 and not is_univariate:
        test = test.reshape(-1, 1)
    elif test.ndim == 1 and is_univariate:
        test = test.reshape(-1, 1)

    T = train.shape[0]
    k = test.shape[0]
    d = train.shape[1]

    # ---------- Calibration: compute residual norms on training data ----------
    # To avoid mutating user's forecaster, recreate a fresh forecaster on train data
    # Note: relies on create_kalman_forecaster being in same module
    calib_forecaster = create_kalman_forecaster(train, observation_lag=observation_lag)

    calib_preds = []
    calib_truth = []

    # Generate one-step ahead predictions across training where possible
    # For predicting y_i (index i), pass previous observations y_{i-1},...,y_{i-l}
    for i in range(observation_lag, T):
        prev_obs = []
        for lag in range(observation_lag):
            prev_obs.append(train[i - 1 - lag])
        # pass as a list (most recent first)
        yhat = calib_forecaster(prev_obs)
        calib_preds.append(np.asarray(yhat).reshape(-1))
        calib_truth.append(train[i].reshape(-1))

    if len(calib_preds) == 0:
        # fallback: small set of residuals from last training point using last obs
        # produce a single-step prediction
        prev_obs = [train[-1 - i] for i in range(min(observation_lag, T))]
        yhat = calib_forecaster(prev_obs)
        calib_preds = [np.asarray(yhat).reshape(-1)]
        calib_truth = [train[-1].reshape(-1)]

    calib_preds = np.vstack(calib_preds)
    calib_truth = np.vstack(calib_truth)

    # Residual norms for conformal calibration
    calib_residuals = calib_truth - calib_preds
    calib_norms = np.linalg.norm(calib_residuals.reshape(len(calib_residuals), -1), axis=1)
    q = np.quantile(calib_norms, 1.0 - alpha)  # radius for ball around prediction

    # ---------- Run forecaster on test data (sequentially using true previous obs) ----------
    # We'll mutate the provided forecaster (intended use)
    history = [row for row in train]  # list of vectors
    yhat_list = []

    for i in range(k):
        # build previous l observations (most recent first)
        prev_obs = []
        for lag in range(observation_lag):
            idx = len(history) - 1 - lag
            if idx < 0:
                # not enough history: pad with zeros
                prev_obs.append(np.zeros(d))
            else:
                prev_obs.append(history[idx])
        yhat = forecaster(prev_obs)
        yhat = np.asarray(yhat).reshape(-1)
        yhat_list.append(yhat)
        # append true test observation for future predictions (forecaster uses true prev obs)
        history.append(test[i].reshape(-1))

    yhat = np.vstack(yhat_list)  # shape (k, d)

    # ---------- Conformal sets ----------
    # For multivariate: ball centered at yhat with radius q
    # For univariate: interval [yhat - q, yhat + q]
    if d == 1:
        lower = (yhat[:, 0] - q)
        upper = (yhat[:, 0] + q)
        intervals = {'lower': lower, 'upper': upper}
        # width per-prediction:
        widths = upper - lower  # constant 2*q
    else:
        intervals = {'radius': q}
        # volume of d-ball with radius q
        vol_const = (pi ** (d / 2.0)) / gamma(d / 2.0 + 1.0)
        vol_q = vol_const * (q ** d)
        widths = np.full(k, vol_q)  # volume per prediction

    # ---------- Rolling metrics ----------
    # For rolling metrics we need element-wise errors for test points (0..k-1)
    errors = test.reshape(k, d) - yhat.reshape(k, d)
    abs_errors = np.linalg.norm(errors.reshape(k, d), axis=1) if d > 1 else np.abs(errors.ravel())
    sq_errors = np.sum(errors ** 2, axis=1) if d > 1 else (errors.ravel() ** 2)

    if k < rolling_window:
        raise ValueError(f"Test length k={k} is smaller than rolling_window={rolling_window}.")

    n_roll = k - (rolling_window - 1)
    rolling_mae = np.empty(n_roll)
    rolling_rmse = np.empty(n_roll)
    rolling_cov = np.empty(n_roll)
    rolling_width = np.empty(n_roll)

    # Precompute indicators for coverage: 1 if true in conformal set
    if d == 1:
        in_set = ((test.ravel() >= intervals['lower']) & (test.ravel() <= intervals['upper'])).astype(float)
    else:
        # check norm <= q
        test_norms = np.linalg.norm((test - yhat).reshape(k, d), axis=1)
        in_set = (test_norms <= q).astype(float)

    for idx in range(n_roll):
        start = idx
        end = idx + rolling_window  # exclusive
        window_abs = abs_errors[start:end]
        window_sq = sq_errors[start:end]
        rolling_mae[idx] = float(np.mean(window_abs))
        rolling_rmse[idx] = float(np.sqrt(np.mean(window_sq)))
        rolling_cov[idx] = float(np.mean(in_set[start:end]))
        rolling_width[idx] = float(np.mean(widths[start:end]))

    times = np.arange(T + 1, T + k + 1)  # global times for test points
    times_rolling = np.arange(T + rolling_window, T + k + 1)  # global times for rolling metrics

    results = {
        'yhat': yhat,
        'intervals': intervals,
        'rolling_mae': rolling_mae,
        'rolling_rmse': rolling_rmse,
        'rolling_cov': rolling_cov,
        'rolling_width': rolling_width,
        'times': times,
        'times_rolling': times_rolling,
        'alpha': alpha,
        'q_radius': q
    }

    # ---------- Plots (four separate figures) ----------
    if show:
        # 1) Rolling MAE & RMSE (figure size 8x6)
        plt.figure(figsize=(8, 6))
        plt.plot(times_rolling, rolling_mae, label=f'Mean absolute error (MAE)', color='C0')
        plt.plot(times_rolling, rolling_rmse, label=f'Root mean square error (RMSE)', color='C1')
        plt.xlabel('Time (t)')
        plt.ylabel('Error')
        plt.xlim(left=times_rolling[0], right=times_rolling[-1])
        plt.ylim(bottom=0)
        plt.suptitle(f'Forecast test error', fontsize=14)
        plt.title(f'Input lag = {observation_lag} // Rolling test window size = {rolling_window}', fontsize=10)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # 2) True vs Predicted (univariate only) (figure size 8x6)
        if d == 1:
            plt.figure(figsize=(8, 6))
            plt.plot(times, test.ravel(), label=r'$y_{t}$', color='black')
            plt.plot(times, yhat.ravel(), label=r'$\hat{y_t}$', color='C2', linestyle='--')
            plt.xlabel('Time (t)')
            plt.ylabel('Outcome (Y)')
            plt.xlim(left=times[0], right=times[-1])
            plt.suptitle('Forecasted vs True values', fontsize=14)
            plt.title(f'Input lag = {observation_lag}', fontsize=10)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()        

        # 3) Conformal intervals (univariate only) (figure size 8x6)
        if d == 1:
            plt.figure(figsize=(8, 6))
            plt.plot(times, test.ravel(), label=r'$y_{t}$', color='black')
            plt.plot(times, yhat.ravel(), label=r'$\hat{y_t}$', color='C2', linestyle='--')
            plt.fill_between(times, intervals['lower'], intervals['upper'], color='C3', alpha=0.25,
                             label=f'{100*(1-alpha):.1f}% conformal interval')
            plt.xlabel('Time (t)')
            plt.ylabel('Outcome (Y)')
            plt.xlim(left=times[0], right=times[-1])
            plt.suptitle('Split conformal prediction intervals', fontsize=14)
            plt.title(f'Input lag = {observation_lag} // Miscoverage level $\\alpha$ = {alpha}', fontsize=10)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        # 4) Rolling coverage and width side-by-side (figure size 10x6)
        plt.figure(figsize=(10, 6))
        ax_left = plt.subplot2grid((1, 2), (0, 0))
        ax_right = plt.subplot2grid((1, 2), (0, 1))

        ax_left.plot(times_rolling, rolling_cov, color='C4')
        ax_left.axhline(1.0 - alpha, color='k', linestyle=':', label=f'Nominal rate = {100*(1-alpha):.0f}\%')
        ax_left.set_ylim(-0.05, 1.05)
        ax_left.set_xlim(left=times_rolling[0], right=times_rolling[-1])
        ax_left.set_xlabel('Time (t)')
        ax_left.set_ylabel('Coverage (\%)')
        ax_left.set_title('Coverage rate', fontsize=10)
        ax_left.grid(True)
        ax_left.legend()

        ax_right.plot(times_rolling, rolling_width, color='C5')
        ax_right.set_xlim(left=times_rolling[0], right=times_rolling[-1])
        ax_right.set_ylim(bottom=0)
        ax_right.set_xlabel('Time (t)')
        ax_right.set_ylabel('Width / Volume of conformal set')
        ax_right.set_title('Set size', fontsize=10)
        ax_right.grid(True)

        plt.suptitle('Split conformal set properties', fontsize=14)
        plt.tight_layout()
        plt.show()

    return results
