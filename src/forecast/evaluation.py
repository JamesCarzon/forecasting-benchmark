"""
Evaluation module for forecasting methods.

This module provides functions to evaluate forecaster performance
on test datasets with diagnostic plots and metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Union, List, Dict, Any
from math import pi, gamma


def evaluate_forecasts(
    forecaster: callable,
    train_data: np.ndarray,
    test_data: np.ndarray,
    observation_lag: int = 1,
    alpha: float = 0.32,
    rolling_window: int = 15,
    show: bool = True,
    create_forecaster_func: Callable = None
) -> Dict[str, Any]:
    """
    Apply a forecaster to a test dataset and produce diagnostic plots and metrics.

    Parameters
    ----------
    forecaster : callable
        One-step ahead forecaster function.
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
    create_forecaster_func : callable, optional
        Function to create a fresh forecaster for calibration.
        Should have signature: create_forecaster_func(data, observation_lag=observation_lag)
        If None, uses the provided forecaster for calibration.

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
    # To avoid mutating user's forecaster, recreate a fresh forecaster on train data if possible
    if create_forecaster_func is not None:
        calib_forecaster = create_forecaster_func(train, observation_lag=observation_lag)
    else:
        # Use the provided forecaster as-is for calibration
        calib_forecaster = forecaster

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
