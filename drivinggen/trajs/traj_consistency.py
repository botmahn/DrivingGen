"""traj_consistency.py – Within-trajectory velocity and acceleration smoothness metrics.

A trajectory is considered "consistent" if its speed and acceleration profiles
are steady over time — i.e. low coefficient-of-variation for both velocity and
acceleration.  The score is computed per trajectory and optionally reduced to a
dataset-level scalar via nanmean.

Implemented metrics
-------------------
1. ``trajectory_consistency`` – Combined velocity + acceleration smoothness score
       S = 0.5 * [exp(−σ_v / (μ_v + ε))  +  exp(−σ_a / (μ_a + ε))]
2. ``get_traj_consistency``   – Dataset-level nanmean wrapper
"""

import numpy as np
from typing import Tuple, Union, Iterable

# Type alias: accepts either a NumPy array or any Python iterable.
ArrayLike = Union[np.ndarray, Iterable]


# ---------------------------------------------------------------------------
# Internal preprocessing helper
# ---------------------------------------------------------------------------

def _prep_xy(arr: np.ndarray, axis: int):
    """Reshape a trajectory array to canonical (N, T, 2) form.

    Steps:
      1. Discard channels beyond the first two (retain only x and y).
      2. Move the specified time axis to position -2 so the layout becomes
         (..., T, 2).
      3. Flatten all leading batch dimensions into a single N axis.

    Args:
        arr  (np.ndarray): Input trajectory array, shape (..., T, C) with C >= 2.
        axis (int): Index of the time dimension in ``arr``.

    Returns:
        tuple:
            - arr (np.ndarray): Flattened x-y array, shape (N, T, 2).
            - batch_shape (tuple): Original leading dimensions, e.g. ``(B,)``
              or ``(B1, B2)``.  Used by callers to restore the output shape.
    """
    # Keep only x and y; discard z or any extra channels.
    # arr: (..., T, C) -> (..., T, 2)
    arr = arr[..., :2]

    # Move the caller-specified time axis to the second-to-last position.
    # arr: (..., T, 2)  (no-op when axis == -2)
    arr = np.moveaxis(arr, axis, -2)          # shape: (..., T, 2)

    # Record the batch prefix so the caller can reshape outputs back.
    batch_shape = arr.shape[:-2]

    T = arr.shape[-2]

    # Collapse all batch dimensions into one for vectorised operations.
    arr = arr.reshape(-1, T, 2)               # shape: (N, T, 2)

    return arr, batch_shape


# ---------------------------------------------------------------------------
# Trajectory consistency score
# ---------------------------------------------------------------------------

def trajectory_consistency(
    traj_xy: ArrayLike,
    *,
    dt: float = 0.1,
    axis: int = -2,
    eps: float = 1e-9,
    v_static: float = 0.1,
    reduce: str = "none",
) -> np.ndarray | float:
    """Compute a velocity + acceleration smoothness score for each trajectory.

    The score measures how *consistent* the speed and acceleration profiles are
    over time.  It is defined as the arithmetic mean of two exponential terms:

        S = 0.5 * exp(-sigma_v / (mu_v + eps))
          + 0.5 * exp(-sigma_a / (mu_a + eps))

    Where:
        - mu_v    = mean speed magnitude over the trajectory
        - sigma_v = standard deviation of speed magnitude
        - mu_a    = mean |acceleration| over the trajectory
        - sigma_a = standard deviation of acceleration

    Each exponential term is the negative coefficient-of-variation mapped to
    [0, 1]:
        - -> 1  when the signal is perfectly constant (sigma = 0)
        - -> 0  when variability dominates (sigma >> mu)

    A score near 1 indicates very smooth, steady motion.  Trajectories whose
    maximum speed never reaches ``v_static`` are classified as "static" and
    assigned NaN (excluded from dataset-level averages).

    Args:
        traj_xy   (ArrayLike): Trajectory array, shape (..., T, C) with C >= 2.
            Only the first two channels (x, y) are used.
        dt        (float): Sampling interval in seconds.  Default 0.1 (10 Hz).
        axis      (int): Index of the time dimension.  Default -2.
        eps       (float): Small constant added to denominators to avoid
            division by zero.  Default 1e-9.
        v_static  (float): Speed threshold (m/s) below which a trajectory is
            considered static.  Default 0.1 m/s.
        reduce    (str): If ``"none"`` (default) return per-trajectory scores;
            any other value returns the scalar nanmean over the batch.

    Returns:
        np.ndarray | float:
            - ``reduce == "none"``: array of shape ``batch_shape``, dtype float.
              NaN for static trajectories.
            - otherwise: scalar float nanmean score across moving trajectories.
    """
    # Normalise input and reshape to (N, T, 2).
    xy, bs = _prep_xy(np.asarray(traj_xy, float), axis)          # shape: (N, T, 2)

    # Finite-difference velocity: central displacement normalised by dt.
    # np.diff(xy, axis=1): shape (N, T-1, 2)  — displacement vectors
    # divided by dt:        shape (N, T-1, 2)  — velocity vectors
    # norm:                 shape (N, T-1)     — speed magnitude at each step
    v = np.linalg.norm(np.diff(xy, axis=1) / dt, axis=-1)        # shape: (N, T-1)

    # Initialise output with NaN so static trajectories remain NaN by default.
    scores = np.full(v.shape[0], np.nan, dtype=float)            # shape: (N,)

    # Determine which trajectories ever exceed the static speed threshold.
    # A trajectory is "moving" if its peak speed is at least v_static.
    moving = v.max(axis=1) >= v_static                           # shape: (N,), bool

    if moving.any():
        # Work only with the M moving trajectories to avoid polluting results.
        v_m = v[moving]                                          # shape: (M, T-1)

        # --- Velocity smoothness term ---
        # mu_v:    mean speed per trajectory,  shape (M,)
        # sigma_v: std  speed per trajectory,  shape (M,)
        mu_v    = v_m.mean(axis=1)                               # shape: (M,)
        sigma_v = v_m.std(axis=1)                                # shape: (M,)

        # Velocity smoothness: exp(-CV_v) where CV_v = sigma_v / mu_v.
        # -> 1 for perfectly constant speed, -> 0 for highly variable speed.
        scores[moving] = np.exp(-sigma_v / (mu_v + eps))        # shape: (M,)

        # --- Acceleration smoothness term ---
        # Scalar acceleration: first difference of speed divided by dt.
        # np.diff(v_m, axis=1): shape (M, T-2)
        # divided by dt:        shape (M, T-2) — acceleration at each step
        a_m = np.diff(v_m, axis=1) / dt                          # shape: (M, T-2)

        # mu_a:    mean |acceleration|,  shape (M,)
        # sigma_a: std  acceleration,    shape (M,)
        mu_a    = np.mean(np.abs(a_m), axis=1)                   # shape: (M,)
        sigma_a = np.std(a_m, axis=1)                            # shape: (M,)

        # Acceleration smoothness: exp(-CV_a).
        # Accumulated additively; then multiplied by 0.5 to form the average.
        scores[moving] += np.exp(-sigma_a / (mu_a + eps))        # shape: (M,)

        # Normalise: average the two smoothness terms (velocity + acceleration).
        scores[moving] *= 0.5                                    # shape: (M,)

    # Restore the original batch shape for the output.
    out = scores.reshape(bs)
    return out if reduce == "none" else float(np.nanmean(out))


# ---------------------------------------------------------------------------
# Convenience wrapper (dataset-level nanmean)
# ---------------------------------------------------------------------------

def get_traj_consistency(preds):
    """Compute the dataset-level mean trajectory consistency, ignoring NaN entries.

    Wraps :func:`trajectory_consistency` with ``reduce="none"`` and applies
    :func:`numpy.nanmean` to safely average over static trajectories (NaN)
    and any padding that may be present in the batch.

    Args:
        preds (ArrayLike): Batch of predicted trajectories, shape (N, T, C).
            Only the first two channels (x, y) are used.

    Returns:
        float: Scalar mean consistency score across all moving trajectories.
            Returns NaN if all trajectories are static.
    """
    # tc: shape (N,)  — per-trajectory consistency scores (NaN for static ones)
    tc = trajectory_consistency(preds, reduce='none')
    tc = np.nanmean(tc)   # scalar, ignores NaN from static trajectories
    return float(tc)
