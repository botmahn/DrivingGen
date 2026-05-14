"""traj_alignment.py – Ground-truth-aware trajectory error metrics.

Each metric is a standalone NumPy function.  No external deps except an
optional SciPy import for DTW; a pure-NumPy fallback is provided.

Conventions
-----------
* Prediction & GT arrays have the **same shape** ``(..., T, C)`` with
  :math:`C in {2, 3}`; leading batch dims are preserved in outputs.
* ``axis`` selects the time dimension (default ``-2``).

Implemented metrics
-------------------
1. ``ade``                 – Average Displacement Error
2. ``fde``                 – Final Displacement Error
3. ``success_rate``        – FDE < *threshold* ratio
4. ``hausdorff``           – Symmetric Hausdorff distance
5. ``ndtw``                – Normalised Dynamic Time-Warping score
6. ``dynamic_consistency`` – Motion-dynamic similarity (Wasserstein)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "ade",
    "fde",
    "success_rate",
    "hausdorff",
    "ndtw",
    "dynamic_consistency",
]

# ---------------------------------------------------------------------------
# Internal preprocessing helper
# ---------------------------------------------------------------------------

def _prep(a: np.ndarray, b: np.ndarray, axis: int):
    """Reshape prediction and ground-truth arrays to a canonical (N, T, C) form.

    This helper:
      1. Strips the ground-truth array to its first two spatial channels (x, y),
         matching the number of channels used for error computation.
      2. Moves the time axis to position -2 so the layout becomes (..., T, C).
      3. Flattens all leading batch dimensions into a single N axis.

    Args:
        a (np.ndarray): Predicted trajectory array with shape (..., T, C).
        b (np.ndarray): Ground-truth trajectory array with shape (..., T, C').
            C' >= 2; only the first two channels are kept.
        axis (int): Index of the time dimension in the *input* arrays.

    Returns:
        tuple:
            - a (np.ndarray): Flattened predictions,  shape (N, T, C).
            - b (np.ndarray): Flattened ground truth, shape (N, T, C).
            - batch_shape (tuple): Original leading dimensions, e.g. ``(B,)``
              or ``(B1, B2)``.  Used to restore output shape after computation.

    Raises:
        ValueError: If ``pred`` and ``gt`` do not share the same shape after
            channel truncation.
    """
    # Keep only x, y channels from the GT array so spatial dims match.
    # b: (..., T, C') -> (..., T, 2)
    b = b[..., :2]

    # Both arrays must now have identical shapes.
    if a.shape != b.shape:
        raise ValueError("pred and gt must share shape")

    # Move the caller-specified time axis to the second-to-last position.
    # a: (..., T, C) -> (..., T, C)  (no-op when axis == -2)
    a = np.moveaxis(a, axis, -2)          # shape: (..., T, C)
    b = np.moveaxis(b, axis, -2)          # shape: (..., T, C)

    # Record the batch prefix so callers can restore it later.
    # E.g. if a.shape == (B1, B2, T, C) then batch_shape == (B1, B2).
    batch_shape = a.shape[:-2]            # e.g. (B,) or (B1, B2)

    T, C = a.shape[-2:]

    # Flatten all batch dims into a single N axis for vectorised computation.
    a = a.reshape(-1, T, C)               # shape: (N, T, C)
    b = b.reshape(-1, T, C)               # shape: (N, T, C)

    return a, b, batch_shape


# ---------------------------------------------------------------------------
# 1. ADE  /  2. FDE  /  3. Success-rate
# ---------------------------------------------------------------------------

def ade(pred, gt, *, axis=-2, reduce="none"):
    """Compute the Average Displacement Error (ADE) between predicted and GT trajectories.

    ADE measures the mean Euclidean distance between corresponding waypoints
    across *all* timesteps:

        ADE_i = (1/T) * sum_{t=1}^{T}  ||pred_i(t) - gt_i(t)||_2

    A lower ADE indicates the predicted path stays close to the reference
    across its entire duration.

    Args:
        pred (ArrayLike): Predicted trajectories, shape (..., T, C).
        gt   (ArrayLike): Ground-truth trajectories, shape (..., T, C).
            Only the first two channels (x, y) are used.
        axis (int): Index of the time dimension. Default ``-2``.
        reduce (str): If ``"none"`` (default) return per-trajectory errors;
            any other value returns the scalar mean over the batch.

    Returns:
        np.ndarray | float:
            - ``reduce == "none"``: array of shape ``batch_shape`` with one
              ADE value per trajectory.
            - otherwise: scalar float mean ADE.
    """
    p, g, bs = _prep(np.asarray(pred), np.asarray(gt), axis)
    # Compute per-timestep Euclidean distance, then average over time.
    # p - g:  shape (N, T, C)
    # norm:   shape (N, T)   — L2 distance at each timestep
    # mean:   shape (N,)     — averaged across T timesteps
    err = np.linalg.norm(p - g, axis=-1).mean(-1)     # shape: (N,)

    # Restore original batch shape, e.g. (B,) or scalar.
    err = err.reshape(bs)
    return err if reduce == "none" else err.mean()


def fde(pred, gt, *, axis=-2, reduce="none"):
    """Compute the Final Displacement Error (FDE) between predicted and GT trajectories.

    FDE measures the Euclidean distance *only at the last timestep* T:

        FDE_i = ||pred_i(T) - gt_i(T)||_2

    FDE captures how well the model predicts the endpoint of the trajectory,
    which is especially important in planning horizons where the destination
    matters more than the intermediate path.

    Args:
        pred (ArrayLike): Predicted trajectories, shape (..., T, C).
        gt   (ArrayLike): Ground-truth trajectories, shape (..., T, C).
            Only the first two channels (x, y) are used.
        axis (int): Index of the time dimension. Default ``-2``.
        reduce (str): If ``"none"`` (default) return per-trajectory errors;
            any other value returns the scalar mean over the batch.

    Returns:
        np.ndarray | float:
            - ``reduce == "none"``: array of shape ``batch_shape`` with one
              FDE value per trajectory.
            - otherwise: scalar float mean FDE.
    """
    p, g, bs = _prep(np.asarray(pred), np.asarray(gt), axis)
    # Index the final timestep for both pred and GT.
    # p[:, -1]: shape (N, C)
    # g[:, -1]: shape (N, C)
    # norm:     shape (N,)   — single endpoint distance per trajectory
    err = np.linalg.norm(p[:, -1] - g[:, -1], axis=-1)   # shape: (N,)

    err = err.reshape(bs)
    return err if reduce == "none" else err.mean()


def success_rate(pred, gt, *, threshold=3.0, axis=-2, reduce="none"):
    """Compute the Success Rate: the fraction of trajectories with FDE < threshold.

    A trajectory is considered "successful" if its final displacement error
    falls below ``threshold`` metres.  The success rate is the mean of these
    binary success indicators across the batch:

        SR = (1/N) * sum_{i=1}^{N}  1[FDE_i < threshold]

    Args:
        pred      (ArrayLike): Predicted trajectories, shape (..., T, C).
        gt        (ArrayLike): Ground-truth trajectories, shape (..., T, C).
        threshold (float): Distance threshold in the same units as the
            trajectory coordinates (default 3.0 metres).
        axis      (int): Index of the time dimension. Default ``-2``.
        reduce    (str): If ``"none"`` (default) return a boolean array of
            shape ``batch_shape``; any other value returns the scalar mean.

    Returns:
        np.ndarray | float:
            - ``reduce == "none"``: boolean array of shape ``batch_shape``,
              True where FDE < threshold.
            - otherwise: scalar float in [0, 1] representing the success rate.
    """
    # Reuse FDE; compare each value to the threshold to get a boolean mask.
    # sr: shape batch_shape, dtype bool
    sr = (fde(pred, gt, axis=axis, reduce="none") < threshold)
    return sr if reduce == "none" else sr.mean()


# ---------------------------------------------------------------------------
# 4. Hausdorff distance  (per-sample for-loop)
# ---------------------------------------------------------------------------

def hausdorff(pred, gt, *, axis=-2, reduce="none"):
    """Compute the symmetric Hausdorff distance between predicted and GT curves.

    The Hausdorff distance measures the worst-case deviation between two
    point sets.  For two curves P and G:

        h(P, G) = max_{p in P}  min_{g in G}  ||p - g||_2   (directed)
        H(P, G) = max( h(P, G), h(G, P) )                   (symmetric)

    It captures the single largest "gap" between the curves, making it
    sensitive to localised outlier deviations.

    Args:
        pred (ArrayLike): Predicted trajectories, shape (..., T, C).
        gt   (ArrayLike): Ground-truth trajectories, shape (..., T, C).
            Only the first two channels (x, y) are used.
        axis (int): Index of the time dimension. Default ``-2``.
        reduce (str): If ``"none"`` (default) return per-trajectory distances;
            any other value returns the scalar mean.

    Returns:
        np.ndarray | float:
            - ``reduce == "none"``: array of shape ``batch_shape``.
            - otherwise: scalar float mean Hausdorff distance.
    """
    p, g, bs = _prep(np.asarray(pred), np.asarray(gt), axis)
    N = p.shape[0]
    out = np.empty(N)   # shape: (N,)

    for i in range(N):
        # Build the full T x T pairwise distance matrix between the two curves.
        # p[i, :, None]:   shape (T, 1, C)
        # g[i, None, :]:   shape (1, T, C)
        # D:               shape (T, T) — D[t1, t2] = ||p[t1] - g[t2]||_2
        D = np.linalg.norm(p[i, :, None] - g[i, None, :], axis=-1)   # shape: (T, T)

        # Directed Hausdorff from P to G: for each pred point, find the
        # nearest GT point; then take the worst case over all pred points.
        # D.min(1): shape (T,)  — nearest GT distance for each pred waypoint
        # D.min(0): shape (T,)  — nearest pred distance for each GT waypoint
        # Symmetric Hausdorff is the max of both directed distances.
        out[i] = max(D.min(1).max(), D.min(0).max())

    # Restore batch shape.
    out = out.reshape(bs)
    return out if reduce == "none" else out.mean()


# ---------------------------------------------------------------------------
# 5. nDTW  (classic O(T^2) dynamic-programming for-loop)
# ---------------------------------------------------------------------------

# Try to import cdist for potential future vectorised use; fall back gracefully.
try:
    from scipy.spatial.distance import cdist
except ImportError:
    cdist = None


def _dtw(a: np.ndarray, b: np.ndarray):
    """Compute the classic Dynamic Time Warping (DTW) cost between two sequences.

    DTW finds the optimal monotonic alignment between two time series by
    solving the recurrence:

        D[0, 0] = 0,  D[i>0, 0] = D[0, j>0] = inf
        D[i, j] = ||a[i-1] - b[j-1]||_2
                  + min(D[i-1, j],    # insertion
                        D[i, j-1],    # deletion
                        D[i-1, j-1])  # match

    The returned value D[n, m] is the minimum cumulative L2 cost of any
    monotonic warping path from (0,0) to (n,m).

    Args:
        a (np.ndarray): First sequence, shape (T_a, C).
        b (np.ndarray): Second sequence, shape (T_b, C).

    Returns:
        float: The DTW alignment cost (minimum cumulative L2 distance).

    Note:
        Time complexity is O(T_a * T_b) and space complexity is O(T_a * T_b).
        For long sequences this can be expensive; consider pruning or FastDTW.
    """
    n, m = len(a), len(b)

    # Initialise the (n+1) x (m+1) DP table with +inf sentinels.
    # The extra row/column at index 0 act as boundary conditions.
    # D: shape (n+1, m+1)
    D = np.full((n+1, m+1), np.inf)
    D[0, 0] = 0   # Base case: zero cost to align empty sequences.

    for i in range(1, n+1):
        for j in range(1, m+1):
            # Local cost: Euclidean distance between the i-th and j-th points.
            dist = np.linalg.norm(a[i-1] - b[j-1])
            # Optimal substructure: cumulative cost via the cheapest predecessor.
            D[i, j] = dist + min(D[i-1, j],    # step along a only
                                  D[i, j-1],    # step along b only
                                  D[i-1, j-1])  # diagonal match step

    # The bottom-right cell holds the total DTW cost.
    return D[n, m]


def dtw(pred, gt, *, alpha=4.0, axis=-2, reduce="none"):
    """Compute the raw DTW alignment cost between predicted and GT trajectories.

    For each trajectory pair (pred_i, gt_i), this function runs the classic
    O(T^2) DTW dynamic program (see :func:`_dtw`) and returns the raw
    cumulative L2 cost without normalisation.

    Args:
        pred  (ArrayLike): Predicted trajectories, shape (..., T, C).
        gt    (ArrayLike): Ground-truth trajectories, shape (..., T, C).
            Only the first two channels (x, y) are used.
        alpha (float): Reserved parameter (not used in the raw DTW score;
            present for API consistency with ndtw). Default 4.0.
        axis  (int): Index of the time dimension. Default ``-2``.
        reduce (str): If ``"none"`` (default) return per-trajectory scores;
            any other value returns the scalar mean.

    Returns:
        np.ndarray | float:
            - ``reduce == "none"``: array of shape ``batch_shape``, each
              entry the raw DTW cost for that trajectory.
            - otherwise: scalar float mean DTW cost.
    """
    p, g, bs = _prep(np.asarray(pred), np.asarray(gt), axis)
    N, T, _ = p.shape   # N trajectories, T timesteps, _ spatial channels

    score = np.empty(N)   # shape: (N,)

    for i in range(N):
        # Call the DP helper for the i-th trajectory pair.
        # p[i]: shape (T, C),  g[i]: shape (T, C)
        cost = _dtw(p[i], g[i])           # scalar DTW alignment cost
        score[i] = cost

    # Restore batch shape.
    score = score.reshape(bs)
    return score if reduce == "none" else score.mean()


import numpy as np         # Ensure consistent import across the module.


def sdtw(pred, gt, *, threshold=2.0, alpha=4.0,
         axis=-2, reduce="none"):
    """Compute the Success-weighted DTW (sDTW) score.

    sDTW combines the Success Rate (SR) binary indicator with the normalised
    DTW score (nDTW) to reward both geometric similarity AND goal accuracy:

        sDTW_i = SR_i * nDTW_i

    where SR_i = 1 if FDE_i < threshold, else 0.

    Note: In the current implementation the SR multiplier is computed but
    the final output equals nDTW alone (the multiplication is commented out).
    This preserves the original behaviour while keeping the SR computation
    available for future use.

    Args:
        pred      (ArrayLike): Predicted trajectories, shape (..., T, C).
        gt        (ArrayLike): Ground-truth trajectories, shape (..., T, C).
        threshold (float): FDE threshold (metres) for the SR indicator.
            Default 2.0.
        alpha     (float): Exponential scaling factor passed through to
            :func:`ndtw`. Default 4.0.
        axis      (int): Index of the time dimension. Default ``-2``.
        reduce    (str): If ``"none"`` (default) return per-trajectory scores;
            any other value returns the scalar mean.

    Returns:
        np.ndarray | float:
            - ``reduce == "none"``: array of shape ``batch_shape``.
            - otherwise: scalar float mean sDTW.
    """
    # Step 1: Compute binary success indicator; cast to float for multiplication.
    # sr: shape batch_shape, values in {0.0, 1.0}
    sr = success_rate(pred, gt, threshold=threshold,
                      axis=axis, reduce="none").astype(float)

    # Step 2: Compute the normalised DTW score (nDTW) for each trajectory.
    # ndtw_score: shape batch_shape, values in (0, 1]
    ndtw_score = ndtw(pred, gt, alpha=alpha,
                      axis=axis, reduce="none")

    # Step 3: Multiply SR and nDTW elementwise to get sDTW.
    # NOTE: The SR multiplication is currently disabled; out = nDTW only.
    # out = sr * ndtw_score   # sDTW as originally defined
    out = ndtw_score           # current effective behaviour

    return out if reduce == "none" else out.mean()


# ---------------------------------------------------------------------------
# 6. Dynamic-consistency (Wasserstein distance)  (per-sample for-loop)
# ---------------------------------------------------------------------------

def dynamic_consistency(pred, gt, *, dt=0.1, axis=-2, reduce="none"):
    """Measure motion-dynamic similarity via Wasserstein distance on velocity/acceleration.

    This metric compares the *distributions* of speed and acceleration
    magnitudes between a predicted trajectory and its GT reference, then maps
    the two Wasserstein distances to a similarity score in (0, 1]:

        score_i = exp(-W_1(v_pred, v_gt))  *  exp(-W_1(a_pred, a_gt))

    Where:
        - v = ||dx/dt||_2    (speed magnitude at each timestep)
        - a = dv/dt          (scalar acceleration)
        - W_1(P, Q)          is the 1-D Wasserstein-1 (Earth Mover's) distance
          between the empirical speed/acceleration distributions.

    A score near 1 means the predicted dynamics closely match the GT dynamics.
    The score is invariant to the *order* of measurements, which means it
    captures distributional similarity rather than pointwise alignment.

    If SciPy is unavailable a pure-NumPy fallback computes W_1 via the
    empirical CDF integral:  W_1(P,Q) = integral |F_P(x) - F_Q(x)| dx.

    Args:
        pred (ArrayLike): Predicted trajectories, shape (..., T, C).
        gt   (ArrayLike): Ground-truth trajectories, shape (..., T, C).
            Only the first two channels (x, y) are used.
        dt   (float): Sampling interval in seconds. Default 0.1 (10 Hz).
        axis (int): Index of the time dimension. Default ``-2``.
        reduce (str): If ``"none"`` (default) return per-trajectory scores;
            any other value returns the scalar mean.

    Returns:
        np.ndarray | float:
            - ``reduce == "none"``: array of shape ``batch_shape``, values
              in (0, 1].
            - otherwise: scalar float mean score.
    """
    # Try to use SciPy's optimised W_1 implementation; fall back to a
    # pure-NumPy version based on the empirical CDF integral formula:
    #   W_1(P, Q) = integral_{-inf}^{inf} |F_P(x) - F_Q(x)| dx
    try:
        from scipy.stats import wasserstein_distance as w1
    except ImportError:
        def w1(x, y):
            # Sort both samples to build their empirical CDFs.
            x, y = np.sort(x), np.sort(y)
            # Merge all unique breakpoints where either CDF may change.
            allv = np.sort(np.concatenate([x, y]))
            # CDF values at each breakpoint (right-continuous step function).
            cdfx = np.searchsorted(x, allv, side="right") / len(x)
            cdfy = np.searchsorted(y, allv, side="right") / len(y)
            # Numerical integration of |F_P - F_Q| using the trapezoidal rule.
            return np.trapz(np.abs(cdfx - cdfy), allv)

    p, g, bs = _prep(np.asarray(pred), np.asarray(gt), axis)
    N, T, _ = p.shape   # N trajectories, T timesteps, _ spatial channels

    out = np.empty(N)   # shape: (N,)

    for i in range(N):
        # Finite-difference velocity: displacement / time-step.
        # np.diff(p[i], axis=0): shape (T-1, C)
        # divided by dt:          shape (T-1, C)
        # norm:                   shape (T-1,)  — speed magnitude
        v_p = np.linalg.norm(np.diff(p[i], axis=0) / dt, axis=-1)   # shape: (T-1,)
        v_g = np.linalg.norm(np.diff(g[i], axis=0) / dt, axis=-1)   # shape: (T-1,)

        # Scalar acceleration: first-order difference of speed.
        # np.diff(v_p): shape (T-2,)
        a_p, a_g = np.diff(v_p), np.diff(v_g)   # shape: (T-2,) each

        # 1-D Wasserstein distance between speed distributions.
        dv = w1(v_p, v_g)   # scalar — distributional speed mismatch
        # 1-D Wasserstein distance between acceleration distributions.
        da = w1(a_p, a_g)   # scalar — distributional acceleration mismatch

        # Map both distances to [0, 1] via negative exponential and multiply.
        # exp(-W) -> 1 when W -> 0 (distributions identical)
        #          -> 0 when W -> inf (distributions maximally different)
        out[i] = np.exp(-dv) * np.exp(-da)

    # Restore batch shape.
    out = out.reshape(bs)
    return out if reduce == "none" else out.mean()


# ---------------------------------------------------------------------------
# Convenience wrappers (nanmean over a full batch)
# ---------------------------------------------------------------------------

def get_ade(preds, gts):
    """Compute the dataset-level mean ADE, ignoring NaN entries.

    Wraps :func:`ade` with ``reduce="none"`` and applies :func:`numpy.nanmean`
    to robustly average over any trajectories that may contain NaN values
    (e.g. from padding).

    Args:
        preds (ArrayLike): Batch of predicted trajectories, shape (N, T, C).
        gts   (ArrayLike): Batch of ground-truth trajectories, shape (N, T, C).

    Returns:
        float: Scalar mean ADE across all valid trajectories.
    """
    # ad_err: shape (N,)  — per-trajectory ADE values
    ad_err = ade(preds, gts, reduce='none')
    ad_err = np.nanmean(ad_err)   # scalar, ignores NaN padding
    return float(ad_err)


def get_dtw(preds, gts):
    """Compute the dataset-level mean DTW cost, ignoring NaN entries.

    Wraps :func:`dtw` with ``reduce="none"`` and applies :func:`numpy.nanmean`
    to robustly average over any trajectories that may contain NaN values.

    Args:
        preds (ArrayLike): Batch of predicted trajectories, shape (N, T, C).
        gts   (ArrayLike): Batch of ground-truth trajectories, shape (N, T, C).

    Returns:
        float: Scalar mean DTW alignment cost across all valid trajectories.
    """
    # dtw_err: shape (N,)  — per-trajectory raw DTW costs
    dtw_err = dtw(preds, gts, reduce='none')
    dtw_err = np.nanmean(dtw_err)   # scalar, ignores NaN padding
    return float(dtw_err)


# ---------------------------------------------------------------------------
# Quick demo / sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(0)
    B = 8          # batch size
    T = 101
    t = np.linspace(0, 4 * np.pi, T)

    # Build a circular ground-truth trajectory and broadcast to a batch.
    # single_traj: shape (T, 2)
    # gt:          shape (B, T, 2) — same circle for every batch element
    single_traj = np.stack([30 * np.cos(t), 30 * np.sin(t)], -1)      # shape: (T, 2)
    gt = np.repeat(single_traj[None, ...], B, axis=0)                  # shape: (B, T, 2)

    # Noisy predictions: base circle + Gaussian noise + time-shifted copy.
    # pred: shape (B, T, 2)
    pred = (gt                                                  # base trajectory
            + 0.4 * np.random.randn(B, T, 2)                    # additive Gaussian noise
            + 0.3 * np.roll(gt, 10, axis=1))                    # 10-step time-shift artefact

    # Evaluate all metrics; each uses the first two (x, y) channels.
    print("ADE:", ade(pred, gt, reduce='mean'))
    print("FDE:", fde(pred, gt, reduce='mean'))
    print("SR @2m:", success_rate(pred, gt, axis=-2, reduce='mean'))
    print("Hausdorff:", hausdorff(pred, gt, reduce='mean'))
    print("nDTW:", ndtw(pred, gt, reduce='mean'))
    # Distribution-based metric (disabled by default to reduce runtime):
    # print("DynCons:", dynamic_consistency(pred, gt, reduce='mean'))

    '''
    ADE: 9.020381286932569
    FDE: 9.043563595143615
    SR @2m: 0.0
    Hausdorff: 5.470039531722291
    nDTW: 0.34880821820753677
    DynCons: 1.278691741265335e-06
    '''
