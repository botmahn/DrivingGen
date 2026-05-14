"""traj_quality.py – Trajectory quality metrics using driving-specific criteria.

All metrics return per-trajectory scores in (0, 1] where higher is better.
Static trajectories (max speed < v_static) or trajectories with insufficient
travel distance are marked NaN and excluded from dataset-level averages.

Implemented metrics
-------------------
1. ``comfort_score_norm`` – Geometric mean of jerk / acceleration / yaw-rate
       sub-scores, each normalised by path length.
       S_comf = (S_j * S_a * S_y)^(1/3),  S_x = 1 / (1 + metric_per_metre)

2. ``curvature_rms``      – RMS Frenet curvature score.
       kappa(t) = |x_dot * y_ddot - y_dot * x_ddot| / (x_dot^2 + y_dot^2)^1.5
       S_curv = 1 / (1 + kappa_rms)

3. ``speed_score``        – Log-linear score comparing mean speed to a reference.
       S_speed = clip(ln(1 + v_stat) / ln(1 + v_max), 0, 1)

4. ``get_traj_quality``   – Dataset-level average of all three scores.
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
# 1. Comfort score (jerk + acceleration + yaw-rate, normalised by path length)
# ---------------------------------------------------------------------------

def comfort_score_norm(
    traj_xy: ArrayLike,
    *,
    dt: float = 0.1,
    axis: int = -2,
    eps: float = 1e-9,
    v_static: float = 0.1,
    pct: float = None,          # Temporal aggregation percentile (e.g. 95); None = mean
    length_eps: float = 1.0,    # Minimum path length threshold; shorter => invalid
    j_scale: float = 1.0,       # Dimensional scale for jerk sub-score
    a_scale: float = 1.0,       # Dimensional scale for acceleration sub-score
    y_scale: float = 1.0,       # Dimensional scale for yaw-rate sub-score
    reduce: str = "none",       # "none" -> per-trajectory; else dataset mean
    return_components: bool = False,  # True -> also return (S_j, S_a, S_y) array
):
    """Compute a normalised comfort score from jerk, acceleration, and yaw-rate.

    Comfort is quantified by three driving-dynamics criteria, each penalising
    large values relative to the total path length travelled:

        jerk_per_m  = agg( ||j(t)||_2 ) / path_length
        acc_per_m   = agg( ||a(t)||_2 ) / path_length
        yaw_per_m   = agg( |psi_dot(t)| ) / path_length

    where ``agg`` is either the mean or a percentile (``pct`` parameter).

    Each raw metric is mapped to a unit-interval sub-score using the same
    transformation as :func:`curvature_rms`:

        S_x = 1 / (1 + metric_per_metre / scale_x)

    S_x -> 1 when the metric is near zero (smooth ride);
    S_x -> 0 when the metric is very large (harsh ride).

    The three sub-scores are combined via their geometric mean (= exp of mean
    of logs), which is more sensitive to any single bad dimension than the
    arithmetic mean:

        S_comf = (S_j * S_a * S_y)^(1/3)
               = exp( (1/3) * (ln S_j + ln S_a + ln S_y) )

    Trajectories that are static (max speed < v_static) or too short
    (path length <= length_eps) are assigned NaN.

    Derivatives are computed using **central differences** to reduce
    one-sided edge bias:

        v(t)  = (x(t+1) - x(t-1)) / (2*dt)       (T-2 interior points)
        a(t)  = (x(t+1) - 2*x(t) + x(t-1)) / dt^2
        j(t)  = (a(t+1) - a(t-1)) / (2*dt)        (T-4 interior points)
        yaw(t)= (theta(t+1) - theta(t-1)) / (2*dt) (angular velocity)

    Args:
        traj_xy   (ArrayLike): Trajectory array, shape (..., T, C) with C >= 2.
            Only the first two channels (x, y) are used.
        dt        (float): Sampling interval in seconds.  Default 0.1 (10 Hz).
        axis      (int): Index of the time dimension.  Default -2.
        eps       (float): Numerical stability constant.  Default 1e-9.
        v_static  (float): Minimum peak speed (m/s) to consider a trajectory
            "moving".  Default 0.1 m/s.
        pct       (float | None): If set, temporal aggregation uses the
            ``pct``-th percentile instead of the mean (e.g. 95 for near-worst-
            case comfort).  Default None (mean).
        length_eps (float): Minimum required path length (metres).  Trajectories
            with path_length <= length_eps are assigned NaN.  Default 1.0.
        j_scale   (float): Dimensional scale applied to jerk sub-score.
            Default 1.0.
        a_scale   (float): Dimensional scale applied to acceleration sub-score.
            Default 1.0.
        y_scale   (float): Dimensional scale applied to yaw-rate sub-score.
            Default 1.0.
        reduce    (str): If ``"none"`` (default) return per-trajectory scores;
            any other value returns the scalar nanmean.
        return_components (bool): If True, also return the stacked sub-scores
            array of shape ``(*batch_shape, 3)``.  Default False.

    Returns:
        np.ndarray | float:
            When ``return_components=False``:
              - ``reduce == "none"``: array of shape ``batch_shape`` with comfort
                scores in (0, 1].  NaN for invalid trajectories.
              - otherwise: scalar float nanmean.
            When ``return_components=True``:
              - Tuple (S_comf, comps) where comps has an extra trailing dim of
                size 3 holding (S_j, S_a, S_y).

    Raises:
        ValueError: If the trajectory has fewer than 5 frames (required for
            the central-difference jerk estimate).
    """
    xy = np.asarray(traj_xy, float)

    # Normalise time axis to -2 without calling _prep_xy so we can extract
    # N, T, C directly from the reshaped array below.
    if axis != -2:
        xy = np.moveaxis(xy, axis, -2)

    # Record the batch prefix for output reshaping.
    bs = xy.shape[:-2]

    # Flatten all batch dimensions into a single N axis.
    N, T, _ = xy.reshape(-1, xy.shape[-2], 2).shape
    xy_s = xy.reshape(-1, xy.shape[-2], 2)   # shape: (N, T, 2)

    if T < 5:
        raise ValueError("Need >= 5 frames")

    # -----------------------------------------------------------------------
    # Central-difference derivatives
    # -----------------------------------------------------------------------

    # Velocity: central difference using frames t-1 and t+1.
    # v[n, t, :] = (x[t+1] - x[t-1]) / (2*dt)
    # Valid at interior indices 1 .. T-2  =>  T-2 velocity vectors.
    v = (xy_s[:, 2:, :] - xy_s[:, :-2, :]) / (2 * dt)                      # shape: (N, T-2, 2)

    # Acceleration: second-order central difference.
    # a[n, t, :] = (x[t+1] - 2*x[t] + x[t-1]) / dt^2
    # Also T-2 vectors, aligned with the velocity indices.
    a = (xy_s[:, 2:, :] - 2*xy_s[:, 1:-1, :] + xy_s[:, :-2, :]) / (dt**2) # shape: (N, T-2, 2)

    # Acceleration interior slice for jerk: trim one frame from each end so
    # jerk indices align with the velocity indices used for yaw-rate.
    a_c = a[:, 1:-1, :]                                                      # shape: (N, T-4, 2)

    # Jerk: central difference of acceleration vectors.
    # j[n, t, :] = (a[t+1] - a[t-1]) / (2*dt)
    # T-4 valid frames (two fewer than velocity).
    j = (a[:, 2:, :] - a[:, :-2, :]) / (2 * dt)                             # shape: (N, T-4, 2)

    # -----------------------------------------------------------------------
    # Static trajectory detection
    # -----------------------------------------------------------------------

    # Speed magnitude at each velocity frame.
    speed = np.linalg.norm(v, axis=-1)                                       # shape: (N, T-2)

    # A trajectory is "moving" if its peak speed >= v_static.
    moving = speed.max(axis=1) >= v_static                                   # shape: (N,), bool

    # -----------------------------------------------------------------------
    # Yaw-rate: angular velocity of the heading angle.
    # theta = arctan2(vy, vx); yaw_rate = d(theta)/dt via central difference.
    # -----------------------------------------------------------------------

    # Heading angle at velocity frames [2:] and [:-2] (skip one on each side
    # so that the central difference spans two velocity steps = 2*dt).
    th2 = np.arctan2(v[:, 2:, 1], v[:, 2:, 0])    # shape: (N, T-4)  — heading at t+1
    th0 = np.arctan2(v[:, :-2, 1], v[:, :-2, 0])  # shape: (N, T-4)  — heading at t-1

    # Wrap the angle difference to [-pi, pi] to handle heading discontinuities
    # (e.g. crossing from 179 degrees to -179 degrees).
    # yaw_rt[n, t] = (theta(t+1) - theta(t-1)) / (2*dt)
    yaw_rt = ((th2 - th0 + np.pi) % (2*np.pi) - np.pi) / (2 * dt)          # shape: (N, T-4)

    # -----------------------------------------------------------------------
    # Temporal aggregation: mean or percentile per trajectory
    # -----------------------------------------------------------------------

    def t_reduce(x):
        """Aggregate a (N, T') array to (N,) via mean or percentile."""
        return np.percentile(x, pct, axis=1) if pct is not None else x.mean(axis=1)

    # Aggregate jerk, acceleration, and yaw-rate magnitudes over time.
    jerk_p = t_reduce(np.linalg.norm(j, axis=-1))            # shape: (N,) — jerk magnitude
    acc_p  = t_reduce(np.linalg.norm(a_c, axis=-1))          # shape: (N,) — accel magnitude
    yaw_p  = t_reduce(np.abs(yaw_rt))                        # shape: (N,) — |yaw-rate|

    # -----------------------------------------------------------------------
    # Path length and validity mask
    # -----------------------------------------------------------------------

    # Cumulative Euclidean path length: sum of step-wise distances.
    # np.diff(xy_s, axis=1): shape (N, T-1, 2)
    # norm:                   shape (N, T-1)
    # sum:                    shape (N,)
    lengths = np.sum(np.linalg.norm(np.diff(xy_s, axis=1), axis=-1), axis=1)  # shape: (N,)

    # A trajectory is valid if it is moving AND has enough path length.
    valid = moving & (lengths > length_eps)                                    # shape: (N,), bool

    # -----------------------------------------------------------------------
    # Normalise aggregated metrics by path length
    # -----------------------------------------------------------------------

    # jerk_per_m, acc_per_m, yaw_per_m: intensity per metre of travel.
    # Invalid trajectories receive NaN so they propagate correctly below.
    jerk_pm = np.where(valid, jerk_p / (lengths + eps), np.nan)  # shape: (N,)
    acc_pm  = np.where(valid, acc_p  / (lengths + eps), np.nan)  # shape: (N,)
    yaw_pm  = np.where(valid, yaw_p  / (lengths + eps), np.nan)  # shape: (N,)

    # -----------------------------------------------------------------------
    # Sub-scores: map each normalised metric to (0, 1] via 1/(1 + x/scale)
    # This is the same form used in curvature_rms for consistent interpretation:
    #   S -> 1 when metric_per_m -> 0 (very smooth)
    #   S -> 0 when metric_per_m -> inf (very harsh)
    # -----------------------------------------------------------------------

    S_j = 1.0 / (1.0 + (jerk_pm / max(j_scale, eps)))   # shape: (N,) — jerk sub-score
    S_a = 1.0 / (1.0 + (acc_pm  / max(a_scale, eps)))   # shape: (N,) — accel sub-score
    S_y = 1.0 / (1.0 + (yaw_pm  / max(y_scale, eps)))   # shape: (N,) — yaw-rate sub-score

    # -----------------------------------------------------------------------
    # Geometric mean of sub-scores (consistent with log-domain averaging)
    #   S_comf = (S_j * S_a * S_y)^(1/3) = exp( mean(ln S_j, ln S_a, ln S_y) )
    # Using nanmean in log-space handles any NaN sub-scores gracefully.
    # -----------------------------------------------------------------------

    # Stack sub-scores into a (3, N) matrix for log-mean computation.
    comp = np.vstack([S_j, S_a, S_y])                       # shape: (3, N)

    # Geometric mean via exp of (nanmean of logs).
    # log(comp): shape (3, N)
    # nanmean along axis=0: shape (N,) — mean log sub-score per trajectory
    # exp: shape (N,) — geometric mean comfort score per trajectory
    S_comf = np.exp(np.nanmean(np.log(comp), axis=0))       # shape: (N,)

    # -----------------------------------------------------------------------
    # Restore batch shape and return
    # -----------------------------------------------------------------------

    S_comf = S_comf.reshape(bs)   # shape: batch_shape

    if return_components:
        # Stack sub-scores along the last dimension for inspection.
        # comps: shape (*batch_shape, 3) — (S_j, S_a, S_y) per trajectory
        comps = np.stack([S_j, S_a, S_y], axis=-1).reshape(*bs, 3)
        if reduce == "none":
            return S_comf, comps
        # Dataset-level mean: scalar S_comf and per-component mean vector.
        return float(np.nanmean(S_comf)), np.nanmean(comps, axis=tuple(range(comps.ndim-1)))
    else:
        if reduce == "none":
            return S_comf
        return float(np.nanmean(S_comf))


# ---------------------------------------------------------------------------
# 2. Curvature RMS score
# ---------------------------------------------------------------------------

def curvature_rms(
    traj_xy: ArrayLike,
    *,
    dt: float = 0.1,
    axis: int = -2,
    eps: float = 1e-9,
    v_static: float = 0.1,
    pct: float = None,          # Percentile threshold for curvature spike removal
    reduce: str = "none",
) -> np.ndarray | float:
    """Compute an RMS Frenet curvature score for each trajectory.

    The Frenet curvature at each timestep quantifies how sharply the path
    bends.  It is computed from first and second derivatives of position:

        kappa(t) = |x_dot * y_ddot - y_dot * x_ddot|
                   -----------------------------------
                   (x_dot^2 + y_dot^2 + eps)^1.5

    where x_dot, y_dot are velocity components and x_ddot, y_ddot are
    acceleration components (estimated via finite differences).

    The RMS curvature is then mapped to a score in (0, 1]:

        kappa_rms = sqrt( mean( kappa(t)^2 ) )
        S_curv    = 1 / (1 + kappa_rms)

    S_curv -> 1 for straight paths (small kappa), -> 0 for highly curved paths.

    Optionally, curvature spike filtering is applied before the RMS: values
    above the ``pct``-th percentile are replaced with NaN so that momentary
    numerical spikes do not dominate the score.

    Trajectories where the maximum speed never reaches ``v_static`` are
    classified as static and assigned NaN.

    Args:
        traj_xy  (ArrayLike): Trajectory array, shape (..., T, C) with C >= 2.
            Only the first two channels (x, y) are used.
        dt       (float): Sampling interval in seconds.  Default 0.1 (10 Hz).
        axis     (int): Index of the time dimension.  Default -2.
        eps      (float): Small constant added to the denominator to prevent
            division by zero when speed is near zero.  Default 1e-9.
        v_static (float): Peak-speed threshold below which a trajectory is
            considered static.  Default 0.1 m/s.
        pct      (float | None): If set, curvature values above the ``pct``-th
            percentile (per trajectory) are treated as spikes and replaced with
            NaN before computing the RMS.  Default None (no filtering).
        reduce   (str): If ``"none"`` (default) return per-trajectory scores;
            any other value returns the scalar nanmean.

    Returns:
        np.ndarray | float:
            - ``reduce == "none"``: array of shape ``batch_shape``, values in
              (0, 1].  NaN for static trajectories.
            - otherwise: scalar float nanmean score.

    Raises:
        ValueError: If the trajectory has fewer than 3 frames (minimum for
            finite-difference velocity and acceleration).
    """
    xy, bs = _prep_xy(np.asarray(traj_xy, float), axis)          # shape: (N, T, 2)
    N, T, _ = xy.shape

    if T < 3:
        raise ValueError("Need >= 3 frames to compute curvature")

    # -----------------------------------------------------------------------
    # Velocity and static-trajectory detection
    # -----------------------------------------------------------------------

    # Forward finite-difference velocity: displacement / dt.
    # np.diff(xy, axis=1): shape (N, T-1, 2)
    v     = np.diff(xy, axis=1) / dt                                  # shape: (N, T-1, 2)
    speed = np.linalg.norm(v, axis=-1)                                # shape: (N, T-1)

    # "Moving" means peak speed >= v_static.
    moving = speed.max(axis=1) >= v_static                            # shape: (N,), bool

    # Initialise output with NaN; only moving trajectories will receive scores.
    scores = np.full(N, np.nan, dtype=float)                          # shape: (N,)

    if moving.any():
        # Select only the M moving trajectories.
        idx  = np.where(moving)[0]
        v_mv = v[idx]                                                 # shape: (M, T-1, 2)

        # Acceleration: forward difference of velocity.
        # np.diff(v_mv, axis=1): shape (M, T-2, 2)
        a_mv = np.diff(v_mv, axis=1) / dt                             # shape: (M, T-2, 2)

        # Align velocity with acceleration: skip the first velocity frame so
        # both arrays cover the same T-2 interior timesteps.
        # x_dot, y_dot: velocity components at times 1 .. T-2
        x_dot, y_dot = v_mv[..., 1:, 0], v_mv[..., 1:, 1]           # shape: (M, T-2) each

        # x_dd, y_dd: acceleration components at times 1 .. T-2
        x_dd , y_dd  = a_mv[..., :, 0],  a_mv[..., :, 1]            # shape: (M, T-2) each

        # -----------------------------------------------------------------------
        # Frenet curvature formula:
        #   kappa = |x_dot * y_ddot - y_dot * x_ddot| / (x_dot^2 + y_dot^2)^1.5
        #
        # Numerator  = cross product of velocity and acceleration (2-D "z-component")
        #            = signed curvature magnitude
        # Denominator = speed^3 (converts from arc-length to time parameterisation)
        # eps in denominator avoids division by zero when speed -> 0.
        # -----------------------------------------------------------------------

        num   = np.abs(x_dot * y_dd - y_dot * x_dd)                  # shape: (M, T-2)
        den   = (x_dot**2 + y_dot**2 + eps) ** 1.5                   # shape: (M, T-2)
        kappa = num / den                                             # shape: (M, T-2)

        # -----------------------------------------------------------------------
        # Optional spike filtering: mask curvature values above the pct-th
        # percentile (per trajectory) to reduce the influence of sudden spikes
        # caused by discretisation noise or near-zero speed frames.
        # -----------------------------------------------------------------------

        if pct is not None:
            # Compute per-trajectory percentile threshold.
            # thr: shape (M, 1) — broadcast-ready
            thr  = np.percentile(kappa, pct, axis=-1, keepdims=True)   # shape: (M, 1)
            # Keep values within threshold; replace spikes with NaN.
            mask = kappa <= thr + eps
            kappa = np.where(mask, kappa, np.nan)                       # shape: (M, T-2)

        # RMS curvature: sqrt of mean of squared curvature values.
        # nanmean ignores any NaN spike-filtered frames.
        # rms: shape (M,)
        rms = np.sqrt(np.nanmean(kappa**2, axis=-1))                  # shape: (M,)

        # Map RMS curvature to a score in (0, 1]:
        #   S_curv = 1 / (1 + kappa_rms)
        # -> 1 for straight paths (kappa_rms ~ 0)
        # -> 0 for very curved paths (kappa_rms -> inf)
        scores[idx] = 1.0 / (1.0 + rms)                              # shape: (M,) -> written back

    # Restore original batch shape.
    out = scores.reshape(bs)
    return out if reduce == "none" else float(np.nanmean(out))


# ---------------------------------------------------------------------------
# 3. Speed score
# ---------------------------------------------------------------------------

def speed_score(
    traj_xy: ArrayLike,
    *,
    dt: float = 0.1,          # Sampling interval; 10 Hz -> 0.1 s
    axis: int = -2,
    v_ref: float = 6.0,       # Reference speed in m/s (approximately 22 km/h)
    k: float = 2.5,           # v_max = k * v_ref (upper bound of the score range)
    v_static: float = 0.1,    # Static threshold in m/s; also serves as v_min
    use_percentile=None,      # None -> use mean speed; or e.g. 90, 95 for a percentile
    reduce: str = "none",
):
    """Compute a log-linear speed score encouraging trajectories to reach a reference speed.

    The score rewards trajectories that achieve higher average speeds up to
    a maximum ``v_max = k * v_ref``, using a logarithmic mapping to give
    diminishing returns as speed grows:

        v_max  = k * v_ref
        S_speed = clip( ln(1 + v_stat) / ln(1 + v_max), 0, 1 )

    where ``v_stat`` is the mean (or percentile) speed of the trajectory.

    The logarithmic denominator ``ln(1 + v_max)`` normalises the score so
    that a trajectory traveling at exactly ``v_max`` receives S_speed = 1.0.
    Trajectories with v_stat > v_max are clipped to 1.0.

    Trajectories that never exceed ``v_static`` receive a score of 0.0
    (rather than NaN) to penalise immobile predictions.

    Note on v_min: unlike the docstring formula, the implementation does NOT
    subtract v_min from the numerator and denominator; the log-linear mapping
    uses raw speed relative to zero, which is the effective v_min = 0.

    Args:
        traj_xy       (ArrayLike): Trajectory array, shape (..., T, C) with C >= 2.
            Only the first two channels (x, y) are used.
        dt            (float): Sampling interval in seconds.  Default 0.1 (10 Hz).
        axis          (int): Index of the time dimension.  Default -2.
        v_ref         (float): Reference / typical driving speed in m/s.
            Default 6.0 m/s (approx. 22 km/h — urban crawl speed).
        k             (float): Multiplier determining the maximum scored speed;
            ``v_max = k * v_ref``.  Default 2.5 (so v_max = 15 m/s = 54 km/h).
        v_static      (float): Speed threshold below which a trajectory is
            classified as static and assigned a score of 0.0.  Default 0.1 m/s.
        use_percentile (float | None): If set, aggregates speed using the
            ``use_percentile``-th percentile instead of the mean.  Useful for
            characterising near-peak speed behaviour.  Default None (mean).
        reduce        (str): If ``"none"`` (default) return per-trajectory
            scores; any other value returns the scalar nanmean.

    Returns:
        np.ndarray | float:
            - ``reduce == "none"``: array of shape ``batch_shape``, values in
              [0, 1].  0.0 for static trajectories.
            - otherwise: scalar float nanmean score.
    """
    xy, bs = _prep_xy(np.asarray(traj_xy, float), axis)        # shape: (N, T, 2)

    # Finite-difference speed: displacement magnitude per time step.
    # np.diff(xy, axis=1): shape (N, T-1, 2)
    # divided by dt:        shape (N, T-1, 2)
    # norm:                 shape (N, T-1)  — speed magnitude at each step
    v = np.linalg.norm(np.diff(xy, axis=1) / dt, axis=-1)      # shape: (N, T-1)

    # -----------------------------------------------------------------------
    # Per-trajectory speed statistic: mean or percentile
    # -----------------------------------------------------------------------

    # v_stat: shape (N,) — representative speed scalar for each trajectory
    v_stat = (v.mean(axis=1) if use_percentile is None
              else np.percentile(v, use_percentile, axis=1))     # shape: (N,)

    # -----------------------------------------------------------------------
    # Classify trajectories as moving or static
    # -----------------------------------------------------------------------

    # "Moving" if the trajectory ever exceeds v_static.
    moving = v.max(axis=1) >= v_static                          # shape: (N,), bool

    # Initialise all scores to NaN; will be overwritten below.
    scores = np.full_like(v_stat, np.nan, dtype=float)          # shape: (N,)

    # -----------------------------------------------------------------------
    # Log-linear mapping:
    #   v_max  = k * v_ref                (upper end of the score range)
    #   denom  = ln(1 + v_max)            (normalisation constant, scalar)
    #   score  = clip( ln(1 + v_stat) / denom, 0, 1 )
    #
    # ln(1 + x) provides sub-linear growth: doubling speed gives less than
    # double the score, reflecting diminishing safety returns at high speed.
    # -----------------------------------------------------------------------

    v_max  = k * v_ref                                          # scalar — maximum scored speed

    # Precompute the fixed denominator for normalisation.
    denom  = np.log1p(v_max)                                    # scalar — ln(1 + v_max)

    # Numerator: log-transformed speed statistic for moving trajectories.
    # np.log1p(x) = ln(1 + x), numerically stable for x near 0.
    num    = np.log1p(v_stat[moving])                           # shape: (M,)

    # Clip to [0, 1]: ensures v_stat > v_max does not exceed 1.0.
    scores[moving] = np.clip(num / denom, 0.0, 1.0)            # shape: (M,) -> written back

    # Static trajectories (never moved) receive a score of 0.0 rather than NaN
    # to explicitly penalise immobile predictions in the dataset average.
    scores[~moving] = 0.0                                       # shape: (~M,)

    # -----------------------------------------------------------------------
    # Restore batch shape and return
    # -----------------------------------------------------------------------

    out = scores.reshape(bs)
    return out if reduce == "none" else float(np.nanmean(out))


# ---------------------------------------------------------------------------
# Dataset-level quality aggregator
# ---------------------------------------------------------------------------

def get_traj_quality(preds):
    """Compute the dataset-level mean trajectory quality score.

    Averages the three quality dimensions — comfort, curvature, and speed —
    per trajectory, then takes the nanmean over the dataset:

        quality_i = nanmean( [S_comf_i, S_curv_i, S_speed_i] )
        quality   = nanmean( quality_i  for all i )

    Each sub-score is in (0, 1]; the joint mean penalises trajectories that
    are poor in any single dimension.  NaN entries (e.g. from static or
    too-short trajectories) are excluded via nanmean at both levels.

    Args:
        preds (ArrayLike): Batch of predicted trajectories, shape (N, T, C).
            Only the first two channels (x, y) are used.

    Returns:
        float: Scalar mean trajectory quality score across the dataset.
    """
    # Compute each quality metric per trajectory (no reduction yet).
    # comfort_norm_: shape (N,) — comfort scores in (0, 1]
    # crms:          shape (N,) — curvature scores in (0, 1]
    # ss:            shape (N,) — speed scores in [0, 1]
    comfort_norm_ = comfort_score_norm(preds, reduce='none')   # shape: (N,)
    crms          = curvature_rms(preds, reduce='none')        # shape: (N,)
    ss            = speed_score(preds, reduce='none')          # shape: (N,)

    # Stack the three score arrays along a new trailing axis.
    # arr: shape (N, 3) — [S_comf, S_curv, S_speed] for each trajectory
    arr = np.stack([comfort_norm_, crms, ss], axis=-1).astype(float)  # shape: (N, 3)

    # Per-trajectory mean across the three quality dimensions (ignore NaN).
    # quality: shape (N,) — combined quality score per trajectory
    quality = np.nanmean(arr, axis=-1)                                 # shape: (N,)

    # Dataset-level mean across all trajectories (ignore NaN).
    quality = np.nanmean(quality)                                      # scalar

    return float(quality)
