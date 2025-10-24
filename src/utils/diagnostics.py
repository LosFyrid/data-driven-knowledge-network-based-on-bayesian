"""
Diagnostic utilities for Bayesian inference quality checks.

This module provides optional diagnostic tools for checking:
- Sample size sufficiency
- Data quality (NaN, Inf, extreme values)
- MCMC convergence (R-hat, ESS)

All functions are purely informational - they return (bool, str) tuples
and never modify state or raise exceptions. Users decide how to act on results.

Example:
    >>> from utils.diagnostics import check_sample_size, check_convergence
    >>>
    >>> # Pre-fit check
    >>> valid, msg = check_sample_size(X, y, n_params=3)
    >>> if not valid:
    ...     print(f"Warning: {msg}")
    >>>
    >>> # Post-fit check
    >>> result = rule.fit(X, y)
    >>> converged, msg = check_convergence(result.trace, result.diagnostics)
    >>> print(msg)
"""

import numpy as np
import arviz as az
from typing import Dict, Any, Tuple


def check_sample_size(
    X: np.ndarray,
    y: np.ndarray,
    n_params: int,
    min_samples: int = 10,
    min_samples_per_param: int = 10
) -> Tuple[bool, str]:
    """
    Check if sample size is sufficient for reliable Bayesian inference.

    Rule of thumb: Need at least 10-20 samples per parameter for stable
    posterior estimation.

    Args:
        X: Input features, shape (n_samples, ...) or (n_samples,)
        y: Target values, shape (n_samples,)
        n_params: Number of model parameters
        min_samples: Absolute minimum number of samples (default: 10)
        min_samples_per_param: Minimum samples per parameter (default: 10)

    Returns:
        (is_valid, message): Tuple of boolean and explanation string

    Example:
        >>> valid, msg = check_sample_size(X, y, n_params=3, min_samples_per_param=15)
        >>> if not valid:
        ...     print(f"⚠️  {msg}")
        ... else:
        ...     print(f"✓ {msg}")
    """
    n_samples = len(y)

    # Check 1: Absolute minimum
    if n_samples < min_samples:
        return False, (
            f"Sample size too small: n={n_samples} < minimum={min_samples}. "
            f"Need at least {min_samples} samples for Bayesian inference."
        )

    # Check 2: Samples per parameter
    samples_per_param = n_samples / n_params
    required_samples = n_params * min_samples_per_param

    if samples_per_param < min_samples_per_param:
        return False, (
            f"Insufficient samples per parameter: {samples_per_param:.1f} < {min_samples_per_param}. "
            f"With {n_params} parameters, need at least {required_samples} samples. "
            f"Current: {n_samples} samples."
        )

    return True, (
        f"Sample size OK: n={n_samples}, {samples_per_param:.1f} samples/param "
        f"(≥ {min_samples_per_param} required)"
    )


def check_data_quality(
    X: np.ndarray,
    y: np.ndarray
) -> Tuple[bool, str]:
    """
    Check for common data quality issues.

    Checks for:
    - NaN values (will crash MCMC)
    - Infinite values (will crash MCMC)
    - Constant target variable (no variation to model)
    - Extreme values (may cause numerical overflow)

    Args:
        X: Input features
        y: Target values

    Returns:
        (is_valid, message): Tuple of boolean and explanation string

    Example:
        >>> valid, msg = check_data_quality(X, y)
        >>> if not valid:
        ...     print(f"⚠️  Data issue: {msg}")
    """
    # Check for NaN
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        return False, "Data contains NaN values (will crash MCMC)"

    # Check for Inf
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        return False, "Data contains infinite values (will crash MCMC)"

    # Check for constant target
    if np.std(y) == 0:
        return False, "Target variable is constant (no variation to model)"

    # Check for constant features (if multivariate)
    if X.ndim > 1 and X.shape[1] > 1:
        constant_features = np.where(np.std(X, axis=0) == 0)[0]
        if len(constant_features) > 0:
            return False, f"Features {list(constant_features)} are constant (no variation)"

    # Check for extreme values that might cause overflow
    x_max = np.abs(X).max()
    y_max = np.abs(y).max()

    if x_max > 1e10 or y_max > 1e10:
        return False, (
            f"Data contains extreme values (X_max={x_max:.2e}, y_max={y_max:.2e}). "
            f"Values > 1e10 may cause numerical overflow. Consider scaling."
        )

    return True, "Data quality OK (no NaN, Inf, constants, or extreme values)"


def check_convergence(
    trace: az.InferenceData,
    diagnostics: Dict[str, Any],
    max_rhat: float = 1.1,
    min_ess: float = 100,
    max_divergence_rate: float = 0.05
) -> Tuple[bool, str]:
    """
    Check MCMC convergence quality.

    Checks:
    - R-hat: Should be < 1.01 (ideal) or < 1.1 (acceptable)
    - ESS: Should be > 400 (good) or > 100 (acceptable)
    - Divergences: Should be < 1% (good) or < 5% (acceptable)

    Args:
        trace: ArviZ InferenceData from sampling
        diagnostics: Dictionary with diagnostic metrics (from RuleClass.fit())
        max_rhat: Maximum acceptable R-hat (default: 1.1)
        min_ess: Minimum acceptable ESS (default: 100)
        max_divergence_rate: Maximum acceptable divergence rate (default: 0.05)

    Returns:
        (is_converged, message): Tuple of boolean and explanation string

    Example:
        >>> result = rule.fit(X, y)
        >>> converged, msg = check_convergence(result.trace, result.diagnostics)
        >>> if not converged:
        ...     print(f"⚠️  Convergence issue: {msg}")
    """
    issues = []

    # Check 1: R-hat (chain convergence)
    rhat_max = diagnostics.get('rhat_max', float('inf'))
    if rhat_max > max_rhat:
        issues.append(
            f"R-hat too high: {rhat_max:.4f} > {max_rhat} "
            f"(chains did not converge; try increasing tune/draws)"
        )
    elif rhat_max > 1.05:
        issues.append(
            f"R-hat borderline: {rhat_max:.4f} > 1.05 "
            f"(convergence questionable; consider more sampling)"
        )

    # Check 2: ESS (effective sample size)
    ess_min = diagnostics.get('ess_min', 0)
    if ess_min < min_ess:
        issues.append(
            f"ESS too low: {ess_min:.0f} < {min_ess} "
            f"(high autocorrelation; try increasing draws)"
        )
    elif ess_min < 400:
        issues.append(
            f"ESS borderline: {ess_min:.0f} < 400 "
            f"(moderate autocorrelation; more draws recommended)"
        )

    # Check 3: Divergences (if available)
    if hasattr(trace, 'sample_stats') and 'diverging' in trace.sample_stats:
        divergences = trace.sample_stats.diverging.values
        n_divergences = int(np.sum(divergences))
        n_total = divergences.size
        divergence_rate = n_divergences / n_total

        if divergence_rate > max_divergence_rate:
            issues.append(
                f"Too many divergences: {n_divergences}/{n_total} ({divergence_rate*100:.1f}%) "
                f"(try reparameterizing model or increasing target_accept)"
            )
        elif n_divergences > 0:
            issues.append(
                f"Some divergences: {n_divergences}/{n_total} ({divergence_rate*100:.1f}%) "
                f"(minor issue; monitor if persists)"
            )

    # Construct message
    if issues:
        return False, "\n  - ".join(["Convergence issues detected:"] + issues)

    # Success message with details
    ess_mean = diagnostics.get('ess_mean', ess_min)
    return True, (
        f"Convergence OK: R-hat={rhat_max:.4f}, ESS={ess_min:.0f}-{ess_mean:.0f}, "
        f"converged={diagnostics.get('converged', True)}"
    )
