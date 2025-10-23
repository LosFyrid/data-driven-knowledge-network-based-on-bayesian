"""
Simple linear rule implementation for testing RuleClass base class.

This implements a white-box model: y = a * x + b
"""

import numpy as np
import pymc as pm
from typing import Dict, Any, Optional, Tuple

from .base import RuleClass, RuleMetadata, ModelingMode


class LinearRule(RuleClass):
    """
    Simple linear rule: y = a * x + b

    This is a white-box model where the formula is known.
    We use Bayesian inference to estimate parameters a and b.

    Current Limitations (TODO for future versions):
    1. Single-variable only: Only supports 1D input (X.shape = (n,))
       Future: Extend to multivariate linear regression y = X @ beta + b

    2. Hard-coded priors: Prior distributions are fixed (Normal(0, 10))
       Future: Allow user-specified priors or LLM-extracted priors

    3. No extrapolation warning: Does not warn when X_test is far from training data
       Future: Add domain checking and uncertainty inflation for extrapolation
    """

    def __init__(self, name: str = "Linear Rule"):
        """
        Initialize a linear rule.

        Args:
            name: Name of the rule
        """
        metadata = RuleMetadata(
            name=name,
            formula="y = a * x + b",
            parameters=["a", "b", "sigma"],
            mode=ModelingMode.WHITE_BOX,
            source="test_implementation"
        )
        super().__init__(metadata)

    def build_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> pm.Model:
        """
        Build PyMC model for linear relationship.

        Args:
            X: Input features, shape (n_samples, 1) or (n_samples,)
            y: Target observations, shape (n_samples,)
            context: Optional context (not used in this simple model)

        Returns:
            PyMC Model
        """
        # Ensure X is 1D
        if X.ndim == 2:
            X = X.flatten()

        with pm.Model() as model:
            # Priors
            a = pm.Normal('a', mu=0, sigma=10)  # Slope
            b = pm.Normal('b', mu=0, sigma=10)  # Intercept
            sigma = pm.HalfNormal('sigma', sigma=1)  # Noise

            # Linear model
            mu = a * X + b

            # Likelihood
            obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=y)

        return model

    def predict(
        self,
        X: np.ndarray,
        use_posterior: bool = True,
        n_samples: int = 500
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty quantification.

        The returned uncertainty (std) includes BOTH:
        1. Epistemic uncertainty: from parameter uncertainty (a, b)
        2. Aleatoric uncertainty: from observation noise (sigma)

        Total uncertainty = sqrt(epistemic^2 + aleatoric^2)

        Args:
            X: Input features for prediction
            use_posterior: If True, use posterior; if False, use prior
            n_samples: Number of samples for prediction

        Returns:
            Tuple of (mean_predictions, std_predictions)
        """
        if use_posterior and not self._is_fitted:
            raise RuntimeError("Model must be fitted before making posterior predictions")

        # Ensure X is 1D
        if X.ndim == 2:
            X = X.flatten()

        if use_posterior:
            # Extract posterior samples
            trace = self._last_fit_result.trace
            a_samples = trace.posterior['a'].values.flatten()[:n_samples]
            b_samples = trace.posterior['b'].values.flatten()[:n_samples]
            sigma_samples = trace.posterior['sigma'].values.flatten()[:n_samples]

            # Compute predictions for each sample
            predictions = np.array([a * X + b for a, b in zip(a_samples, b_samples)])

            # Compute mean prediction
            mean_pred = predictions.mean(axis=0)

            # Compute total uncertainty = epistemic + aleatoric
            # Epistemic uncertainty: due to parameter uncertainty (a, b)
            std_epistemic = predictions.std(axis=0)

            # Aleatoric uncertainty: due to observation noise (sigma)
            std_aleatoric = np.mean(sigma_samples)

            # Total predictive uncertainty (variance adds)
            std_pred = np.sqrt(std_epistemic**2 + std_aleatoric**2)
        else:
            # Use prior (just return zeros with high uncertainty)
            mean_pred = np.zeros_like(X)
            std_pred = np.ones_like(X) * 10

        return mean_pred, std_pred
