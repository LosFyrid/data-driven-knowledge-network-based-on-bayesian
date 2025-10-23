"""
Base RuleClass for Aetherium cognitive architecture.

This module defines the abstract base class for all scientific laws and rules
in the system. It bridges three types of knowledge:
- Probabilistic knowledge (LLMs)
- Symbolic knowledge (formulas/laws)
- Numerical data (experiments)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pymc as pm
import arviz as az


class ModelingMode(Enum):
    """
    Three modeling modes representing different levels of prior knowledge.

    WHITE_BOX: Exact formula is known (e.g., F=kx). Use Bayesian parameter estimation.
    GRAY_BOX: Qualitative properties known (monotonicity, smoothness). Use constrained fitting.
    BLACK_BOX: No formula or properties known. Use unconstrained function fitting (e.g., GP).
    """
    WHITE_BOX = "white_box"
    GRAY_BOX = "gray_box"
    BLACK_BOX = "black_box"


@dataclass
class RuleMetadata:
    """
    Metadata describing a rule's properties and characteristics.

    Attributes:
        name: Human-readable name of the rule (e.g., "Hooke's Law")
        formula: Mathematical formula as string (e.g., "F = k * x")
        domain: Description of applicability domain
        parameters: List of internal parameter names (e.g., ["k", "x0"])
        mode: Modeling mode (white/gray/black box)
        source: Origin of the rule (e.g., "literature", "discovered", "LLM-extracted")
        constraints: Qualitative constraints (for gray-box mode)
    """
    name: str
    formula: Optional[str] = None
    domain: Optional[str] = None
    parameters: List[str] = field(default_factory=list)
    mode: ModelingMode = ModelingMode.WHITE_BOX
    source: str = "unknown"
    constraints: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"RuleMetadata(name='{self.name}', mode={self.mode.value}, formula='{self.formula}')"


@dataclass
class FitResult:
    """
    Result from fitting a rule to data via Bayesian updating.

    Attributes:
        trace: PyMC trace object containing posterior samples
        log_likelihood: Log-likelihood score (for outer loop model selection)
        posterior_predictive: Posterior predictive samples
        diagnostics: Convergence diagnostics (e.g., R-hat, ESS)
        metadata: Additional information about the fit
    """
    trace: az.InferenceData
    log_likelihood: float
    posterior_predictive: Optional[np.ndarray] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_parameter_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Extract summary statistics for each parameter.

        Returns:
            Dictionary mapping parameter names to their statistics (mean, std, HDI).
        """
        summary = az.summary(self.trace)
        return summary.to_dict('index')

    def __repr__(self) -> str:
        return f"FitResult(log_likelihood={self.log_likelihood:.2f}, n_params={len(self.trace.posterior.data_vars)})"


class RuleClass(ABC):
    """
    Abstract base class for all scientific rules in Aetherium.

    A RuleClass represents a scientific law or hypothesis that can:
    1. Be fit to experimental data via Bayesian updating (Inner Loop)
    2. Provide likelihood scores for meta-cognitive model selection (Outer Loop)
    3. Make predictions with uncertainty quantification
    4. Adapt its complexity based on available data and knowledge

    The outer loop (Bayesian Network) is agnostic to the modeling mode—it only
    needs the likelihood score to learn when/where this rule applies.

    **Implementation Notes**:
    - Default implementation uses **PyMC for inference** and **ArviZ for diagnostics**
    - Subclasses can override `_compute_log_likelihood()` and `_compute_diagnostics()`
      to use different inference engines (e.g., NumPyro, Stan, TensorFlow Probability)
    - The `fit()` method assumes PyMC workflow; override entirely if using different backend
    - API compatibility: Written for PyMC >= 5.10.0 and ArviZ >= 0.17.0
    """

    def __init__(self, metadata: RuleMetadata):
        """
        Initialize a RuleClass with metadata.

        Args:
            metadata: RuleMetadata object describing the rule
        """
        self.metadata = metadata
        self._last_fit_result: Optional[FitResult] = None
        self._is_fitted: bool = False

    @abstractmethod
    def build_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> pm.Model:
        """
        Build PyMC probabilistic model for this rule.

        This is the core method that defines the rule's probabilistic structure.
        Subclasses must implement this to specify:
        - Prior distributions on parameters
        - Likelihood function connecting parameters to observations
        - Any constraints or regularization

        Args:
            X: Input features, shape (n_samples, n_features)
            y: Target observations, shape (n_samples,)
            context: Optional context parameters (e.g., temperature, scale)

        Returns:
            PyMC Model object ready for inference
        """
        pass

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
        **inference_kwargs
    ) -> FitResult:
        """
        Fit the rule to data via Bayesian inference (Inner Loop operation).

        This method:
        1. Builds the probabilistic model
        2. Runs MCMC sampling to obtain posterior
        3. Computes log-likelihood for outer loop
        4. Generates posterior predictive samples
        5. Computes convergence diagnostics

        Args:
            X: Input features, shape (n_samples, n_features)
            y: Target observations, shape (n_samples,)
            context: Optional context parameters
            **inference_kwargs: Additional arguments for pm.sample() (e.g., draws, tune, chains)

        Returns:
            FitResult object containing trace, likelihood, and diagnostics
        """
        # Set default inference parameters
        inference_params = {
            'draws': 1000,
            'tune': 1000,
            'chains': 4,
            'return_inferencedata': True,
            'random_seed': 42,
        }
        inference_params.update(inference_kwargs)

        # Build the model
        model = self.build_model(X, y, context)

        # Run MCMC sampling
        with model:
            trace = pm.sample(**inference_params, idata_kwargs={'log_likelihood': True})

            # Generate posterior predictive samples
            posterior_predictive = pm.sample_posterior_predictive(
                trace,
                var_names=['obs'],
                random_seed=42
            )

        # Compute log-likelihood for model selection
        log_likelihood = self._compute_log_likelihood(trace, model)

        # Compute diagnostics
        diagnostics = self._compute_diagnostics(trace)

        # Store result
        result = FitResult(
            trace=trace,
            log_likelihood=log_likelihood,
            posterior_predictive=posterior_predictive.posterior_predictive['obs'].values,
            diagnostics=diagnostics,
            metadata={
                'rule_name': self.metadata.name,
                'n_samples': len(y),
                'context': context
            }
        )

        self._last_fit_result = result
        self._is_fitted = True

        return result

    @abstractmethod
    def predict(
        self,
        X: np.ndarray,
        use_posterior: bool = True,
        n_samples: int = 500
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty quantification.

        Args:
            X: Input features for prediction, shape (n_test, n_features)
            use_posterior: If True, use posterior samples; if False, use prior
            n_samples: Number of posterior samples to use for prediction

        Returns:
            Tuple of (mean_predictions, std_predictions), each shape (n_test,)
        """
        pass

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate the fitted rule on test data.

        Args:
            X: Test input features
            y: Test target values
            metrics: List of metric names (e.g., ['rmse', 'mae', 'r2'])

        Returns:
            Dictionary mapping metric names to values
        """
        if not self._is_fitted:
            raise RuntimeError("Rule must be fitted before evaluation. Call fit() first.")

        if metrics is None:
            metrics = ['rmse', 'mae', 'r2']

        # Get predictions
        y_pred, y_std = self.predict(X)

        results = {}

        if 'rmse' in metrics:
            results['rmse'] = np.sqrt(np.mean((y - y_pred) ** 2))

        if 'mae' in metrics:
            results['mae'] = np.mean(np.abs(y - y_pred))

        if 'r2' in metrics:
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            results['r2'] = 1 - (ss_res / ss_tot)

        if 'nll' in metrics:
            # Negative log-likelihood (lower is better)
            nll = -np.mean(
                -0.5 * np.log(2 * np.pi * y_std**2) - 0.5 * ((y - y_pred) / y_std) ** 2
            )
            results['nll'] = nll

        return results

    def get_applicability_score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
        **inference_kwargs
    ) -> float:
        """
        Compute an applicability score for this rule in given context.

        This is the key interface for the outer loop. It quantifies how well
        this rule explains the data in a specific context.

        Args:
            X: Input features
            y: Target observations
            context: Context parameters
            **inference_kwargs: Additional arguments for MCMC sampling

        Returns:
            Log-likelihood score (higher = rule more applicable)
        """
        # If already fitted with same context, return cached likelihood
        if (self._is_fitted and
            self._last_fit_result is not None and
            self._last_fit_result.metadata.get('context') == context):
            return self._last_fit_result.log_likelihood

        # Otherwise, fit and return likelihood
        result = self.fit(X, y, context, **inference_kwargs)
        return result.log_likelihood

    def _compute_log_likelihood(self, trace: az.InferenceData, model: pm.Model) -> float:
        """
        Compute log-likelihood for model selection.

        Uses the log-pointwise predictive density (lppd) via LOO-CV.

        **Default implementation**: Uses ArviZ LOO with PyMC InferenceData.
        **Override this method** if using a different inference engine or preferring WAIC.

        Args:
            trace: Inference data from sampling
            model: PyMC model (unused in default implementation, kept for compatibility)

        Returns:
            Log-likelihood score (ELPD LOO) - higher is better

        Notes:
            - Requires log_likelihood to be computed during sampling:
              `pm.sample(..., idata_kwargs={'log_likelihood': True})`
            - Compatible with ArviZ >= 0.17.0 (returns ELPDData as pandas.Series)
        """
        # Compute LOO using ArviZ
        loo_result = az.loo(trace, pointwise=False)
        # Access elpd_loo from ELPDData object (inherits from pandas.Series)
        log_likelihood = loo_result['elpd_loo']
        return float(log_likelihood)

    def _compute_diagnostics(self, trace: az.InferenceData) -> Dict[str, Any]:
        """
        Compute convergence diagnostics for MCMC sampling.

        **Default implementation**: Uses ArviZ diagnostics (R-hat, ESS) for PyMC traces.
        **Override this method** if using a different inference engine or need custom diagnostics.

        Args:
            trace: Inference data from sampling

        Returns:
            Dictionary of diagnostic metrics:
                - rhat_max: Maximum R-hat across all parameters (should be < 1.1)
                - rhat_mean: Mean R-hat across all parameters
                - ess_min: Minimum effective sample size
                - ess_mean: Mean effective sample size
                - converged: Boolean indicating if sampling converged

        Notes:
            - Compatible with ArviZ >= 0.17.0 (xarray Dataset format)
            - Extraction from xarray: `.to_array().values` to get numpy array
        """
        diagnostics = {}

        # R-hat (should be close to 1.0)
        rhat = az.rhat(trace)
        diagnostics['rhat_max'] = float(rhat.max().to_array().values.max())
        diagnostics['rhat_mean'] = float(rhat.to_array().values.mean())

        # Effective sample size
        ess = az.ess(trace)
        diagnostics['ess_min'] = float(ess.min().to_array().values.min())
        diagnostics['ess_mean'] = float(ess.to_array().values.mean())

        # Check for convergence issues
        diagnostics['converged'] = diagnostics['rhat_max'] < 1.1

        return diagnostics

    @property
    def is_fitted(self) -> bool:
        """Check if the rule has been fitted to data."""
        return self._is_fitted

    @property
    def last_fit_result(self) -> Optional[FitResult]:
        """Get the most recent fit result."""
        return self._last_fit_result

    def __repr__(self) -> str:
        return f"RuleClass(name='{self.metadata.name}', mode={self.metadata.mode.value}, fitted={self._is_fitted})"
