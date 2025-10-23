"""
Fast CI/CD tests for LinearRule implementation.

These tests verify interface contracts, basic functionality, and edge cases.
They use fast MCMC sampling (draws=50) to keep runtime under 30 seconds.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

from rules import LinearRule, ModelingMode


class TestLinearRuleInterface:
    """Test LinearRule interface contracts"""

    def test_initialization(self):
        """Test rule can be initialized"""
        rule = LinearRule(name="Test Linear Rule")
        assert rule.metadata.name == "Test Linear Rule"
        assert rule.metadata.mode == ModelingMode.WHITE_BOX
        assert rule.metadata.formula == "y = a * x + b"
        assert 'a' in rule.metadata.parameters
        assert 'b' in rule.metadata.parameters
        assert 'sigma' in rule.metadata.parameters

    def test_cannot_predict_before_fit(self):
        """Test that predict() raises error before fit()"""
        rule = LinearRule()
        X_test = np.array([1.0, 2.0, 3.0])

        with pytest.raises(RuntimeError, match="must be fitted"):
            rule.predict(X_test)

    def test_is_fitted_flag(self):
        """Test is_fitted property works correctly"""
        rule = LinearRule()
        assert not rule.is_fitted

        # After fitting, should be True
        X = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])
        rule.fit(X, y, draws=50, tune=50, chains=1)

        assert rule.is_fitted


class TestLinearRuleBasicFunctionality:
    """Test basic fit and predict functionality"""

    def test_fit_returns_fit_result(self, synthetic_linear_data, sampling_params_fast):
        """Test that fit() returns a FitResult object"""
        X, y = synthetic_linear_data
        rule = LinearRule()

        result = rule.fit(X, y, **sampling_params_fast)

        assert result is not None
        assert hasattr(result, 'log_likelihood')
        assert hasattr(result, 'trace')
        assert hasattr(result, 'diagnostics')

    def test_predict_returns_correct_shape(self, synthetic_linear_data, sampling_params_fast):
        """Test that predict() returns correct output shapes"""
        X_train, y_train = synthetic_linear_data
        rule = LinearRule()
        rule.fit(X_train, y_train, **sampling_params_fast)

        X_test = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean, std = rule.predict(X_test)

        assert mean.shape == (5,)
        assert std.shape == (5,)
        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(std))
        assert np.all(std > 0)  # Uncertainty must be positive

    def test_predict_handles_2d_input(self, synthetic_linear_data, sampling_params_fast):
        """Test that predict() handles 2D input correctly"""
        X_train, y_train = synthetic_linear_data
        rule = LinearRule()
        rule.fit(X_train, y_train, **sampling_params_fast)

        X_test = np.array([[1.0], [2.0], [3.0]])  # 2D shape
        mean, std = rule.predict(X_test)

        assert mean.shape == (3,)
        assert std.shape == (3,)


class TestLinearRuleEdgeCases:
    """Test edge cases and error handling"""

    def test_handles_empty_input(self):
        """Test that empty input raises appropriate error"""
        rule = LinearRule()
        X = np.array([])
        y = np.array([])

        # PyMC should raise an error with empty data
        with pytest.raises(Exception):  # Could be ValueError or PyMC error
            rule.fit(X, y, draws=10, tune=10, chains=1)

    def test_handles_mismatched_input(self):
        """Test that mismatched X and y shapes raise error"""
        rule = LinearRule()
        X = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0])  # Wrong length

        with pytest.raises(Exception):
            rule.fit(X, y, draws=10, tune=10, chains=1)

    def test_handles_nan_input(self):
        """Test that NaN values in input raise error"""
        rule = LinearRule()
        X = np.array([1.0, 2.0, np.nan])
        y = np.array([2.0, 4.0, 6.0])

        # Should fail during sampling
        with pytest.raises(Exception):
            rule.fit(X, y, draws=10, tune=10, chains=1)


class TestLinearRuleEvaluate:
    """Test evaluation metrics"""

    def test_evaluate_computes_metrics(self, synthetic_linear_data, sampling_params_fast):
        """Test that evaluate() computes all requested metrics"""
        X, y = synthetic_linear_data
        rule = LinearRule()
        rule.fit(X, y, **sampling_params_fast)

        metrics = rule.evaluate(X, y, metrics=['rmse', 'mae', 'r2'])

        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert all(np.isfinite(v) for v in metrics.values())

    def test_evaluate_before_fit_raises_error(self):
        """Test that evaluate() raises error before fit()"""
        rule = LinearRule()
        X = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])

        with pytest.raises(RuntimeError):
            rule.evaluate(X, y)


class TestLinearRuleApplicabilityScore:
    """Test applicability score computation"""

    def test_get_applicability_score(self, synthetic_linear_data, sampling_params_fast):
        """Test that get_applicability_score() returns a finite number"""
        X, y = synthetic_linear_data
        rule = LinearRule()

        score = rule.get_applicability_score(X, y, draws=50, tune=50, chains=1)

        assert isinstance(score, (int, float))
        assert np.isfinite(score)

    def test_applicability_score_caching(self, synthetic_linear_data, sampling_params_fast):
        """Test that applicability score is cached after fit"""
        X, y = synthetic_linear_data
        rule = LinearRule()

        # First call should fit
        score1 = rule.get_applicability_score(X, y, **sampling_params_fast)

        # Second call with same data should return cached value
        score2 = rule.get_applicability_score(X, y, **sampling_params_fast)

        assert score1 == score2


class TestLinearRuleSmokeTest:
    """End-to-end smoke test"""

    def test_complete_workflow(self, synthetic_linear_data, sampling_params_fast):
        """Test complete workflow: create -> fit -> predict -> evaluate"""
        X_train, y_train = synthetic_linear_data
        X_test = np.linspace(0, 10, 10)

        # Create rule
        rule = LinearRule(name="Smoke Test Rule")
        assert not rule.is_fitted

        # Fit
        result = rule.fit(X_train, y_train, **sampling_params_fast)
        assert rule.is_fitted
        assert result.log_likelihood is not None

        # Predict
        y_pred, y_std = rule.predict(X_test)
        assert y_pred.shape == X_test.shape
        assert y_std.shape == X_test.shape

        # Evaluate
        y_test = 2.0 * X_test + 1.0  # True function
        metrics = rule.evaluate(X_test, y_test)
        assert 'rmse' in metrics
        assert metrics['rmse'] < 1.0  # Should fit reasonably well

        print(f"\n✓ Smoke test passed!")
        print(f"  Log-likelihood: {result.log_likelihood:.2f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  R²: {metrics['r2']:.4f}")
