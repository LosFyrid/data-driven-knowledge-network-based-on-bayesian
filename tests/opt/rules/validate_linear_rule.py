"""
Deep validation script for LinearRule implementation.

This script performs thorough validation of the RuleClass base class
and LinearRule concrete implementation, including:
- Bayesian inference correctness
- Uncertainty quantification
- Convergence diagnostics
- Visual inspection of results

This is intended for manual validation during development, not CI/CD.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

from rules.linear_rule import LinearRule

# Optional: Import diagnostic utilities (comment out if not needed)
from utils.diagnostics import check_sample_size, check_data_quality, check_convergence


def generate_synthetic_data(n_samples=50, a_true=2.5, b_true=1.0, noise_std=0.5, seed=42):
    """
    Generate synthetic linear data: y = a_true * x + b_true + noise

    Args:
        n_samples: Number of data points
        a_true: True slope
        b_true: True intercept
        noise_std: Standard deviation of Gaussian noise
        seed: Random seed

    Returns:
        X, y, true_params dict
    """
    np.random.seed(seed)
    X = np.linspace(-5, 5, n_samples)
    y = a_true * X + b_true + np.random.normal(0, noise_std, n_samples)

    true_params = {'a': a_true, 'b': b_true, 'sigma': noise_std}

    return X, y, true_params


def test_basic_functionality():
    """Test 1: Basic fit and predict functionality"""
    print("=" * 60)
    print("Test 1: Basic Functionality")
    print("=" * 60)

    # Generate data
    X_train, y_train, true_params = generate_synthetic_data(n_samples=50)
    X_test = np.linspace(-6, 6, 100)

    print(f"\nTrue parameters: a={true_params['a']}, b={true_params['b']}, sigma={true_params['sigma']}")

    # Create rule
    rule = LinearRule(name="Test Linear Rule")
    print(f"\nCreated rule: {rule}")
    print(f"Metadata: {rule.metadata}")

    # Optional: Pre-fit diagnostic checks (comment out if not needed)
    print("\n--- Optional: Pre-fit Diagnostics ---")
    valid, msg = check_sample_size(X_train, y_train, n_params=3)
    print(f"Sample size: {msg}")
    valid, msg = check_data_quality(X_train, y_train)
    print(f"Data quality: {msg}")

    # Fit the rule
    print("\n--- Fitting the rule (running MCMC sampling) ---")
    print("This may take 30-60 seconds...")
    result = rule.fit(X_train, y_train, draws=500, tune=500, chains=2)

    print("\n--- Fit completed! ---")
    print(f"Result: {result}")
    print(f"Log-likelihood: {result.log_likelihood:.2f}")

    # Check convergence
    print("\n--- Convergence Diagnostics ---")
    print(f"R-hat max: {result.diagnostics['rhat_max']:.4f} (should be < 1.1)")
    print(f"R-hat mean: {result.diagnostics['rhat_mean']:.4f}")
    print(f"ESS min: {result.diagnostics['ess_min']:.0f}")
    print(f"Converged: {result.diagnostics['converged']}")

    # Optional: Post-fit convergence check (comment out if not needed)
    print("\n--- Optional: Post-fit Diagnostics ---")
    converged, msg = check_convergence(result.trace, result.diagnostics)
    print(msg)

    # Get parameter estimates
    print("\n--- Parameter Estimates ---")
    param_summary = result.get_parameter_summary()
    for param_name, stats in param_summary.items():
        true_val = true_params.get(param_name, None)
        mean_val = stats['mean']
        std_val = stats['sd']
        if true_val is not None:
            error = abs(mean_val - true_val)
            print(f"{param_name}: {mean_val:.3f} ± {std_val:.3f} (true={true_val}, error={error:.3f})")
        else:
            print(f"{param_name}: {mean_val:.3f} ± {std_val:.3f}")

    # Make predictions
    print("\n--- Making Predictions ---")
    y_pred, y_std = rule.predict(X_test)
    print(f"Predicted on {len(X_test)} test points")
    print(f"Mean prediction range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
    print(f"Mean uncertainty: {y_std.mean():.3f}")

    # Evaluate on test data
    print("\n--- Evaluation Metrics ---")
    y_test_true = true_params['a'] * X_test + true_params['b']
    metrics = rule.evaluate(X_test, y_test_true, metrics=['rmse', 'mae', 'r2'])
    for metric_name, value in metrics.items():
        print(f"{metric_name.upper()}: {value:.4f}")

    return rule, X_train, y_train, X_test, y_pred, y_std, true_params, result


def test_uncertainty_quantification():
    """Test 2: Uncertainty increases with less data"""
    print("\n" + "=" * 60)
    print("Test 2: Uncertainty Quantification")
    print("=" * 60)

    X_test = np.array([0.0])  # Single test point

    # Scenario 1: Lots of data, low noise
    print("\nScenario 1: Lots of data (n=100), low noise (sigma=0.1)")
    X1, y1, _ = generate_synthetic_data(n_samples=100, noise_std=0.1, seed=1)
    rule1 = LinearRule()
    rule1.fit(X1, y1, draws=500, tune=500, chains=2)
    _, std1 = rule1.predict(X_test)
    print(f"Prediction uncertainty: {std1[0]:.4f}")

    # Scenario 2: Less data, high noise
    print("\nScenario 2: Less data (n=20), high noise (sigma=1.0)")
    X2, y2, _ = generate_synthetic_data(n_samples=20, noise_std=1.0, seed=2)
    rule2 = LinearRule()
    rule2.fit(X2, y2, draws=500, tune=500, chains=2)
    _, std2 = rule2.predict(X_test)
    print(f"Prediction uncertainty: {std2[0]:.4f}")

    print(f"\nUncertainty ratio (scenario2 / scenario1): {std2[0] / std1[0]:.2f}x")
    print("Expected: Scenario 2 should have MUCH higher uncertainty ✓" if std2[0] > std1[0] * 2 else "⚠ Unexpected result")


def test_applicability_score():
    """Test 3: Applicability score distinguishes good vs bad fit"""
    print("\n" + "=" * 60)
    print("Test 3: Applicability Score")
    print("=" * 60)

    rule = LinearRule()

    # Good fit: linear data
    print("\nScenario 1: Linear data (should have HIGH applicability)")
    X_linear, y_linear, _ = generate_synthetic_data(n_samples=50, noise_std=0.5)
    score_linear = rule.get_applicability_score(X_linear, y_linear)
    print(f"Applicability score: {score_linear:.2f}")

    # Bad fit: quadratic data
    print("\nScenario 2: Quadratic data (should have LOW applicability)")
    X_quad = np.linspace(-5, 5, 50)
    y_quad = 0.5 * X_quad**2 + np.random.normal(0, 0.5, 50)
    rule_new = LinearRule()  # Need new instance
    score_quad = rule_new.get_applicability_score(X_quad, y_quad)
    print(f"Applicability score: {score_quad:.2f}")

    print(f"\nScore difference: {score_linear - score_quad:.2f}")
    print("Expected: Linear data should have HIGHER score ✓" if score_linear > score_quad else "⚠ Unexpected result")


def visualize_results(rule, X_train, y_train, X_test, y_pred, y_std, true_params, result):
    """Create visualization of results"""
    print("\n" + "=" * 60)
    print("Creating Visualizations")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Posterior distributions
    ax = axes[0, 0]
    trace = result.trace
    a_samples = trace.posterior['a'].values.flatten()
    b_samples = trace.posterior['b'].values.flatten()

    ax.scatter(a_samples, b_samples, alpha=0.3, s=5)
    ax.axvline(true_params['a'], color='red', linestyle='--', linewidth=2, label='True a')
    ax.axhline(true_params['b'], color='blue', linestyle='--', linewidth=2, label='True b')
    ax.set_xlabel('a (slope)')
    ax.set_ylabel('b (intercept)')
    ax.set_title('Posterior Distribution of Parameters')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Marginal posteriors
    ax = axes[0, 1]
    ax.hist(a_samples, bins=30, alpha=0.6, label='a (slope)', density=True)
    ax.axvline(true_params['a'], color='red', linestyle='--', linewidth=2, label=f"True a={true_params['a']}")
    ax.set_xlabel('Parameter value')
    ax.set_ylabel('Density')
    ax.set_title('Marginal Posterior: a (slope)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Predictions with uncertainty
    ax = axes[1, 0]
    ax.scatter(X_train, y_train, alpha=0.6, s=30, label='Training data')
    ax.plot(X_test, y_pred, 'r-', linewidth=2, label='Prediction (mean)')
    ax.fill_between(X_test, y_pred - 2*y_std, y_pred + 2*y_std,
                     alpha=0.3, label='95% prediction interval')

    # True line
    y_true = true_params['a'] * X_test + true_params['b']
    ax.plot(X_test, y_true, 'g--', linewidth=2, label='True function')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Predictions with Uncertainty')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Residuals
    ax = axes[1, 1]
    y_train_pred, _ = rule.predict(X_train)
    residuals = y_train - y_train_pred
    ax.scatter(X_train, residuals, alpha=0.6)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('x')
    ax.set_ylabel('Residuals')
    ax.set_title('Training Residuals')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).parent.parent.parent / 'experiments'
    output_path = output_dir / f'linear_rule_validation_{timestamp}.png'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    # Close figure to free memory (non-interactive mode)
    plt.close(fig)


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("TESTING RuleClass Base + LinearRule Implementation")
    print("=" * 60)

    try:
        # Test 1: Basic functionality
        rule, X_train, y_train, X_test, y_pred, y_std, true_params, result = test_basic_functionality()

        # Test 2: Uncertainty quantification
        test_uncertainty_quantification()

        # Test 3: Applicability score
        test_applicability_score()

        # Visualize
        visualize_results(rule, X_train, y_train, X_test, y_pred, y_std, true_params, result)

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
