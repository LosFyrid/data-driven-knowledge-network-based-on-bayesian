# Tests Directory Structure

This directory contains all testing and validation code for the Aetherium project, organized by purpose and usage pattern.

## Directory Organization

```
tests/
├── cicd/                    # CI/CD automated tests (mirrors src/)
│   ├── conftest.py          # Global fixtures
│   ├── rules/               # Tests for src/rules/
│   │   ├── conftest.py      # rules-specific fixtures
│   │   ├── test_rule_base.py
│   │   └── test_linear_rule.py
│   ├── core/                # Tests for src/core/ (future)
│   ├── knowledge_base/      # Tests for src/knowledge_base/ (future)
│   ├── acquisition/         # Tests for src/acquisition/ (future)
│   └── discovery/           # Tests for src/discovery/ (future)
│
└── opt/                     # Optimization/validation tests (mirrors src/)
    ├── rules/               # Validation for src/rules/
    │   └── validate_linear_rule.py
    ├── core/                # Validation for src/core/ (future)
    └── knowledge_base/      # Validation for src/knowledge_base/ (future)
```

**Design principle**: The test directory structure **mirrors the src/ directory structure**.
This makes it easy to find tests for any given module.

---

## tests/cicd/ - CI/CD Automated Tests

**Purpose**: Fast, automated tests suitable for CI/CD pipelines.

**Characteristics**:
- ⚡ **Fast**: Total runtime < 1 minute
- 🤖 **Automated**: Run on every commit/PR
- ✅ **Pass/Fail**: Clear success criteria (assert statements)
- 🎯 **Focus**: Interface contracts, edge cases, smoke tests

**What to test here**:
- Interface contracts (e.g., "cannot predict before fit")
- Input validation and edge cases (empty input, NaN values)
- Output shape and type correctness
- Basic workflow smoke tests (with minimal sampling)
- Numerical stability

**What NOT to test here**:
- Scientific correctness of Bayesian inference (too slow)
- Convergence quality (requires many MCMC samples)
- Visual inspection (not automatable)
- Performance optimization (separate concern)

**Example usage**:
```bash
# Run all CI/CD tests
pytest tests/cicd/ -v

# Run all tests for rules module
pytest tests/cicd/rules/ -v

# Run specific test file
pytest tests/cicd/rules/test_linear_rule.py -v

# Run specific test class
pytest tests/cicd/rules/test_linear_rule.py::TestLinearRuleInterface -v

# Run with coverage
pytest tests/cicd/ --cov=src --cov-report=html
```

**Directory structure**:
```
tests/cicd/
├── conftest.py              # Global fixtures (available to all tests)
├── rules/                   # Tests for src/rules/
│   ├── conftest.py          # rules-specific fixtures
│   ├── test_rule_base.py    # Tests for RuleClass base
│   └── test_linear_rule.py  # Tests for LinearRule
├── core/                    # Tests for src/core/ (future)
├── knowledge_base/          # Tests for src/knowledge_base/ (future)
├── acquisition/             # Tests for src/acquisition/ (future)
└── discovery/               # Tests for src/discovery/ (future)
```

---

## tests/opt/ - Optimization and Validation Tests

**Purpose**: Deep validation and optimization during development.

**Characteristics**:
- 🐢 **Slow**: Runtime 1-60 minutes per test
- 👤 **Manual**: Run by developers when needed
- 👁️ **Human judgment**: Results require interpretation
- 🔬 **Focus**: Scientific correctness, convergence, visualization

**What to test here**:
- Bayesian inference correctness (posterior contains true parameters)
- MCMC convergence diagnostics (R-hat, ESS)
- Uncertainty quantification (epistemic vs aleatoric)
- Visual inspection (posterior distributions, predictions)
- Performance benchmarking
- Prior sensitivity analysis

**Example usage**:
```bash
# Run validation script for LinearRule
python tests/opt/rules/validate_linear_rule.py

# Results include:
# - Printed diagnostics (parameter estimates, convergence metrics)
# - Visualizations (saved to experiments/)
# - Manual inspection required
```

**Directory structure**:
```
tests/opt/
├── rules/                           # Validation for src/rules/
│   └── validate_linear_rule.py      # Deep validation of LinearRule
├── core/                            # Validation for src/core/ (future)
│   └── validate_inner_loop.py
├── knowledge_base/                  # Validation for src/knowledge_base/ (future)
│   └── validate_bn_learning.py
└── ...
```

---

## Design Philosophy

### Why separate cicd/ and opt/?

Traditional software testing (unit tests, mocks, coverage) has **limited value** for scientific computing projects like Aetherium because:

1. **Uncertain outputs**: Bayesian models produce probability distributions, not deterministic results
2. **Stochastic algorithms**: MCMC has inherent randomness
3. **Scientific correctness ≠ code correctness**: 100% code coverage doesn't mean the model converged
4. **Visual inspection required**: Posterior distributions must be manually inspected

Therefore, we split tests by **purpose**:

| Aspect | cicd/ | opt/ |
|--------|-------|------|
| **Question** | "Does it crash?" | "Is it scientifically correct?" |
| **Speed** | <1 min | 1-60 min |
| **Automation** | ✅ Fully automated | ❌ Manual execution |
| **Judgment** | ✅ Pass/Fail | 👁️ Human interpretation |
| **Sampling** | Fast (50 draws) | Full (1000+ draws) |
| **Visualization** | None | Extensive |

### When to use each?

**Use cicd/ when**:
- Making changes to interfaces or contracts
- Refactoring code
- Adding new features (smoke test)
- Running pre-commit checks

**Use opt/ when**:
- Implementing new RuleClass
- Tuning priors or sampling parameters
- Debugging convergence issues
- Validating scientific correctness
- Benchmarking performance

---

## Integration with CI/CD Pipeline

### Recommended GitHub Actions workflow:

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.10

      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-cov

      - name: Run CI/CD tests
        run: pytest tests/cicd/ -v --cov=src

      # Note: tests/opt/ is NOT run in CI
      # Run manually during development
```

---

## Adding New Tests

### Adding a cicd/ test:

1. **Identify the module**: Find which src/ module you're testing (e.g., `src/rules/`)
2. **Create/find the mirror directory**: `tests/cicd/rules/`
3. **Create test file**: `tests/cicd/rules/test_your_rule.py`
4. **Use fast sampling**: Get `sampling_params_fast` from global conftest
5. **Focus on contracts and edge cases**, not scientific correctness
6. **Keep runtime < 10 seconds** per test
7. **Use clear assert statements** for automated pass/fail

**Example**:
```python
# tests/cicd/rules/test_hookes_law.py
import pytest
from rules import HookesLaw

def test_hookes_law_interface(sampling_params_fast):
    rule = HookesLaw()
    X = np.array([0.1, 0.2, 0.3])
    F = np.array([0.5, 1.0, 1.5])

    rule.fit(X, F, **sampling_params_fast)
    pred, std = rule.predict(np.array([0.4]))

    assert pred.shape == (1,)
    assert std.shape == (1,)
```

### Adding an opt/ test:

1. **Identify the module**: Find which src/ module you're validating
2. **Create/find the mirror directory**: `tests/opt/rules/`
3. **Create validation script**: `tests/opt/rules/validate_your_rule.py`
4. **Use full sampling**: 1000+ draws, 4 chains
5. **Generate visualizations**: Posterior distributions, predictions, residuals
6. **Save results**: To `experiments/` directory
7. **Document expectations**: What should the results look like?

**Example**:
```python
# tests/opt/rules/validate_hookes_law.py
def main():
    # Generate spring system data
    X, F = generate_spring_data()

    # Fit with full sampling
    rule = HookesLaw()
    result = rule.fit(X, F, draws=1000, tune=1000, chains=4)

    # Print diagnostics
    print(f"R-hat: {result.diagnostics['rhat_max']}")
    print(f"ESS: {result.diagnostics['ess_min']}")

    # Visualize
    plot_posterior_distributions(result)
    plot_predictions(rule, X, F)

    # Save
    plt.savefig('experiments/hookes_law_validation.png')
```

---

## Summary

- **tests/cicd/**: Fast automated tests for regression prevention (CI/CD)
- **tests/opt/**: Slow manual tests for scientific validation (development)
- **Mirror structure**: Test directories mirror src/ directories for easy navigation
- **Both are necessary**: cicd/ catches bugs, opt/ ensures correctness
- **Different purposes**: Don't try to automate scientific judgment

### Key Design Principles

1. **Mirror src/ structure**: `tests/cicd/rules/` tests `src/rules/`
2. **Layered fixtures**: Global fixtures in top-level conftest.py, module-specific fixtures in subdirectory conftest.py
3. **Fast CI tests**: cicd/ tests use minimal sampling (50 draws) for speed
4. **Deep validation**: opt/ tests use full sampling (1000+ draws) for correctness
5. **Human judgment**: opt/ results require visual inspection and interpretation
