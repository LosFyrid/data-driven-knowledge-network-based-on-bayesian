# Diagnostic Utilities

Optional diagnostic tools for checking Bayesian inference quality.

## Location

`src/utils/diagnostics.py`

## Philosophy

**Tool, not enforcer**: These functions provide information, never force behavior.
- Return `(bool, str)` tuples
- Never raise exceptions
- Never modify state
- **User decides what to do**

## Available Functions

### 1. `check_sample_size(X, y, n_params, ...)`

Check if you have enough data for reliable inference.

**Rule of thumb**: Need 10-20 samples per parameter.

```python
from utils.diagnostics import check_sample_size

valid, msg = check_sample_size(X, y, n_params=3)
print(msg)
# Output: "Sample size OK: n=50, 16.7 samples/param (≥ 10 required)"
```

**Parameters**:
- `min_samples`: Absolute minimum (default: 10)
- `min_samples_per_param`: Per-parameter minimum (default: 10)

---

### 2. `check_data_quality(X, y)`

Check for common data problems.

**Checks**:
- NaN values (will crash MCMC)
- Infinite values (will crash MCMC)
- Constant variables (no variation)
- Extreme values (>1e10, may overflow)

```python
from utils.diagnostics import check_data_quality

valid, msg = check_data_quality(X, y)
if not valid:
    print(f"⚠️  {msg}")
```

---

### 3. `check_convergence(trace, diagnostics, ...)`

Check MCMC convergence after sampling.

**Checks**:
- **R-hat**: < 1.01 (ideal), < 1.1 (acceptable)
- **ESS**: > 400 (good), > 100 (acceptable)
- **Divergences**: < 1% (good), < 5% (acceptable)

```python
from utils.diagnostics import check_convergence

result = rule.fit(X, y)
converged, msg = check_convergence(result.trace, result.diagnostics)
print(msg)
# Output: "Convergence OK: R-hat=1.0043, ESS=984-1205, converged=True"
```

**Parameters**:
- `max_rhat`: Maximum R-hat (default: 1.1)
- `min_ess`: Minimum ESS (default: 100)
- `max_divergence_rate`: Maximum divergence rate (default: 0.05)

---

## Usage Example

```python
from rules import LinearRule
from utils.diagnostics import check_sample_size, check_data_quality, check_convergence

# Pre-fit checks (optional)
valid, msg = check_sample_size(X, y, n_params=3)
print(f"Sample size: {msg}")

valid, msg = check_data_quality(X, y)
print(f"Data quality: {msg}")

# Fit
rule = LinearRule()
result = rule.fit(X, y)

# Post-fit check (optional)
converged, msg = check_convergence(result.trace, result.diagnostics)
print(f"Convergence: {msg}")
```

---

## When to Use

### ✅ Use when:
- Developing new models
- Debugging convergence issues
- Validating results for publication
- Teaching/learning Bayesian inference

### ⚠️ Skip when:
- Rapid prototyping
- Exploring parameter space deliberately
- You know what you're doing
- Running on trusted pipelines

---

## Design Rationale

**Why tool-style, not automatic?**

1. **User autonomy**: Researchers may intentionally use small samples, high divergences, etc. for exploration
2. **No false sense of security**: Passing checks ≠ scientifically valid results
3. **Transparency**: User sees and understands the checks
4. **Flexibility**: Can be used selectively or not at all

**Why not in base class?**

- Keeps `RuleClass` simple and focused
- Avoids imposing workflow on users
- Easy to test and maintain separately
- Can be evolved independently

---

## See Also

- Test example: `tests/opt/rules/validate_linear_rule.py`
- ArviZ documentation: https://python.arviz.org/en/stable/
- MCMC diagnostics: https://mc-stan.org/misc/warnings.html
