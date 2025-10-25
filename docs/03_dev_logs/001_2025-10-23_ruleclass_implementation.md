# 001 - RuleClass Implementation

**Created**: 2025-10-23
**Updated**: 2025-10-23
**Status**: ✅ Archived
**Related**: [Design Philosophy](../00_guides/design_philosophy.md)

---

## Summary

Implemented the foundational RuleClass framework for Aetherium's inner loop. This establishes the base abstraction for representing scientific laws with three modeling modes (white-box, gray-box, black-box).

---

## Current Status

### ✅ Completed
- **RuleClass base framework** ([src/rules/base.py](../../src/rules/base.py))
  - Abstract base class with three modeling modes
  - FitResult and RuleMetadata dataclasses
  - Integration with PyMC and ArviZ
  - Bayesian inference workflow (fit, predict, evaluate)
  - Log-likelihood computation for outer loop integration

- **LinearRule implementation** ([src/rules/linear_rule.py](../../src/rules/linear_rule.py))
  - White-box mode example (y = ax + b)
  - Full Bayesian parameter estimation
  - Posterior predictive sampling

- **Test suite**
  - Unit tests: [tests/cicd/rules/](../../tests/cicd/rules/)
  - Validation tests: [tests/opt/rules/](../../tests/opt/rules/)
  - Test coverage for core functionality

- **Diagnostic utilities** ([src/utils/diagnostics.py](../../src/utils/diagnostics.py))
  - MCMC convergence checks
  - Posterior analysis tools

---

## Key Decisions Made

### Decision 1: Three Modeling Modes
**Decision**: Support white-box, gray-box, and black-box modeling modes in RuleClass.

**Reason**:
- Reflects varying levels of prior knowledge about scientific laws
- White-box: Known formulas (F=kx) → parameter estimation
- Gray-box: Known constraints (monotonicity) → constrained fitting
- Black-box: No prior knowledge → GP/flexible fitting
- Outer loop remains agnostic to mode—only needs log_likelihood

**Impact**: Flexible framework that can handle different knowledge states

---

### Decision 2: PyMC as Inference Engine
**Decision**: Use PyMC (>= 5.10.0) for all probabilistic programming.

**Reason**:
- Modern, actively maintained
- Excellent MCMC samplers (NUTS)
- Native ArviZ integration for diagnostics
- Supports complex probabilistic models
- Python-native (no external compilation needed)

**Alternatives Considered**:
- NumPyro: Faster but requires JAX (deferred for optimization phase)
- Stan: Mature but requires external compilation

**Impact**: Clean API, good diagnostics, suitable for MVP

---

### Decision 3: Log-Likelihood as Outer Loop Interface
**Decision**: Inner loop returns single scalar `log_likelihood` score.

**Reason**:
- Simple, universal interface for outer loop
- Model-agnostic (works with any modeling mode)
- Enables model comparison and selection
- Used ArviZ LOO (Leave-One-Out Cross-Validation) for robust estimation

**Implementation**:
```python
def get_applicability_score(X, y, context) -> float:
    result = self.fit(X, y, context)
    return result.log_likelihood  # ELPD LOO score
```

---

## Implementation Highlights

### RuleClass API Design
```python
class RuleClass(ABC):
    @abstractmethod
    def build_model(X, y, context) -> pm.Model:
        """Define probabilistic model"""

    def fit(X, y, context, **kwargs) -> FitResult:
        """Run Bayesian inference"""

    @abstractmethod
    def predict(X, use_posterior=True) -> (mean, std):
        """Make predictions with uncertainty"""

    def get_applicability_score(X, y, context) -> float:
        """Interface for outer loop"""
```

### FitResult Structure
```python
@dataclass
class FitResult:
    trace: az.InferenceData           # Posterior samples
    log_likelihood: float              # Model evidence
    posterior_predictive: np.ndarray   # Predictions
    diagnostics: Dict[str, Any]        # R-hat, ESS, etc.
    metadata: Dict[str, Any]           # Context info
```

---

## Tests Written

### Unit Tests (tests/cicd/rules/)
- `test_rule_base.py`: RuleMetadata, FitResult dataclasses
- `test_linear_rule.py`:
  - Model building
  - Parameter estimation accuracy
  - Prediction functionality
  - Diagnostics computation

### Validation Tests (tests/opt/rules/)
- `validate_linear_rule.py`:
  - End-to-end workflow validation
  - Posterior recovery from synthetic data
  - Visual diagnostics (trace plots, posterior plots)

---

## Technical Challenges & Solutions

### Challenge 1: ArviZ API Changes
**Problem**: ArviZ 0.17+ changed return types for `loo()` and diagnostics.

**Solution**:
- Updated to use pandas.Series/xarray accessors
- `loo_result['elpd_loo']` instead of `loo_result.elpd_loo`
- `.to_array().values` for xarray Dataset extraction

### Challenge 2: Log-Likelihood Computation
**Problem**: Need robust model evidence for outer loop comparison.

**Solution**: Use LOO-CV instead of simple log-likelihood
- More robust to overfitting
- Penalizes model complexity
- Well-supported in ArviZ

---

## Lessons Learned

1. **Start Simple**: LinearRule was perfect MVP—validated workflow before complexity
2. **ArviZ Integration**: Built-in diagnostics saved significant development time
3. **Abstract Interface**: Separating `build_model()` from `fit()` enables testing
4. **Documentation**: Extensive docstrings crucial for future development

---

## Next Phase Dependencies

This implementation enables:
- ✅ Inner loop is ready for outer loop integration
- ✅ Can test meta_model with real RuleClass instances
- ⏳ Gray-box and black-box implementations (deferred to Phase 2)
- ⏳ InnerLoopEngine wrapper (deferred, may not be needed)

---

## Files Created

```
src/rules/
├── __init__.py
├── base.py          # RuleClass, FitResult, RuleMetadata
└── linear_rule.py   # LinearRule implementation

tests/cicd/rules/
├── __init__.py
├── conftest.py
├── test_rule_base.py
└── test_linear_rule.py

tests/opt/rules/
├── __init__.py
└── validate_linear_rule.py

src/utils/
└── diagnostics.py
```

---

## Context for Next Session

**What was accomplished**: Complete inner loop foundation with working example.

**What's next**: Focus shifts to outer loop (meta_model/) implementation.

**Key takeaway**: The `get_applicability_score()` method is the critical interface—outer loop only needs this single float value.

---

**Session End**: 2025-10-23
**Next Log**: [002 - Directory Restructure ADR](002_2025-10-24_directory_restructure_adr.md)
