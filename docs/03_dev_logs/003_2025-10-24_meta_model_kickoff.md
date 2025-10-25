# 003 - Meta Model Kickoff

**Created**: 2025-10-24
**Updated**: 2025-10-24
**Status**: 🚧 Active
**Related**: [Meta Model Design](../02_planning/meta_model_design.md)

---

## Summary

Starting implementation of the outer loop (meta_model/) with SuperBN (Hybrid Bayesian Network). This log tracks the initial setup and MVP implementation progress.

---

## Current Status

### ✅ Completed
- ✅ Created src/meta_model/ directory structure
- ✅ Created src/meta_model/__init__.py with module documentation
- ✅ Finalized directory structure (ADR-002)
- ✅ Updated CLAUDE.md to reflect new structure
- ✅ Created comprehensive design doc: [meta_model_design.md](../02_planning/meta_model_design.md)
- ✅ Reorganized documentation structure
  - Created docs/00_guides/, 01_implemented/, 02_planning/, 03_dev_logs/
  - Moved existing docs to appropriate locations
  - Created docs/README.md navigation index

### ⏳ In Progress
- 🚧 Reorganizing documentation (nearly complete)
- ⏳ **NEXT**: Implement src/meta_model/nodes.py

### 📋 Pending
- ⏳ Implement src/meta_model/hybrid_bn.py (MVP version)
- ⏳ Implement src/meta_model/super_bn.py (simplified version)
- ⏳ Write tests in tests/cicd/meta_model/
- ⏳ Integration test with RuleClass

---

## Decisions Made

### Decision 1: SuperBN Technical Stack
**Decision**: Use hybrid pgmpy + PyMC architecture.

**Architecture**:
```
SuperBN (high-level API)
   ↓
HybridBayesianNetwork (mixed distribution management)
   ↓
pgmpy (DAG structure) + PyMC (conditional distributions)
```

**Reason**:
- **pgmpy strengths**: Structure learning, causal inference, DAG management
- **PyMC strengths**: Flexible conditional distributions (CLG, non-linear, GP)
- **Evolution path**: Standard BN → Hybrid BN → SCM
- **Modularity**: Can test DAG separately from distributions

**Key Technical Points**:
- Conditional Linear Gaussian (CLG): Use `pm.switch()` for discrete state switching
- Non-linear relationships: Use PyMC GP or polynomials
- Mixed inference: Variable Elimination for discrete, MCMC for continuous

**Reference**: [meta_model_design.md §2.2](../02_planning/meta_model_design.md)

---

### Decision 2: MVP Scope
**Decision**: Implement minimal viable version first, defer advanced features.

**MVP Includes**:
- ✅ Node management (add_node, add_edge)
- ✅ Manual structure definition (no auto structure learning yet)
- ✅ Simple parameter learning (pure discrete OR pure continuous)
- ✅ MCMC-based inference
- ✅ Integration with RuleClass.get_applicability_score()

**Deferred to Later**:
- ❌ Automatic structure learning (hill-climbing, MMPC)
- ❌ CLG (Conditional Linear Gaussian) models
- ❌ Active learning integration
- ❌ Parameter promotion
- ❌ Rule switching

**Reason**: Validate core concept before adding complexity.

---

### Decision 3: Documentation Reorganization
**Decision**: Restructure docs/ into four categories with dev logs.

**New Structure**:
```
docs/
├── 00_guides/         # Timeless architectural guides
├── 01_implemented/    # Completed features documentation
├── 02_planning/       # Design specs for upcoming features
└── 03_dev_logs/       # Chronological development logs
```

**Benefits**:
- ✅ Clear separation: static (guides) vs dynamic (dev logs)
- ✅ Time-stamped progress tracking
- ✅ New sessions can quickly find current status
- ✅ CLAUDE.md stays stable as "command center"

**Reference**: User feedback on documentation organization

---

## Implementation Plan

### Phase 1: Foundation (Current)

**Step 1: Node Definitions** (`nodes.py`)
```python
class NodeType(Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    LIKELIHOOD = "likelihood"
    APPLICABILITY = "applicability"

class ConditionalDistributionType(Enum):
    CATEGORICAL = "categorical"
    LINEAR_GAUSSIAN = "linear_gaussian"
    # CLG and non-linear deferred to Phase 2

@dataclass
class SuperBNNode:
    name: str
    node_type: NodeType
    distribution_type: ConditionalDistributionType
    parents: List[str]
    # ... other fields
```

**Step 2: Hybrid BN Core** (`hybrid_bn.py` MVP)
```python
class HybridBayesianNetwork:
    def __init__(self):
        self.nodes: Dict[str, SuperBNNode] = {}
        self.dag: pgmpy.DAGModel = None
        self._pymc_model: Optional[pm.Model] = None

    def add_node(node: SuperBNNode) -> None:
        """Add node to network"""

    def add_edge(parent: str, child: str) -> None:
        """Add directed edge"""

    def learn_parameters(data: pd.DataFrame) -> None:
        """Learn conditional distributions (simple version)"""

    def infer(evidence: Dict, query_vars: List[str]) -> Dict:
        """MCMC-based inference"""
```

**Step 3: High-Level API** (`super_bn.py` simplified)
```python
class SuperBN:
    def __init__(self, rules: List[RuleClass]):
        self.rules = rules
        self.network = HybridBayesianNetwork()

    def update_with_experiment(context: Dict, X: np.ndarray, y: np.ndarray):
        """Update BN with new experiment data"""
        # 1. Get log_likelihood from each rule
        # 2. Add to dataset
        # 3. Update BN parameters
```

**Step 4: Testing**
- Unit tests for nodes
- Unit tests for HybridBN (3-node toy network)
- Integration test: SuperBN + LinearRule

---

## Next Actions

### Immediate (Today/This Session)
1. **Finish documentation reorganization**
   - ✅ Create dev logs 001, 002, 003
   - ⏳ Refactor CLAUDE.md to static command center
   - ⏳ Remove/archive DEVELOPMENT.md

2. **Implement nodes.py**
   - Define NodeType enum
   - Define ConditionalDistributionType enum
   - Implement SuperBNNode dataclass
   - Write docstrings

### Short-term (Next 1-2 Sessions)
3. **Implement hybrid_bn.py MVP**
   - Basic node management
   - Simple parameter learning
   - MCMC inference

4. **Implement super_bn.py MVP**
   - update_with_experiment()
   - Integration with RuleClass

5. **Write tests**
   - tests/cicd/meta_model/test_nodes.py
   - tests/cicd/meta_model/test_hybrid_bn.py
   - Synthetic test case (Hooke's Law applicability)

### Medium-term (Next Week)
6. **Integration testing**
   - End-to-end: RuleClass → SuperBN → Applicability assessment
   - Document usage examples

7. **Plan Phase 2**
   - CLG implementation
   - Structure learning
   - knowledge_base/ integration

---

## Technical Challenges Expected

### Challenge 1: Mixed Inference
**Problem**: How to efficiently infer over mixed discrete-continuous distributions?

**Strategy**:
- Start with simple case: All discrete OR all continuous
- For mixed: Use MCMC over full joint (slow but works)
- Future optimization: Hybrid inference (exact for discrete, MCMC for continuous)

---

### Challenge 2: Interfacing pgmpy + PyMC
**Problem**: pgmpy and PyMC have different data structures.

**Strategy**:
- Use pgmpy only for DAG structure
- Build PyMC model programmatically from DAG
- Keep clear separation between structure and parameters

---

### Challenge 3: Testing Without Real Data
**Problem**: No experimental data available yet.

**Strategy**:
- Generate synthetic data from known distributions
- Test case: Hooke's Law (elastic) vs Plastic deformation
  - Context: force, displacement
  - Hooke's Law works when force < threshold
  - Should learn P(applicability | force, displacement)

---

## Context for Next Session

### What We've Accomplished Today
1. ✅ Finalized directory structure (meta_model/ separation)
2. ✅ Created comprehensive design document
3. ✅ Set up meta_model/ module structure
4. ✅ Reorganized all documentation into clean hierarchy
5. 🚧 About to start nodes.py implementation

### What's Next
**Immediate task**: Implement `src/meta_model/nodes.py`

**Key files to reference**:
- [meta_model_design.md](../02_planning/meta_model_design.md) - Section 3.2 for class definitions
- [src/rules/base.py](../../src/rules/base.py) - Example of good dataclass design

**Key interface to remember**:
```python
# Inner loop provides this
log_likelihood = rule.get_applicability_score(X, y, context)

# Outer loop will use this
applicability = super_bn.update_with_experiment(context, X, y)
```

### Important Notes
- MVP scope is intentionally limited—don't implement CLG yet
- Focus on end-to-end workflow before optimization
- Keep testing in mind from the start

---

## Files Created This Session

```
src/meta_model/
└── __init__.py

docs/
├── README.md
├── 00_guides/
│   └── design_philosophy.md (moved)
├── 01_implemented/
│   └── diagnostics.md (moved)
├── 02_planning/
│   └── meta_model_design.md (moved & renamed)
└── 03_dev_logs/
    ├── README.md
    ├── 001_2025-10-23_ruleclass_implementation.md
    ├── 002_2025-10-24_directory_restructure_adr.md
    └── 003_2025-10-24_meta_model_kickoff.md (this file)
```

---

**Session Status**: 🚧 Active - Documentation reorganization nearly complete, ready to start coding
**Next Milestone**: Complete nodes.py implementation
