# 002 - Directory Restructure ADR

**Created**: 2025-10-24
**Updated**: 2025-10-24
**Status**: ✅ Archived
**Related**: [Meta Model Design](../02_planning/meta_model_design.md)

---

## Summary

Restructured project directories to separate concerns: meta_model (algorithms), knowledge_base (data), and acquisition (strategies). This ADR documents the architectural decision and rationale.

---

## Background

### The Problem
Original CLAUDE.md defined:
```
src/knowledge_base/    # Outer loop engine (Bayesian Network / SCM)
```

**Issues identified**:
1. **Naming confusion**: "knowledge_base" implies data storage, not algorithm implementation
2. **Responsibility mixing**: Combining BN engine with LLM prior extraction
3. **Unclear relationship**: How does acquisition/ relate to knowledge_base/?

**Trigger**: User raised critical questions:
- "knowledge_base should store LLM-extracted priors, not implement BN engine"
- "Should acquisition/ be tightly coupled with outer loop?"

---

## Options Considered

### Option 1: Separate Engine and Knowledge ✅ **ADOPTED**
```
src/
├── meta_model/        # Outer loop engine (algorithms)
├── knowledge_base/    # Knowledge extraction & storage (data)
└── acquisition/       # Active learning (strategies)
```

**Pros**:
- ✅ Clear separation of concerns
- ✅ meta_model testable without LLM dependency
- ✅ acquisition can use meta_model via dependency injection
- ✅ Intuitive naming

**Cons**:
- ❌ Requires CLAUDE.md update
- ❌ Deviates from original structure

---

### Option 2: Composite knowledge_base/
```
src/
└── knowledge_base/
    ├── meta_model/    # Sub-module: BN engine
    ├── priors/        # Sub-module: prior management
    └── domain/        # Sub-module: domain knowledge
```

**Pros**:
- ✅ Aligns with original CLAUDE.md
- ✅ Groups related functionality

**Cons**:
- ❌ Still confusing name
- ❌ Overly nested structure
- ❌ Tight coupling

---

### Option 3: Rename to outer_loop/
```
src/
└── outer_loop/
    ├── bn/            # BN engine
    ├── priors/        # Prior management
    └── llm/           # LLM integration
```

**Pros**:
- ✅ Descriptive name

**Cons**:
- ❌ Mixes algorithm and data concerns
- ❌ Not as clear as Option 1

---

## Decision

**Adopted: Option 1 - Separate Engine and Knowledge**

---

## Rationale

### Principle: Separation of Concerns

**meta_model/** - Pure Algorithm Implementation
- Manages BN structure, learning, inference
- Receives priors through **interfaces**
- No direct LLM or literature parsing dependency
- **Testable in isolation**

**knowledge_base/** - Knowledge Extraction & Storage
- Extracts causal priors from literature (LangChain)
- Manages domain knowledge (physics laws, constraints)
- Provides **standardized prior interfaces**
- LLM integration lives here

**acquisition/** - Active Learning Strategies
- Uses meta_model via **dependency injection**
- Independent module, reusable across architectures
- Can query both meta_model and inner loop

---

### Design Pattern: Dependency Injection

```python
# knowledge_base provides priors
from knowledge_base.literature import LiteratureParser
from knowledge_base.priors import PriorBuilder

parser = LiteratureParser(llm_api="gpt-4")
causal_graph = parser.extract_causal_structure("Hooke's Law")
priors = PriorBuilder.from_causal_graph(causal_graph)

# meta_model uses priors (interface-based)
from meta_model.super_bn import SuperBN

super_bn = SuperBN(rules=[hookes_law])
super_bn.build_initial_structure(
    context_vars=["force", "displacement"],
    priors=priors  # Injected dependency
)

# acquisition uses meta_model (dependency injection)
from acquisition.uncertainty import MaxUncertaintyStrategy

strategy = MaxUncertaintyStrategy(meta_model=super_bn)
next_experiment = strategy.recommend()
```

---

## Implementation Changes

### Directory Structure
```
src/
├── meta_model/        # NEW: Outer loop engine
│   ├── __init__.py
│   ├── nodes.py
│   ├── hybrid_bn.py
│   ├── super_bn.py
│   └── inference.py
│
├── knowledge_base/    # CLARIFIED: Knowledge management
│   ├── priors.py
│   ├── literature.py
│   ├── llm_integration.py
│   └── domain_knowledge/
│
└── acquisition/       # UNCHANGED: Active learning
    ├── strategies.py
    ├── uncertainty.py
    └── boundary_exploration.py
```

### Updated Documentation
- ✅ CLAUDE.md: Updated "Project Structure" section
- ✅ CLAUDE.md: Added "Module Responsibilities" section
- ✅ docs/outer_loop_super_bn_design.md: Added Section 0 (directory structure explanation)

---

## Consequences

### Positive
1. **Modularity**: Each module has single responsibility
2. **Testability**: meta_model can be tested with mock priors
3. **Flexibility**: Can swap knowledge_base implementation (e.g., manual priors, different LLM)
4. **Clarity**: Intuitive naming for new contributors

### Negative
1. **Documentation debt**: Must update all references to old structure
2. **Initial overhead**: More directories to navigate initially

### Neutral
1. **Learning curve**: Developers must understand dependency injection pattern

---

## Validation

### Interface Contracts

**knowledge_base → meta_model**:
```python
@dataclass
class CausalPrior:
    edges: List[Tuple[str, str]]
    forbidden_edges: List[Tuple[str, str]]
    edge_probs: Dict[Tuple[str, str], float]
```

**meta_model → acquisition**:
```python
class MetaModel(ABC):
    @abstractmethod
    def get_uncertainty(context: Dict) -> float:
        """Return BN uncertainty at context"""
```

---

## Timeline

- **2025-10-24 Morning**: User raised concerns
- **2025-10-24 Afternoon**: Discussed 3 options
- **2025-10-24**: Decision made, implemented
  - Created src/meta_model/
  - Updated CLAUDE.md
  - Updated design docs

---

## Related Decisions

- **ADR-002**: SuperBN technical stack (pgmpy + PyMC)
- **Future**: Interface design for knowledge_base/ (when implementing Phase 2)

---

## Context for Next Session

**Key Insight**: Separation of concerns enables parallel development—meta_model/ can progress without waiting for knowledge_base/ LLM integration.

**Next Step**: Implement meta_model/nodes.py with the understanding that priors will come via interface.

---

**Session End**: 2025-10-24
**Next Log**: [003 - Meta Model Kickoff](003_2025-10-24_meta_model_kickoff.md)
