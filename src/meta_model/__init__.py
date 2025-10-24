"""
Meta-Model Module: Outer Loop Engine for Aetherium

This module implements the outer loop of Aetherium's dual-loop cognitive architecture.
It answers "WHEN does a rule apply?" and "WHY does it fail?" through meta-cognitive modeling.

Key Components:
- HybridBayesianNetwork: Core mixed discrete-continuous Bayesian network
- SuperBN: High-level API for scientific discovery tasks
- Inference engines: MCMC/VI-based approximate inference

Architecture:
    SuperBN (high-level API)
        ↓
    HybridBayesianNetwork (mixed distribution management)
        ↓
    pgmpy.BayesianNetwork (DAG structure) + PyMC (conditional distributions)

Separation of Concerns:
- meta_model/: Pure algorithm implementation (this module)
- knowledge_base/: Knowledge extraction and storage (LLM priors, literature)
- acquisition/: Active learning strategies (uses meta_model through dependency injection)
"""

from src.meta_model.nodes import (
    NodeType,
    ConditionalDistributionType,
    SuperBNNode,
)

# HybridBayesianNetwork and SuperBN will be imported when implemented
# from src.meta_model.hybrid_bn import HybridBayesianNetwork
# from src.meta_model.super_bn import SuperBN

__all__ = [
    "NodeType",
    "ConditionalDistributionType",
    "SuperBNNode",
    # "HybridBayesianNetwork",
    # "SuperBN",
]

__version__ = "0.1.0"
