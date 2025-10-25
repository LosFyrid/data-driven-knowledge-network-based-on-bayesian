"""
Node definitions for SuperBN (Hybrid Bayesian Network).

This module defines the type system and data structures for nodes in the outer loop's
meta-model. The SuperBN manages causal relationships between context variables and
rule applicability, answering "WHEN does this law apply?" and "WHY does it fail?"

Key Components:
    - NodeType: Enum defining the four types of nodes (discrete, continuous, likelihood, applicability)
    - ConditionalDistributionType: Enum defining supported conditional probability distributions
    - SuperBNNode: Dataclass representing a single node in the network
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Callable


class NodeType(Enum):
    """
    Types of nodes in the Hybrid Bayesian Network.

    DISCRETE: Categorical variables (e.g., "regime: elastic/plastic/viscous")
    CONTINUOUS: Real-valued variables (e.g., velocity, temperature, force)
    LIKELIHOOD: Log-likelihood scores from inner loop (RuleClass.get_applicability_score)
    APPLICABILITY: Target variable indicating whether a rule applies in a given context
    """
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    LIKELIHOOD = "likelihood"
    APPLICABILITY = "applicability"


class ConditionalDistributionType(Enum):
    """
    Types of conditional probability distributions for nodes.

    MVP (Phase 1) - Implemented:
        CATEGORICAL: P(discrete | parents) - Categorical distribution
        LINEAR_GAUSSIAN: P(y | x) = N(α + βx, σ²) - Linear Gaussian relationship

    Phase 2 - Deferred:
        CONDITIONAL_LINEAR_GAUSSIAN: P(y | x, discrete_parent) - CLG with discrete switching
        NONLINEAR_GAUSSIAN: P(y | x) with non-linear relationships (via PyMC GP/polynomials)
        MIXTURE: Mixture distributions for complex dependencies
    """
    CATEGORICAL = "categorical"
    LINEAR_GAUSSIAN = "linear_gaussian"
    CONDITIONAL_LINEAR_GAUSSIAN = "clg"  # Phase 2
    NONLINEAR_GAUSSIAN = "nonlinear_gaussian"  # Phase 2
    MIXTURE = "mixture"  # Phase 2


@dataclass
class SuperBNNode:
    """
    A node in the Super Bayesian Network.

    Represents a variable in the outer loop's causal model, which can be:
    - Context variables (input): velocity, temperature, pressure, etc.
    - Likelihood variables: scores from RuleClass.get_applicability_score()
    - Target variables (output): rule applicability

    Attributes:
        name: Unique identifier for the node (e.g., "velocity", "hookes_law_likelihood")
        node_type: Type of the node (discrete, continuous, likelihood, applicability)
        distribution_type: Type of conditional probability distribution
        categories: For discrete nodes, list of possible categories (e.g., ["elastic", "plastic"])
        prior_params: Prior distribution parameters (for Bayesian parameter learning)
                     Example: {"alpha_mu": 0, "alpha_sigma": 10, "beta_mu": 1, "beta_sigma": 5}
        pymc_model_fn: Optional callable to build custom PyMC model for complex distributions (Phase 2)
        learned_params: Learned parameters after fitting to data (populated by HybridBN.learn_parameters)
                       Example: {"alpha": 2.3, "beta": 1.7, "sigma": 0.5}
        metadata: Additional information (e.g., units, description, source)

    Note:
        Parent-child relationships are managed by HybridBayesianNetwork.dag, not stored in nodes.
        Use HybridBayesianNetwork.get_parents(node_name) to query parent nodes.

    Example:
        >>> # Continuous context variable (root node)
        >>> velocity_node = SuperBNNode(
        ...     name="velocity",
        ...     node_type=NodeType.CONTINUOUS,
        ...     distribution_type=ConditionalDistributionType.LINEAR_GAUSSIAN,
        ...     prior_params={"mu": 0, "sigma": 100}
        ... )

        >>> # Likelihood node (structure defined separately in HybridBN.dag)
        >>> likelihood_node = SuperBNNode(
        ...     name="hookes_law_likelihood",
        ...     node_type=NodeType.LIKELIHOOD,
        ...     distribution_type=ConditionalDistributionType.LINEAR_GAUSSIAN,
        ...     prior_params={"alpha_mu": 0, "alpha_sigma": 5, "beta_mu": 1, "beta_sigma": 2}
        ... )

        >>> # Discrete regime variable
        >>> regime_node = SuperBNNode(
        ...     name="deformation_regime",
        ...     node_type=NodeType.DISCRETE,
        ...     distribution_type=ConditionalDistributionType.CATEGORICAL,
        ...     categories=["elastic", "plastic", "fracture"]
        ... )
    """
    name: str
    node_type: NodeType
    distribution_type: ConditionalDistributionType

    # Discrete node properties
    categories: Optional[List[str]] = None

    # Continuous node properties
    prior_params: Optional[Dict[str, Any]] = None
    pymc_model_fn: Optional[Callable] = None

    # Learned parameters (populated after fitting)
    learned_params: Optional[Dict[str, Any]] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate node configuration."""
        # Validate discrete nodes have categories
        if self.node_type == NodeType.DISCRETE and self.categories is None:
            raise ValueError(f"Discrete node '{self.name}' must have 'categories' defined")

        # Validate distribution type matches node type
        if self.node_type == NodeType.DISCRETE:
            if self.distribution_type not in [ConditionalDistributionType.CATEGORICAL,
                                             ConditionalDistributionType.CONDITIONAL_LINEAR_GAUSSIAN]:
                raise ValueError(f"Discrete node '{self.name}' must use CATEGORICAL or CLG distribution")

        # Validate categories are unique
        if self.categories is not None and len(self.categories) != len(set(self.categories)):
            raise ValueError(f"Node '{self.name}' has duplicate categories")

    def __repr__(self) -> str:
        """Concise string representation."""
        return (f"SuperBNNode(name='{self.name}', "
                f"type={self.node_type.value}, "
                f"dist={self.distribution_type.value})")

    def has_learned_params(self) -> bool:
        """Check if parameters have been learned for this node."""
        return self.learned_params is not None and len(self.learned_params) > 0
