"""
Fast CI/CD tests for RuleClass base class.

These tests focus on interface contracts, edge cases, and basic functionality.
They run quickly (<1 minute total) and are suitable for automated CI/CD pipelines.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

from rules.base import RuleClass, RuleMetadata, ModelingMode, FitResult


def test_rule_metadata_creation():
    """Test creating RuleMetadata with various configurations"""
    # Minimal metadata
    metadata = RuleMetadata(name="Test Rule")
    assert metadata.name == "Test Rule"
    assert metadata.mode == ModelingMode.WHITE_BOX  # Default
    assert metadata.parameters == []

    # Full metadata
    metadata = RuleMetadata(
        name="Hooke's Law",
        formula="F = k * x",
        parameters=["k", "x0"],
        mode=ModelingMode.WHITE_BOX,
        source="literature"
    )
    assert metadata.formula == "F = k * x"
    assert "k" in metadata.parameters
    assert metadata.source == "literature"


def test_modeling_mode_enum():
    """Test ModelingMode enum values"""
    assert ModelingMode.WHITE_BOX.value == "white_box"
    assert ModelingMode.GRAY_BOX.value == "gray_box"
    assert ModelingMode.BLACK_BOX.value == "black_box"


def test_fit_result_parameter_summary():
    """Test FitResult can extract parameter summary"""
    # This would require a real trace, so we'll test it in test_linear_rule.py
    pass
