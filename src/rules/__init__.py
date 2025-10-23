"""
Rules module for Aetherium project.

This module contains the RuleClass abstraction and concrete implementations
for representing scientific laws and rules in different modeling modes.
"""

from .base import RuleClass, ModelingMode, RuleMetadata, FitResult
from .linear_rule import LinearRule

__all__ = [
    'RuleClass',
    'ModelingMode',
    'RuleMetadata',
    'FitResult',
    'LinearRule',
]
