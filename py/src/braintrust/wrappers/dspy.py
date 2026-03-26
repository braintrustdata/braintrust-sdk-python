"""Backward-compatible re-exports — implementation lives in braintrust.integrations.dspy."""

from braintrust.integrations.dspy import BraintrustDSpyCallback, patch_dspy


__all__ = ["BraintrustDSpyCallback", "patch_dspy"]
