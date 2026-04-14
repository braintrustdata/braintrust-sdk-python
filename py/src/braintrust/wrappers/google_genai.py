"""Compatibility shim — real implementation lives in braintrust.integrations.google_genai."""

from braintrust.integrations.google_genai import setup_genai  # noqa: F401


__all__ = ["setup_genai"]
