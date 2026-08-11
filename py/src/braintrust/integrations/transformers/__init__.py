"""Braintrust integration for local Hugging Face Transformers pipelines."""

from .integration import TransformersIntegration
from .patchers import wrap_transformers


__all__ = [
    "TransformersIntegration",
    "setup_transformers",
    "wrap_transformers",
]


def setup_transformers() -> bool:
    """Instrument supported Transformers pipeline classes."""
    return TransformersIntegration.setup()
