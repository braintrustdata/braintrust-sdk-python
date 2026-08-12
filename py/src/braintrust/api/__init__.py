"""Public Braintrust API client package."""

from ._routing import EndpointRouter, RequestTarget
from ._service import ClientContext
from .attachments import AttachmentsAPI
from .auth import AuthAPI, LoginResult, OrganizationInfo
from .client import BraintrustClient
from .datasets import DatasetsAPI
from .errors import (
    BraintrustAPIError,
    BraintrustHTTPError,
    BraintrustResponseError,
    BraintrustRetryExhaustedError,
    BraintrustTransportError,
    BraintrustTransportRetryExhaustedError,
)
from .experiments import (
    BaseExperiment,
    ExperimentComparison,
    ExperimentMetric,
    ExperimentRecord,
    ExperimentsAPI,
    ExperimentScore,
)
from .functions import FunctionsAPI
from .policies import RetryMode, RetryPolicy
from .projects import ProjectsAPI
from .prompts import PromptsAPI
from .queries import QueriesAPI


__all__ = [
    "AttachmentsAPI",
    "AuthAPI",
    "BaseExperiment",
    "BraintrustAPIError",
    "BraintrustClient",
    "BraintrustHTTPError",
    "BraintrustResponseError",
    "BraintrustRetryExhaustedError",
    "BraintrustTransportError",
    "BraintrustTransportRetryExhaustedError",
    "ClientContext",
    "DatasetsAPI",
    "EndpointRouter",
    "ExperimentComparison",
    "ExperimentMetric",
    "ExperimentRecord",
    "ExperimentScore",
    "ExperimentsAPI",
    "FunctionsAPI",
    "LoginResult",
    "OrganizationInfo",
    "ProjectsAPI",
    "PromptsAPI",
    "QueriesAPI",
    "RequestTarget",
    "RetryMode",
    "RetryPolicy",
]
