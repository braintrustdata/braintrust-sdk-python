"""Experiment routing for Agno eval suite runs.

An agno suite run is the natural analogue of a Braintrust experiment: a fixed set of
cases, run together, scored, and worth comparing against the previous run. When a
suite runs in a process that has a Braintrust project in scope but no experiment,
this module opens one for the duration of the suite so each case lands as an
experiment row instead of a log.

Individual evals (``AccuracyEval`` and friends) deliberately do not get this
treatment: a script running five of them would produce five one-row experiments.
They log to whatever is already current, which is an experiment if the caller opened
one themselves.
"""

import contextvars
import logging
from contextlib import contextmanager
from typing import Any

from braintrust.env import EnvParser, EnvVar
from braintrust.logger import current_experiment, current_logger, init


logger = logging.getLogger(__name__)

# Declared here rather than in braintrust.env: this is one integration's behavior, and
# the knob should live next to the code that reads it.
_EVAL_EXPERIMENTS_ENV = EnvVar("BRAINTRUST_AGNO_EVAL_EXPERIMENTS", EnvParser.BOOL)

# Set by ``setup_agno(eval_experiments=...)``; ``None`` defers to the environment.
_override: bool | None = None

_active: contextvars.ContextVar[Any | None] = contextvars.ContextVar("braintrust_agno_eval_experiment", default=None)


def configure(eval_experiments: bool | None) -> None:
    """Record the caller's opt in/out of per-suite experiments."""
    global _override  # pylint: disable=global-statement
    _override = eval_experiments


def _enabled() -> bool:
    if _override is not None:
        return _override
    return _EVAL_EXPERIMENTS_ENV.get(True)


def active_experiment() -> Any | None:
    """Return the experiment opened for the running suite, if any."""
    return _active.get()


def _logger_project() -> dict[str, Any] | None:
    """Return ``init()`` project kwargs taken from the current logger.

    Read off the logger's stored metadata args rather than its ``project`` property,
    which would force a login and a project lookup just to answer the question. Core
    has no non-resolving accessor for this today.
    """
    active_logger = current_logger()
    if active_logger is None:
        return None
    args = getattr(active_logger, "_compute_metadata_args", None) or {}
    project_id = args.get("project_id")
    if project_id:
        return {"project_id": project_id}
    project_name = args.get("project_name")
    if project_name:
        return {"project": project_name}
    return None


@contextmanager
def suite_experiment(metadata: dict[str, Any] | None = None):
    """Open an experiment for a suite run, or leave rows to be routed as usual.

    Does nothing — so cases land wherever any other span would — when per-suite
    experiments are turned off, when the caller already opened an experiment, when no
    project is in scope, or when opening one fails. An eval suite must still run when
    Braintrust is misconfigured or offline. Callers read the result through
    ``active_experiment()``.
    """
    if not _enabled() or current_experiment() is not None:
        yield
        return

    project_args = _logger_project()
    if project_args is None:
        # Reachable when the logger was built without a project (init_logger() with no
        # arguments logs to the Global project), so say why nothing happened.
        logger.debug("No Braintrust project in scope; agno eval suite rows will be logged, not run as an experiment")
        yield
        return

    try:
        # set_current=False: ``state.current_experiment`` is process-global, and rows are
        # routed explicitly through ``experiment.start_span()`` instead, so a suite cannot
        # leak its experiment into unrelated tracing.
        experiment = init(**project_args, set_current=False, metadata=metadata or None)
    except Exception:
        logger.warning("Failed to open a Braintrust experiment for the agno eval suite", exc_info=True)
        yield
        return

    token = _active.set(experiment)
    try:
        yield
    finally:
        _active.reset(token)
        try:
            experiment.flush()
        except Exception:
            logger.warning("Failed to flush the agno eval suite experiment", exc_info=True)
