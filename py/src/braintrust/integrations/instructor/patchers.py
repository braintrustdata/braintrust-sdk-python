"""Patchers for the Instructor integration.

Targets:

* ``instructor.core.client.Instructor`` — sync structured-output client.
* ``instructor.core.client.AsyncInstructor`` — async structured-output client.

Each client exposes four entry points:
``create``, ``create_with_completion``, ``create_partial``, ``create_iterable``.

Only the Instructor layer is patched.  The underlying provider client
(``OpenAI``, ``Anthropic``, ``Cohere`` …) is left to the existing provider
integrations to instrument so that the LLM span stays a single child of the
Instructor parent and token usage is not double-counted.
"""

from braintrust.integrations.base import CompositeFunctionWrapperPatcher, FunctionWrapperPatcher

from .tracing import (
    _async_create_iterable_wrapper,
    _async_create_partial_wrapper,
    _async_create_with_completion_wrapper,
    _async_create_wrapper,
    _create_iterable_wrapper,
    _create_partial_wrapper,
    _create_with_completion_wrapper,
    _create_wrapper,
)


_TARGET_MODULE = "instructor.core.client"


# ---------------------------------------------------------------------------
# sync — Instructor
# ---------------------------------------------------------------------------


class InstructorCreatePatcher(FunctionWrapperPatcher):
    name = "instructor.create"
    target_module = _TARGET_MODULE
    target_path = "Instructor.create"
    wrapper = _create_wrapper


class InstructorCreateWithCompletionPatcher(FunctionWrapperPatcher):
    name = "instructor.create_with_completion"
    target_module = _TARGET_MODULE
    target_path = "Instructor.create_with_completion"
    wrapper = _create_with_completion_wrapper


class InstructorCreatePartialPatcher(FunctionWrapperPatcher):
    name = "instructor.create_partial"
    target_module = _TARGET_MODULE
    target_path = "Instructor.create_partial"
    wrapper = _create_partial_wrapper


class InstructorCreateIterablePatcher(FunctionWrapperPatcher):
    name = "instructor.create_iterable"
    target_module = _TARGET_MODULE
    target_path = "Instructor.create_iterable"
    wrapper = _create_iterable_wrapper


# ---------------------------------------------------------------------------
# async — AsyncInstructor
# ---------------------------------------------------------------------------


class AsyncInstructorCreatePatcher(FunctionWrapperPatcher):
    name = "instructor.async.create"
    target_module = _TARGET_MODULE
    target_path = "AsyncInstructor.create"
    wrapper = _async_create_wrapper


class AsyncInstructorCreateWithCompletionPatcher(FunctionWrapperPatcher):
    name = "instructor.async.create_with_completion"
    target_module = _TARGET_MODULE
    target_path = "AsyncInstructor.create_with_completion"
    wrapper = _async_create_with_completion_wrapper


class AsyncInstructorCreatePartialPatcher(FunctionWrapperPatcher):
    name = "instructor.async.create_partial"
    target_module = _TARGET_MODULE
    target_path = "AsyncInstructor.create_partial"
    wrapper = _async_create_partial_wrapper


class AsyncInstructorCreateIterablePatcher(FunctionWrapperPatcher):
    name = "instructor.async.create_iterable"
    target_module = _TARGET_MODULE
    target_path = "AsyncInstructor.create_iterable"
    wrapper = _async_create_iterable_wrapper


# ---------------------------------------------------------------------------
# Composite — one patcher per logical Instructor surface
# ---------------------------------------------------------------------------


class InstructorPatcher(CompositeFunctionWrapperPatcher):
    """Patches every Instructor / AsyncInstructor create-style entry point."""

    name = "instructor"
    sub_patchers = (
        InstructorCreatePatcher,
        InstructorCreateWithCompletionPatcher,
        InstructorCreatePartialPatcher,
        InstructorCreateIterablePatcher,
        AsyncInstructorCreatePatcher,
        AsyncInstructorCreateWithCompletionPatcher,
        AsyncInstructorCreatePartialPatcher,
        AsyncInstructorCreateIterablePatcher,
    )
