"""Shared decision span structure from braintrust/typespecs/src/question.ts.

Provider adapters normalize their question and answer types before using these
helpers. Named maps become arrays linked by ID; arrays already carry their IDs.
"""

from collections.abc import Mapping
from typing import Any

from .utils import _try_to_dict


_QUESTION_FIELDS = ("type", "instructions", "criteria")
_ANSWER_FIELDS = ("type", "noul", "probability", "choice", "score", "confidence", "legend", "probabilities")


def _question_items(values: Any, fields: tuple[str, ...]) -> Any:
    if not isinstance(values, Mapping):
        return values
    items = []
    for identifier, value in values.items():
        value = _try_to_dict(value)
        if isinstance(value, dict):
            items.append({**{key: value[key] for key in fields if key in value}, "id": identifier})
    return items


def _question_span_input(state: Any, questions: Any) -> dict[str, Any]:
    return {"state": state, "questions": _question_items(questions, _QUESTION_FIELDS)}


def _question_span_output(answers: Any) -> dict[str, Any]:
    return {"answers": _question_items(answers, _ANSWER_FIELDS)}
