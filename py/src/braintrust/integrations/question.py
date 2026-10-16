"""Shared decision span structure from braintrust/typespecs/src/question.ts.

Provider adapters normalize their question and answer types before using these
helpers. Named maps become arrays linked by ID; arrays already carry their IDs.
"""

import logging
from collections.abc import Callable, Mapping
from typing import Any

from braintrust.logger import Attachment

from .utils import _materialize_attachment, _try_to_dict


_QUESTION_FIELDS = ("type", "instructions", "criteria")
_ANSWER_FIELDS = (
    "type",
    "noul",
    "probability",
    "choice",
    "score",
    "confidence",
    "legend",
    "probabilities",
    "box",
    "image_path",
)
_ATTACHMENT_TYPES = ("braintrust_attachment", "external_attachment", "inline_attachment")
_log = logging.getLogger(__name__)


def _decision_media(value: Any, media_transform: Callable[[Any], Any] | None = None) -> Any:
    """Replace inline media in the logged payload without changing request values."""
    if isinstance(value, Mapping) and value.get("type") in _ATTACHMENT_TYPES:
        return value
    if media_transform is not None:
        transformed = media_transform(value)
        if transformed is not value:
            return transformed
    if isinstance(value, Mapping):
        return {key: _decision_media(item, media_transform) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_decision_media(item, media_transform) for item in value]
    if isinstance(value, str) and value.startswith("data:"):
        resolved = _materialize_attachment(value)
        if resolved is not None:
            return resolved.attachment
    return value


def _decision_image_paths(value: Any, path: str = "") -> list[str]:
    """Find image occurrences, retaining repeated locations and JSON Pointer escaping."""
    if isinstance(value, Attachment):
        return [path] if value.reference.get("content_type", "").startswith("image/") else []
    if isinstance(value, Mapping):
        if value.get("type") in _ATTACHMENT_TYPES:
            return [path] if str(value.get("content_type", "")).startswith("image/") else []
        if value.get("type") == "input_image" and isinstance(value.get("image_url"), str):
            return [path + "/image_url"]
        if value.get("type") == "image_url" and isinstance(value.get("image_url"), Mapping):
            return [path + "/image_url/url"]
        return [
            image_path
            for key, item in value.items()
            for image_path in _decision_image_paths(item, path + "/" + str(key).replace("~", "~0").replace("/", "~1"))
        ]
    if isinstance(value, (list, tuple)):
        return [
            image_path
            for index, item in enumerate(value)
            for image_path in _decision_image_paths(item, f"{path}/{index}")
        ]
    if isinstance(value, str) and value.startswith("data:image/"):
        return [path]
    return []


def _question_items(values: Any, fields: tuple[str, ...]) -> Any:
    if not isinstance(values, Mapping):
        return values
    items = []
    for identifier, value in values.items():
        value = _try_to_dict(value)
        if isinstance(value, dict):
            payload = {key: value[key] for key in fields if key in value}
            if "box" in payload:
                payload["box"] = _try_to_dict(payload["box"])
            items.append({**payload, "id": identifier})
    return items


def _question_span_input(
    state: Any, questions: Any, *, media_transform: Callable[[Any], Any] | None = None
) -> dict[str, Any]:
    payload = {"state": state, "questions": _question_items(questions, _QUESTION_FIELDS)}
    try:
        return {
            "state": _decision_media(state, media_transform),
            "questions": [
                {
                    key: _decision_media(value, media_transform) if key in ("instructions", "criteria") else value
                    for key, value in question.items()
                }
                for question in payload["questions"]
            ]
            if isinstance(payload["questions"], list)
            else payload["questions"],
        }
    except Exception:
        # Abort the whole rewrite if a conversion fails; retain all original media.
        _log.debug("Failed to capture decision media", exc_info=True)
        return payload


def _question_span_output(answers: Any, span_input: Any = None) -> dict[str, Any]:
    items = _question_items(answers, _ANSWER_FIELDS)
    if not isinstance(items, list) or not any(
        isinstance(answer, dict) and answer.get("type") == "box" and "image_path" not in answer for answer in items
    ):
        return {"answers": items}
    image_paths = _decision_image_paths(span_input)
    if len(image_paths) == 1:
        items = [
            {**answer, "image_path": image_paths[0]}
            if isinstance(answer, dict) and answer.get("type") == "box" and "image_path" not in answer
            else answer
            for answer in items
        ]
    return {"answers": items}
