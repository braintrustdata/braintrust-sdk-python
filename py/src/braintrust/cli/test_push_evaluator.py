"""Tests for evaluator definition collection in push."""

from unittest.mock import MagicMock

import slugify

from ..test_helpers import assert_dict_matches
from .push import _collect_evaluator_defs


def _make_scorer(name):
    scorer = MagicMock()
    scorer.__name__ = name
    del scorer._name
    return scorer


def _make_evaluator(project_name, scorer_names, parameters=None, classifier_names=None):
    evaluator = MagicMock()
    evaluator.project_name = project_name
    evaluator.scores = [_make_scorer(n) for n in scorer_names]
    evaluator.classifiers = [_make_scorer(n) for n in (classifier_names or [])]
    evaluator.parameters = parameters

    instance = MagicMock()
    instance.evaluator = evaluator
    return instance


class TestCollectEvaluatorDefs:
    def test_basic_evaluator_def_structure(self, mock_project_ids):
        evaluators = {"my_eval": _make_evaluator("test-project", ["accuracy"])}

        functions = []
        _collect_evaluator_defs(mock_project_ids, functions, "bundle-abc", "replace", "evals/my_eval.py", evaluators)

        assert len(functions) == 1
        assert_dict_matches(
            functions[0],
            {
                "project_id": "project-123",
                "name": "Eval my_eval sandbox",
                "slug": slugify.slugify("my_eval-my_eval-sandbox"),
                "function_type": "sandbox",
                "function_data": {
                    "type": "code",
                    "data": {
                        "type": "bundle",
                        "location": {
                            "type": "sandbox",
                            "sandbox_spec": {"provider": "lambda"},
                            "entrypoints": ["evals/my_eval.py"],
                            "eval_name": "my_eval",
                            "evaluator_definition": {
                                "scores": [{"name": "accuracy"}],
                                "classifiers": [],
                            },
                        },
                        "bundle_id": "bundle-abc",
                    },
                },
                "metadata": {"_bt_sandbox_group_name": "my_eval"},
                "if_exists": "replace",
            },
        )

    def test_multiple_scorers(self, mock_project_ids):
        scorer_with_name_method = MagicMock()
        scorer_with_name_method._name.return_value = "relevance"
        del scorer_with_name_method.__name__

        evaluator = MagicMock()
        evaluator.project_name = "test-project"
        evaluator.scores = [
            _make_scorer("accuracy"),
            scorer_with_name_method,
            lambda output: 1.0,
        ]
        evaluator.parameters = None

        instance = MagicMock()
        instance.evaluator = evaluator
        evaluators = {"eval1": instance}

        functions = []
        _collect_evaluator_defs(mock_project_ids, functions, "bundle-1", "replace", "eval.py", evaluators)

        scores = functions[0]["function_data"]["data"]["location"]["evaluator_definition"]["scores"]
        assert scores == [{"name": "accuracy"}, {"name": "relevance"}, {"name": "scorer_2"}]

    def test_evaluator_with_parameters(self, mock_project_ids):
        params = {"prompt": {"type": "prompt", "default": None}}
        evaluators = {"eval1": _make_evaluator("test-project", ["accuracy"], parameters=params)}

        functions = []
        _collect_evaluator_defs(mock_project_ids, functions, "bundle-1", "replace", "eval.py", evaluators)

        eval_def = functions[0]["function_data"]["data"]["location"]["evaluator_definition"]
        assert "scores" in eval_def
        parameters = eval_def["parameters"]
        assert parameters["type"] == "braintrust.staticParameters"
        assert parameters["source"] is None
        assert parameters["schema"]["prompt"]["type"] == "prompt"

    def test_evaluator_with_classifiers(self, mock_project_ids):
        evaluators = {"eval1": _make_evaluator("test-project", ["accuracy"], classifier_names=["category"])}

        functions = []
        _collect_evaluator_defs(mock_project_ids, functions, "bundle-1", "replace", "eval.py", evaluators)

        eval_def = functions[0]["function_data"]["data"]["location"]["evaluator_definition"]
        assert eval_def["scores"] == [{"name": "accuracy"}]
        assert eval_def["classifiers"] == [{"name": "category"}]

    def test_slug_from_source_file(self, mock_project_ids):
        evaluators = {"Test Eval": _make_evaluator("test-project", ["accuracy"])}

        functions = []
        _collect_evaluator_defs(mock_project_ids, functions, "bundle-1", "replace", "path/to/my-eval.py", evaluators)

        assert functions[0]["slug"] == slugify.slugify("my-eval-Test Eval-sandbox")
        assert functions[0]["metadata"]["_bt_sandbox_group_name"] == "my-eval"

    def test_bundle_id_none(self, mock_project_ids):
        evaluators = {"eval1": _make_evaluator("test-project", ["accuracy"])}

        functions = []
        _collect_evaluator_defs(mock_project_ids, functions, None, "replace", "eval.py", evaluators)

        assert functions[0]["function_data"]["data"]["bundle_id"] is None
