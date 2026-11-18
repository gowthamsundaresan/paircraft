"""Standalone Paircraft benchmark contract for untrusted agent deployments.

This module intentionally uses only the Python standard library. It defines
the visible prompt and query RPC messages, but contains no evaluator, answer,
oracle, spin solver, sweep plan, or scientific generator implementation.
Public initial observations can support scientific inference; direct simulator
execution is unavailable in the deployed agent artifact.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite
from string import hexdigits
from typing import Final

_PROMPT_SCHEMA: Final = "paircraft-benchmark-task-prompt-v5"
_SCORING_POLICY_ID: Final = "paircraft-yield-prediction-score-v1"
_ABSOLUTE_ERROR_TOLERANCE: Final = 1e-4
_RELATIVE_ERROR_TOLERANCE: Final = 1e-3
_PASS_NORMALIZED_ERROR: Final = 1.0
_SCORING_INTERPRETATION: Final = (
    "Numerical prediction grading only. The tolerances are not experimental, "
    "biological, parameter, model-form, or predictive uncertainty."
)


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(payload: object) -> str:
    return sha256(_canonical_bytes(payload)).hexdigest()


def _text(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be explicit")


def _digest(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in hexdigits for character in value)
        or value != value.lower()
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _finite(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value):
        raise ValueError(f"{label} must be a finite real scalar")
    return float(value)


@dataclass(frozen=True, slots=True)
class QueryBudget:
    """Agent-visible hard query limit."""

    maximum_queries: int

    def __post_init__(self) -> None:
        if isinstance(self.maximum_queries, bool) or not isinstance(self.maximum_queries, int):
            raise ValueError("maximum queries must be an integer")
        if self.maximum_queries < 0:
            raise ValueError("maximum queries cannot be negative")


@dataclass(frozen=True, slots=True)
class NumericalScoringPolicy:
    """Visible deterministic prediction grading; never an uncertainty estimate."""

    policy_id: str = _SCORING_POLICY_ID
    absolute_error_tolerance: float = _ABSOLUTE_ERROR_TOLERANCE
    relative_error_tolerance: float = _RELATIVE_ERROR_TOLERANCE
    pass_normalized_error: float = _PASS_NORMALIZED_ERROR
    interpretation: str = _SCORING_INTERPRETATION

    def __post_init__(self) -> None:
        _text(self.policy_id, "scoring-policy ID")
        _text(self.interpretation, "scoring-policy interpretation")
        absolute = _finite(self.absolute_error_tolerance, "absolute error tolerance")
        relative = _finite(self.relative_error_tolerance, "relative error tolerance")
        threshold = _finite(self.pass_normalized_error, "normalized-error pass threshold")
        if (
            absolute != _ABSOLUTE_ERROR_TOLERANCE
            or relative != _RELATIVE_ERROR_TOLERANCE
            or threshold != _PASS_NORMALIZED_ERROR
            or self.policy_id != _SCORING_POLICY_ID
            or self.interpretation != _SCORING_INTERPRETATION
        ):
            raise ValueError("v1 scoring policy is fixed and cannot be caller-tuned")
        object.__setattr__(self, "absolute_error_tolerance", absolute)
        object.__setattr__(self, "relative_error_tolerance", relative)
        object.__setattr__(self, "pass_normalized_error", threshold)

    def allowed_error(self, reference_answer: float) -> float:
        """Return the fixed absolute-plus-relative numerical grading scale."""

        reference = _finite(reference_answer, "reference answer")
        return max(
            self.absolute_error_tolerance,
            self.relative_error_tolerance * abs(reference),
        )


@dataclass(frozen=True, slots=True)
class VisibleObservation:
    """Public simulator observation carrying only an opaque evaluator receipt."""

    field_tesla: float
    measurement_type: str
    value: float
    unit: str
    observation_receipt_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "field_tesla", _finite(self.field_tesla, "observation field"))
        object.__setattr__(self, "value", _finite(self.value, "observation value"))
        _text(self.measurement_type, "observation measurement type")
        _text(self.unit, "observation unit")
        _digest(self.observation_receipt_sha256, "observation receipt sha256")


@dataclass(frozen=True, slots=True)
class FieldQueryPolicy:
    """Structured visible action policy for one withheld-field task."""

    measurement_type: str
    allowed_fields_tesla: tuple[float, ...]
    withheld_target_field_tesla: float
    field_unit: str = "T"

    def __post_init__(self) -> None:
        _text(self.measurement_type, "query-policy measurement type")
        _text(self.field_unit, "query-policy field unit")
        if self.field_unit != "T":
            raise ValueError("query-policy magnetic flux density must use tesla")
        if not isinstance(self.allowed_fields_tesla, tuple) or not self.allowed_fields_tesla:
            raise ValueError("query policy requires an immutable nonempty field tuple")
        allowed = tuple(
            _finite(value, "allowed query field") for value in self.allowed_fields_tesla
        )
        target = _finite(self.withheld_target_field_tesla, "withheld target field")
        if allowed != tuple(sorted(set(allowed))) or target in allowed:
            raise ValueError("query fields must be unique, sorted, and exclude the target")
        object.__setattr__(self, "allowed_fields_tesla", allowed)
        object.__setattr__(self, "withheld_target_field_tesla", target)


@dataclass(frozen=True, slots=True)
class SensitivityQueryAction:
    """Typed agent request accepted by the remote evaluator gateway."""

    field_tesla: float
    measurement_type: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "field_tesla", _finite(self.field_tesla, "query field"))
        _text(self.measurement_type, "query measurement type")


def sensitivity_query_action_document(action: SensitivityQueryAction) -> dict[str, object]:
    """Return the canonical JSON-compatible query request."""

    if not isinstance(action, SensitivityQueryAction):
        raise ValueError("query-action serialization requires a SensitivityQueryAction")
    return asdict(action)


@dataclass(frozen=True, slots=True)
class SensitivityQueryResponse:
    """Public response carrying a value and opaque evaluator receipt."""

    field_tesla: float
    measurement_type: str
    value: float
    unit: str
    query_receipt_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "field_tesla", _finite(self.field_tesla, "response field"))
        object.__setattr__(self, "value", _finite(self.value, "response value"))
        _text(self.measurement_type, "response measurement type")
        _text(self.unit, "response unit")
        _digest(self.query_receipt_sha256, "query receipt sha256")


def sensitivity_query_response_document(response: SensitivityQueryResponse) -> dict[str, object]:
    """Return the canonical JSON-compatible query response."""

    if not isinstance(response, SensitivityQueryResponse):
        raise ValueError("query-response serialization requires a SensitivityQueryResponse")
    return asdict(response)


class SensitivityQueryToolAdapter:
    """Model-vendor-neutral JSON/RPC adapter around a remote query handler."""

    def __init__(
        self,
        handler: Callable[[SensitivityQueryAction], SensitivityQueryResponse],
    ) -> None:
        if not callable(handler):
            raise ValueError("query-tool adapter requires a callable handler")
        self._handler = handler

    def invoke(self, request: Mapping[str, object]) -> dict[str, object]:
        """Validate a JSON-like request and return a canonical response document."""

        if not isinstance(request, Mapping) or set(request) != {
            "field_tesla",
            "measurement_type",
        }:
            raise ValueError("query-tool request has unsupported or missing fields")
        field = request["field_tesla"]
        measurement = request["measurement_type"]
        if isinstance(field, bool) or not isinstance(field, int | float):
            raise ValueError("query-tool field_tesla must be a real scalar")
        if not isinstance(measurement, str):
            raise ValueError("query-tool measurement_type must be text")
        response = self._handler(SensitivityQueryAction(float(field), measurement))
        if not isinstance(response, SensitivityQueryResponse):
            raise ValueError("query-tool handler returned an invalid response")
        return sensitivity_query_response_document(response)


@dataclass(frozen=True, slots=True)
class TaskPrompt:
    """Complete benchmark state visible inside the untrusted agent process."""

    task_id: str
    system_description: str
    objective: str
    measurement_type: str
    answer_unit: str
    query_budget: QueryBudget
    interaction_constraints: tuple[str, ...]
    initial_observations: tuple[VisibleObservation, ...]
    action_policy: FieldQueryPolicy
    scoring_policy: NumericalScoringPolicy
    schema_version: str = _PROMPT_SCHEMA

    def __post_init__(self) -> None:
        for value, label in (
            (self.task_id, "task ID"),
            (self.system_description, "system description"),
            (self.objective, "objective"),
            (self.measurement_type, "measurement type"),
            (self.answer_unit, "answer unit"),
        ):
            _text(value, label)
        if not isinstance(self.query_budget, QueryBudget):
            raise ValueError("task prompt requires a typed query budget")
        if not isinstance(self.initial_observations, tuple) or any(
            not isinstance(item, VisibleObservation) for item in self.initial_observations
        ):
            raise ValueError("initial observations must be an immutable typed tuple")
        if not isinstance(self.action_policy, FieldQueryPolicy):
            raise ValueError("task prompt requires a typed action policy")
        if not isinstance(self.scoring_policy, NumericalScoringPolicy):
            raise ValueError("task prompt requires the fixed numerical scoring policy")
        if not isinstance(self.interaction_constraints, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.interaction_constraints
        ):
            raise ValueError("interaction constraints must be an immutable tuple of IDs")
        if len(set(self.interaction_constraints)) != len(self.interaction_constraints):
            raise ValueError("interaction constraint IDs must be unique")
        if not self.initial_observations:
            raise ValueError("task prompt requires at least one initial observation")
        observation_fields = tuple(item.field_tesla for item in self.initial_observations)
        if len(set(observation_fields)) != len(observation_fields):
            raise ValueError("initial observation fields must be unique")
        if (
            any(
                item.measurement_type != self.measurement_type for item in self.initial_observations
            )
            or self.action_policy.measurement_type != self.measurement_type
        ):
            raise ValueError(
                "prompt, observations, and action policy must share a measurement type"
            )
        if any(item.unit != self.answer_unit for item in self.initial_observations):
            raise ValueError("initial observations and answer must share one unit")
        visible_fields = set(observation_fields)
        allowed_fields = set(self.action_policy.allowed_fields_tesla)
        target = self.action_policy.withheld_target_field_tesla
        if visible_fields & allowed_fields or target in visible_fields or target in allowed_fields:
            raise ValueError("observation, allowed-query, and target fields must not overlap")
        if self.query_budget.maximum_queries > len(allowed_fields):
            raise ValueError("query budget cannot exceed the number of declared query fields")
        if self.schema_version != _PROMPT_SCHEMA:
            raise ValueError("task-prompt schema version is unsupported")

    @property
    def prompt_sha256(self) -> str:
        return _sha(task_prompt_document(self))


def task_prompt_document(prompt: TaskPrompt) -> dict[str, object]:
    """Serialize agent-visible state without evaluator answers or commitments."""

    if not isinstance(prompt, TaskPrompt):
        raise ValueError("prompt document requires a TaskPrompt")
    return {
        "schema_version": prompt.schema_version,
        "task_id": prompt.task_id,
        "system_description": prompt.system_description,
        "objective": prompt.objective,
        "measurement_type": prompt.measurement_type,
        "answer_unit": prompt.answer_unit,
        "query_budget": asdict(prompt.query_budget),
        "interaction_constraints": list(prompt.interaction_constraints),
        "initial_observations": [asdict(item) for item in prompt.initial_observations],
        "action_policy": {
            "measurement_type": prompt.action_policy.measurement_type,
            "allowed_fields_tesla": list(prompt.action_policy.allowed_fields_tesla),
            "withheld_target_field_tesla": prompt.action_policy.withheld_target_field_tesla,
            "field_unit": prompt.action_policy.field_unit,
        },
        "scoring_policy": asdict(prompt.scoring_policy),
    }


def task_prompt_from_document(document: Mapping[str, object]) -> TaskPrompt:
    """Reconstruct a prompt without coercing or dropping untrusted values."""

    expected_prompt_keys = {
        "schema_version",
        "task_id",
        "system_description",
        "objective",
        "measurement_type",
        "answer_unit",
        "query_budget",
        "interaction_constraints",
        "initial_observations",
        "action_policy",
        "scoring_policy",
    }
    try:
        if set(document) != expected_prompt_keys:
            raise TypeError
        schema_version = document["schema_version"]
        task_id = document["task_id"]
        system_description = document["system_description"]
        objective = document["objective"]
        measurement_type = document["measurement_type"]
        answer_unit = document["answer_unit"]
        if not isinstance(schema_version, str):
            raise TypeError
        if not isinstance(task_id, str):
            raise TypeError
        if not isinstance(system_description, str):
            raise TypeError
        if not isinstance(objective, str):
            raise TypeError
        if not isinstance(measurement_type, str):
            raise TypeError
        if not isinstance(answer_unit, str):
            raise TypeError

        budget = document["query_budget"]
        if not isinstance(budget, Mapping) or set(budget) != {"maximum_queries"}:
            raise TypeError
        maximum_queries = budget["maximum_queries"]
        if isinstance(maximum_queries, bool) or not isinstance(maximum_queries, int):
            raise TypeError

        constraints = document["interaction_constraints"]
        if not isinstance(constraints, list) or any(
            not isinstance(item, str) for item in constraints
        ):
            raise TypeError

        observations_payload = document["initial_observations"]
        if not isinstance(observations_payload, list) or any(
            not isinstance(item, Mapping) for item in observations_payload
        ):
            raise TypeError
        observations = tuple(_observation_from_document(item) for item in observations_payload)

        policy_payload = document["action_policy"]
        scoring_payload = document["scoring_policy"]
        if not isinstance(policy_payload, Mapping) or not isinstance(scoring_payload, Mapping):
            raise TypeError
        policy = _field_query_policy_from_document(policy_payload)
        scoring = _scoring_policy_from_document(scoring_payload)
        return TaskPrompt(
            task_id=task_id,
            system_description=system_description,
            objective=objective,
            measurement_type=measurement_type,
            answer_unit=answer_unit,
            query_budget=QueryBudget(maximum_queries),
            interaction_constraints=tuple(constraints),
            initial_observations=observations,
            action_policy=policy,
            scoring_policy=scoring,
            schema_version=schema_version,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("prompt document is malformed or unsupported") from error


def _observation_from_document(document: Mapping[str, object]) -> VisibleObservation:
    expected = {
        "field_tesla",
        "measurement_type",
        "value",
        "unit",
        "observation_receipt_sha256",
    }
    if set(document) != expected:
        raise TypeError
    field = document["field_tesla"]
    value = document["value"]
    measurement = document["measurement_type"]
    unit = document["unit"]
    receipt = document["observation_receipt_sha256"]
    if isinstance(field, bool) or not isinstance(field, int | float):
        raise TypeError
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError
    if (
        not isinstance(measurement, str)
        or not isinstance(unit, str)
        or not isinstance(receipt, str)
    ):
        raise TypeError
    return VisibleObservation(float(field), measurement, float(value), unit, receipt)


def _field_query_policy_from_document(document: Mapping[str, object]) -> FieldQueryPolicy:
    expected = {
        "measurement_type",
        "allowed_fields_tesla",
        "withheld_target_field_tesla",
        "field_unit",
    }
    if set(document) != expected:
        raise TypeError
    measurement = document["measurement_type"]
    fields = document["allowed_fields_tesla"]
    target = document["withheld_target_field_tesla"]
    unit = document["field_unit"]
    if not isinstance(measurement, str) or not isinstance(unit, str):
        raise TypeError
    if not isinstance(fields, list) or any(
        isinstance(item, bool) or not isinstance(item, int | float) for item in fields
    ):
        raise TypeError
    if isinstance(target, bool) or not isinstance(target, int | float):
        raise TypeError
    return FieldQueryPolicy(
        measurement,
        tuple(float(item) for item in fields),
        float(target),
        unit,
    )


def _scoring_policy_from_document(document: Mapping[str, object]) -> NumericalScoringPolicy:
    expected = {
        "policy_id",
        "absolute_error_tolerance",
        "relative_error_tolerance",
        "pass_normalized_error",
        "interpretation",
    }
    if set(document) != expected:
        raise TypeError
    policy_id = document["policy_id"]
    absolute = document["absolute_error_tolerance"]
    relative = document["relative_error_tolerance"]
    threshold = document["pass_normalized_error"]
    interpretation = document["interpretation"]
    if not isinstance(policy_id, str) or not isinstance(interpretation, str):
        raise TypeError
    if isinstance(absolute, bool) or not isinstance(absolute, int | float):
        raise TypeError
    if isinstance(relative, bool) or not isinstance(relative, int | float):
        raise TypeError
    if isinstance(threshold, bool) or not isinstance(threshold, int | float):
        raise TypeError
    return NumericalScoringPolicy(
        policy_id=policy_id,
        absolute_error_tolerance=float(absolute),
        relative_error_tolerance=float(relative),
        pass_normalized_error=float(threshold),
        interpretation=interpretation,
    )


def sensitivity_query_action_schema() -> dict[str, object]:
    """Return the transport-neutral request JSON schema."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["field_tesla", "measurement_type"],
        "properties": {
            "field_tesla": {"type": "number"},
            "measurement_type": {"type": "string", "minLength": 1},
        },
    }


def sensitivity_query_response_schema() -> dict[str, object]:
    """Return the transport-neutral response JSON schema."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "field_tesla",
            "measurement_type",
            "value",
            "unit",
            "query_receipt_sha256",
        ],
        "properties": {
            "field_tesla": {"type": "number"},
            "measurement_type": {"type": "string", "minLength": 1},
            "value": {"type": "number"},
            "unit": {"type": "string", "minLength": 1},
            "query_receipt_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
    }


__all__ = [
    "FieldQueryPolicy",
    "NumericalScoringPolicy",
    "QueryBudget",
    "SensitivityQueryAction",
    "SensitivityQueryResponse",
    "SensitivityQueryToolAdapter",
    "TaskPrompt",
    "VisibleObservation",
    "sensitivity_query_action_document",
    "sensitivity_query_action_schema",
    "sensitivity_query_response_document",
    "sensitivity_query_response_schema",
    "task_prompt_document",
    "task_prompt_from_document",
]
