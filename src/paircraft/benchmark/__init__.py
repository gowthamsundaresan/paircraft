"""Agent-facing benchmark API.

Prompts, typed query messages, canonical documents, and the transport-neutral
tool adapter are safe interface objects. The adapter's handler, harness,
transcripts, evaluator store, answers, and scores must remain in the trusted
server process; Python module visibility does not enforce that boundary.
"""

from paircraft_benchmark_contract import (
    FieldQueryPolicy,
    NumericalScoringPolicy,
    QueryBudget,
    SensitivityQueryAction,
    SensitivityQueryResponse,
    SensitivityQueryToolAdapter,
    TaskPrompt,
    VisibleObservation,
    sensitivity_query_action_document,
    sensitivity_query_action_schema,
    sensitivity_query_response_document,
    sensitivity_query_response_schema,
    task_prompt_document,
    task_prompt_from_document,
)

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
