from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from hashlib import sha256

import pytest

import paircraft.benchmark as agent_api
from paircraft.benchmark.model import (
    EpisodeResult,
    EvaluationStore,
    OutcomeScore,
    ReliabilityClass,
    episode_document,
)
from paircraft.benchmark.runner import (
    AgentRunIdentity,
    LocalCompletedTrial,
    LocalFailedTrial,
    LocalTrialFailureCategory,
    SensitivityAgent,
    run_local_five_trial_reliability,
    run_local_sensitivity_episode,
)
from paircraft_benchmark_contract import (
    FieldQueryPolicy,
    NumericalScoringPolicy,
    QueryBudget,
    SensitivityQueryAction,
    SensitivityQueryResponse,
    TaskPrompt,
    VisibleObservation,
    task_prompt_document,
    task_prompt_from_document,
)


@pytest.fixture
def identity() -> AgentRunIdentity:
    return AgentRunIdentity(
        model_id="local-test-agent",
        model_version="1.0.0",
        model_artifact_sha256="a" * 64,
        configuration_sha256="b" * 64,
        harness_id="paircraft-local-runner",
        harness_version="1.0.0",
        harness_artifact_sha256="c" * 64,
    )


class FakeGateway:
    def __init__(self, *, run_id: str, identity: AgentRunIdentity) -> None:
        self._run_id = run_id
        self._identity = identity
        self.queries: list[SensitivityQueryAction] = []
        self.submissions: list[float] = []

    def query(self, action: SensitivityQueryAction) -> SensitivityQueryResponse:
        self.queries.append(action)
        return SensitivityQueryResponse(
            field_tesla=action.field_tesla,
            measurement_type=action.measurement_type,
            value=0.0,
            unit="1",
            query_receipt_sha256="d" * 64,
        )

    def submit_final_answer(self, answer: float) -> EpisodeResult:
        self.submissions.append(answer)
        error = abs(answer)
        query_count = len(self.queries)
        correct = error <= 1.0
        budget = query_count <= 1
        passed = correct and budget
        return EpisodeResult(
            case_sha256="e" * 64,
            transcript_sha256=sha256(self._run_id.encode()).hexdigest(),
            run_id=self._run_id,
            model_id=self._identity.model_id,
            model_version=self._identity.model_version,
            model_artifact_sha256=self._identity.model_artifact_sha256,
            configuration_sha256=self._identity.configuration_sha256,
            score=OutcomeScore(
                predicted_answer=answer,
                absolute_error=error,
                allowed_error=1.0,
                normalized_error=error,
                query_count=query_count,
                maximum_queries=1,
                hard_constraint_violations=(),
                answer_correct=correct,
                budget_compliant=budget,
                interaction_policy_compliant=True,
                passed=passed,
                reward=1.0 / (1.0 + error) if budget else 0.0,
            ),
        )


@pytest.fixture
def local_store(
    monkeypatch: pytest.MonkeyPatch,
    identity: AgentRunIdentity,
) -> tuple[EvaluationStore, TaskPrompt, list[FakeGateway]]:
    measurement = "simulated absolute secondary-pair yield"
    prompt = TaskPrompt(
        task_id="case-local-runner",
        system_description="Opaque simulator-only sensitivity system.",
        objective="Predict one withheld scalar outcome.",
        measurement_type=measurement,
        answer_unit="1",
        query_budget=QueryBudget(1),
        interaction_constraints=("declared-query-field",),
        initial_observations=(
            VisibleObservation(
                field_tesla=0.0,
                measurement_type=measurement,
                value=0.0,
                unit="1",
                observation_receipt_sha256="a" * 64,
            ),
        ),
        action_policy=FieldQueryPolicy(
            measurement_type=measurement,
            allowed_fields_tesla=(50e-6,),
            withheld_target_field_tesla=200e-6,
        ),
        scoring_policy=NumericalScoringPolicy(),
    )
    store = EvaluationStore(evaluator_nonce_sha256="f" * 64)
    gateways: list[FakeGateway] = []

    def stored_prompt(task_id: str) -> TaskPrompt:
        if task_id != prompt.task_id:
            raise ValueError("unknown task")
        return prompt

    def case_manifest(task_id: str) -> dict[str, object]:
        if task_id != prompt.task_id:
            raise ValueError("unknown task")
        return {"case_sha256": "e" * 64}

    def open_gateway(task_id: str, **kwargs: object) -> FakeGateway:
        if task_id != prompt.task_id:
            raise ValueError("unknown task")
        run_id = kwargs["run_id"]
        if not isinstance(run_id, str):
            raise ValueError("run ID must be text")
        gateway = FakeGateway(run_id=run_id, identity=identity)
        gateways.append(gateway)
        return gateway

    monkeypatch.setattr(store, "prompt", stored_prompt)
    monkeypatch.setattr(store, "case_manifest", case_manifest)
    monkeypatch.setattr(store, "open_sensitivity_gateway", open_gateway)
    return store, prompt, gateways


class QueryingAgent:
    def run(
        self,
        prompt: TaskPrompt,
        query: Callable[[SensitivityQueryAction], SensitivityQueryResponse],
    ) -> float:
        response = query(
            SensitivityQueryAction(
                field_tesla=50e-6,
                measurement_type=prompt.measurement_type,
            )
        )
        return response.value


class ConstantAgent:
    def __init__(self, answer: float) -> None:
        self._answer = answer

    def run(
        self,
        prompt: TaskPrompt,
        query: Callable[[SensitivityQueryAction], SensitivityQueryResponse],
    ) -> float:
        del prompt, query
        return self._answer


class FailingAgent:
    def run(
        self,
        prompt: TaskPrompt,
        query: Callable[[SensitivityQueryAction], SensitivityQueryResponse],
    ) -> float:
        del prompt, query
        raise RuntimeError("agent failed")


def test_local_runner_accepts_model_agnostic_agent_and_records_query(
    local_store: tuple[EvaluationStore, TaskPrompt, list[FakeGateway]],
    identity: AgentRunIdentity,
) -> None:
    store, prompt, gateways = local_store
    agent: SensitivityAgent = QueryingAgent()

    episode = run_local_sensitivity_episode(
        store,
        task_id=prompt.task_id,
        run_id="querying-agent",
        identity=identity,
        agent=agent,
    )

    assert episode.run_id == "querying-agent"
    assert episode.model_id == identity.model_id
    assert episode.score.query_count == 1
    assert len(episode.transcript_sha256) == 64
    assert len(gateways) == 1
    assert gateways[0].queries[0].field_tesla == 50e-6


def test_interaction_policy_wire_names_reject_legacy_protocol_shape(
    local_store: tuple[EvaluationStore, TaskPrompt, list[FakeGateway]],
) -> None:
    _, prompt, _ = local_store
    document = task_prompt_document(prompt)
    legacy = dict(document)
    legacy["protocol_constraints"] = legacy.pop("interaction_constraints")

    assert document["schema_version"] == "paircraft-benchmark-task-prompt-v5"
    assert "interaction_constraints" in document
    with pytest.raises(ValueError, match="malformed or unsupported"):
        task_prompt_from_document(legacy)


def test_local_runner_does_not_change_agent_safe_exports(
    local_store: tuple[EvaluationStore, TaskPrompt, list[FakeGateway]],
    identity: AgentRunIdentity,
) -> None:
    store, prompt, _ = local_store
    episode = run_local_sensitivity_episode(
        store,
        task_id=prompt.task_id,
        run_id="constant-agent",
        identity=identity,
        agent=ConstantAgent(0.0),
    )

    assert episode.score.query_count == 0
    serialized = episode_document(episode)
    assert serialized["schema_version"] == "paircraft-benchmark-episode-v3"
    assert serialized["score"]["interaction_policy_compliant"] is True  # type: ignore[index]
    assert "protocol_valid" not in serialized["score"]  # type: ignore[operator]
    assert "EvaluationStore" not in agent_api.__all__
    assert "run_local_sensitivity_episode" not in agent_api.__all__


def test_five_trial_runner_uses_fresh_agents_and_distinct_run_ids(
    local_store: tuple[EvaluationStore, TaskPrompt, list[FakeGateway]],
    identity: AgentRunIdentity,
) -> None:
    store, prompt, gateways = local_store
    created: list[int] = []

    def factory(index: int) -> ConstantAgent:
        created.append(index)
        return ConstantAgent(0.0)

    summary = run_local_five_trial_reliability(
        store,
        task_id=prompt.task_id,
        run_id_prefix="reliability",
        identity=identity,
        agent_factory=factory,
    )

    assert created == [0, 1, 2, 3, 4]
    assert tuple(trial.run_id for trial in summary.trials) == (
        "reliability-1",
        "reliability-2",
        "reliability-3",
        "reliability-4",
        "reliability-5",
    )
    completed = tuple(trial for trial in summary.trials if isinstance(trial, LocalCompletedTrial))
    assert len({trial.transcript_sha256 for trial in completed}) == 5
    assert summary.classification is ReliabilityClass.RELIABLE
    assert summary.execution_failure_trials == 0
    assert summary.case_sha256 == "e" * 64
    assert summary.harness_id == identity.harness_id
    assert summary.harness_version == identity.harness_version
    assert summary.harness_artifact_sha256 == identity.harness_artifact_sha256
    assert len(gateways) == 5


@pytest.mark.parametrize("failed_index", [0, 2, 4])
def test_five_trial_runner_retains_early_middle_and_final_agent_failures(
    local_store: tuple[EvaluationStore, TaskPrompt, list[FakeGateway]],
    identity: AgentRunIdentity,
    failed_index: int,
) -> None:
    store, prompt, _ = local_store

    def factory(index: int) -> SensitivityAgent:
        return FailingAgent() if index == failed_index else ConstantAgent(0.0)

    summary = run_local_five_trial_reliability(
        store,
        task_id=prompt.task_id,
        run_id_prefix="agent-failure",
        identity=identity,
        agent_factory=factory,
    )

    failures = tuple(trial for trial in summary.trials if isinstance(trial, LocalFailedTrial))
    assert tuple(trial.run_id for trial in summary.trials) == tuple(
        f"agent-failure-{index}" for index in range(1, 6)
    )
    assert len(failures) == 1
    assert failures[0].run_id == f"agent-failure-{failed_index + 1}"
    assert failures[0].category is LocalTrialFailureCategory.AGENT_EXECUTION_FAILURE
    assert failures[0].exception_type == "RuntimeError"
    assert summary.passed_trials == 4
    assert summary.failed_trials == 1
    assert summary.execution_failure_trials == 1
    assert summary.classification is ReliabilityClass.RELIABLE


def test_five_trial_runner_counts_nonfinite_output_without_inventing_episode(
    local_store: tuple[EvaluationStore, TaskPrompt, list[FakeGateway]],
    identity: AgentRunIdentity,
) -> None:
    store, prompt, gateways = local_store

    def factory(index: int) -> ConstantAgent:
        return ConstantAgent(float("nan") if index == 2 else 0.0)

    summary = run_local_five_trial_reliability(
        store,
        task_id=prompt.task_id,
        run_id_prefix="invalid-answer",
        identity=identity,
        agent_factory=factory,
    )
    failed = summary.trials[2]

    assert isinstance(failed, LocalFailedTrial)
    assert failed.category is LocalTrialFailureCategory.INVALID_FINAL_ANSWER
    assert gateways[2].submissions == []
    assert summary.passed_trials == 4
    assert summary.execution_failure_trials == 1


def test_five_trial_runner_rejects_reused_agent_instance_before_execution(
    local_store: tuple[EvaluationStore, TaskPrompt, list[FakeGateway]],
    identity: AgentRunIdentity,
) -> None:
    store, prompt, gateways = local_store
    shared = ConstantAgent(0.0)

    with pytest.raises(ValueError, match="distinct successfully constructed agents"):
        run_local_five_trial_reliability(
            store,
            task_id=prompt.task_id,
            run_id_prefix="reused-agent",
            identity=identity,
            agent_factory=lambda index: shared,
        )
    assert gateways == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("case_sha256", "0" * 64),
        ("run_id", "wrong-run"),
        ("model_id", "wrong-model"),
        ("model_version", "wrong-version"),
        ("model_artifact_sha256", "1" * 64),
        ("configuration_sha256", "2" * 64),
    ],
)
def test_five_trial_runner_rejects_contradictory_episode_identity(
    local_store: tuple[EvaluationStore, TaskPrompt, list[FakeGateway]],
    identity: AgentRunIdentity,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    store, prompt, _ = local_store
    original_submit = FakeGateway.submit_final_answer

    def contradictory_submit(gateway: FakeGateway, answer: float) -> EpisodeResult:
        episode = original_submit(gateway, answer)
        if field == "case_sha256":
            return replace(episode, case_sha256=value)
        if field == "run_id":
            return replace(episode, run_id=value)
        if field == "model_id":
            return replace(episode, model_id=value)
        if field == "model_version":
            return replace(episode, model_version=value)
        if field == "model_artifact_sha256":
            return replace(episode, model_artifact_sha256=value)
        if field == "configuration_sha256":
            return replace(episode, configuration_sha256=value)
        raise AssertionError("test requested an unsupported identity field")

    monkeypatch.setattr(FakeGateway, "submit_final_answer", contradictory_submit)
    summary = run_local_five_trial_reliability(
        store,
        task_id=prompt.task_id,
        run_id_prefix="identity-mismatch",
        identity=identity,
        agent_factory=lambda index: ConstantAgent(float(index) / 10),
    )

    assert all(isinstance(trial, LocalFailedTrial) for trial in summary.trials)
    assert all(
        trial.category is LocalTrialFailureCategory.EVALUATOR_EXECUTION_FAILURE
        for trial in summary.trials
        if isinstance(trial, LocalFailedTrial)
    )
    assert tuple(trial.run_id for trial in summary.trials) == tuple(
        f"identity-mismatch-{index}" for index in range(1, 6)
    )
    assert summary.classification is ReliabilityClass.UNSOLVED


@pytest.mark.parametrize("failed_index", [0, 2, 4])
def test_five_trial_runner_retains_factory_failures_at_their_run_ids(
    local_store: tuple[EvaluationStore, TaskPrompt, list[FakeGateway]],
    identity: AgentRunIdentity,
    failed_index: int,
) -> None:
    store, prompt, gateways = local_store

    def factory(index: int) -> ConstantAgent:
        if index == failed_index:
            raise LookupError("factory failed")
        return ConstantAgent(0.0)

    summary = run_local_five_trial_reliability(
        store,
        task_id=prompt.task_id,
        run_id_prefix="factory-failure",
        identity=identity,
        agent_factory=factory,
    )
    failed = summary.trials[failed_index]

    assert isinstance(failed, LocalFailedTrial)
    assert failed.run_id == f"factory-failure-{failed_index + 1}"
    assert failed.category is LocalTrialFailureCategory.AGENT_FACTORY_FAILURE
    assert failed.exception_type == "LookupError"
    assert summary.passed_trials == 4
    assert summary.failed_trials == 1
    assert summary.execution_failure_trials == 1
    assert len(gateways) == 4


@pytest.mark.parametrize("answer", [True, float("nan"), float("inf")])
def test_local_runner_rejects_invalid_final_answer_without_episode(
    local_store: tuple[EvaluationStore, TaskPrompt, list[FakeGateway]],
    identity: AgentRunIdentity,
    answer: float,
) -> None:
    store, prompt, gateways = local_store
    with pytest.raises(ValueError, match="finite real scalar"):
        run_local_sensitivity_episode(
            store,
            task_id=prompt.task_id,
            run_id=f"invalid-{answer}",
            identity=identity,
            agent=ConstantAgent(answer),
        )
    assert gateways[-1].submissions == []


def test_agent_exception_propagates_without_scored_result(
    local_store: tuple[EvaluationStore, TaskPrompt, list[FakeGateway]],
    identity: AgentRunIdentity,
) -> None:
    store, prompt, gateways = local_store

    with pytest.raises(RuntimeError, match="agent failed"):
        run_local_sensitivity_episode(
            store,
            task_id=prompt.task_id,
            run_id="agent-failure",
            identity=identity,
            agent=FailingAgent(),
        )
    assert gateways[-1].submissions == []
