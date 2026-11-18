"""Evaluator-side local runner for model-agnostic sensitivity agents.

This module is deliberately absent from :mod:`paircraft.benchmark` exports and
from agent deployment bundles.  Passing Python callables is useful for local
integration and tests, but it is not a process-security boundary; deployed
agents must use the standalone contract over an isolated transport.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from string import hexdigits
from typing import Protocol

from paircraft.benchmark.model import (
    EpisodeResult,
    EvaluationStore,
    ReliabilityClass,
    SensitivityQueryGateway,
)
from paircraft_benchmark_contract import (
    SensitivityQueryAction,
    SensitivityQueryResponse,
    TaskPrompt,
)


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


class SensitivityAgent(Protocol):
    """Minimal vendor-neutral callback implemented by one local agent adapter."""

    def run(
        self,
        prompt: TaskPrompt,
        query: Callable[[SensitivityQueryAction], SensitivityQueryResponse],
    ) -> float:
        """Use the typed query tool and return one final scalar prediction."""
        ...


@dataclass(frozen=True, slots=True)
class AgentRunIdentity:
    """Exact evaluated-model, configuration, and local-harness identity."""

    model_id: str
    model_version: str
    model_artifact_sha256: str
    configuration_sha256: str
    harness_id: str
    harness_version: str
    harness_artifact_sha256: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.model_id, "model ID"),
            (self.model_version, "model version"),
            (self.harness_id, "harness ID"),
            (self.harness_version, "harness version"),
        ):
            _text(value, label)
        for value, label in (
            (self.model_artifact_sha256, "model artifact sha256"),
            (self.configuration_sha256, "configuration sha256"),
            (self.harness_artifact_sha256, "harness artifact sha256"),
        ):
            _digest(value, label)


class LocalTrialFailureCategory(StrEnum):
    """Failure stage for a local trial that produced no scientific answer."""

    AGENT_FACTORY_FAILURE = "agent_factory_failure"
    AGENT_EXECUTION_FAILURE = "agent_execution_failure"
    INVALID_FINAL_ANSWER = "invalid_final_answer"
    EVALUATOR_EXECUTION_FAILURE = "evaluator_execution_failure"


@dataclass(frozen=True, slots=True)
class LocalCompletedTrial:
    """Identity and pass status of one evaluator-completed local episode."""

    run_id: str
    transcript_sha256: str
    episode_sha256: str
    passed: bool

    def __post_init__(self) -> None:
        _text(self.run_id, "run ID")
        _digest(self.transcript_sha256, "transcript sha256")
        _digest(self.episode_sha256, "episode sha256")
        if not isinstance(self.passed, bool):
            raise ValueError("completed-trial pass status must be Boolean")


@dataclass(frozen=True, slots=True)
class LocalFailedTrial:
    """One attempted run that failed before an authenticated episode existed."""

    run_id: str
    category: LocalTrialFailureCategory
    exception_type: str

    def __post_init__(self) -> None:
        _text(self.run_id, "run ID")
        _text(self.exception_type, "exception type")
        if not isinstance(self.category, LocalTrialFailureCategory):
            raise ValueError("failed trial requires a typed failure category")


type LocalTrialResult = LocalCompletedTrial | LocalFailedTrial


@dataclass(frozen=True, slots=True)
class LocalReliabilitySummary:
    """Five attempted local runs, including failures without invented answers."""

    task_id: str
    case_sha256: str
    model_id: str
    model_version: str
    model_artifact_sha256: str
    configuration_sha256: str
    harness_id: str
    harness_version: str
    harness_artifact_sha256: str
    trials: tuple[LocalTrialResult, ...]
    passed_trials: int
    failed_trials: int
    execution_failure_trials: int
    success_rate: float
    classification: ReliabilityClass

    def __post_init__(self) -> None:
        for value, label in (
            (self.task_id, "task ID"),
            (self.model_id, "model ID"),
            (self.model_version, "model version"),
            (self.harness_id, "harness ID"),
            (self.harness_version, "harness version"),
        ):
            _text(value, label)
        _digest(self.case_sha256, "case sha256")
        _digest(self.model_artifact_sha256, "model artifact sha256")
        _digest(self.configuration_sha256, "configuration sha256")
        _digest(self.harness_artifact_sha256, "harness artifact sha256")
        if (
            not isinstance(self.trials, tuple)
            or len(self.trials) != 5
            or any(
                not isinstance(trial, LocalCompletedTrial | LocalFailedTrial)
                for trial in self.trials
            )
        ):
            raise ValueError("local reliability requires exactly five typed trials")
        if len({trial.run_id for trial in self.trials}) != 5:
            raise ValueError("local reliability requires five distinct run IDs")
        completed = tuple(trial for trial in self.trials if isinstance(trial, LocalCompletedTrial))
        if len({trial.transcript_sha256 for trial in completed}) != len(completed) or len(
            {trial.episode_sha256 for trial in completed}
        ) != len(completed):
            raise ValueError("completed local trial identities must be distinct")
        expected_passed = sum(trial.passed for trial in completed)
        expected_execution_failures = sum(
            isinstance(trial, LocalFailedTrial) for trial in self.trials
        )
        expected_class = (
            ReliabilityClass.RELIABLE
            if expected_passed >= 4
            else ReliabilityClass.MIXED
            if expected_passed == 3
            else ReliabilityClass.BRITTLE
            if expected_passed >= 1
            else ReliabilityClass.UNSOLVED
        )
        if (
            self.passed_trials != expected_passed
            or self.failed_trials != 5 - expected_passed
            or self.execution_failure_trials != expected_execution_failures
            or self.success_rate != expected_passed / 5
            or self.classification is not expected_class
        ):
            raise ValueError("local reliability summary contradicts its trials")


def _open_gateway(
    store: EvaluationStore,
    *,
    task_id: str,
    run_id: str,
    identity: AgentRunIdentity,
) -> SensitivityQueryGateway:
    return store.open_sensitivity_gateway(
        task_id,
        run_id=run_id,
        model_id=identity.model_id,
        model_version=identity.model_version,
        model_artifact_sha256=identity.model_artifact_sha256,
        configuration_sha256=identity.configuration_sha256,
        harness_id=identity.harness_id,
        harness_version=identity.harness_version,
        harness_artifact_sha256=identity.harness_artifact_sha256,
    )


def _final_answer(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value):
        raise ValueError("agent final answer must be a finite real scalar")
    return float(value)


def _expected_case_sha256(store: EvaluationStore, task_id: str) -> str:
    manifest = store.case_manifest(task_id)
    if not isinstance(manifest, dict):
        raise ValueError("evaluator case manifest must be an object")
    case_sha256 = manifest.get("case_sha256")
    if not isinstance(case_sha256, str):
        raise ValueError("evaluator case manifest lacks a case sha256")
    _digest(case_sha256, "case sha256")
    return case_sha256


def run_local_sensitivity_episode(
    store: EvaluationStore,
    *,
    task_id: str,
    run_id: str,
    identity: AgentRunIdentity,
    agent: SensitivityAgent,
) -> EpisodeResult:
    """Run one local callback against an evaluator-owned query gateway.

    Agent exceptions propagate and do not create an ``EpisodeResult``.  This
    same-process convenience API is not isolation: Python introspection of the
    bound query callback or imported modules can reach evaluator state. Use an
    isolated transport for untrusted agents.
    """

    if not isinstance(store, EvaluationStore):
        raise ValueError("local runner requires an EvaluationStore")
    if not isinstance(identity, AgentRunIdentity):
        raise ValueError("local runner requires a typed run identity")
    _text(task_id, "task ID")
    _text(run_id, "run ID")
    prompt = store.prompt(task_id)
    gateway = _open_gateway(
        store,
        task_id=task_id,
        run_id=run_id,
        identity=identity,
    )
    answer = _final_answer(agent.run(prompt, gateway.query))
    return gateway.submit_final_answer(answer)


def run_local_five_trial_reliability(
    store: EvaluationStore,
    *,
    task_id: str,
    run_id_prefix: str,
    identity: AgentRunIdentity,
    agent_factory: Callable[[int], SensitivityAgent],
) -> LocalReliabilitySummary:
    """Attempt five fresh local agents and classify completed and failed trials.

    A factory exception is retained at that trial's run ID. Successfully
    constructed agents must be distinct objects so state cannot be silently
    reused across the five reliability attempts.
    """

    _text(run_id_prefix, "run-ID prefix")
    if not callable(agent_factory):
        raise ValueError("reliability runner requires an agent factory")
    if not isinstance(store, EvaluationStore):
        raise ValueError("local runner requires an EvaluationStore")
    if not isinstance(identity, AgentRunIdentity):
        raise ValueError("local runner requires a typed run identity")
    _text(task_id, "task ID")
    case_sha256 = _expected_case_sha256(store, task_id)
    prepared: list[tuple[str, SensitivityAgent] | LocalFailedTrial] = []
    agent_object_ids: set[int] = set()
    for index in range(5):
        run_id = f"{run_id_prefix}-{index + 1}"
        try:
            agent = agent_factory(index)
        except Exception as error:
            prepared.append(
                LocalFailedTrial(
                    run_id,
                    LocalTrialFailureCategory.AGENT_FACTORY_FAILURE,
                    type(error).__qualname__,
                )
            )
            continue
        agent_object_id = id(agent)
        if agent_object_id in agent_object_ids:
            raise ValueError("reliability requires distinct successfully constructed agents")
        agent_object_ids.add(agent_object_id)
        prepared.append((run_id, agent))

    trials: list[LocalTrialResult] = []
    for item in prepared:
        if isinstance(item, LocalFailedTrial):
            trials.append(item)
            continue
        run_id, agent = item
        try:
            prompt = store.prompt(task_id)
            gateway = _open_gateway(
                store,
                task_id=task_id,
                run_id=run_id,
                identity=identity,
            )
        except Exception as error:
            trials.append(
                LocalFailedTrial(
                    run_id,
                    LocalTrialFailureCategory.EVALUATOR_EXECUTION_FAILURE,
                    type(error).__qualname__,
                )
            )
            continue
        try:
            raw_answer = agent.run(prompt, gateway.query)
        except Exception as error:
            trials.append(
                LocalFailedTrial(
                    run_id,
                    LocalTrialFailureCategory.AGENT_EXECUTION_FAILURE,
                    type(error).__qualname__,
                )
            )
            continue
        try:
            answer = _final_answer(raw_answer)
        except ValueError as error:
            trials.append(
                LocalFailedTrial(
                    run_id,
                    LocalTrialFailureCategory.INVALID_FINAL_ANSWER,
                    type(error).__qualname__,
                )
            )
            continue
        try:
            episode = gateway.submit_final_answer(answer)
            if (
                not isinstance(episode, EpisodeResult)
                or episode.case_sha256 != case_sha256
                or episode.run_id != run_id
                or episode.model_id != identity.model_id
                or episode.model_version != identity.model_version
                or episode.model_artifact_sha256 != identity.model_artifact_sha256
                or episode.configuration_sha256 != identity.configuration_sha256
            ):
                raise ValueError("completed episode contradicts its requested run identity")
        except Exception as error:
            trials.append(
                LocalFailedTrial(
                    run_id,
                    LocalTrialFailureCategory.EVALUATOR_EXECUTION_FAILURE,
                    type(error).__qualname__,
                )
            )
            continue
        trials.append(
            LocalCompletedTrial(
                run_id=episode.run_id,
                transcript_sha256=episode.transcript_sha256,
                episode_sha256=episode.episode_sha256,
                passed=episode.score.passed,
            )
        )
    passed = sum(trial.passed for trial in trials if isinstance(trial, LocalCompletedTrial))
    classification = (
        ReliabilityClass.RELIABLE
        if passed >= 4
        else ReliabilityClass.MIXED
        if passed == 3
        else ReliabilityClass.BRITTLE
        if passed >= 1
        else ReliabilityClass.UNSOLVED
    )
    return LocalReliabilitySummary(
        task_id=task_id,
        case_sha256=case_sha256,
        model_id=identity.model_id,
        model_version=identity.model_version,
        model_artifact_sha256=identity.model_artifact_sha256,
        configuration_sha256=identity.configuration_sha256,
        harness_id=identity.harness_id,
        harness_version=identity.harness_version,
        harness_artifact_sha256=identity.harness_artifact_sha256,
        trials=tuple(trials),
        passed_trials=passed,
        failed_trials=5 - passed,
        execution_failure_trials=sum(isinstance(trial, LocalFailedTrial) for trial in trials),
        success_rate=passed / 5,
        classification=classification,
    )
