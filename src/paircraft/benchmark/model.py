"""Process-separated benchmark prompts, harness transcripts, and evaluation.

Only prompts, typed query messages, their canonical documents, and the
transport-neutral query adapter are safe agent/harness interface objects.
``HarnessRecorder``, ``EvaluationStore``, sealed cases, generator replays,
scores, and reliability summaries belong in a trusted evaluator process.
Python visibility is not a security boundary; deployments must enforce this
process separation and never serialize evaluator state into an agent request.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from hashlib import sha256
from hmac import new as hmac_new
from math import isfinite
from pathlib import Path
from string import hexdigits
from typing import Final

import paircraft_benchmark_contract as benchmark_contract_module
from paircraft.audit import AuditablePrediction, auditable_prediction_from_simulation
from paircraft.oracle.maeda_sensitivity import execute_maeda_sensitivity_oracle_job
from paircraft.oracle.simulation import SimulationDatum
from paircraft.sweeps.maeda_sensitivity import (
    MaedaSensitivityJob,
    MaedaSensitivityPlan,
)
from paircraft_benchmark_contract import (
    FieldQueryPolicy,
    NumericalScoringPolicy,
    QueryBudget,
    SensitivityQueryAction,
    SensitivityQueryResponse,
    TaskPrompt,
    VisibleObservation,
    sensitivity_query_action_schema,
    sensitivity_query_response_schema,
    task_prompt_document,
)

_TRANSCRIPT_SCHEMA: Final = "paircraft-benchmark-transcript-v2"
_EPISODE_SCHEMA: Final = "paircraft-benchmark-episode-v3"
_RELIABILITY_SCHEMA: Final = "paircraft-benchmark-reliability-v3"
_CASE_SCHEMA: Final = "paircraft-benchmark-sealed-case-v3"
_REPLAY_SCHEMA: Final = "paircraft-benchmark-generator-replay-v1"
_AUTHENTICATED_REPLAY_TOKEN: Final = object()
_MEASUREMENT_TYPE: Final = "simulated absolute secondary-pair yield"
_INITIAL_FIELDS: Final = (0.0, 10e-6)
_QUERY_FIELDS: Final = (20e-6, 50e-6, 100e-6)
_TARGET_FIELD: Final = 200e-6
_CONSTRAINTS: Final = (
    "declared-measurement-type",
    "declared-query-field",
    "query-budget",
    "withheld-target-field",
)
_NUMERICAL_SCORING_POLICY: Final = NumericalScoringPolicy()


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


class TranscriptEventKind(StrEnum):
    """Event classes observed by the trusted harness."""

    ORACLE_QUERY = "oracle_query"
    HARD_CONSTRAINT_VIOLATION = "hard_constraint_violation"
    FINAL_ANSWER = "final_answer"


@dataclass(frozen=True, slots=True)
class TranscriptEvent:
    """One harness-observed event with exactly one kind-specific payload."""

    sequence: int
    kind: TranscriptEventKind
    interaction_sha256: str | None = None
    constraint_id: str | None = None
    final_answer: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise ValueError("transcript-event sequence must be an integer")
        if self.sequence < 0 or not isinstance(self.kind, TranscriptEventKind):
            raise ValueError("transcript-event identity is invalid")
        if self.kind is TranscriptEventKind.ORACLE_QUERY:
            if self.interaction_sha256 is None:
                raise ValueError("query event requires an interaction hash")
            _digest(self.interaction_sha256, "query interaction sha256")
            if self.constraint_id is not None or self.final_answer is not None:
                raise ValueError("query event carries an incompatible payload")
        elif self.kind is TranscriptEventKind.HARD_CONSTRAINT_VIOLATION:
            if self.constraint_id is None:
                raise ValueError("violation event requires a constraint ID")
            _text(self.constraint_id, "constraint ID")
            if self.interaction_sha256 is not None or self.final_answer is not None:
                raise ValueError("violation event carries an incompatible payload")
        else:
            if self.final_answer is None:
                raise ValueError("final-answer event requires a numeric answer")
            object.__setattr__(self, "final_answer", _finite(self.final_answer, "final answer"))
            if self.interaction_sha256 is not None or self.constraint_id is not None:
                raise ValueError("final-answer event carries an incompatible payload")


@dataclass(frozen=True, slots=True, init=False)
class EpisodeTranscript:
    """Immutable content-addressed transcript created only by ``HarnessRecorder``."""

    task_id: str
    run_id: str
    model_id: str
    model_version: str
    model_artifact_sha256: str
    configuration_sha256: str
    harness_id: str
    harness_version: str
    harness_artifact_sha256: str
    events: tuple[TranscriptEvent, ...]
    schema_version: str = _TRANSCRIPT_SCHEMA

    def __init__(
        self,
        *,
        task_id: str,
        run_id: str,
        model_id: str,
        model_version: str,
        model_artifact_sha256: str,
        configuration_sha256: str,
        harness_id: str,
        harness_version: str,
        harness_artifact_sha256: str,
        events: tuple[TranscriptEvent, ...],
        _harness_token: object,
    ) -> None:
        if _harness_token is not _AUTHENTICATED_REPLAY_TOKEN:
            raise ValueError("episode transcripts must be sealed by the trusted harness")
        for value, label in (
            (task_id, "task ID"),
            (run_id, "run ID"),
            (model_id, "model ID"),
            (model_version, "model version"),
            (harness_id, "harness ID"),
            (harness_version, "harness version"),
        ):
            _text(value, label)
        for value, label in (
            (model_artifact_sha256, "model artifact sha256"),
            (configuration_sha256, "configuration sha256"),
            (harness_artifact_sha256, "harness artifact sha256"),
        ):
            _digest(value, label)
        if not isinstance(events, tuple) or any(
            not isinstance(event, TranscriptEvent) for event in events
        ):
            raise ValueError("transcript events must be an immutable typed tuple")
        if tuple(event.sequence for event in events) != tuple(range(len(events))):
            raise ValueError("transcript events must have contiguous ordered sequences")
        final_positions = tuple(
            index
            for index, event in enumerate(events)
            if event.kind is TranscriptEventKind.FINAL_ANSWER
        )
        if final_positions != (len(events) - 1,):
            raise ValueError("transcript requires exactly one terminal final answer")
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "model_version", model_version)
        object.__setattr__(self, "model_artifact_sha256", model_artifact_sha256)
        object.__setattr__(self, "configuration_sha256", configuration_sha256)
        object.__setattr__(self, "harness_id", harness_id)
        object.__setattr__(self, "harness_version", harness_version)
        object.__setattr__(self, "harness_artifact_sha256", harness_artifact_sha256)
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "schema_version", _TRANSCRIPT_SCHEMA)

    @property
    def query_count(self) -> int:
        return sum(event.kind is TranscriptEventKind.ORACLE_QUERY for event in self.events)

    @property
    def hard_constraint_violations(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    event.constraint_id
                    for event in self.events
                    if event.kind is TranscriptEventKind.HARD_CONSTRAINT_VIOLATION
                    and event.constraint_id is not None
                }
            )
        )

    @property
    def final_answer(self) -> float:
        answer = self.events[-1].final_answer
        if answer is None:  # pragma: no cover - constructor proves this invariant
            raise AssertionError("terminal transcript event lost its final answer")
        return answer

    @property
    def transcript_sha256(self) -> str:
        return _sha(transcript_document(self))


def transcript_document(transcript: EpisodeTranscript) -> dict[str, object]:
    """Return the canonical evaluator-side transcript document."""

    if not isinstance(transcript, EpisodeTranscript):
        raise ValueError("transcript serialization requires an EpisodeTranscript")
    return {
        "schema_version": transcript.schema_version,
        "task_id": transcript.task_id,
        "run_id": transcript.run_id,
        "model_id": transcript.model_id,
        "model_version": transcript.model_version,
        "model_artifact_sha256": transcript.model_artifact_sha256,
        "configuration_sha256": transcript.configuration_sha256,
        "harness_id": transcript.harness_id,
        "harness_version": transcript.harness_version,
        "harness_artifact_sha256": transcript.harness_artifact_sha256,
        "events": [
            {
                "sequence": event.sequence,
                "kind": event.kind,
                "interaction_sha256": event.interaction_sha256,
                "constraint_id": event.constraint_id,
                "final_answer": event.final_answer,
            }
            for event in transcript.events
        ],
        "derived": {
            "query_count": transcript.query_count,
            "hard_constraint_violations": list(transcript.hard_constraint_violations),
        },
    }


class HarnessRecorder:
    """Trusted-process recorder that owns query and violation accounting."""

    def __init__(
        self,
        *,
        task_id: str,
        run_id: str,
        model_id: str,
        model_version: str,
        model_artifact_sha256: str,
        configuration_sha256: str,
        harness_id: str,
        harness_version: str,
        harness_artifact_sha256: str,
    ) -> None:
        self._identity = {
            "task_id": task_id,
            "run_id": run_id,
            "model_id": model_id,
            "model_version": model_version,
            "model_artifact_sha256": model_artifact_sha256,
            "configuration_sha256": configuration_sha256,
            "harness_id": harness_id,
            "harness_version": harness_version,
            "harness_artifact_sha256": harness_artifact_sha256,
        }
        self._events: list[TranscriptEvent] = []
        self._sealed = False

    @property
    def query_count(self) -> int:
        return sum(event.kind is TranscriptEventKind.ORACLE_QUERY for event in self._events)

    def _append(self, **payload: object) -> None:
        if self._sealed or any(
            event.kind is TranscriptEventKind.FINAL_ANSWER for event in self._events
        ):
            raise ValueError("a sealed or finalized transcript cannot accept more events")
        self._events.append(TranscriptEvent(sequence=len(self._events), **payload))  # type: ignore[arg-type]

    def _record_query(self, *, request: object, response: object) -> None:
        self._append(
            kind=TranscriptEventKind.ORACLE_QUERY,
            interaction_sha256=_sha({"request": request, "response": response}),
        )

    def _record_hard_constraint_violation(self, constraint_id: str) -> None:
        self._append(
            kind=TranscriptEventKind.HARD_CONSTRAINT_VIOLATION,
            constraint_id=constraint_id,
        )

    def _record_final_answer(self, answer: float) -> None:
        self._append(kind=TranscriptEventKind.FINAL_ANSWER, final_answer=answer)

    def seal(self) -> EpisodeTranscript:
        if self._sealed:
            raise ValueError("transcript recorder is already sealed")
        self._sealed = True
        return EpisodeTranscript(
            **self._identity,
            events=tuple(self._events),
            _harness_token=_AUTHENTICATED_REPLAY_TOKEN,
        )


class GroundTruthOrigin(StrEnum):
    """Supported and reserved evaluator-side truth origins."""

    SIMULATION_REPLAY = "simulation_replay"
    FUTURE_WET_LAB = "future_wet_lab"


def _visible_observation(
    job: MaedaSensitivityJob,
    datum: SimulationDatum,
    receipt_sha256: str,
) -> VisibleObservation:
    return VisibleObservation(
        field_tesla=job.field_point.magnitude_tesla,
        measurement_type=_MEASUREMENT_TYPE,
        value=datum.prediction.distribution.mean,
        unit="1",
        observation_receipt_sha256=receipt_sha256,
    )


@dataclass(frozen=True, slots=True)
class _GeneratorReplay:
    """Authenticated replay record constructed by a concrete generator adapter."""

    generator_id: str
    generator_version: str
    plan_sha256: str
    specification_sha256: str
    source_job_sha256: str
    parameter_family_sha256: str
    design_dependency_sha256: str
    oracle_artifact_sha256: str
    target_execution_sha256: str
    target_result_sha256: str
    initial_result_sha256: tuple[str, ...]
    origin: GroundTruthOrigin
    schema_version: str = _REPLAY_SCHEMA

    def __post_init__(self) -> None:
        for value, label in (
            (self.generator_id, "generator ID"),
            (self.generator_version, "generator version"),
        ):
            _text(value, label)
        for value, label in (
            (self.plan_sha256, "plan sha256"),
            (self.specification_sha256, "specification sha256"),
            (self.source_job_sha256, "source job sha256"),
            (self.parameter_family_sha256, "parameter family sha256"),
            (self.design_dependency_sha256, "design dependency sha256"),
            (self.oracle_artifact_sha256, "oracle artifact sha256"),
            (self.target_execution_sha256, "target execution sha256"),
            (self.target_result_sha256, "target result sha256"),
        ):
            _digest(value, label)
        if len(self.initial_result_sha256) != 2:
            raise ValueError("generator replay requires exactly two initial-result hashes")
        for digest in self.initial_result_sha256:
            _digest(digest, "initial result sha256")
        if self.origin is not GroundTruthOrigin.SIMULATION_REPLAY:
            raise ValueError(
                "wet-lab truth is a future boundary requiring an authenticated evidence adapter"
            )
        if self.schema_version != _REPLAY_SCHEMA:
            raise ValueError("generator-replay schema is unsupported")

    @property
    def replay_sha256(self) -> str:
        return _sha(asdict(self))


@dataclass(frozen=True, slots=True, init=False)
class _SealedEvaluationCase:
    """Evaluator-only answer, nonce, and authenticated generator replay."""

    prompt: TaskPrompt
    replay: _GeneratorReplay
    _answer: float = field(repr=False)
    _nonce: str = field(repr=False)
    schema_version: str = _CASE_SCHEMA

    def __init__(
        self,
        *,
        prompt: TaskPrompt,
        answer: float,
        replay: _GeneratorReplay,
        nonce: str,
        _adapter_token: object,
    ) -> None:
        if _adapter_token is not _AUTHENTICATED_REPLAY_TOKEN:
            raise ValueError("sealed cases require an authenticated generator adapter")
        if not isinstance(prompt, TaskPrompt) or not isinstance(replay, _GeneratorReplay):
            raise ValueError("sealed case inputs are invalid")
        resolved_answer = _finite(answer, "sealed answer")
        _digest(nonce, "case nonce")
        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(self, "replay", replay)
        object.__setattr__(self, "_answer", resolved_answer)
        object.__setattr__(self, "_nonce", nonce)
        object.__setattr__(self, "schema_version", _CASE_SCHEMA)

    @property
    def case_sha256(self) -> str:
        return _sha(
            {
                "schema_version": self.schema_version,
                "prompt_sha256": self.prompt.prompt_sha256,
                "answer_commitment": _sha(
                    {
                        "answer": self._answer,
                        "unit": self.prompt.answer_unit,
                        "scoring_policy": asdict(self.prompt.scoring_policy),
                        "nonce": self._nonce,
                    }
                ),
                "replay_sha256": self.replay.replay_sha256,
            }
        )

    @property
    def answer_commitment_sha256(self) -> str:
        return _sha(
            {
                "answer": self._answer,
                "unit": self.prompt.answer_unit,
                "scoring_policy": asdict(self.prompt.scoring_policy),
                "nonce": self._nonce,
            }
        )


@dataclass(frozen=True, slots=True)
class OutcomeScore:
    """Objective answer result and transcript-derived hard constraints."""

    predicted_answer: float
    absolute_error: float
    allowed_error: float
    normalized_error: float
    query_count: int
    maximum_queries: int
    hard_constraint_violations: tuple[str, ...]
    answer_correct: bool
    budget_compliant: bool
    interaction_policy_compliant: bool
    passed: bool
    reward: float

    def __post_init__(self) -> None:
        predicted = _finite(self.predicted_answer, "predicted answer")
        error = _finite(self.absolute_error, "absolute error")
        allowed = _finite(self.allowed_error, "allowed error")
        normalized = _finite(self.normalized_error, "normalized error")
        if error < 0 or allowed <= 0 or normalized < 0:
            raise ValueError("score errors must be nonnegative and scale must be positive")
        if normalized != error / allowed:
            raise ValueError("normalized error contradicts the absolute error and scale")
        for value, label in (
            (self.query_count, "query count"),
            (self.maximum_queries, "maximum queries"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be a nonnegative integer")
        if (
            not isinstance(self.hard_constraint_violations, tuple)
            or tuple(sorted(set(self.hard_constraint_violations)))
            != self.hard_constraint_violations
        ):
            raise ValueError("hard violations must be a unique sorted tuple")
        expected_correct = normalized <= _NUMERICAL_SCORING_POLICY.pass_normalized_error
        expected_budget = self.query_count <= self.maximum_queries
        expected_interaction = not self.hard_constraint_violations
        expected_pass = expected_correct and expected_budget and expected_interaction
        expected_reward = (
            1.0 / (1.0 + normalized) if expected_budget and expected_interaction else 0.0
        )
        if (
            self.answer_correct is not expected_correct
            or self.budget_compliant is not expected_budget
            or self.interaction_policy_compliant is not expected_interaction
            or self.passed is not expected_pass
            or self.reward != expected_reward
        ):
            raise ValueError("outcome score contradicts its objective components")
        object.__setattr__(self, "predicted_answer", predicted)
        object.__setattr__(self, "absolute_error", error)
        object.__setattr__(self, "allowed_error", allowed)
        object.__setattr__(self, "normalized_error", normalized)


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Evaluator result bound to transcript, case, run, model, and configuration."""

    case_sha256: str
    transcript_sha256: str
    run_id: str
    model_id: str
    model_version: str
    model_artifact_sha256: str
    configuration_sha256: str
    score: OutcomeScore
    schema_version: str = _EPISODE_SCHEMA

    def __post_init__(self) -> None:
        for value, label in (
            (self.case_sha256, "case sha256"),
            (self.transcript_sha256, "transcript sha256"),
            (self.model_artifact_sha256, "model artifact sha256"),
            (self.configuration_sha256, "configuration sha256"),
        ):
            _digest(value, label)
        for value, label in (
            (self.run_id, "run ID"),
            (self.model_id, "model ID"),
            (self.model_version, "model version"),
        ):
            _text(value, label)
        if not isinstance(self.score, OutcomeScore):
            raise ValueError("episode result requires an OutcomeScore")
        if self.schema_version != _EPISODE_SCHEMA:
            raise ValueError("episode schema is unsupported")

    @property
    def episode_sha256(self) -> str:
        return _sha(episode_document(self))


def episode_document(episode: EpisodeResult) -> dict[str, object]:
    """Return the canonical evaluator-side episode document."""

    if not isinstance(episode, EpisodeResult):
        raise ValueError("episode serialization requires an EpisodeResult")
    return {
        "schema_version": episode.schema_version,
        "case_sha256": episode.case_sha256,
        "transcript_sha256": episode.transcript_sha256,
        "run_id": episode.run_id,
        "model_id": episode.model_id,
        "model_version": episode.model_version,
        "model_artifact_sha256": episode.model_artifact_sha256,
        "configuration_sha256": episode.configuration_sha256,
        "score": {
            "predicted_answer": episode.score.predicted_answer,
            "absolute_error": episode.score.absolute_error,
            "allowed_error": episode.score.allowed_error,
            "normalized_error": episode.score.normalized_error,
            "query_count": episode.score.query_count,
            "maximum_queries": episode.score.maximum_queries,
            "hard_constraint_violations": list(episode.score.hard_constraint_violations),
            "answer_correct": episode.score.answer_correct,
            "budget_compliant": episode.score.budget_compliant,
            "interaction_policy_compliant": episode.score.interaction_policy_compliant,
            "passed": episode.score.passed,
            "reward": episode.score.reward,
        },
    }


class ReliabilityClass(StrEnum):
    """Exact five-trial outcome classification."""

    RELIABLE = "reliable"
    MIXED = "mixed"
    BRITTLE = "brittle"
    UNSOLVED = "unsolved"


@dataclass(frozen=True, slots=True)
class ReliabilityTrialIdentity:
    """Exact run-to-transcript-to-episode mapping for one reliability trial."""

    run_id: str
    transcript_sha256: str
    episode_sha256: str
    passed: bool

    def __post_init__(self) -> None:
        _text(self.run_id, "run ID")
        _digest(self.transcript_sha256, "transcript sha256")
        _digest(self.episode_sha256, "episode sha256")
        if not isinstance(self.passed, bool):
            raise ValueError("trial pass status must be Boolean")


@dataclass(frozen=True, slots=True)
class ReliabilitySummary:
    """Five distinct runs under one exact model and configuration identity."""

    case_sha256: str
    model_id: str
    model_version: str
    model_artifact_sha256: str
    configuration_sha256: str
    trials: tuple[ReliabilityTrialIdentity, ...]
    passed_trials: int
    failed_trials: int
    success_rate: float
    classification: ReliabilityClass
    schema_version: str = _RELIABILITY_SCHEMA

    def __post_init__(self) -> None:
        for value, label in (
            (self.case_sha256, "case sha256"),
            (self.model_artifact_sha256, "model artifact sha256"),
            (self.configuration_sha256, "configuration sha256"),
        ):
            _digest(value, label)
        for value, label in (
            (self.model_id, "model ID"),
            (self.model_version, "model version"),
        ):
            _text(value, label)
        if not isinstance(self.trials, tuple) or any(
            not isinstance(trial, ReliabilityTrialIdentity) for trial in self.trials
        ):
            raise ValueError("reliability trials must be an immutable typed tuple")
        if len(self.trials) != 5 or self.trials != tuple(
            sorted(self.trials, key=lambda trial: trial.run_id)
        ):
            raise ValueError("reliability requires five canonically ordered trials")
        if (
            len({trial.run_id for trial in self.trials}) != 5
            or len({trial.transcript_sha256 for trial in self.trials}) != 5
            or len({trial.episode_sha256 for trial in self.trials}) != 5
        ):
            raise ValueError("reliability trial identities must be distinct")
        if not 0 <= self.passed_trials <= 5 or self.failed_trials != 5 - self.passed_trials:
            raise ValueError("reliability trial counts are inconsistent")
        if self.success_rate != self.passed_trials / 5:
            raise ValueError("reliability success rate contradicts its counts")
        expected_class = (
            ReliabilityClass.RELIABLE
            if self.passed_trials >= 4
            else ReliabilityClass.MIXED
            if self.passed_trials == 3
            else ReliabilityClass.BRITTLE
            if self.passed_trials >= 1
            else ReliabilityClass.UNSOLVED
        )
        if self.classification is not expected_class:
            raise ValueError("reliability classification contradicts its counts")
        if self.schema_version != _RELIABILITY_SCHEMA:
            raise ValueError("reliability schema is unsupported")

    @property
    def reliability_sha256(self) -> str:
        return _sha(reliability_document(self))


def reliability_document(summary: ReliabilitySummary) -> dict[str, object]:
    """Return exact counts and all five distinct run identities."""

    if not isinstance(summary, ReliabilitySummary):
        raise ValueError("reliability serialization requires a ReliabilitySummary")
    return {
        "schema_version": summary.schema_version,
        "case_sha256": summary.case_sha256,
        "model_id": summary.model_id,
        "model_version": summary.model_version,
        "model_artifact_sha256": summary.model_artifact_sha256,
        "configuration_sha256": summary.configuration_sha256,
        "trials": [asdict(trial) for trial in summary.trials],
        "passed_trials": summary.passed_trials,
        "failed_trials": summary.failed_trials,
        "trial_count": 5,
        "success_rate": summary.success_rate,
        "classification": summary.classification,
    }


@dataclass(frozen=True, slots=True)
class _SensitivityRuntime:
    """Evaluator-only same-family jobs available through one gateway."""

    plan: MaedaSensitivityPlan
    target_job: MaedaSensitivityJob
    query_jobs: tuple[MaedaSensitivityJob, ...]

    def job_at(self, field_tesla: float) -> MaedaSensitivityJob:
        matches = tuple(
            job for job in self.query_jobs if job.field_point.magnitude_tesla == field_tesla
        )
        if len(matches) != 1:
            raise ValueError("gateway field does not resolve to one exact planned job")
        return matches[0]


class EvaluationStore:
    """Trusted evaluator store; never instantiate or serialize in the agent process."""

    def __init__(self, *, evaluator_nonce_sha256: str) -> None:
        _digest(evaluator_nonce_sha256, "evaluator nonce sha256")
        self._nonce = evaluator_nonce_sha256
        self._cases: dict[str, _SealedEvaluationCase] = {}
        self._sensitivity_runtimes: dict[str, _SensitivityRuntime] = {}

    def _opaque_digest(self, purpose: str, payload: object) -> str:
        _text(purpose, "opaque digest purpose")
        message = _canonical_bytes({"purpose": purpose, "payload": payload})
        return hmac_new(bytes.fromhex(self._nonce), message, sha256).hexdigest()

    def _opaque_task_id(self, job: MaedaSensitivityJob) -> str:
        return f"case-{self._opaque_digest('task-id', job.job_sha256)}"

    def _observation_receipt(
        self,
        job: MaedaSensitivityJob,
        datum: SimulationDatum,
    ) -> str:
        return self._opaque_digest(
            "initial-observation",
            {
                "job_sha256": job.job_sha256,
                "execution_sha256": datum.execution.execution_sha256,
                "datum_sha256": datum.datum_sha256,
            },
        )

    def _query_receipt(self, job: MaedaSensitivityJob, datum: SimulationDatum) -> str:
        return self._opaque_digest(
            "query-response",
            {
                "job_sha256": job.job_sha256,
                "execution_sha256": datum.execution.execution_sha256,
                "datum_sha256": datum.datum_sha256,
            },
        )

    def _register_replay(
        self,
        *,
        prompt: TaskPrompt,
        answer: float,
        replay: _GeneratorReplay,
        _adapter_token: object,
    ) -> TaskPrompt:
        if _adapter_token is not _AUTHENTICATED_REPLAY_TOKEN:
            raise ValueError("case registration requires an authenticated generator adapter")
        if prompt.task_id in self._cases:
            raise ValueError("evaluation store already contains this task ID")
        nonce = _sha(
            {
                "evaluator_nonce_sha256": self._nonce,
                "task_id": prompt.task_id,
                "replay_sha256": replay.replay_sha256,
            }
        )
        self._cases[prompt.task_id] = _SealedEvaluationCase(
            prompt=prompt,
            answer=answer,
            replay=replay,
            nonce=nonce,
            _adapter_token=_AUTHENTICATED_REPLAY_TOKEN,
        )
        return prompt

    def prompt(self, task_id: str) -> TaskPrompt:
        try:
            return self._cases[task_id].prompt
        except KeyError as error:
            raise ValueError("evaluation store has no such task ID") from error

    def case_manifest(self, task_id: str) -> dict[str, object]:
        """Return evaluator-side provenance without the answer or nonce."""

        try:
            case = self._cases[task_id]
        except KeyError as error:
            raise ValueError("evaluation store has no such task ID") from error
        payload: dict[str, object] = {
            "schema_version": "paircraft-benchmark-case-manifest-v2",
            "prompt": task_prompt_document(case.prompt),
            "case_sha256": case.case_sha256,
            "answer_commitment_sha256": case.answer_commitment_sha256,
            "generator_replay": asdict(case.replay),
        }
        return {**payload, "manifest_sha256": _sha(payload)}

    def wet_lab_audit_target(self, task_id: str) -> AuditablePrediction:
        """Replay one exact target for trusted-side future wet-lab auditing.

        The returned target is never part of the agent deployment bundle and
        deliberately carries no calibrated interval for simulator-only cases.
        """

        try:
            case = self._cases[task_id]
            runtime = self._sensitivity_runtimes[task_id]
        except KeyError as error:
            raise ValueError("evaluation store has no auditable sensitivity task") from error
        datum = execute_maeda_sensitivity_oracle_job(runtime.plan, runtime.target_job)
        if (
            datum.prediction.oracle.artifact_sha256 != case.replay.oracle_artifact_sha256
            or datum.execution.execution_sha256 != case.replay.target_execution_sha256
            or datum.datum_sha256 != case.replay.target_result_sha256
            or datum.prediction.distribution.mean != case._answer
        ):
            raise ValueError("auditable target replay contradicts the sealed evaluation case")
        return auditable_prediction_from_simulation(
            case_sha256=case.case_sha256,
            datum=datum,
        )

    def open_sensitivity_gateway(
        self,
        task_id: str,
        *,
        run_id: str,
        model_id: str,
        model_version: str,
        model_artifact_sha256: str,
        configuration_sha256: str,
        harness_id: str,
        harness_version: str,
        harness_artifact_sha256: str,
    ) -> SensitivityQueryGateway:
        """Create one evaluator-owned gateway and private transcript recorder."""

        if task_id not in self._cases or task_id not in self._sensitivity_runtimes:
            raise ValueError("evaluation store has no sensitivity task")
        return SensitivityQueryGateway(
            store=self,
            task_id=task_id,
            runtime=self._sensitivity_runtimes[task_id],
            run_id=run_id,
            model_id=model_id,
            model_version=model_version,
            model_artifact_sha256=model_artifact_sha256,
            configuration_sha256=configuration_sha256,
            harness_id=harness_id,
            harness_version=harness_version,
            harness_artifact_sha256=harness_artifact_sha256,
            _gateway_token=_AUTHENTICATED_REPLAY_TOKEN,
        )

    def score(self, transcript: EpisodeTranscript) -> EpisodeResult:
        """Score using transcript-derived counts and violations only."""

        if not isinstance(transcript, EpisodeTranscript):
            raise ValueError("evaluation requires a harness-sealed transcript")
        try:
            case = self._cases[transcript.task_id]
        except KeyError as missing_error:
            raise ValueError("transcript task is not registered") from missing_error
        violations = transcript.hard_constraint_violations
        if any(item not in case.prompt.interaction_constraints for item in violations):
            raise ValueError("transcript contains an unknown hard-constraint ID")
        absolute_error = abs(transcript.final_answer - case._answer)
        allowed_error = case.prompt.scoring_policy.allowed_error(case._answer)
        normalized_error = absolute_error / allowed_error
        correct = normalized_error <= case.prompt.scoring_policy.pass_normalized_error
        budget = transcript.query_count <= case.prompt.query_budget.maximum_queries
        interaction_compliant = not violations
        passed = correct and budget and interaction_compliant
        reward = 1.0 / (1.0 + normalized_error) if budget and interaction_compliant else 0.0
        score = OutcomeScore(
            predicted_answer=transcript.final_answer,
            absolute_error=absolute_error,
            allowed_error=allowed_error,
            normalized_error=normalized_error,
            query_count=transcript.query_count,
            maximum_queries=case.prompt.query_budget.maximum_queries,
            hard_constraint_violations=violations,
            answer_correct=correct,
            budget_compliant=budget,
            interaction_policy_compliant=interaction_compliant,
            passed=passed,
            reward=reward,
        )
        return EpisodeResult(
            case_sha256=case.case_sha256,
            transcript_sha256=transcript.transcript_sha256,
            run_id=transcript.run_id,
            model_id=transcript.model_id,
            model_version=transcript.model_version,
            model_artifact_sha256=transcript.model_artifact_sha256,
            configuration_sha256=transcript.configuration_sha256,
            score=score,
        )


@dataclass(frozen=True, slots=True)
class AgentDeploymentManifest:
    """Content-addressed inventory of one oracle-free agent deployment bundle."""

    prompt_sha256: str
    files: tuple[tuple[str, str], ...]
    schema_version: str = "paircraft-agent-deployment-bundle-v1"

    def __post_init__(self) -> None:
        _digest(self.prompt_sha256, "deployment prompt sha256")
        if self.files != tuple(sorted(self.files)) or not self.files:
            raise ValueError("deployment files must be a nonempty canonical tuple")
        for name, digest in self.files:
            _text(name, "deployment filename")
            _digest(digest, "deployment file sha256")

    @property
    def bundle_sha256(self) -> str:
        return _sha(agent_deployment_manifest_document(self, include_bundle_hash=False))


def agent_deployment_manifest_document(
    manifest: AgentDeploymentManifest,
    *,
    include_bundle_hash: bool = True,
) -> dict[str, object]:
    """Return the canonical non-secret deployment inventory."""

    if not isinstance(manifest, AgentDeploymentManifest):
        raise ValueError("deployment manifest requires an AgentDeploymentManifest")
    document: dict[str, object] = {
        "schema_version": manifest.schema_version,
        "prompt_sha256": manifest.prompt_sha256,
        "files": dict(manifest.files),
        "scientific_boundary": (
            "Public observations may support scientific inference. The bundle contains no "
            "Paircraft evaluator, answer, oracle, spin solver, sweep plan, or direct simulator."
        ),
    }
    if include_bundle_hash:
        document["bundle_sha256"] = manifest.bundle_sha256
    return document


def build_agent_deployment_bundle(
    destination: Path,
    prompt: TaskPrompt,
) -> AgentDeploymentManifest:
    """Write the standalone contract, prompt, and RPC schemas for an agent process."""

    if not isinstance(destination, Path) or not isinstance(prompt, TaskPrompt):
        raise ValueError("deployment builder requires a Path and TaskPrompt")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("deployment destination must be absent or empty")
    destination.mkdir(parents=True, exist_ok=True)
    contract_path_value = benchmark_contract_module.__file__
    if contract_path_value is None:
        raise ValueError("standalone benchmark contract has no filesystem artifact")
    payloads = {
        "paircraft_benchmark_contract.py": Path(contract_path_value).read_bytes(),
        "prompt.json": _canonical_bytes(task_prompt_document(prompt)),
        "query-action.schema.json": _canonical_bytes(sensitivity_query_action_schema()),
        "query-response.schema.json": _canonical_bytes(sensitivity_query_response_schema()),
    }
    files = tuple(sorted((name, sha256(payload).hexdigest()) for name, payload in payloads.items()))
    manifest = AgentDeploymentManifest(prompt_sha256=prompt.prompt_sha256, files=files)
    for name, payload in payloads.items():
        (destination / name).write_bytes(payload)
    (destination / "manifest.json").write_bytes(
        _canonical_bytes(agent_deployment_manifest_document(manifest))
    )
    return manifest


class SensitivityQueryRejected(ValueError):
    """A hard policy rejection already recorded in the private transcript."""

    def __init__(self, constraint_id: str, message: str) -> None:
        self.constraint_id = constraint_id
        super().__init__(message)


class SensitivityQueryGateway:
    """Evaluator-owned typed query gateway for one registered sensitivity task."""

    def __init__(
        self,
        *,
        store: EvaluationStore,
        task_id: str,
        runtime: _SensitivityRuntime,
        run_id: str,
        model_id: str,
        model_version: str,
        model_artifact_sha256: str,
        configuration_sha256: str,
        harness_id: str,
        harness_version: str,
        harness_artifact_sha256: str,
        _gateway_token: object,
    ) -> None:
        if _gateway_token is not _AUTHENTICATED_REPLAY_TOKEN:
            raise ValueError("sensitivity gateway requires the evaluator store")
        self._store = store
        self._task_id = task_id
        self._runtime = runtime
        self._recorder = HarnessRecorder(
            task_id=task_id,
            run_id=run_id,
            model_id=model_id,
            model_version=model_version,
            model_artifact_sha256=model_artifact_sha256,
            configuration_sha256=configuration_sha256,
            harness_id=harness_id,
            harness_version=harness_version,
            harness_artifact_sha256=harness_artifact_sha256,
        )
        self._transcript: EpisodeTranscript | None = None

    def _reject(self, constraint_id: str, message: str) -> None:
        self._recorder._record_hard_constraint_violation(constraint_id)
        raise SensitivityQueryRejected(constraint_id, message)

    def query(self, action: SensitivityQueryAction) -> SensitivityQueryResponse:
        """Execute one allowed exact field job or record a hard rejection."""

        prompt = self._store.prompt(self._task_id)
        if not isinstance(action, SensitivityQueryAction) or (
            action.measurement_type != prompt.action_policy.measurement_type
        ):
            self._reject(
                "declared-measurement-type",
                "query measurement type does not match the declared task",
            )
        if action.field_tesla == prompt.action_policy.withheld_target_field_tesla:
            self._reject("withheld-target-field", "withheld target field cannot be queried")
        if action.field_tesla not in prompt.action_policy.allowed_fields_tesla:
            self._reject("declared-query-field", "query field is not declared as available")
        if self._recorder.query_count >= prompt.query_budget.maximum_queries:
            self._reject("query-budget", "query budget is exhausted")
        job = self._runtime.job_at(action.field_tesla)
        datum = execute_maeda_sensitivity_oracle_job(self._runtime.plan, job)
        response = SensitivityQueryResponse(
            field_tesla=job.field_point.magnitude_tesla,
            measurement_type=_MEASUREMENT_TYPE,
            value=datum.prediction.distribution.mean,
            unit="1",
            query_receipt_sha256=self._store._query_receipt(job, datum),
        )
        self._recorder._record_query(request=asdict(action), response=asdict(response))
        return response

    def submit_final_answer(self, answer: float) -> EpisodeResult:
        """Seal the private transcript and ask the evaluator store to score it."""

        self._recorder._record_final_answer(answer)
        self._transcript = self._recorder.seal()
        return self._store.score(self._transcript)

    def transcript_document(self) -> dict[str, object]:
        """Return the sealed transcript to the evaluator, not the agent process."""

        if self._transcript is None:
            raise ValueError("gateway transcript is not sealed")
        return transcript_document(self._transcript)


def register_maeda_sensitivity_case(
    store: EvaluationStore,
    *,
    plan: MaedaSensitivityPlan,
    target_job: MaedaSensitivityJob,
) -> TaskPrompt:
    """Register one perturbed-family 200 µT task by exact oracle replay."""

    if not isinstance(store, EvaluationStore):
        raise ValueError("sensitivity benchmark registration requires an EvaluationStore")
    if not isinstance(plan, MaedaSensitivityPlan) or not isinstance(
        target_job, MaedaSensitivityJob
    ):
        raise ValueError("sensitivity benchmark registration requires a typed plan and job")
    if target_job not in plan.jobs:
        raise ValueError("benchmark target job is not an exact plan member")
    if target_job.family.family_id == "baseline":
        raise ValueError("benchmark task requires a non-baseline sensitivity family")
    if target_job.field_point.magnitude_tesla != _TARGET_FIELD:
        raise ValueError("benchmark target job must be the withheld 200 µT field")
    family_jobs = tuple(
        job
        for job in plan.jobs
        if job.parameter_family_sha256 == target_job.parameter_family_sha256
    )
    jobs_by_field = {job.field_point.magnitude_tesla: job for job in family_jobs}
    required_fields = {*_INITIAL_FIELDS, *_QUERY_FIELDS, _TARGET_FIELD}
    if set(jobs_by_field) != required_fields:
        raise ValueError("benchmark family does not contain the exact declared field grid")
    initial_data = tuple(
        execute_maeda_sensitivity_oracle_job(plan, jobs_by_field[field])
        for field in _INITIAL_FIELDS
    )
    target_datum = execute_maeda_sensitivity_oracle_job(plan, target_job)
    initial_observations = tuple(
        _visible_observation(
            jobs_by_field[field],
            datum,
            store._observation_receipt(jobs_by_field[field], datum),
        )
        for field, datum in zip(_INITIAL_FIELDS, initial_data, strict=True)
    )
    prompt = TaskPrompt(
        task_id=store._opaque_task_id(target_job),
        system_description=(
            "Hypothetical simulator-only radical-pair sensitivity system with one fixed "
            "hidden parameter family. Public observations may support scientific inference; "
            "direct simulator execution is unavailable to the agent."
        ),
        objective=(
            "Choose up to two declared field measurements, then predict the withheld absolute "
            "secondary-pair yield at 0.0002 T."
        ),
        measurement_type=_MEASUREMENT_TYPE,
        answer_unit="1",
        query_budget=QueryBudget(2),
        interaction_constraints=_CONSTRAINTS,
        initial_observations=initial_observations,
        action_policy=FieldQueryPolicy(
            measurement_type=_MEASUREMENT_TYPE,
            allowed_fields_tesla=_QUERY_FIELDS,
            withheld_target_field_tesla=_TARGET_FIELD,
        ),
        scoring_policy=_NUMERICAL_SCORING_POLICY,
    )
    replay = _GeneratorReplay(
        generator_id=target_datum.prediction.oracle.oracle_id,
        generator_version=target_datum.prediction.oracle.oracle_version,
        plan_sha256=plan.plan_sha256,
        specification_sha256=plan.specification.specification_sha256,
        source_job_sha256=target_job.job_sha256,
        parameter_family_sha256=target_job.parameter_family_sha256,
        design_dependency_sha256=target_job.design_dependency_sha256,
        oracle_artifact_sha256=target_datum.prediction.oracle.artifact_sha256,
        target_execution_sha256=target_datum.execution.execution_sha256,
        target_result_sha256=target_datum.datum_sha256,
        initial_result_sha256=tuple(datum.datum_sha256 for datum in initial_data),
        origin=GroundTruthOrigin.SIMULATION_REPLAY,
    )
    registered = store._register_replay(
        prompt=prompt,
        answer=target_datum.prediction.distribution.mean,
        replay=replay,
        _adapter_token=_AUTHENTICATED_REPLAY_TOKEN,
    )
    store._sensitivity_runtimes[prompt.task_id] = _SensitivityRuntime(
        plan=plan,
        target_job=target_job,
        query_jobs=tuple(jobs_by_field[field] for field in _QUERY_FIELDS),
    )
    return registered


def summarize_reliability(episodes: tuple[EpisodeResult, ...]) -> ReliabilitySummary:
    """Classify exactly five distinct runs of one case/model/configuration."""

    if (
        not isinstance(episodes, tuple)
        or len(episodes) != 5
        or any(not isinstance(episode, EpisodeResult) for episode in episodes)
    ):
        raise ValueError("reliability requires exactly five EpisodeResult values")
    case_ids = {episode.case_sha256 for episode in episodes}
    identities = {
        (
            episode.model_id,
            episode.model_version,
            episode.model_artifact_sha256,
            episode.configuration_sha256,
        )
        for episode in episodes
    }
    ordered = tuple(sorted(episodes, key=lambda episode: episode.run_id))
    if len(case_ids) != 1 or len(identities) != 1:
        raise ValueError("reliability trials must share one case, model, and configuration")
    if (
        len({episode.run_id for episode in ordered}) != 5
        or len({episode.transcript_sha256 for episode in ordered}) != 5
        or len({episode.episode_sha256 for episode in ordered}) != 5
    ):
        raise ValueError("reliability requires five distinct run, transcript, and episode IDs")
    passed = sum(episode.score.passed for episode in episodes)
    classification = (
        ReliabilityClass.RELIABLE
        if passed >= 4
        else ReliabilityClass.MIXED
        if passed == 3
        else ReliabilityClass.BRITTLE
        if passed >= 1
        else ReliabilityClass.UNSOLVED
    )
    identity = next(iter(identities))
    return ReliabilitySummary(
        case_sha256=next(iter(case_ids)),
        model_id=identity[0],
        model_version=identity[1],
        model_artifact_sha256=identity[2],
        configuration_sha256=identity[3],
        trials=tuple(
            ReliabilityTrialIdentity(
                run_id=episode.run_id,
                transcript_sha256=episode.transcript_sha256,
                episode_sha256=episode.episode_sha256,
                passed=episode.score.passed,
            )
            for episode in ordered
        ),
        passed_trials=passed,
        failed_trials=5 - passed,
        success_rate=passed / 5,
        classification=classification,
    )
