from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

import paircraft.benchmark as agent_api
import paircraft.benchmark.model as benchmark_model
from paircraft.audit import PhysicalEvidenceLayer
from paircraft.benchmark import (
    NumericalScoringPolicy,
    SensitivityQueryAction,
    SensitivityQueryToolAdapter,
    TaskPrompt,
    sensitivity_query_action_document,
    task_prompt_document,
    task_prompt_from_document,
)
from paircraft.benchmark.model import (
    AgentDeploymentManifest,
    EpisodeResult,
    EpisodeTranscript,
    EvaluationStore,
    HarnessRecorder,
    ReliabilityClass,
    ReliabilitySummary,
    ReliabilityTrialIdentity,
    SensitivityQueryGateway,
    SensitivityQueryRejected,
    TranscriptEvent,
    TranscriptEventKind,
    agent_deployment_manifest_document,
    build_agent_deployment_bundle,
    episode_document,
    register_maeda_sensitivity_case,
    reliability_document,
    summarize_reliability,
    transcript_document,
)
from paircraft.oracle.maeda_sensitivity import execute_maeda_sensitivity_oracle_job
from paircraft.oracle.model import CalibrationTier
from paircraft.sweeps.maeda_sensitivity import (
    MaedaSensitivityJob,
    MaedaSensitivityPlan,
    default_maeda_local_sensitivity,
    materialize_maeda_local_sensitivity,
)

RegisteredCase = tuple[
    EvaluationStore,
    TaskPrompt,
    MaedaSensitivityPlan,
    MaedaSensitivityJob,
    float,
]


@pytest.fixture(scope="module")
def registered_case() -> RegisteredCase:
    plan = materialize_maeda_local_sensitivity(default_maeda_local_sensitivity())
    target = next(
        job
        for job in plan.jobs
        if job.family.family_id != "baseline" and job.field_point.magnitude_tesla == 200e-6
    )
    store = EvaluationStore(evaluator_nonce_sha256="e" * 64)
    prompt = register_maeda_sensitivity_case(
        store,
        plan=plan,
        target_job=target,
    )
    exact = execute_maeda_sensitivity_oracle_job(plan, target).prediction.distribution.mean
    return store, prompt, plan, target, exact


def _gateway(
    store: EvaluationStore,
    prompt: TaskPrompt,
    run_id: str,
) -> SensitivityQueryGateway:
    return store.open_sensitivity_gateway(
        prompt.task_id,
        run_id=run_id,
        model_id="evaluated-model",
        model_version="1.0.0",
        model_artifact_sha256="a" * 64,
        configuration_sha256="b" * 64,
        harness_id="paircraft-harness",
        harness_version="1.0.0",
        harness_artifact_sha256="c" * 64,
    )


def _episode(
    store: EvaluationStore,
    prompt: TaskPrompt,
    answer: float,
    run_index: int,
) -> EpisodeResult:
    return _gateway(store, prompt, f"run-{run_index}").submit_final_answer(answer)


def test_prompt_is_a_perturbed_withheld_field_task_with_structured_policy(
    registered_case: RegisteredCase,
) -> None:
    _, prompt, _, target, exact = registered_case
    document = task_prompt_document(prompt)
    encoded = json.dumps(document, sort_keys=True)

    assert set(agent_api.__all__) == {
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
    }
    assert target.family.family_id != "baseline"
    assert target.field_point.magnitude_tesla == 200e-6
    assert tuple(item.field_tesla for item in prompt.initial_observations) == (0.0, 10e-6)
    assert prompt.action_policy.allowed_fields_tesla == (20e-6, 50e-6, 100e-6)
    assert prompt.action_policy.withheld_target_field_tesla == 200e-6
    assert prompt.query_budget.maximum_queries == 2
    assert document["schema_version"] == "paircraft-benchmark-task-prompt-v5"
    assert "interaction_constraints" in document
    assert "protocol_constraints" not in document
    assert prompt.scoring_policy.absolute_error_tolerance == 1e-4
    assert prompt.scoring_policy.relative_error_tolerance == 1e-3
    assert "not experimental" in prompt.scoring_policy.interpretation
    assert "choose up to two" in prompt.objective.lower()
    assert "objective_answer" not in document
    assert "nonce" not in encoded
    assert str(exact) not in encoded
    assert target.job_sha256 not in encoded
    assert target.family.family_id not in prompt.system_description
    assert target.parameter_family_sha256 not in encoded
    assert prompt.task_id.startswith("case-")


def test_trusted_audit_bridge_replays_exact_simulator_target_without_uncertainty(
    registered_case: RegisteredCase,
) -> None:
    store, prompt, plan, job, _ = registered_case
    expected = execute_maeda_sensitivity_oracle_job(plan, job)

    target = store.wet_lab_audit_target(prompt.task_id)

    assert target.case_sha256 == store.case_manifest(prompt.task_id)["case_sha256"]
    assert target.query_sha256 == expected.query_sha256
    assert target.prediction_sha256 == expected.prediction_sha256
    assert target.execution_sha256 == expected.execution.execution_sha256
    assert target.model_family is expected.execution.model_family
    assert target.model_id == expected.execution.model_id
    assert target.evidence_layer is PhysicalEvidenceLayer.SPIN_CHEMICAL_YIELD
    assert target.calibration_tier is CalibrationTier.SIMULATOR_ONLY
    assert target.interval is None


def test_prompt_cross_validates_visible_measurements_units_and_field_partitions(
    registered_case: RegisteredCase,
) -> None:
    _, prompt, _, _, _ = registered_case
    first, second = prompt.initial_observations

    with pytest.raises(ValueError, match="share a measurement type"):
        replace(
            prompt,
            initial_observations=(
                replace(first, measurement_type="different measurement"),
                second,
            ),
        )
    with pytest.raises(ValueError, match="share a measurement type"):
        replace(
            prompt,
            action_policy=replace(
                prompt.action_policy,
                measurement_type="different measurement",
            ),
        )
    with pytest.raises(ValueError, match="share one unit"):
        replace(prompt, initial_observations=(replace(first, unit="percent"), second))
    with pytest.raises(ValueError, match="fields must be unique"):
        replace(
            prompt,
            initial_observations=(first, replace(second, field_tesla=first.field_tesla)),
        )
    with pytest.raises(ValueError, match="must not overlap"):
        replace(
            prompt,
            action_policy=replace(
                prompt.action_policy,
                allowed_fields_tesla=(10e-6, 20e-6, 50e-6),
            ),
        )
    with pytest.raises(ValueError, match="must not overlap"):
        replace(
            prompt,
            action_policy=replace(
                prompt.action_policy,
                withheld_target_field_tesla=10e-6,
            ),
        )
    with pytest.raises(ValueError, match="exclude the target"):
        replace(
            prompt.action_policy,
            allowed_fields_tesla=(20e-6, 50e-6, 200e-6),
        )


def test_v1_scoring_policy_is_fixed_and_not_caller_tunable() -> None:
    with pytest.raises(ValueError, match="fixed"):
        NumericalScoringPolicy(absolute_error_tolerance=0.0)
    with pytest.raises(ValueError, match="fixed"):
        NumericalScoringPolicy(absolute_error_tolerance=1e-10)
    with pytest.raises(ValueError, match="fixed"):
        NumericalScoringPolicy(interpretation="This is predictive uncertainty.")


def test_prompt_parser_rejects_coercion_extra_missing_and_dropped_values(
    registered_case: RegisteredCase,
) -> None:
    _, prompt, _, _, _ = registered_case
    canonical = task_prompt_document(prompt)

    bool_budget = copy.deepcopy(canonical)
    bool_budget_value = bool_budget["query_budget"]
    assert isinstance(bool_budget_value, dict)
    bool_budget_value["maximum_queries"] = True
    extra_key = copy.deepcopy(canonical)
    extra_key["unexpected"] = "value"
    missing_key = copy.deepcopy(canonical)
    del missing_key["objective"]
    non_mapping_observation = copy.deepcopy(canonical)
    non_mapping_observations = non_mapping_observation["initial_observations"]
    assert isinstance(non_mapping_observations, list)
    non_mapping_observations[0] = "drop me"
    coerced_field = copy.deepcopy(canonical)
    coerced_action_policy = coerced_field["action_policy"]
    assert isinstance(coerced_action_policy, dict)
    coerced_action_policy["withheld_target_field_tesla"] = "0.0002"
    observation_extra = copy.deepcopy(canonical)
    extra_observations = observation_extra["initial_observations"]
    assert isinstance(extra_observations, list)
    first_observation = extra_observations[0]
    assert isinstance(first_observation, dict)
    first_observation["job_sha256"] = "a" * 64
    legacy_constraint_name = copy.deepcopy(canonical)
    legacy_constraint_name["protocol_constraints"] = legacy_constraint_name.pop(
        "interaction_constraints"
    )

    for malformed in (
        bool_budget,
        extra_key,
        missing_key,
        non_mapping_observation,
        coerced_field,
        observation_extra,
        legacy_constraint_name,
    ):
        with pytest.raises(ValueError, match="malformed or unsupported"):
            task_prompt_from_document(malformed)

    assert task_prompt_from_document(canonical) == prompt


def test_built_wheel_contains_and_imports_standalone_contract(tmp_path: Path) -> None:
    wheel_directory = tmp_path / "wheel"
    completed = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheel_directory)],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    wheels = tuple(wheel_directory.glob("*.whl"))
    assert len(wheels) == 1
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    assert "paircraft_benchmark_contract.py" in names
    assert "paircraft/benchmark/model.py" in names

    imported = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "import paircraft_benchmark_contract as contract; "
                "assert contract.SensitivityQueryAction(5e-5, 'yield').field_tesla == 5e-5; "
                "print(contract.__file__)"
            ),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(wheel), "PYTHONNOUSERSITE": "1"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert imported.returncode == 0, imported.stderr
    assert "paircraft_benchmark_contract.py" in imported.stdout


def test_case_manifest_binds_replay_provenance_without_answer_or_nonce(
    registered_case: RegisteredCase,
) -> None:
    store, prompt, plan, target, exact = registered_case
    manifest = store.case_manifest(prompt.task_id)
    encoded = json.dumps(manifest, sort_keys=True)
    replay = manifest["generator_replay"]

    assert replay["plan_sha256"] == plan.plan_sha256  # type: ignore[index]
    assert manifest["schema_version"] == "paircraft-benchmark-case-manifest-v2"
    assert replay["specification_sha256"] == plan.specification.specification_sha256  # type: ignore[index]
    assert replay["source_job_sha256"] == target.job_sha256  # type: ignore[index]
    assert replay["parameter_family_sha256"] == target.parameter_family_sha256  # type: ignore[index]
    assert len(replay["oracle_artifact_sha256"]) == 64  # type: ignore[index]
    assert len(replay["target_execution_sha256"]) == 64  # type: ignore[index]
    assert len(replay["target_result_sha256"]) == 64  # type: ignore[index]
    assert "answer_commitment_sha256" in manifest
    assert "nonce" not in encoded
    assert "objective_answer" not in encoded
    assert str(exact) not in encoded
    assert len(manifest["manifest_sha256"]) == 64  # type: ignore[arg-type]


def test_agent_deployment_bundle_is_oracle_free_and_importable_in_isolation(
    registered_case: RegisteredCase,
    tmp_path: Path,
) -> None:
    _, prompt, _, target, _ = registered_case
    bundle = tmp_path / "agent-bundle"

    manifest = build_agent_deployment_bundle(bundle, prompt)
    document = agent_deployment_manifest_document(manifest)
    deployed_files = document["files"]

    assert isinstance(manifest, AgentDeploymentManifest)
    assert isinstance(deployed_files, dict)
    assert set(deployed_files) == {
        "paircraft_benchmark_contract.py",
        "prompt.json",
        "query-action.schema.json",
        "query-response.schema.json",
    }
    forbidden = (
        target.job_sha256,
        target.parameter_family_sha256,
        target.family.family_id,
        "paircraft.oracle",
        "paircraft.spin",
        "paircraft.sweeps",
    )
    deployed_text = "\n".join(
        path.read_text(encoding="utf-8") for path in bundle.iterdir() if path.is_file()
    )
    assert all(value not in deployed_text for value in forbidden)

    script = """
import importlib.util
import json
from pathlib import Path

from paircraft_benchmark_contract import (
    SensitivityQueryAction,
    SensitivityQueryResponse,
    SensitivityQueryToolAdapter,
    sensitivity_query_action_document,
    task_prompt_from_document,
)

prompt = task_prompt_from_document(json.loads(Path("prompt.json").read_text()))
action = SensitivityQueryAction(50e-6, prompt.measurement_type)

def remote_stub(request):
    return SensitivityQueryResponse(
        request.field_tesla,
        request.measurement_type,
        0.25,
        prompt.answer_unit,
        "a" * 64,
    )

response = SensitivityQueryToolAdapter(remote_stub).invoke(
    sensitivity_query_action_document(action)
)
assert response["value"] == 0.25
assert importlib.util.find_spec("paircraft") is None
try:
    from paircraft.benchmark.model import EvaluationStore
except ModuleNotFoundError:
    pass
else:
    raise AssertionError("deployed artifact exposed Paircraft evaluator code")
print(prompt.task_id)
"""
    environment = {
        "PYTHONPATH": str(bundle),
        "PYTHONNOUSERSITE": "1",
    }
    completed = subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=bundle,
        env={**os.environ, **environment},
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == prompt.task_id


def test_allowed_query_executes_the_exact_authenticated_family_job(
    registered_case: RegisteredCase,
) -> None:
    store, prompt, plan, target, _ = registered_case
    gateway = _gateway(store, prompt, "allowed-query")
    action = SensitivityQueryAction(
        field_tesla=50e-6,
        measurement_type=prompt.measurement_type,
    )
    adapter = SensitivityQueryToolAdapter(gateway.query)
    response_document = adapter.invoke(sensitivity_query_action_document(action))
    expected_job = next(
        job
        for job in plan.jobs
        if job.parameter_family_sha256 == target.parameter_family_sha256
        and job.field_point.magnitude_tesla == 50e-6
    )
    expected = execute_maeda_sensitivity_oracle_job(plan, expected_job)

    assert response_document["field_tesla"] == 50e-6
    assert response_document["value"] == pytest.approx(
        expected.prediction.distribution.mean,
        abs=1e-12,
    )
    receipt = response_document["query_receipt_sha256"]
    assert isinstance(receipt, str) and len(receipt) == 64
    assert expected_job.job_sha256 not in json.dumps(response_document, sort_keys=True)


def test_forbidden_target_query_is_recorded_and_hard_fails_episode(
    registered_case: RegisteredCase,
) -> None:
    store, prompt, _, _, exact = registered_case
    gateway = _gateway(store, prompt, "target-rejection")

    with pytest.raises(SensitivityQueryRejected, match="cannot be queried") as rejected:
        gateway.query(
            SensitivityQueryAction(
                field_tesla=200e-6,
                measurement_type=prompt.measurement_type,
            )
        )
    episode = gateway.submit_final_answer(exact + 0.5 * prompt.scoring_policy.allowed_error(exact))

    assert rejected.value.constraint_id == "withheld-target-field"
    assert episode.score.answer_correct
    assert episode.score.query_count == 0
    assert episode.score.hard_constraint_violations == ("withheld-target-field",)
    assert not episode.score.interaction_policy_compliant and not episode.score.passed


def test_gateway_enforces_budget_and_records_attempted_third_query(
    registered_case: RegisteredCase,
) -> None:
    store, prompt, _, _, exact = registered_case
    gateway = _gateway(store, prompt, "budget-rejection")
    for field in (20e-6, 50e-6):
        gateway.query(
            SensitivityQueryAction(
                field_tesla=field,
                measurement_type=prompt.measurement_type,
            )
        )
    with pytest.raises(SensitivityQueryRejected, match="budget is exhausted") as rejected:
        gateway.query(
            SensitivityQueryAction(
                field_tesla=100e-6,
                measurement_type=prompt.measurement_type,
            )
        )
    episode = gateway.submit_final_answer(exact + 0.5 * prompt.scoring_policy.allowed_error(exact))
    transcript = gateway.transcript_document()

    assert rejected.value.constraint_id == "query-budget"
    assert episode.score.query_count == 2
    assert episode.score.budget_compliant
    assert episode.score.hard_constraint_violations == ("query-budget",)
    assert transcript["derived"]["query_count"] == 2  # type: ignore[index]
    assert transcript["schema_version"] == "paircraft-benchmark-transcript-v2"
    assert not episode.score.passed


@pytest.mark.parametrize(
    "action, constraint",
    [
        (
            SensitivityQueryAction(field_tesla=20e-6, measurement_type="wrong-measurement"),
            "declared-measurement-type",
        ),
        (
            SensitivityQueryAction(
                field_tesla=5e-6,
                measurement_type="simulated absolute secondary-pair yield",
            ),
            "declared-query-field",
        ),
    ],
)
def test_gateway_records_measurement_and_field_policy_violations(
    registered_case: RegisteredCase,
    action: SensitivityQueryAction,
    constraint: str,
) -> None:
    store, prompt, _, _, exact = registered_case
    gateway = _gateway(store, prompt, f"policy-{constraint}")

    with pytest.raises(SensitivityQueryRejected) as rejected:
        gateway.query(action)
    episode = gateway.submit_final_answer(exact + 0.5 * prompt.scoring_policy.allowed_error(exact))

    assert rejected.value.constraint_id == constraint
    assert episode.score.hard_constraint_violations == (constraint,)
    assert not episode.score.passed


def test_objective_score_comes_from_gateway_trajectory_and_withheld_answer(
    registered_case: RegisteredCase,
) -> None:
    store, prompt, _, _, exact = registered_case
    gateway = _gateway(store, prompt, "successful-trajectory")
    gateway.query(
        SensitivityQueryAction(
            field_tesla=100e-6,
            measurement_type=prompt.measurement_type,
        )
    )

    prediction = exact + 0.5 * prompt.scoring_policy.allowed_error(exact)
    episode = gateway.submit_final_answer(prediction)

    assert episode.score.query_count == 1
    assert episode.score.answer_correct
    assert episode.score.budget_compliant
    assert episode.score.interaction_policy_compliant
    assert episode.score.normalized_error == pytest.approx(0.5)
    assert episode.score.passed
    assert episode.score.reward == pytest.approx(2 / 3)
    assert len(episode.transcript_sha256) == 64
    assert len(episode.episode_sha256) == 64
    episode_payload = episode_document(episode)
    assert episode_payload["schema_version"] == "paircraft-benchmark-episode-v3"
    assert episode_payload["score"]["query_count"] == 1  # type: ignore[index]
    assert episode_payload["score"]["interaction_policy_compliant"] is True  # type: ignore[index]
    assert "protocol_valid" not in episode_payload["score"]  # type: ignore[operator]


@pytest.mark.parametrize(
    "passed, expected",
    [
        (5, ReliabilityClass.RELIABLE),
        (4, ReliabilityClass.RELIABLE),
        (3, ReliabilityClass.MIXED),
        (2, ReliabilityClass.BRITTLE),
        (1, ReliabilityClass.BRITTLE),
        (0, ReliabilityClass.UNSOLVED),
    ],
)
def test_reliability_classification_and_distinct_identity_mapping(
    registered_case: RegisteredCase,
    passed: int,
    expected: ReliabilityClass,
) -> None:
    store, prompt, _, _, exact = registered_case
    episodes = tuple(
        _episode(
            store,
            prompt,
            exact + (0.5 if index < passed else 2.0) * prompt.scoring_policy.allowed_error(exact),
            index,
        )
        for index in range(5)
    )

    summary = summarize_reliability(episodes)
    document = reliability_document(summary)

    assert summary.passed_trials == passed
    assert summary.failed_trials == 5 - passed
    assert summary.classification is expected
    assert len(summary.trials) == 5
    assert len({trial.run_id for trial in summary.trials}) == 5
    assert sum(trial.passed for trial in summary.trials) == passed
    assert document["trial_count"] == 5
    assert document["schema_version"] == "paircraft-benchmark-reliability-v3"
    assert len(summary.reliability_sha256) == 64


def test_reliability_rejects_reused_or_changed_run_identities(
    registered_case: RegisteredCase,
) -> None:
    store, prompt, _, _, exact = registered_case
    prediction = exact + 0.5 * prompt.scoring_policy.allowed_error(exact)
    episodes = tuple(_episode(store, prompt, prediction, index) for index in range(5))

    with pytest.raises(ValueError, match="five distinct"):
        summarize_reliability((*episodes[:4], episodes[0]))
    with pytest.raises(ValueError, match="one case, model, and configuration"):
        summarize_reliability((*episodes[:4], replace(episodes[4], configuration_sha256="d" * 64)))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"sequence": True, "kind": TranscriptEventKind.FINAL_ANSWER, "final_answer": 0.1},
            "sequence must be an integer",
        ),
        (
            {"sequence": -1, "kind": TranscriptEventKind.FINAL_ANSWER, "final_answer": 0.1},
            "identity is invalid",
        ),
        ({"sequence": 0, "kind": TranscriptEventKind.ORACLE_QUERY}, "requires an interaction hash"),
        (
            {
                "sequence": 0,
                "kind": TranscriptEventKind.ORACLE_QUERY,
                "interaction_sha256": "a" * 64,
                "constraint_id": "unexpected",
            },
            "incompatible payload",
        ),
        (
            {"sequence": 0, "kind": TranscriptEventKind.HARD_CONSTRAINT_VIOLATION},
            "requires a constraint ID",
        ),
        (
            {
                "sequence": 0,
                "kind": TranscriptEventKind.HARD_CONSTRAINT_VIOLATION,
                "constraint_id": "constraint",
                "final_answer": 0.1,
            },
            "incompatible payload",
        ),
        ({"sequence": 0, "kind": TranscriptEventKind.FINAL_ANSWER}, "requires a numeric answer"),
        (
            {
                "sequence": 0,
                "kind": TranscriptEventKind.FINAL_ANSWER,
                "final_answer": 0.1,
                "interaction_sha256": "a" * 64,
            },
            "incompatible payload",
        ),
    ],
)
def test_transcript_events_refuse_ambiguous_kind_specific_payloads(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TranscriptEvent(**kwargs)  # type: ignore[arg-type]


def _recorder() -> HarnessRecorder:
    return HarnessRecorder(
        task_id="task",
        run_id="run",
        model_id="model",
        model_version="1",
        model_artifact_sha256="a" * 64,
        configuration_sha256="b" * 64,
        harness_id="harness",
        harness_version="1",
        harness_artifact_sha256="c" * 64,
    )


def test_harness_refuses_post_finalization_events_and_double_seal() -> None:
    finalized = _recorder()
    finalized._record_final_answer(0.1)
    with pytest.raises(ValueError, match="finalized transcript"):
        finalized._record_query(request={}, response={})

    sealed = _recorder()
    sealed._record_final_answer(0.1)
    transcript = sealed.seal()
    assert transcript.final_answer == pytest.approx(0.1)
    with pytest.raises(ValueError, match="already sealed"):
        sealed.seal()
    with pytest.raises(ValueError, match="sealed or finalized"):
        sealed._record_hard_constraint_violation("constraint")


def test_transcript_constructor_and_serializer_require_trusted_complete_history() -> None:
    identity: dict[str, object] = {
        "task_id": "task",
        "run_id": "run",
        "model_id": "model",
        "model_version": "1",
        "model_artifact_sha256": "a" * 64,
        "configuration_sha256": "b" * 64,
        "harness_id": "harness",
        "harness_version": "1",
        "harness_artifact_sha256": "c" * 64,
    }
    final = TranscriptEvent(0, TranscriptEventKind.FINAL_ANSWER, final_answer=0.1)

    with pytest.raises(ValueError, match="trusted harness"):
        EpisodeTranscript(**identity, events=(final,), _harness_token=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="immutable typed tuple"):
        EpisodeTranscript(
            **identity,  # type: ignore[arg-type]
            events=[final],  # type: ignore[arg-type]
            _harness_token=benchmark_model._AUTHENTICATED_REPLAY_TOKEN,
        )
    with pytest.raises(ValueError, match="contiguous ordered"):
        EpisodeTranscript(
            **identity,  # type: ignore[arg-type]
            events=(replace(final, sequence=1),),
            _harness_token=benchmark_model._AUTHENTICATED_REPLAY_TOKEN,
        )
    with pytest.raises(ValueError, match="exactly one terminal"):
        EpisodeTranscript(
            **identity,  # type: ignore[arg-type]
            events=(),
            _harness_token=benchmark_model._AUTHENTICATED_REPLAY_TOKEN,
        )
    with pytest.raises(ValueError, match="transcript serialization"):
        transcript_document(object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"absolute_error": -1.0}, "nonnegative"),
        ({"allowed_error": 0.0}, "scale must be positive"),
        ({"normalized_error": 2.0}, "contradicts"),
        ({"query_count": True}, "nonnegative integer"),
        ({"maximum_queries": -1}, "nonnegative integer"),
        ({"hard_constraint_violations": ["x"]}, "unique sorted tuple"),
        ({"hard_constraint_violations": ("x", "x")}, "unique sorted tuple"),
        ({"answer_correct": False}, "objective components"),
        ({"budget_compliant": False}, "objective components"),
        ({"interaction_policy_compliant": False}, "objective components"),
        ({"passed": False}, "objective components"),
        ({"reward": 0.0}, "objective components"),
    ],
)
def test_outcome_score_refuses_contradictory_derived_claims(
    registered_case: RegisteredCase, changes: dict[str, object], message: str
) -> None:
    store, prompt, _, _, exact = registered_case
    score = _episode(store, prompt, exact, 99).score

    with pytest.raises(ValueError, match=message):
        replace(score, **changes)  # type: ignore[arg-type]


def test_episode_and_reliability_serializers_refuse_untyped_values() -> None:
    with pytest.raises(ValueError, match="episode serialization"):
        episode_document(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reliability serialization"):
        reliability_document(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="deployment manifest"):
        agent_deployment_manifest_document(object())  # type: ignore[arg-type]


def test_episode_result_rejects_untyped_score_and_unknown_schema(
    registered_case: RegisteredCase,
) -> None:
    store, prompt, _, _, exact = registered_case
    episode = _episode(store, prompt, exact, 100)

    with pytest.raises(ValueError, match="requires an OutcomeScore"):
        replace(episode, score=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="schema is unsupported"):
        replace(episode, schema_version="future")


def _summary(registered_case: RegisteredCase) -> ReliabilitySummary:
    store, prompt, _, _, exact = registered_case
    return summarize_reliability(
        tuple(_episode(store, prompt, exact, 200 + index) for index in range(5))
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"trials": []}, "immutable typed tuple"),
        ({"trials": ()}, "five canonically ordered"),
        ({"passed_trials": 6}, "counts are inconsistent"),
        ({"failed_trials": 1}, "counts are inconsistent"),
        ({"success_rate": 0.0}, "success rate contradicts"),
        ({"classification": ReliabilityClass.UNSOLVED}, "classification contradicts"),
        ({"schema_version": "future"}, "schema is unsupported"),
    ],
)
def test_reliability_summary_refuses_inconsistent_aggregate_claims(
    registered_case: RegisteredCase, changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_summary(registered_case), **changes)  # type: ignore[arg-type]


def test_reliability_trial_identity_requires_boolean_pass_status() -> None:
    with pytest.raises(ValueError, match="must be Boolean"):
        ReliabilityTrialIdentity("run", "a" * 64, "b" * 64, 1)  # type: ignore[arg-type]


def test_store_refuses_unknown_tasks_transcripts_and_constraints(
    registered_case: RegisteredCase,
) -> None:
    store, prompt, _, _, exact = registered_case
    with pytest.raises(ValueError, match="no such task ID"):
        store.prompt("missing")
    with pytest.raises(ValueError, match="no auditable sensitivity task"):
        store.wet_lab_audit_target("missing")
    with pytest.raises(ValueError, match="no sensitivity task"):
        store.open_sensitivity_gateway(
            "missing",
            run_id="run",
            model_id="model",
            model_version="1",
            model_artifact_sha256="a" * 64,
            configuration_sha256="b" * 64,
            harness_id="harness",
            harness_version="1",
            harness_artifact_sha256="c" * 64,
        )
    with pytest.raises(ValueError, match="harness-sealed transcript"):
        store.score(object())  # type: ignore[arg-type]

    recorder = _recorder()
    recorder._record_final_answer(exact)
    sealed = recorder.seal()
    unknown_task = EpisodeTranscript(
        task_id="missing",
        run_id=sealed.run_id,
        model_id=sealed.model_id,
        model_version=sealed.model_version,
        model_artifact_sha256=sealed.model_artifact_sha256,
        configuration_sha256=sealed.configuration_sha256,
        harness_id=sealed.harness_id,
        harness_version=sealed.harness_version,
        harness_artifact_sha256=sealed.harness_artifact_sha256,
        events=sealed.events,
        _harness_token=benchmark_model._AUTHENTICATED_REPLAY_TOKEN,
    )
    with pytest.raises(ValueError, match="task is not registered"):
        store.score(unknown_task)

    gateway = _gateway(store, prompt, "unknown-constraint")
    gateway._recorder._record_hard_constraint_violation("not-declared")
    gateway._recorder._record_final_answer(exact)
    with pytest.raises(ValueError, match="unknown hard-constraint"):
        store.score(gateway._recorder.seal())


def test_gateway_transcript_is_unavailable_before_final_answer(
    registered_case: RegisteredCase,
) -> None:
    store, prompt, _, _, _ = registered_case
    with pytest.raises(ValueError, match="not sealed"):
        _gateway(store, prompt, "unsealed").transcript_document()


def test_deployment_manifest_and_builder_refuse_ambiguous_destinations(
    registered_case: RegisteredCase, tmp_path: Path
) -> None:
    _, prompt, _, _, _ = registered_case
    with pytest.raises(ValueError, match="nonempty canonical tuple"):
        AgentDeploymentManifest(prompt.prompt_sha256, ())
    with pytest.raises(ValueError, match="nonempty canonical tuple"):
        AgentDeploymentManifest(
            prompt.prompt_sha256,
            (("z", "a" * 64), ("a", "b" * 64)),
        )
    destination = tmp_path / "occupied"
    destination.mkdir()
    (destination / "existing").write_text("occupied")
    with pytest.raises(ValueError, match="absent or empty"):
        build_agent_deployment_bundle(destination, prompt)
    with pytest.raises(ValueError, match="requires a Path and TaskPrompt"):
        build_agent_deployment_bundle("not-a-path", prompt)  # type: ignore[arg-type]


def test_registration_refuses_wrong_store_plan_and_target_roles(
    registered_case: RegisteredCase,
) -> None:
    _, _, plan, _, _ = registered_case
    baseline = next(
        job
        for job in plan.jobs
        if job.family.family_id == "baseline" and job.field_point.magnitude_tesla == 200e-6
    )
    wrong_field = next(
        job
        for job in plan.jobs
        if job.family.family_id != "baseline" and job.field_point.magnitude_tesla == 100e-6
    )

    with pytest.raises(ValueError, match="requires an EvaluationStore"):
        register_maeda_sensitivity_case(object(), plan=plan, target_job=wrong_field)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="typed plan and job"):
        register_maeda_sensitivity_case(
            EvaluationStore(evaluator_nonce_sha256="d" * 64),
            plan=object(),  # type: ignore[arg-type]
            target_job=wrong_field,
        )
    with pytest.raises(ValueError, match="non-baseline"):
        register_maeda_sensitivity_case(
            EvaluationStore(evaluator_nonce_sha256="d" * 64),
            plan=plan,
            target_job=baseline,
        )
    with pytest.raises(ValueError, match="withheld 200"):
        register_maeda_sensitivity_case(
            EvaluationStore(evaluator_nonce_sha256="d" * 64),
            plan=plan,
            target_job=wrong_field,
        )


def test_summarize_reliability_refuses_non_episode_collections() -> None:
    with pytest.raises(ValueError, match="exactly five EpisodeResult"):
        summarize_reliability(())
    with pytest.raises(ValueError, match="exactly five EpisodeResult"):
        summarize_reliability((object(),) * 5)  # type: ignore[arg-type]
