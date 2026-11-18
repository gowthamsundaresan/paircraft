from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import paircraft.benchmark.demo as demo_module
from paircraft.benchmark.demo import NearestFieldPersistenceAgent, run_benchmark_demo
from paircraft.benchmark.model import (
    AgentDeploymentManifest,
    EpisodeResult,
    OutcomeScore,
    build_agent_deployment_bundle,
)
from paircraft.benchmark.runner import AgentRunIdentity
from paircraft.sweeps.storage import canonical_json_bytes, canonical_sha256
from paircraft_benchmark_contract import (
    FieldQueryPolicy,
    NumericalScoringPolicy,
    QueryBudget,
    SensitivityQueryAction,
    SensitivityQueryResponse,
    TaskPrompt,
    VisibleObservation,
    task_prompt_document,
)


def _prompt() -> TaskPrompt:
    measurement = "simulated absolute secondary-pair yield"
    return TaskPrompt(
        task_id="case-demo",
        system_description="Opaque simulator-only sensitivity system.",
        objective="Predict one withheld scalar outcome.",
        measurement_type=measurement,
        answer_unit="1",
        query_budget=QueryBudget(2),
        interaction_constraints=("declared-query-field",),
        initial_observations=(VisibleObservation(0.0, measurement, 0.2, "1", "a" * 64),),
        action_policy=FieldQueryPolicy(
            measurement,
            (20e-6, 50e-6, 100e-6),
            200e-6,
        ),
        scoring_policy=NumericalScoringPolicy(),
    )


def _score() -> OutcomeScore:
    return OutcomeScore(
        predicted_answer=0.3,
        absolute_error=0.1,
        allowed_error=1.0,
        normalized_error=0.1,
        query_count=1,
        maximum_queries=2,
        hard_constraint_violations=(),
        answer_correct=True,
        budget_compliant=True,
        interaction_policy_compliant=True,
        passed=True,
        reward=1 / 1.1,
    )


def _install_demo_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mismatch: str | None = None,
) -> tuple[TaskPrompt, list[str], str]:
    prompt = _prompt()
    selected: list[str] = []
    generated_nonce = "7" * 64
    jobs = (
        SimpleNamespace(
            family=SimpleNamespace(family_id="baseline"),
            field_point=SimpleNamespace(magnitude_tesla=200e-6),
            job_sha256="f" * 64,
        ),
        SimpleNamespace(
            family=SimpleNamespace(family_id="perturbed-b"),
            field_point=SimpleNamespace(magnitude_tesla=200e-6),
            job_sha256="2" * 64,
        ),
        SimpleNamespace(
            family=SimpleNamespace(family_id="perturbed-a"),
            field_point=SimpleNamespace(magnitude_tesla=200e-6),
            job_sha256="1" * 64,
        ),
    )
    plan = SimpleNamespace(jobs=jobs, plan_sha256="3" * 64)

    class FakeStore:
        def __init__(self, *, evaluator_nonce_sha256: str) -> None:
            assert evaluator_nonce_sha256 == generated_nonce

        def case_manifest(self, task_id: str) -> dict[str, object]:
            assert task_id == prompt.task_id
            manifest_prompt: object = task_prompt_document(prompt)
            replay_plan = plan.plan_sha256
            replay_source = "1" * 64
            if mismatch == "case_prompt":
                manifest_prompt = {"wrong": True}
            elif mismatch == "replay_plan":
                replay_plan = "0" * 64
            elif mismatch == "replay_source":
                replay_source = "0" * 64
            return {
                "schema_version": "fake-case-v1",
                "case_sha256": "e" * 64,
                "manifest_sha256": "4" * 64,
                "prompt": manifest_prompt,
                "generator_replay": {
                    "plan_sha256": replay_plan,
                    "source_job_sha256": replay_source,
                },
            }

    def register(store: object, *, plan: object, target_job: object) -> TaskPrompt:
        del store
        assert plan is not None
        job_sha256 = cast(SimpleNamespace, target_job).job_sha256
        assert isinstance(job_sha256, str)
        selected.append(job_sha256)
        return prompt

    def run_local(*args: object, **kwargs: object) -> EpisodeResult:
        del args
        assert kwargs["task_id"] == prompt.task_id
        assert isinstance(kwargs["agent"], NearestFieldPersistenceAgent)
        identity = cast(AgentRunIdentity, kwargs["identity"])
        episode = EpisodeResult(
            case_sha256="e" * 64,
            transcript_sha256="f" * 64,
            run_id="nearest-field-persistence-1",
            model_id=identity.model_id,
            model_version=identity.model_version,
            model_artifact_sha256=identity.model_artifact_sha256,
            configuration_sha256=identity.configuration_sha256,
            score=_score(),
        )
        if mismatch == "episode_case":
            return replace(episode, case_sha256="0" * 64)
        if mismatch == "episode_model_id":
            return replace(episode, model_id="wrong-model")
        if mismatch == "episode_model_version":
            return replace(episode, model_version="wrong-version")
        if mismatch == "episode_model_artifact":
            return replace(episode, model_artifact_sha256="0" * 64)
        if mismatch == "episode_configuration":
            return replace(episode, configuration_sha256="0" * 64)
        return episode

    real_build = build_agent_deployment_bundle

    def build_deployment(destination: Path, task: TaskPrompt) -> AgentDeploymentManifest:
        deployment = real_build(destination, task)
        if mismatch == "deployment_prompt":
            return replace(deployment, prompt_sha256="0" * 64)
        return deployment

    monkeypatch.setattr(demo_module, "token_hex", lambda count: generated_nonce)
    monkeypatch.setattr(demo_module, "EvaluationStore", FakeStore)
    monkeypatch.setattr(demo_module, "default_maeda_local_sensitivity", lambda: object())
    monkeypatch.setattr(demo_module, "materialize_maeda_local_sensitivity", lambda value: plan)
    monkeypatch.setattr(demo_module, "register_maeda_sensitivity_case", register)
    monkeypatch.setattr(demo_module, "run_local_sensitivity_episode", run_local)
    monkeypatch.setattr(demo_module, "build_agent_deployment_bundle", build_deployment)
    return prompt, selected, generated_nonce


def test_nearest_field_baseline_queries_one_closest_declared_field() -> None:
    prompt = _prompt()
    actions: list[SensitivityQueryAction] = []

    def query(action: SensitivityQueryAction) -> SensitivityQueryResponse:
        actions.append(action)
        return SensitivityQueryResponse(
            action.field_tesla,
            action.measurement_type,
            0.37,
            "1",
            "b" * 64,
        )

    prediction = NearestFieldPersistenceAgent().run(prompt, query)

    assert prediction == 0.37
    assert len(actions) == 1
    assert actions[0].field_tesla == 100e-6


def test_demo_publishes_isolated_linked_content_without_nonce(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, selected, generated_nonce = _install_demo_fakes(monkeypatch)

    artifacts = run_benchmark_demo(tmp_path / "outputs")
    raw = artifacts.summary_path.read_bytes()
    document = json.loads(raw)
    claimed_hash = document.pop("summary_sha256")

    assert selected == ["1" * 64]
    assert artifacts.summary_sha256 == claimed_hash == canonical_sha256(document)
    assert raw == canonical_json_bytes({**document, "summary_sha256": claimed_hash})
    assert generated_nonce not in raw.decode()
    assert document["baseline"]["learned"] is False
    assert document["local_harness"]["security_status"] == "same_process_not_isolated"
    reproducibility = document["reproducibility"]
    assert reproducibility["scientific_target_deterministic"] is True
    assert reproducibility["evaluator_nonce_serialized"] is False
    assert reproducibility["exact_task_prompt_bundle_commitment_summary_reproducible"] is False
    assert reproducibility["answer_commitment_reopenable_after_run"] is False
    assert "not learned" in document["scientific_boundary"]
    assert "not experimentally calibrated" in document["scientific_boundary"]
    assert artifacts.agent_bundle_directory.is_dir()
    deployed = {path.name for path in artifacts.agent_bundle_directory.iterdir()}
    assert deployed == {
        "manifest.json",
        "paircraft_benchmark_contract.py",
        "prompt.json",
        "query-action.schema.json",
        "query-response.schema.json",
    }
    deployed_text = "".join(path.read_text() for path in artifacts.agent_bundle_directory.iterdir())
    assert generated_nonce not in deployed_text
    assert "paircraft.benchmark.model" not in deployed_text


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("deployment_prompt", "deployment prompt identity"),
        ("case_prompt", "case prompt"),
        ("replay_plan", "replay plan"),
        ("replay_source", "replay source job"),
        ("episode_case", "episode case identity"),
        ("episode_model_id", "episode model identity"),
        ("episode_model_version", "episode model identity"),
        ("episode_model_artifact", "episode model identity"),
        ("episode_configuration", "episode model identity"),
    ],
)
def test_demo_rejects_each_cross_artifact_identity_mismatch_before_summary_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
    message: str,
) -> None:
    _install_demo_fakes(monkeypatch, mismatch=mismatch)
    output = tmp_path / mismatch

    with pytest.raises(ValueError, match=message):
        run_benchmark_demo(output)

    assert tuple(output.glob("benchmark-demo-*.json")) == ()
