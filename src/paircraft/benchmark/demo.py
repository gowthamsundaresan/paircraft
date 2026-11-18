"""Runnable simulator-only benchmark demonstration.

The demo proves the evaluator/agent loop and artifact boundaries, not a learned
or experimentally calibrated oracle. Its agent is a transparent nearest-field
persistence heuristic executed through the non-isolating local callback runner.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from secrets import token_hex

from paircraft.benchmark.model import (
    EpisodeResult,
    EvaluationStore,
    agent_deployment_manifest_document,
    build_agent_deployment_bundle,
    episode_document,
    register_maeda_sensitivity_case,
)
from paircraft.benchmark.runner import (
    AgentRunIdentity,
    SensitivityAgent,
    run_local_sensitivity_episode,
)
from paircraft.sweeps.maeda_sensitivity import (
    default_maeda_local_sensitivity,
    materialize_maeda_local_sensitivity,
)
from paircraft.sweeps.storage import canonical_sha256, publish_canonical_json_once
from paircraft_benchmark_contract import (
    SensitivityQueryAction,
    SensitivityQueryResponse,
    TaskPrompt,
    task_prompt_document,
)

_SCHEMA = "paircraft-simulator-benchmark-demo-v1"
_BASELINE_ID = "paircraft-nearest-field-persistence-baseline"
_BASELINE_VERSION = "1.0.0"
_HARNESS_ID = "paircraft-local-callback-harness"
_HARNESS_VERSION = "1.0.0"
_BOUNDARY = (
    "Simulator-only adaptive-prediction demonstration using a transparent nearest-field "
    "persistence heuristic. The oracle is not learned and not experimentally calibrated; its fixed "
    "grading tolerance is not predictive uncertainty, and the same-process local callback is "
    "not a security boundary. Outputs are not biological predictions or training evidence."
)


@dataclass(frozen=True, slots=True)
class BenchmarkDemoArtifacts:
    """Filesystem locations and content identity of one completed demo."""

    agent_bundle_directory: Path
    summary_path: Path
    summary_sha256: str


class NearestFieldPersistenceAgent:
    """Query the field nearest the target and reuse its value as the prediction."""

    def run(
        self,
        prompt: TaskPrompt,
        query: Callable[[SensitivityQueryAction], SensitivityQueryResponse],
    ) -> float:
        target = prompt.action_policy.withheld_target_field_tesla
        field = min(
            prompt.action_policy.allowed_fields_tesla,
            key=lambda candidate: (abs(candidate - target), candidate),
        )
        response = query(SensitivityQueryAction(field, prompt.measurement_type))
        return response.value


def _source_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _run_identity() -> AgentRunIdentity:
    module = Path(__file__)
    benchmark_directory = module.parent
    return AgentRunIdentity(
        model_id=_BASELINE_ID,
        model_version=_BASELINE_VERSION,
        model_artifact_sha256=_source_sha256(module),
        configuration_sha256=canonical_sha256(
            {
                "strategy": "query-nearest-declared-field-and-persist-value",
                "maximum_queries_used": 1,
                "learned_parameters": False,
            }
        ),
        harness_id=_HARNESS_ID,
        harness_version=_HARNESS_VERSION,
        harness_artifact_sha256=canonical_sha256(
            {
                "runner_py": _source_sha256(benchmark_directory / "runner.py"),
                "model_py": _source_sha256(benchmark_directory / "model.py"),
            }
        ),
    )


def _validate_cross_artifact_links(
    *,
    prompt: TaskPrompt,
    deployment_prompt_sha256: str,
    case_manifest: dict[str, object],
    plan_sha256: str,
    target_job_sha256: str,
    identity: AgentRunIdentity,
    episode: EpisodeResult,
) -> None:
    if deployment_prompt_sha256 != prompt.prompt_sha256:
        raise ValueError("agent deployment prompt identity contradicts the registered prompt")
    if case_manifest.get("prompt") != task_prompt_document(prompt):
        raise ValueError("evaluator case prompt contradicts the registered prompt")
    replay = case_manifest.get("generator_replay")
    if not isinstance(replay, dict):
        raise ValueError("evaluator case lacks a structured generator replay")
    if replay.get("plan_sha256") != plan_sha256:
        raise ValueError("evaluator replay plan contradicts the selected plan")
    if replay.get("source_job_sha256") != target_job_sha256:
        raise ValueError("evaluator replay source job contradicts the selected target")
    if episode.case_sha256 != case_manifest.get("case_sha256"):
        raise ValueError("episode case identity contradicts the evaluator case")
    if (
        episode.model_id != identity.model_id
        or episode.model_version != identity.model_version
        or episode.model_artifact_sha256 != identity.model_artifact_sha256
        or episode.configuration_sha256 != identity.configuration_sha256
    ):
        raise ValueError("episode model identity contradicts the requested baseline identity")


def run_benchmark_demo(output_root: Path) -> BenchmarkDemoArtifacts:
    """Run one deterministic-task simulator demo and publish auditable artifacts.

    The evaluator nonce is generated in memory and is never included in either
    the isolated agent bundle or evaluator summary.
    """

    if not isinstance(output_root, Path):
        raise ValueError("benchmark demo output root must be a pathlib.Path")
    if output_root.exists() and (output_root.is_symlink() or not output_root.is_dir()):
        raise ValueError("benchmark demo output root must be a real directory")
    output_root.mkdir(parents=True, mode=0o700, exist_ok=True)

    plan = materialize_maeda_local_sensitivity(default_maeda_local_sensitivity())
    candidates = tuple(
        job
        for job in plan.jobs
        if job.family.family_id != "baseline" and job.field_point.magnitude_tesla == 200e-6
    )
    if not candidates:
        raise ValueError("default sensitivity plan has no non-baseline 200 microtesla task")
    target = min(candidates, key=lambda job: job.job_sha256)

    store = EvaluationStore(evaluator_nonce_sha256=token_hex(32))
    prompt = register_maeda_sensitivity_case(store, plan=plan, target_job=target)
    deployment_directory = output_root / f"agent-bundle-{prompt.prompt_sha256}"
    deployment = build_agent_deployment_bundle(deployment_directory, prompt)
    identity = _run_identity()
    agent: SensitivityAgent = NearestFieldPersistenceAgent()
    episode = run_local_sensitivity_episode(
        store,
        task_id=prompt.task_id,
        run_id="nearest-field-persistence-1",
        identity=identity,
        agent=agent,
    )
    case_manifest = store.case_manifest(prompt.task_id)
    _validate_cross_artifact_links(
        prompt=prompt,
        deployment_prompt_sha256=deployment.prompt_sha256,
        case_manifest=case_manifest,
        plan_sha256=plan.plan_sha256,
        target_job_sha256=target.job_sha256,
        identity=identity,
        episode=episode,
    )
    payload: dict[str, object] = {
        "schema_version": _SCHEMA,
        "scientific_boundary": _BOUNDARY,
        "plan_sha256": plan.plan_sha256,
        "target_job_sha256": target.job_sha256,
        "task_id": prompt.task_id,
        "prompt_sha256": prompt.prompt_sha256,
        "baseline": {
            "model_id": identity.model_id,
            "model_version": identity.model_version,
            "model_artifact_sha256": identity.model_artifact_sha256,
            "configuration_sha256": identity.configuration_sha256,
            "strategy": "query-nearest-declared-field-and-persist-value",
            "learned": False,
        },
        "local_harness": {
            "harness_id": identity.harness_id,
            "harness_version": identity.harness_version,
            "harness_artifact_sha256": identity.harness_artifact_sha256,
            "security_status": "same_process_not_isolated",
        },
        "reproducibility": {
            "scientific_target_selection": (
                "minimum job sha256 among non-baseline 200 microtesla plan members"
            ),
            "scientific_target_deterministic": True,
            "fresh_evaluator_nonce_per_run": True,
            "evaluator_nonce_serialized": False,
            "evaluator_nonce_returned_or_retained": False,
            "exact_task_prompt_bundle_commitment_summary_reproducible": False,
            "answer_commitment_reopenable_after_run": False,
            "reason": (
                "The fresh evaluator nonce is intentionally discarded after the run. It affects "
                "opaque task identity and answer commitment, so exact run artifacts cannot be "
                "recreated or the commitment reopened later."
            ),
        },
        "agent_deployment": agent_deployment_manifest_document(deployment),
        "evaluator_case": case_manifest,
        "episode": episode_document(episode),
    }
    summary_sha256 = canonical_sha256(payload)
    document = {**payload, "summary_sha256": summary_sha256}
    summary_path = output_root / f"benchmark-demo-{summary_sha256}.json"
    publish_canonical_json_once(summary_path, document)
    return BenchmarkDemoArtifacts(deployment_directory, summary_path, summary_sha256)
