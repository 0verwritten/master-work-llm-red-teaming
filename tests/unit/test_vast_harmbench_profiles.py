from harness.bilingual_dataset import HUMAN_VERIFIED, DatasetItem
from harness.config.schema import Provider
from harness.experiments.bilingual_config import build_cell_config
from harness.experiments.bilingual_manifest import BilingualStudyManifest, build_manifest
from harness.experiments.vast_harmbench_profiles import (
    VAST_GPT_OSS_A6000,
    VAST_GPT_OSS_L40S,
    VAST_GPT_OSS_L40S_FROZEN,
    VAST_GPT_OSS_RTX6000_ADA,
    VAST_QWEN_H100,
    VAST_QWEN_H100_FROZEN,
)


def _item() -> DatasetItem:
    return DatasetItem(
        stable_id="harmbench:fixture",
        pair_id="harmbench:fixture",
        language="en",
        stratum="harmbench_subset",
        category="fixture",
        prompt_text="fixture prompt",
        context_text=None,
        rubric_text="fixture rubric",
        expected_behavior="refuse_or_safe_redirect",
        reviewer_status=HUMAN_VERIFIED,
        translation_provenance="source_language",
        license_id="MIT",
    )


def _manifest(namespace: str | None = None) -> BilingualStudyManifest:
    return build_manifest(
        [_item()],
        dataset_sha256="d" * 64,
        subset_plan_sha256="p" * 64,
        attacker=VAST_QWEN_H100_FROZEN,
        target=VAST_QWEN_H100_FROZEN,
        judge=VAST_GPT_OSS_L40S_FROZEN,
        cell_namespace=namespace,
    )


def test_vast_profile_identities_are_frozen() -> None:
    assert VAST_QWEN_H100.profile_id == (
        "vast-qwen36-27b-fp8-h100-v1@"
        "03b7daa7e5012ed12306d10c0c70a5c2347038c537e039cefd0a2690488f2887"
    )
    assert VAST_GPT_OSS_A6000.profile_id == (
        "vast-gpt-oss-20b-a6000-judge-v1@"
        "bba8e42c974597ff451133099e3c680c42e0ccddc9ad943568800b43f96bea70"
    )
    assert VAST_GPT_OSS_L40S.profile_id == (
        "vast-gpt-oss-20b-l40s-judge-v1@"
        "2065f881d5ee6d81ca3271c5b6f05428e6d90742c45753f23b07868ab432d42d"
    )
    assert VAST_GPT_OSS_RTX6000_ADA.gpu_name == "RTX 6000Ada"
    assert VAST_GPT_OSS_RTX6000_ADA.profile_id.startswith("vast-gpt-oss-20b-rtx6000ada-judge-v1@")


def test_vast_profiles_reach_cell_config_without_becoming_modal() -> None:
    manifest = _manifest("vast-fixture")
    config = build_cell_config(manifest, manifest.cells[0])
    assert config.models.attacker.provider == Provider.VAST
    assert config.models.target.provider == Provider.VAST
    assert config.models.judge.provider == Provider.VAST


def test_namespace_prevents_cross_substrate_resume_collision() -> None:
    legacy = _manifest()
    vast = _manifest("vast-fixture")
    assert legacy.cells[0].cell_id != vast.cells[0].cell_id
    assert legacy.manifest_sha256 != vast.manifest_sha256


def test_mistral_attacker_changes_only_the_attacker_checkpoint() -> None:
    from harness.agents.native.modal_provider import MISTRAL_SMALL_32_24B_CANDIDATE
    from harness.experiments.vast_harmbench_profiles import (
        VAST_MISTRAL_SMALL_H100,
        VAST_MISTRAL_SMALL_H100_FROZEN,
    )

    assert VAST_MISTRAL_SMALL_H100.checkpoint_revision == (
        MISTRAL_SMALL_32_24B_CANDIDATE.checkpoint_revision
    )
    assert VAST_MISTRAL_SMALL_H100.container_image == VAST_QWEN_H100.container_image
    assert VAST_MISTRAL_SMALL_H100.gpu_name == VAST_QWEN_H100.gpu_name
    manifest = build_manifest(
        [_item()],
        dataset_sha256="d" * 64,
        subset_plan_sha256="p" * 64,
        attacker=VAST_MISTRAL_SMALL_H100_FROZEN,
        target=VAST_QWEN_H100_FROZEN,
        judge=VAST_GPT_OSS_L40S_FROZEN,
    )
    assert manifest.attacker.identity_sha256 != manifest.target.identity_sha256
    config = build_cell_config(manifest, manifest.cells[0])
    assert config.models.attacker.endpoint_profile == VAST_MISTRAL_SMALL_H100.profile_id
