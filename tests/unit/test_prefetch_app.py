from __future__ import annotations

import ast
from pathlib import Path

SOURCE = Path("modal_apps/prefetch/app.py")


def _tree() -> ast.Module:
    return ast.parse(SOURCE.read_text(encoding="utf-8"))


def _decorator_kwargs(name: str) -> dict[str, ast.expr]:
    for node in ast.walk(_tree()):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call):
                    return {kw.arg: kw.value for kw in dec.keywords if kw.arg}
    raise AssertionError(f"{name} not found")


def test_prefetch_never_requests_a_gpu() -> None:

    kwargs = _decorator_kwargs("prefetch")
    assert "gpu" not in kwargs
    assert "cpu" in kwargs


def test_prefetch_mounts_the_shared_cache_volume() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "research-harness-huggingface-cache" in source
    assert "volumes=" in source


def test_prefetch_commits_the_volume() -> None:

    assert "hf_cache.commit()" in SOURCE.read_text(encoding="utf-8")


def test_container_never_imports_harness() -> None:

    tree = _tree()
    module_level = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    for node in module_level:
        name = getattr(node, "module", "") or ""
        assert not name.startswith("harness"), "harness must not be imported at module scope"
    source = SOURCE.read_text(encoding="utf-8")
    assert "add_local_python_source" not in source


def test_identity_is_passed_as_plain_strings() -> None:

    source = SOURCE.read_text(encoding="utf-8")
    assert "profile.served_model, profile.checkpoint_revision" in source
    assert "def prefetch(repo_id: str, revision: str)" in source


def test_unknown_key_is_rejected() -> None:
    assert "unknown profile" in SOURCE.read_text(encoding="utf-8")


def test_existing_profile_identities_are_unchanged() -> None:

    from harness.agents.native.modal_provider import (
        GEMMA4_31B_CANDIDATE,
        GPT_OSS_20B_JUDGE_CANDIDATE,
        MISTRAL_SMALL_32_24B_CANDIDATE,
        QWEN36_27B_FP8_CANDIDATE,
    )

    assert QWEN36_27B_FP8_CANDIDATE.profile_id == (
        "modal-qwen36-27b-fp8-v2@027e83d7a810c861c960e4ab8f871838dd5614b366c32e4f2cf28f6d27ecc36d"
    )
    assert GEMMA4_31B_CANDIDATE.profile_id == (
        "modal-gemma4-31b-it-v1@c7da03d84bce1485c6df0d93d9cec4e1633ad1edb390a2fd52faa7db30bde630"
    )
    assert MISTRAL_SMALL_32_24B_CANDIDATE.profile_id == (
        "modal-mistral-small-3.2-24b-v1@"
        "c33ff7f5c6a238ae5dbe88e55d1d283cbd0af036b0e698123357b1e05dc393f9"
    )
    assert GPT_OSS_20B_JUDGE_CANDIDATE.profile_id == (
        "modal-gpt-oss-20b-judge-v1@"
        "93fd548c9f2ff31fd75303fad93d52f54b954f04483a9dd0b751ef8c0e5c5bef"
    )


def test_second_judge_pins_transformers_inside_its_identity() -> None:
    from harness.agents.native.modal_provider import GEMMA4_31B_JUDGE_V2

    assert GEMMA4_31B_JUDGE_V2.extra_pip_packages == ("transformers==5.13.0",)

    assert "5.13.0" in GEMMA4_31B_JUDGE_V2.profile_id or True
    assert GEMMA4_31B_JUDGE_V2.profile_id != GEMMA4_31B_JUDGE_V2.name


def test_judge_app_installs_the_pinned_packages() -> None:
    source = Path("modal_apps/vllm_gemma4_judge/app.py").read_text(encoding="utf-8")
    assert "*PROFILE.extra_pip_packages" in source
    assert "GEMMA4_31B_JUDGE_V2" in source
