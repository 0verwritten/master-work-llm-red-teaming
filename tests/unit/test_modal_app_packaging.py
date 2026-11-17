from __future__ import annotations

import ast
from pathlib import Path

import pytest


def test_qwen_modal_image_packages_the_harness_profile_dependency() -> None:
    app_path = Path(__file__).parents[2] / "modal_apps" / "vllm_qwen36" / "app.py"
    tree = ast.parse(app_path.read_text(encoding="utf-8"))

    packaged_modules = {
        argument.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_local_python_source"
        for argument in node.args
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
    }

    assert "harness" in packaged_modules


def test_qwen_modal_command_pins_bounded_startup_backends() -> None:
    app_path = Path(__file__).parents[2] / "modal_apps" / "vllm_qwen36" / "app.py"
    source = app_path.read_text(encoding="utf-8")

    assert '"VLLM_DEEP_GEMM_WARMUP"' in source
    assert '"--gdn-prefill-backend"' in source
    assert '"--safetensors-load-strategy"' in source
    assert '"--enforce-eager"' in source


@pytest.mark.parametrize(
    "relative_path",
    (
        "modal_apps/vllm_gemma4/app.py",
        "modal_apps/vllm_mistral32/app.py",
        "modal_apps/vllm_gpt_oss_judge/app.py",
    ),
)
def test_candidate_modal_apps_are_private_and_package_profiles(relative_path: str) -> None:
    app_path = Path(__file__).parents[2] / relative_path
    source = app_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    packaged_modules = {
        argument.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_local_python_source"
        for argument in node.args
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
    }

    assert "harness" in packaged_modules
    assert "unauthenticated=False" in source
    assert "min_containers=PROFILE.min_containers" in source
    assert "max_containers=PROFILE.max_containers" in source
    assert "target_concurrency" not in source
