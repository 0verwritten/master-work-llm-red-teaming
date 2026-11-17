from __future__ import annotations

from pathlib import Path

import pytest

from apps.measurement_validation.main import _load_page, _repository_path
from apps.measurement_validation.workspace import ADJUDICATIONS

PAGE = Path("apps/measurement_validation/app.html")


def test_page_is_self_contained() -> None:

    html = PAGE.read_text(encoding="utf-8")
    for marker in ("http://", "https://", 'src="//'):
        assert marker not in html, f"page references {marker}"


def test_token_is_injected_not_hardcoded() -> None:
    html = PAGE.read_text(encoding="utf-8")
    assert "__SESSION_TOKEN__" in html
    page = _load_page("abc123").decode()
    assert "__SESSION_TOKEN__" not in page
    assert "abc123" in page


def test_page_never_requests_the_proposal_before_locking() -> None:

    html = PAGE.read_text(encoding="utf-8")
    reveal_calls = html.count('api("/api/reveal"')
    assert reveal_calls == 1, "reveal should be reachable from exactly one place"

    assert 'STATE = await api("/api/state")' in html


def test_paths_outside_the_repository_are_refused() -> None:
    with pytest.raises(ValueError, match="escapes the repository"):
        _repository_path(Path("/etc"))


def test_repository_relative_paths_are_accepted() -> None:
    assert _repository_path(Path(".translation-workspace")).name == ".translation-workspace"


def test_server_declares_the_security_headers() -> None:
    source = Path("apps/measurement_validation/main.py").read_text(encoding="utf-8")
    for header in (
        "Content-Security-Policy",
        "X-Content-Type-Options",
        "Referrer-Policy",
        "Cache-Control",
    ):
        assert header in source
    assert 'HOST = "127.0.0.1"' in source
    assert "frame-ancestors 'none'" in source


def test_request_logging_is_suppressed() -> None:

    source = Path("apps/measurement_validation/main.py").read_text(encoding="utf-8")
    assert "def log_message" in source
    assert "return" in source.split("def log_message")[1][:200]


def test_body_size_is_bounded() -> None:
    source = Path("apps/measurement_validation/main.py").read_text(encoding="utf-8")
    assert "MAX_BODY_BYTES" in source
    assert "REQUEST_ENTITY_TOO_LARGE" in source


def test_token_comparison_is_constant_time() -> None:
    source = Path("apps/measurement_validation/main.py").read_text(encoding="utf-8")
    assert "secrets.compare_digest" in source


def test_page_does_not_duplicate_the_schema() -> None:

    from harness.artificial_reviewer import DIMENSIONS

    html = PAGE.read_text(encoding="utf-8")
    for dimension in DIMENSIONS:
        assert f'"{dimension}"' not in html, f"{dimension} hardcoded in the page"
    for decision in ADJUDICATIONS:
        assert f'value="{decision}"' not in html, f"{decision} hardcoded in the page"
    assert 'api("/api/schema")' in html


def test_server_publishes_the_schema() -> None:
    from harness.artificial_reviewer import DIMENSIONS

    source = Path("apps/measurement_validation/main.py").read_text(encoding="utf-8")
    assert "/api/schema" in source
    assert "list(DIMENSIONS)" in source
    assert len(DIMENSIONS) == 5


def test_state_endpoint_carries_no_proposal() -> None:

    import ast

    module = ast.parse(Path("apps/measurement_validation/workspace.py").read_text(encoding="utf-8"))
    function = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == "public_state"
    )
    body = function.body[1:] if ast.get_docstring(function) else function.body
    code = "\n".join(ast.unparse(node) for node in body)
    assert "artificial" not in code.lower()
    assert "_queue.find" not in code, "must not reach into the queue for a proposal"


def test_page_renders_disagreement_against_the_locked_verdict() -> None:

    html = PAGE.read_text(encoding="utf-8")
    assert "disagrees with you on" in html
    assert "agrees with your verdict" in html


def test_decision_control_does_not_use_hidden_radios() -> None:

    html = PAGE.read_text(encoding="utf-8")
    assert '<select id="decision">' in html
    assert 'input[name="decision"]' not in html
    assert 'document.getElementById("decision").value' in html


def test_page_exposes_private_correction_and_rereview_routes() -> None:
    html = PAGE.read_text(encoding="utf-8")
    source = Path("apps/measurement_validation/main.py").read_text(encoding="utf-8")
    for route in ("/api/correct", "/api/correction-review"):
        assert route in html
        assert route in source
    assert "corrector ID different from the reviewer" in html
    assert 'dimGrid("rereview"' in html


def test_state_carries_the_texts_under_review() -> None:

    import ast

    module = ast.parse(Path("apps/measurement_validation/workspace.py").read_text(encoding="utf-8"))
    function = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == "public_state"
    )
    code = "\n".join(ast.unparse(node) for node in function.body)
    for field in ("en_prompt", "uk_prompt", "en_rubric", "uk_rubric"):
        assert field in code, f"{field} missing from the state the page reads"


def test_page_renders_both_sides() -> None:
    html = PAGE.read_text(encoding="utf-8")
    assert "English source" in html
    assert "Ukrainian translation" in html
    assert "English rubric" in html and "Ukrainian rubric" in html


def test_page_explains_each_dimension() -> None:

    html = PAGE.read_text(encoding="utf-8")
    assert "What each dimension asks" in html
    assert "translationese" in html
    assert "not_judgeable" in html


def test_english_sources_are_optional() -> None:

    source = Path("apps/measurement_validation/queue_source.py").read_text(encoding="utf-8")
    block = source.split("def _english_sources")[1].split("\ndef ")[0]
    assert "if not path.exists():" in block
    assert "return {}" in block
