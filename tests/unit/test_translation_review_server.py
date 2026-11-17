from __future__ import annotations

import threading
from pathlib import Path

import httpx

from apps.translation_review.main import Handler, TranslationReviewServer
from apps.translation_review.workspace import TranslationSourceItem, TranslationWorkspace


def test_server_requires_token_and_persists_translation(tmp_path: Path) -> None:
    source = TranslationSourceItem(
        stable_id="source-1",
        pair_id="pair-1",
        source_content_sha256="1" * 64,
        prompt_text="English fixture",
        context_text=None,
        category="fixture",
        rubric_text="Fixture rubric",
        expected_behavior="refuse_or_safe_redirect",
    )
    server = TranslationReviewServer(("127.0.0.1", 0), Handler)
    server.workspace = TranslationWorkspace(tmp_path / "private", (source,))
    server.token = "test-token"
    server.page = b"fixture page"
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    try:
        with httpx.Client(base_url=base_url) as client:
            assert client.get("/").text == "fixture page"
            assert client.get("/api/state").status_code == 403
            response = client.patch(
                "/api/items/pair-1/translation",
                headers={"X-Translation-App-Token": "test-token"},
                json={
                    "translator_id": "translator-a",
                    "uk_prompt": "Український тест",
                    "uk_image_text": None,
                    "uk_rubric_text": "Українська рубрика",
                },
            )
            assert response.status_code == 200
            assert response.json()["summary"]["translated"] == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
