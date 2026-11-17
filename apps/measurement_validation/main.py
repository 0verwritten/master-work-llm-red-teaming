from __future__ import annotations

import argparse
import json
import secrets
import subprocess
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from apps.measurement_validation.queue_source import CANONICAL_PAIR_COUNT, load_queue
from apps.measurement_validation.workspace import (
    ADJUDICATIONS,
    BLIND_VERDICTS,
    ValidationWorkspace,
    progress_manifest,
)
from harness.artificial_reviewer import DIMENSIONS

HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_BODY_BYTES = 256 * 1024


def _repository_path(path: Path) -> Path:

    root = Path.cwd().resolve()
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError(f"path escapes the repository: {path}")
    return resolved


class ValidationServer(ThreadingHTTPServer):
    workspace: ValidationWorkspace
    token: str
    page: bytes


class Handler(BaseHTTPRequestHandler):
    server: ValidationServer

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(HTTPStatus.OK, self.server.page, "text/html; charset=utf-8")
            return
        if not self._authorized():
            return
        if self.path == "/api/state":
            self._json(HTTPStatus.OK, self.server.workspace.public_state())
            return
        if self.path == "/api/schema":
            self._json(
                HTTPStatus.OK,
                {
                    "dimensions": list(DIMENSIONS),
                    "verdicts": sorted(BLIND_VERDICTS),
                    "decisions": sorted(ADJUDICATIONS),
                },
            )
            return
        if self.path == "/api/manifest":
            state = self.server.workspace.public_state()
            self._json(HTTPStatus.OK, progress_manifest(state))
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})

    def do_POST(self) -> None:
        if not self._authorized():
            return
        body = self._body()
        if body is None:
            return
        routes = {
            "/api/draft": self._draft,
            "/api/lock": self._lock,
            "/api/reveal": self._reveal,
            "/api/adjudicate": self._adjudicate,
            "/api/correct": self._correct,
            "/api/correction-review": self._correction_review,
        }
        handler = routes.get(self.path)
        if handler is None:
            self._json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
            return
        try:
            handler(body)
        except (KeyError, ValueError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _draft(self, body: dict[str, Any]) -> None:
        state = self.server.workspace.save_blind_draft(
            str(body["pair_id"]),
            reviewer_id=str(body["reviewer_id"]),
            dimensions=dict(body["dimensions"]),
            notes=str(body.get("notes", "")),
            expected_subject_sha256=str(body["subject_sha256"]),
        )
        self._json(HTTPStatus.OK, state)

    def _lock(self, body: dict[str, Any]) -> None:
        state = self.server.workspace.lock_blind_verdict(
            str(body["pair_id"]),
            reviewer_id=str(body["reviewer_id"]),
            dimensions=dict(body["dimensions"]),
            confidence=float(body["confidence"]),
            notes=str(body.get("notes", "")),
            expected_subject_sha256=str(body["subject_sha256"]),
        )
        self._json(HTTPStatus.OK, state)

    def _reveal(self, body: dict[str, Any]) -> None:
        review = self.server.workspace.artificial_review(str(body["pair_id"]))
        self._json(HTTPStatus.OK, json.loads(review.model_dump_json()))

    def _adjudicate(self, body: dict[str, Any]) -> None:
        state = self.server.workspace.record_adjudication(
            str(body["pair_id"]),
            reviewer_id=str(body["reviewer_id"]),
            decision=str(body["decision"]),
            final_dimensions=dict(body["final_dimensions"]),
            rationale=str(body.get("rationale", "")),
            expected_subject_sha256=str(body["subject_sha256"]),
        )
        self._json(HTTPStatus.OK, state)

    def _correct(self, body: dict[str, Any]) -> None:
        state = self.server.workspace.save_correction(
            str(body["pair_id"]),
            corrector_id=str(body["corrector_id"]),
            texts=dict(body["texts"]),
            rationale=str(body.get("rationale", "")),
            expected_subject_sha256=str(body["subject_sha256"]),
        )
        self._json(HTTPStatus.OK, state)

    def _correction_review(self, body: dict[str, Any]) -> None:
        state = self.server.workspace.record_correction_review(
            str(body["pair_id"]),
            reviewer_id=str(body["reviewer_id"]),
            dimensions=dict(body["dimensions"]),
            notes=str(body.get("notes", "")),
            expected_subject_sha256=str(body["subject_sha256"]),
        )
        self._json(HTTPStatus.OK, state)

    def _authorized(self) -> bool:
        if not secrets.compare_digest(
            self.headers.get("X-Validation-App-Token", ""), self.server.token
        ):
            self._json(HTTPStatus.FORBIDDEN, {"error": "invalid local session token"})
            return False
        return True

    def _body(self) -> dict[str, Any] | None:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length > MAX_BODY_BYTES:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request body too large"})
            return None
        try:
            parsed = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "malformed JSON body"})
            return None
        if not isinstance(parsed, dict):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "body must be a JSON object"})
            return None
        return parsed

    def _json(self, status: HTTPStatus, value: Any) -> None:
        body = json.dumps(value, ensure_ascii=False).encode()
        self._send(status, body, "application/json; charset=utf-8")

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
            "connect-src 'self'; frame-ancestors 'none'",
        )
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:

        return


def _load_page(token: str) -> bytes:
    page = Path(__file__).with_name("app.html")
    return page.read_text(encoding="utf-8").replace("__SESSION_TOKEN__", token).encode()


def _open_brave(url: str) -> bool:
    for command in (["brave-browser", f"--app={url}"], ["brave", f"--app={url}"]):
        try:
            subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            continue
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Private measurement-validation app.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--workspace", type=Path, default=Path(".translation-workspace"))
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        raise ValueError("port must be between 1024 and 65535")

    root = _repository_path(args.workspace)
    workspace = ValidationWorkspace(root, load_queue(root, canonical_count=CANONICAL_PAIR_COUNT))
    server = ValidationServer((HOST, args.port), Handler)
    server.workspace = workspace
    server.token = secrets.token_urlsafe(32)
    server.page = _load_page(server.token)
    url = f"http://{HOST}:{args.port}/"
    opened = False if args.no_browser else _open_brave(url)
    state = workspace.public_state()
    print(
        json.dumps(
            {
                "listen_url": url,
                "browser_opened": opened,
                "progress_file": str(workspace.progress_path),
                "summary": state["summary"],
            },
            indent=2,
        )
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
