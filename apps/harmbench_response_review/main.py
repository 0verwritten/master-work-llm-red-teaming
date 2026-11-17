"""Loopback-only interface for blind human response annotation."""

from __future__ import annotations

import argparse
import json
import secrets
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from apps.harmbench_response_review.workspace import ResponseReviewWorkspace

ROOT = Path(__file__).resolve().parent


def _load_page(token: str) -> bytes:
    return (ROOT / "app.html").read_text().replace("__TOKEN__", token).encode()


class ReviewServer(ThreadingHTTPServer):
    workspace: ResponseReviewWorkspace
    token: str


class Handler(BaseHTTPRequestHandler):
    server: ReviewServer

    def _json(self, status: HTTPStatus, value: Any) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _authorized(self) -> bool:
        return secrets.compare_digest(self.headers.get("X-Review-Token", ""), self.server.token)

    def do_GET(self) -> None:
        if self.path == "/":
            page = _load_page(self.server.token)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
            return
        if self.path == "/api/state" and self._authorized():
            self._json(HTTPStatus.OK, self.server.workspace.public_state())
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/api/review" or not self._authorized():
            self._json(HTTPStatus.FORBIDDEN, {"error": "forbidden"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            state = self.server.workspace.save_review(
                str(body.get("annotation_id", "")),
                reviewer_id=str(body.get("reviewer_id", "")),
                label=str(body.get("label", "")),
                confidence=str(body.get("confidence", "")),
                partial_subtype=(
                    str(body["partial_subtype"]) if body.get("partial_subtype") else None
                ),
                notes=str(body.get("notes", "")),
                expected_subject_sha256=str(body.get("subject_sha256", "")),
            )
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        self._json(HTTPStatus.OK, state)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path(".translation-workspace/harmbench-response-review-queue-v1.json"),
    )
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    progress = args.progress or Path(
        f".translation-workspace/harmbench-response-review-{args.reviewer_id}-v1.json"
    )
    server = ReviewServer(("127.0.0.1", args.port), Handler)
    server.workspace = ResponseReviewWorkspace(args.queue, progress, args.reviewer_id)
    server.token = secrets.token_urlsafe(24)
    print(f"Response review: http://127.0.0.1:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
