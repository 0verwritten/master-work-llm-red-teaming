from __future__ import annotations

import argparse
import json
import secrets
import shutil
import subprocess
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from apps.harmbench_uk_review.workspace import (
    REVIEW_DIMENSIONS,
    SELECTIONS,
    VERDICTS,
    ReviewWorkspace,
)

HOST = "127.0.0.1"
DEFAULT_PORT = 8770
MAX_BODY_BYTES = 512 * 1024
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _repository_path(path: Path) -> Path:
    resolved = (REPOSITORY_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if resolved != REPOSITORY_ROOT and REPOSITORY_ROOT not in resolved.parents:
        raise ValueError(f"path escapes the repository: {path}")
    return resolved


class ReviewServer(ThreadingHTTPServer):
    workspace: ReviewWorkspace
    token: str
    page: bytes


class Handler(BaseHTTPRequestHandler):
    server: ReviewServer

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
                    "dimensions": list(REVIEW_DIMENSIONS),
                    "verdicts": list(VERDICTS),
                    "selections": list(SELECTIONS),
                },
            )
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})

    def do_POST(self) -> None:
        if not self._authorized():
            return
        body = self._body()
        if body is None:
            return
        try:
            if self.path == "/api/decision":
                manual = body.get("manual_fields")
                state = self.server.workspace.save_decision(
                    str(body["pair_id"]),
                    reviewer_id=str(body.get("reviewer_id", "")),
                    selection=str(body["selection"]),
                    dimensions={
                        str(key): str(value) for key, value in dict(body["dimensions"]).items()
                    },
                    notes=str(body.get("notes", "")),
                    manual_fields=dict(manual) if isinstance(manual, dict) else None,
                    expected_machine_sha256=str(body["machine_content_sha256"]),
                    expected_proposal_sha256=str(body["proposal_content_sha256"]),
                )
                self._json(HTTPStatus.OK, state)
                return
            if self.path == "/api/rubric-approval":
                state = self.server.workspace.approve_rubric(
                    str(body["rubric_id"]),
                    reviewer_id=str(body.get("reviewer_id", "")),
                    rationale=str(body.get("rationale", "")),
                )
                self._json(HTTPStatus.OK, state)
                return
            if self.path == "/api/finalize":
                path = self.server.workspace.finalize()
                self._json(HTTPStatus.OK, {"status": "finalized", "path": str(path)})
                return
        except (KeyError, ValueError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})

    def _authorized(self) -> bool:
        if not secrets.compare_digest(
            self.headers.get("X-Review-App-Token", ""), self.server.token
        ):
            self._json(HTTPStatus.FORBIDDEN, {"error": "invalid local session token"})
            return False
        return True

    def _body(self) -> dict[str, Any] | None:
        length = int(self.headers.get("Content-Length", "0"))
        if length < 1 or length > MAX_BODY_BYTES:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid request body length"})
            return None
        try:
            value = json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "request body must be JSON"})
            return None
        if not isinstance(value, dict):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "request body must be an object"})
            return None
        return value

    def _json(self, status: HTTPStatus, value: Any) -> None:
        self._send(
            status,
            json.dumps(value, ensure_ascii=False).encode(),
            "application/json; charset=utf-8",
        )

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
    page_path = Path(__file__).with_name("quick_app.html")
    return page_path.read_text().replace("__SESSION_TOKEN__", token).encode()


def _open_brave(url: str) -> bool:
    executable = shutil.which("brave-browser") or shutil.which("brave")
    if executable is not None:
        command = [executable, f"--app={url}", "--new-window"]
    elif (flatpak := shutil.which("flatpak")) is not None:
        command = [flatpak, "run", "com.brave.Browser", f"--app={url}", "--new-window"]
    else:
        return False
    subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Private Ukrainian HarmBench correction review.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--proposal",
        type=Path,
        default=Path(".translation-workspace/harmbench-uk-correction-proposal-v1.json"),
    )
    parser.add_argument(
        "--overlay",
        type=Path,
        default=Path(".translation-workspace/harmbench-uk-human-review-v2.json"),
    )
    parser.add_argument(
        "--legacy-overlay",
        type=Path,
        default=Path(".translation-workspace/harmbench-uk-human-review-v1.json"),
    )
    parser.add_argument(
        "--reviewed-seed",
        type=Path,
        help="exact prior reviewed artifact used only when creating a new overlay",
    )
    parser.add_argument(
        "--reviewed-output",
        type=Path,
        help="separate finalized output path for this review collection",
    )
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        raise ValueError("port must be between 1024 and 65535")
    workspace = ReviewWorkspace(
        _repository_path(args.proposal),
        _repository_path(args.overlay),
        legacy_overlay_path=_repository_path(args.legacy_overlay),
        reviewed_seed_path=(
            _repository_path(args.reviewed_seed) if args.reviewed_seed is not None else None
        ),
        reviewed_output_path=(
            _repository_path(args.reviewed_output) if args.reviewed_output is not None else None
        ),
    )
    server = ReviewServer((HOST, args.port), Handler)
    server.workspace = workspace
    server.token = secrets.token_urlsafe(32)
    server.page = _load_page(server.token)
    url = f"http://{HOST}:{args.port}/"
    opened = False if args.no_browser else _open_brave(url)
    state = workspace.public_state()
    print(
        json.dumps(
            {
                "browser_opened": opened,
                "listen_url": url,
                "proposal_file": str(args.proposal),
                "overlay_file": str(args.overlay),
                "item_count": state["count"],
                "completed": state["summary"]["completed"],
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
