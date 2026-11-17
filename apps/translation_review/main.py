from __future__ import annotations

import argparse
import asyncio
import json
import secrets
import shutil
import subprocess
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from sqlalchemy import select

from apps.translation_review.workspace import TranslationSourceItem, TranslationWorkspace
from harness.benchmark_import import BENCHMARK_NAME, BENCHMARK_VERSION
from harness.persistence.database import create_engine, session_factory
from harness.persistence.models import BenchmarkItem, BenchmarkVersion
from harness.settings import Settings

HOST = "127.0.0.1"
DEFAULT_PORT = 8765
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _repository_path(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(REPOSITORY_ROOT):
        raise ValueError("translation files must remain inside the repository")
    return resolved


async def load_sources(*, split: str) -> tuple[TranslationSourceItem, ...]:
    engine = create_engine(Settings().database_url)
    try:
        async with session_factory(engine)() as session:
            version = await session.scalar(
                select(BenchmarkVersion).where(
                    BenchmarkVersion.name == BENCHMARK_NAME,
                    BenchmarkVersion.version == BENCHMARK_VERSION,
                )
            )
            if version is None:
                raise RuntimeError("draft benchmark version is unavailable")
            items = tuple(
                (
                    await session.scalars(
                        select(BenchmarkItem)
                        .where(
                            BenchmarkItem.benchmark_version_id == version.id,
                            BenchmarkItem.language == "en",
                            BenchmarkItem.split == split,
                            BenchmarkItem.pair_id.is_not(None),
                        )
                        .order_by(BenchmarkItem.stable_id)
                    )
                ).all()
            )
            return tuple(
                TranslationSourceItem(
                    stable_id=item.stable_id,
                    pair_id=item.pair_id or "",
                    source_content_sha256=item.content_sha256,
                    prompt_text=item.prompt_text,
                    context_text=item.context_text,
                    category=item.category,
                    rubric_text=item.rubric_text,
                    expected_behavior=item.expected_behavior,
                )
                for item in items
            )
    finally:
        await engine.dispose()


class TranslationReviewServer(ThreadingHTTPServer):
    workspace: TranslationWorkspace
    token: str
    page: bytes


class Handler(BaseHTTPRequestHandler):
    server: TranslationReviewServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(HTTPStatus.OK, self.server.page, "text/html; charset=utf-8")
            return
        if parsed.path == "/api/state":
            if not self._authorized():
                return
            self._json(HTTPStatus.OK, self.server.workspace.public_state())
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_PATCH(self) -> None:
        if not self._authorized():
            return
        parsed = urlparse(self.path)
        parts = parsed.path.strip("/").split("/")
        if len(parts) != 4 or parts[:2] != ["api", "items"]:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        pair_id = unquote(parts[2])
        try:
            body = self._body()
            if parts[3] == "translation":
                state = self.server.workspace.save_translation(
                    pair_id,
                    translator_id=str(body.get("translator_id", "")),
                    uk_prompt=str(body.get("uk_prompt", "")),
                    uk_image_text=(
                        str(body["uk_image_text"])
                        if body.get("uk_image_text") is not None
                        else None
                    ),
                    uk_rubric_text=str(body.get("uk_rubric_text", "")),
                    translation_method="human",
                )
            elif parts[3] == "review":
                dimensions = body.get("dimensions")
                if not isinstance(dimensions, dict):
                    raise ValueError("review dimensions are required")
                state = self.server.workspace.save_review(
                    pair_id,
                    reviewer_id=str(body.get("reviewer_id", "")),
                    dimensions={str(key): str(value) for key, value in dimensions.items()},
                    review_notes=str(body.get("review_notes", "")),
                )
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            self._json(HTTPStatus.OK, state)
        except (KeyError, ValueError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def do_POST(self) -> None:
        if not self._authorized():
            return
        if urlparse(self.path).path != "/api/finalize":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            path = self.server.workspace.finalize()
            self._json(HTTPStatus.OK, {"status": "finalized", "path": str(path)})
        except ValueError as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _authorized(self) -> bool:
        if not secrets.compare_digest(
            self.headers.get("X-Translation-App-Token", ""), self.server.token
        ):
            self._json(HTTPStatus.FORBIDDEN, {"error": "invalid local session token"})
            return False
        return True

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length < 1 or length > 1_000_000:
            raise ValueError("invalid request body length")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("request body must be an object")
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
    page_path = Path(__file__).with_name("app.html")
    return page_path.read_text().replace("__SESSION_TOKEN__", token).encode()


def _open_brave(url: str) -> bool:
    executable = shutil.which("brave-browser") or shutil.which("brave")
    command: list[str]
    if executable is not None:
        command = [executable, f"--app={url}", "--new-window"]
    elif (flatpak := shutil.which("flatpak")) is not None:
        command = [flatpak, "run", "com.brave.Browser", f"--app={url}", "--new-window"]
    else:
        return False
    subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Private bilingual translation review app.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--split", default="paired_held_out")
    parser.add_argument("--workspace", type=Path, default=Path(".translation-workspace"))
    parser.add_argument("--import-draft", type=Path)
    parser.add_argument("--translator-id")
    parser.add_argument(
        "--translation-method",
        choices=("human", "machine_deepl", "machine_deepl_ai_revised"),
        default="machine_deepl",
    )
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        raise ValueError("port must be between 1024 and 65535")
    sources = asyncio.run(load_sources(split=args.split))
    workspace = TranslationWorkspace(_repository_path(args.workspace), sources)
    if args.import_draft is not None:
        if not args.translator_id:
            raise ValueError("--translator-id is required with --import-draft")
        draft = json.loads(_repository_path(args.import_draft).read_text())
        if not isinstance(draft, list) or not all(isinstance(row, dict) for row in draft):
            raise ValueError("translation draft must be a JSON array of objects")
        workspace.import_translation_draft(
            draft,
            translator_id=args.translator_id,
            translation_method=args.translation_method,
        )
    server = TranslationReviewServer((HOST, args.port), Handler)
    server.workspace = workspace
    server.token = secrets.token_urlsafe(32)
    server.page = _load_page(server.token)
    url = f"http://{HOST}:{args.port}/"
    opened = False if args.no_browser else _open_brave(url)
    print(
        json.dumps(
            {
                "browser_opened": opened,
                "english_sources_file": str(workspace.english_sources_path),
                "item_count": len(sources),
                "listen_url": url,
                "progress_file": str(workspace.progress_path),
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
