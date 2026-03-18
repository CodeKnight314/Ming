#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DEFAULT_STATE_FILE = ROOT / "reports" / "run_state.json"
DEFAULT_DASHBOARD_FILE = ROOT / "run_state_dashboard.html"
DEFAULT_API_KEYS_FILE = REPO_ROOT / "reports" / "api_keys.txt"
_TAVILY_USAGE_URL = "https://api.tavily.com/usage"
_TAVILY_CACHE_TTL = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a live dashboard for deepresearch-bench run_state.json."
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8123,
        help="Port to bind.",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=DEFAULT_STATE_FILE,
        help="Path to run_state.json.",
    )
    parser.add_argument(
        "--dashboard-file",
        type=Path,
        default=DEFAULT_DASHBOARD_FILE,
        help="Path to the dashboard HTML file.",
    )
    parser.add_argument(
        "--check-interval",
        type=float,
        default=1.0,
        help="Seconds between file change checks for SSE clients.",
    )
    parser.add_argument(
        "--api-keys-file",
        type=Path,
        default=DEFAULT_API_KEYS_FILE,
        help="Path to Tavily API keys file (Label: key per line) for usage polling.",
    )
    return parser.parse_args()


def parse_api_keys(path: Path) -> list[tuple[str, str]]:
    """Return [(label, key), ...] from api_keys.txt format."""
    if not path.exists():
        return []
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            label, _, raw_key = line.partition(":")
            key = raw_key.strip()
        else:
            label = f"key-{len(result) + 1}"
            key = line
        if key and key not in seen:
            seen.add(key)
            result.append((label.strip(), key))
    return result


def fetch_tavily_usage(api_key: str) -> dict[str, Any] | None:
    """Fetch usage from Tavily API. Returns dict or None on failure."""
    try:
        resp = requests.get(
            _TAVILY_USAGE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        try:
            body = resp.json()
        except Exception:
            body = {"detail": resp.text[:500]}
        return {"error": f"HTTP {resp.status_code}", "detail": body}
    try:
        data = resp.json()
        # Dashboard mirrors the batch hotswap logic: `plan_limit` is the total
        # allowance and `plan_usage` is consumed credits. This workflow only
        # uses search credits, so we ignore other usage buckets.
        account = data.get("account") or {}
        raw_plan_usage = account.get("plan_usage", None)
        if raw_plan_usage is None:
            raw_plan_usage = account.get("search_usage", 0)
        plan_usage = int(raw_plan_usage or 0)
        plan_limit = account.get("plan_limit")
        if plan_limit is None:
            return {"usage": plan_usage, "limit": None, "remaining": 999_999}
        limit_int = int(plan_limit)
        return {"usage": plan_usage, "limit": limit_int, "remaining": limit_int - plan_usage}
    except Exception:
        return None


def read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"updated_at": None, "runs": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def stat_signature(path: Path) -> tuple[int, int] | None:
    if not path.exists():
        return None
    stat = path.stat()
    return (stat.st_mtime_ns, stat.st_size)


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler_class: type[BaseHTTPRequestHandler],
        *,
        state_file: Path,
        dashboard_file: Path,
        check_interval: float,
        api_keys_file: Path,
    ) -> None:
        super().__init__(server_address, request_handler_class)
        self.state_file = state_file
        self.dashboard_file = dashboard_file
        self.check_interval = max(0.25, check_interval)
        self.api_keys_file = api_keys_file
        self._tavily_cache: dict[str, Any] | None = None
        self._tavily_cache_at: float = 0
        self._tavily_cache_lock = threading.Lock()

    def get_tavily_usage(self) -> dict[str, Any]:
        """Return Tavily usage for the active key, with short-lived cache."""
        state = read_state(self.state_file)
        tavily_state = state.get("tavily") or {}
        active_label = tavily_state.get("active_key_label")
        keys = parse_api_keys(self.api_keys_file)
        label_to_key = dict(keys)
        if active_label == "env":
            api_key = os.environ.get("TAVILY_API_KEY", "").strip()
            if not api_key:
                api_key = None
        elif keys:
            if not active_label or active_label not in label_to_key:
                active_label = keys[0][0]
            api_key = label_to_key.get(active_label)
        else:
            api_key = os.environ.get("TAVILY_API_KEY", "").strip() or None
            active_label = active_label or "env"

        if not api_key:
            return {
                "error": "No API keys configured (api_keys.txt or TAVILY_API_KEY)",
                "active_key_label": active_label,
                "key_preview": None,
                "usage": None,
                "limit": None,
                "remaining": None,
                "refreshed_at": datetime.now(timezone.utc).isoformat(),
            }

        with self._tavily_cache_lock:
            now = time.monotonic()
            if (
                self._tavily_cache is not None
                and now - self._tavily_cache_at < _TAVILY_CACHE_TTL
                and self._tavily_cache.get("active_key_label") == active_label
            ):
                return {**self._tavily_cache, "refreshed_at": self._tavily_cache.get("refreshed_at")}

        usage_data = fetch_tavily_usage(api_key)
        refreshed_at = datetime.now(timezone.utc).isoformat()
        if usage_data is None:
            result = {
                "error": "Failed to fetch usage",
                "active_key_label": active_label,
                "key_preview": f"{api_key[:16]}…",
                "usage": None,
                "limit": None,
                "remaining": None,
                "refreshed_at": refreshed_at,
            }
        elif "error" in usage_data and ("usage" not in usage_data and "limit" not in usage_data):
            result = {
                "error": usage_data.get("error") or "Failed to fetch usage",
                "detail": usage_data.get("detail"),
                "active_key_label": active_label,
                "key_preview": f"{api_key[:16]}…",
                "usage": None,
                "limit": None,
                "remaining": None,
                "refreshed_at": refreshed_at,
            }
        else:
            result = {
                "error": None,
                "active_key_label": active_label,
                "key_preview": f"{api_key[:16]}…",
                **usage_data,
                "refreshed_at": refreshed_at,
            }
        with self._tavily_cache_lock:
            self._tavily_cache = result
            self._tavily_cache_at = time.monotonic()
        return result


class DashboardHandler(BaseHTTPRequestHandler):
    server: DashboardServer

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route in ("/", "/index.html"):
            self._serve_dashboard()
            return
        if route == "/api/run-state":
            self._serve_state()
            return
        if route == "/api/tavily-usage":
            self._serve_tavily_usage()
            return
        if route == "/events":
            self._serve_events()
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _serve_dashboard(self) -> None:
        html = self.server.dashboard_file.read_text(encoding="utf-8")
        payload = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _serve_state(self) -> None:
        state = read_state(self.server.state_file)
        payload = (json.dumps(state, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _serve_tavily_usage(self) -> None:
        data = self.server.get_tavily_usage()
        payload = (json.dumps(data, ensure_ascii=False) + "\n").encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _serve_events(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        last_signature = object()
        last_ping = time.monotonic()

        try:
            while True:
                current_signature = stat_signature(self.server.state_file)
                if current_signature != last_signature:
                    event = {
                        "signature": current_signature,
                        "updated_at": read_state(self.server.state_file).get("updated_at"),
                    }
                    self.wfile.write(b"event: updated\n")
                    self.wfile.write(f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    last_signature = current_signature
                    last_ping = time.monotonic()
                elif time.monotonic() - last_ping >= 15:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    last_ping = time.monotonic()

                time.sleep(self.server.check_interval)
        except (BrokenPipeError, ConnectionResetError):
            return


def main() -> int:
    args = parse_args()
    server = DashboardServer(
        (args.host, args.port),
        DashboardHandler,
        state_file=args.state_file.resolve(),
        dashboard_file=args.dashboard_file.resolve(),
        check_interval=args.check_interval,
        api_keys_file=args.api_keys_file.resolve(),
    )
    url = f"http://{args.host}:{args.port}"
    print(f"Serving dashboard at {url}")
    print(f"Reading state from {server.state_file}")
    print(f"Tavily keys from {server.api_keys_file}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server...")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
