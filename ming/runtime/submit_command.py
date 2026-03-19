from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import redis

from ming.runtime.contracts import (
    CommandSource,
    RunBatchPayload,
    RunQueryPayload,
    RuntimeCommand,
    RuntimeCommandType,
    runtime_commands_stream_key,
    utc_now_iso,
)
from ming.runtime.emitter import RuntimeEmitter


def build_query_command(
    prompt: str,
    *,
    command_id: str | None = None,
    client_id: str = "cli",
    user: str = "local",
    metadata: dict[str, Any] | None = None,
) -> RuntimeCommand:
    return RuntimeCommand(
        command_id=command_id or f"cmd_{uuid4().hex}",
        type=RuntimeCommandType.RUN_QUERY,
        submitted_at=utc_now_iso(),
        source=CommandSource(kind="tui", client_id=client_id, user=user),
        payload=RunQueryPayload(prompt=prompt.strip(), metadata=metadata or {}),
    )


def build_batch_command(
    items: list[dict[str, Any]],
    *,
    command_id: str | None = None,
    client_id: str = "cli",
    user: str = "local",
) -> RuntimeCommand:
    return RuntimeCommand.from_dict(
        {
            "command_id": command_id or f"cmd_{uuid4().hex}",
            "type": RuntimeCommandType.RUN_BATCH.value,
            "submitted_at": utc_now_iso(),
            "source": {
                "kind": "tui",
                "client_id": client_id,
                "user": user,
            },
            "payload": {
                "mode": "sequential",
                "items": items,
            },
        }
    )


def submit_runtime_command(
    client: redis.Redis,
    command: RuntimeCommand,
    *,
    namespace: str = "runtime",
    stream_maxlen: int | None = None,
) -> str:
    emitter = RuntimeEmitter(client, namespace=namespace, stream_maxlen=stream_maxlen)
    return emitter.append_command(command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit runtime commands to Ming.")
    parser.add_argument("--host", default="127.0.0.1", help="Redis host.")
    parser.add_argument("--port", type=int, default=6379, help="Redis port.")
    parser.add_argument("--db", type=int, default=0, help="Redis database.")
    parser.add_argument("--namespace", default="runtime", help="Runtime namespace prefix.")
    parser.add_argument("--client-id", default="cli", help="Client identifier.")
    parser.add_argument("--user", default="local", help="Submitting user label.")

    subparsers = parser.add_subparsers(dest="command_type", required=True)

    query_parser = subparsers.add_parser("query", help="Submit a single query command.")
    query_parser.add_argument("prompt", help="Prompt to research.")
    query_parser.add_argument(
        "--metadata-json",
        default=None,
        help="Optional JSON object string for payload metadata.",
    )

    batch_parser = subparsers.add_parser("batch", help="Submit a sequential batch command.")
    batch_parser.add_argument(
        "--json-file",
        type=Path,
        required=True,
        help="Path to JSON file containing a list of {id, prompt} objects.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = redis.Redis(host=args.host, port=args.port, db=args.db, decode_responses=True)

    if args.command_type == "query":
        metadata = json.loads(args.metadata_json) if args.metadata_json else {}
        command = build_query_command(
            args.prompt,
            client_id=args.client_id,
            user=args.user,
            metadata=metadata,
        )
    else:
        items = json.loads(args.json_file.read_text(encoding="utf-8"))
        if not isinstance(items, list):
            raise ValueError("Batch JSON file must contain a list of {id, prompt} objects.")
        command = build_batch_command(
            items,
            client_id=args.client_id,
            user=args.user,
        )

    stream_id = submit_runtime_command(client, command, namespace=args.namespace)
    print(
        json.dumps(
            {
                "stream_id": stream_id,
                "command_id": command.command_id,
                "stream_key": runtime_commands_stream_key(args.namespace),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
