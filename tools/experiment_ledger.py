#!/usr/bin/env python3
"""Validated append-only helper for the generative experiment ledger.

The helper is intentionally standard-library only. Existing schema-v1 history
keeps its descriptive IDs and original timestamp ordering; newly appended
events receive a UUIDv4 and system UTC timestamp inside an exclusive sidecar
lock.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "reports" / "experiment_ledger.jsonl"

FIELDS = {
    "schema_version",
    "event_id",
    "timestamp_utc",
    "experiment_id",
    "event_type",
    "status",
    "git_commit",
    "config_path",
    "config_sha256",
    "dataset_fingerprint",
    "checkpoint_path",
    "checkpoint_step",
    "checkpoint_sha256",
    "exact_command",
    "runtime",
    "metrics",
    "artifacts",
    "decision",
    "decision_reason",
    "notes",
}
GENERATED_FIELDS = {"event_id", "timestamp_utc"}
EVENT_TYPES = {
    "experiment_created",
    "data_preflight",
    "cache_created",
    "smoke_test",
    "benchmark",
    "training",
    "training_milestone",
    "evaluation",
    "decision",
    "experiment_closed",
    "correction",
}
STATUSES = {"completed", "failed", "skipped", "pending"}
DECISIONS = {"continue", "stop", "change", "freeze", None}
HEX64_OR_UNKNOWN = re.compile(r"^(?:[A-Fa-f0-9]{64}|unknown)$")
HEX64 = re.compile(r"^[A-Fa-f0-9]{64}$")
TIMESTAMP = re.compile(
    r"^(?P<base>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?P<fraction>\.\d+)?(?P<zone>Z|[+-]\d{2}:\d{2})$"
)
RUNTIME_FIELDS = {
    "device",
    "gpu",
    "dtype",
    "batch",
    "effective_batch",
    "duration_seconds",
}


class LedgerError(ValueError):
    """Rejected metadata or ledger state; no append occurred."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _nullable_string(event: dict[str, Any], key: str) -> None:
    if event[key] is not None and not isinstance(event[key], str):
        raise LedgerError(f"{key} must be a string or null")


def _validate_timestamp(value: Any) -> None:
    if not isinstance(value, str):
        raise LedgerError("timestamp_utc must be a string")
    match = TIMESTAMP.fullmatch(value)
    if match is None:
        raise LedgerError("timestamp_utc must be an RFC 3339 date-time")
    fraction = match.group("fraction") or ""
    normalized_fraction = fraction[:7] if fraction else ""
    normalized = (
        match.group("base")
        + normalized_fraction
        + ("+00:00" if match.group("zone") == "Z" else match.group("zone"))
    )
    try:
        dt.datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise LedgerError("timestamp_utc must be an RFC 3339 date-time") from exc


def validate_event(event: dict[str, Any]) -> None:
    if not isinstance(event, dict):
        raise LedgerError("event must be an object")
    missing = FIELDS - event.keys()
    extra = event.keys() - FIELDS
    if missing or extra:
        raise LedgerError(
            f"schema fields mismatch; missing={sorted(missing)} extra={sorted(extra)}"
        )
    if event["schema_version"] != "1.0":
        raise LedgerError("schema_version must be 1.0")
    if not isinstance(event["event_id"], str) or not event["event_id"]:
        raise LedgerError("event_id must be a non-empty string")
    _validate_timestamp(event["timestamp_utc"])
    if not isinstance(event["experiment_id"], str) or not event["experiment_id"]:
        raise LedgerError("experiment_id must be a non-empty string")
    if event["event_type"] not in EVENT_TYPES:
        raise LedgerError("invalid event_type")
    if event["status"] not in STATUSES:
        raise LedgerError("invalid status")
    for key in (
        "git_commit",
        "config_path",
        "dataset_fingerprint",
        "checkpoint_path",
        "exact_command",
        "decision_reason",
        "notes",
    ):
        _nullable_string(event, key)
    if event["config_sha256"] is not None and (
        not isinstance(event["config_sha256"], str)
        or HEX64_OR_UNKNOWN.fullmatch(event["config_sha256"]) is None
    ):
        raise LedgerError("config_sha256 must be a 64-hex string, unknown, or null")
    if event["checkpoint_sha256"] is not None and (
        not isinstance(event["checkpoint_sha256"], str)
        or HEX64.fullmatch(event["checkpoint_sha256"]) is None
    ):
        raise LedgerError("checkpoint_sha256 must be a 64-hex string or null")
    if event["checkpoint_step"] is not None and (
        not _is_integer(event["checkpoint_step"]) or event["checkpoint_step"] < 0
    ):
        raise LedgerError("checkpoint_step must be a non-negative integer or null")
    runtime = event["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != RUNTIME_FIELDS:
        raise LedgerError("runtime must be a closed object with all schema fields")
    for key in ("device", "gpu", "dtype"):
        if runtime[key] is not None and not isinstance(runtime[key], str):
            raise LedgerError(f"runtime.{key} must be a string or null")
    for key in ("batch", "effective_batch"):
        if runtime[key] is not None and (
            not _is_integer(runtime[key]) or runtime[key] < 1
        ):
            raise LedgerError(f"runtime.{key} must be a positive integer or null")
    duration = runtime["duration_seconds"]
    if duration is not None and (not _is_number(duration) or duration < 0):
        raise LedgerError("runtime.duration_seconds must be non-negative or null")
    if not isinstance(event["metrics"], dict):
        raise LedgerError("metrics must be an object")
    if not isinstance(event["artifacts"], dict):
        raise LedgerError("artifacts must be an object")
    if event["decision"] not in DECISIONS:
        raise LedgerError("invalid decision")


def read_events(ledger: Path = DEFAULT_LEDGER) -> list[dict[str, Any]]:
    if not ledger.exists():
        return []
    raw = ledger.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise LedgerError("ledger must end with a newline")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LedgerError("ledger must be UTF-8") from exc
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LedgerError(
                f"line {line_number}: invalid JSON: {exc.msg}"
            ) from exc
        try:
            validate_event(event)
        except LedgerError as exc:
            raise LedgerError(f"line {line_number}: {exc}") from exc
        events.append(event)
    validate_history(events)
    return events


def validate_history(events: list[dict[str, Any]]) -> None:
    ids: set[str] = set()
    for index, event in enumerate(events, 1):
        validate_event(event)
        if event["event_id"] in ids:
            raise LedgerError(
                f"line {index}: duplicate event_id {event['event_id']}"
            )
        ids.add(event["event_id"])


def _lock_path(ledger: Path) -> Path:
    return ledger.with_name(ledger.name + ".lock")


def append_event(
    metadata: dict[str, Any],
    ledger: Path = DEFAULT_LEDGER,
) -> dict[str, Any]:
    """Validate history and append one helper-authored event.

    ``metadata`` must contain every schema field except ``event_id`` and
    ``timestamp_utc``. Those fields are rejected when supplied by the caller.
    """
    if not isinstance(metadata, dict):
        raise LedgerError("metadata must be an object")
    if GENERATED_FIELDS & metadata.keys():
        raise LedgerError("event_id and timestamp_utc are helper-generated")
    expected = FIELDS - GENERATED_FIELDS
    missing = expected - metadata.keys()
    extra = metadata.keys() - expected
    if missing or extra:
        raise LedgerError(
            f"metadata fields mismatch; missing={sorted(missing)} extra={sorted(extra)}"
        )

    ledger = Path(ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    lock = _lock_path(ledger)
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise LedgerError(f"ledger lock already exists: {lock.name}") from exc
    os.close(lock_fd)

    write_started = False
    succeeded = False
    descriptor: int | None = None
    try:
        history = read_events(ledger)
        event = {
            **metadata,
            "event_id": str(uuid.uuid4()),
            "timestamp_utc": utc_now(),
        }
        validate_event(event)
        if event["event_id"] in {item["event_id"] for item in history}:
            raise LedgerError("generated duplicate event_id")
        validate_history([*history, event])
        payload = (
            json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        descriptor = os.open(
            ledger,
            os.O_CREAT | os.O_WRONLY | os.O_APPEND,
            0o600,
        )
        write_started = True
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError(f"uncertain partial append: {written}/{len(payload)} bytes")
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        succeeded = True
        lock.unlink()
        return event
    except Exception:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if not write_started and lock.exists():
            lock.unlink()
        raise
    finally:
        if succeeded and descriptor is not None:
            os.close(descriptor)


def _load_metadata(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LedgerError(f"invalid metadata file: {exc}") from exc
    if not isinstance(value, dict):
        raise LedgerError("metadata file must contain an object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    append_parser = subparsers.add_parser("append")
    append_parser.add_argument("--metadata-file", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            events = read_events(args.ledger)
            print(f"valid: experiment ledger ({len(events)} events)")
        else:
            event = append_event(_load_metadata(args.metadata_file), args.ledger)
            print(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
        return 0
    except (LedgerError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
