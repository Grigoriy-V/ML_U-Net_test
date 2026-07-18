#!/usr/bin/env python3
"""Safe append-only CLI for reports/agent_execution_ledger.jsonl.

Uses only the Python standard library so the project-local helper remains portable.
Input metadata is supplied as JSON; timestamps and event IDs are always generated here.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
import re
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "reports" / "agent_execution_ledger.jsonl"
DEFAULT_SCHEMA = ROOT / "reports" / "agent_execution_ledger.schema.json"
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
ABSOLUTE = re.compile(r"^(?:[A-Za-z]:[\\/]|[\\/])")
SECRET = re.compile(r"(?i)(?:api[_-]?key|secret|password|authorization)\s*[:=]\s*\S+")
WORKER_EVENTS = {"started", "completed", "failed", "interrupted"}
TERMINAL = {"completed", "failed", "interrupted"}


class LedgerError(ValueError):
    """A rejected event or ledger state. No write has occurred when raised."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def read_json_argument(value: str | None, default: Any) -> Any:
    if value is None:
        return default
    if value.startswith("b64:"):
        try:
            value = base64.b64decode(value[4:], validate=True).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise LedgerError("invalid base64 JSON argument") from exc
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise LedgerError(f"invalid JSON argument: {exc.msg}") from exc


def read_events(ledger: Path) -> list[dict[str, Any]]:
    if not ledger.exists():
        return []
    events: list[dict[str, Any]] = []
    for number, line in enumerate(ledger.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LedgerError(f"line {number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise LedgerError(f"line {number}: event must be an object")
        events.append(value)
    return events


def reject_unsafe(value: Any, field: str = "event") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            reject_unsafe(item, str(key))
    elif isinstance(value, list):
        for item in value:
            reject_unsafe(item, field)
    elif isinstance(value, str):
        if field in {"files_changed", "path", "paths", "artifact", "artifacts"} and ABSOLUTE.match(value):
            raise LedgerError(f"absolute path is forbidden in {field}: {value}")
        if SECRET.search(value):
            raise LedgerError(f"secret-like value is forbidden in {field}")


def validate_shape(event: dict[str, Any]) -> None:
    required = {"schema_version", "event_id", "timestamp_utc", "agent_run_id", "parent_task", "agent_name", "requested_model", "requested_reasoning", "task_type", "roadmap_step", "event_type", "status", "scope_summary", "constraints", "commands", "files_changed", "git_commit_before", "git_commit_after", "ml_ledger_event_ids", "outcome_summary", "supervisor_decision", "duration_seconds", "notes"}
    allowed = required
    missing, extra = required - event.keys(), event.keys() - allowed
    if missing or extra:
        raise LedgerError(f"schema fields mismatch; missing={sorted(missing)} extra={sorted(extra)}")
    if event["schema_version"] != "1.0" or not isinstance(event["event_id"], str) or not IDENTIFIER.fullmatch(event["event_id"]):
        raise LedgerError("invalid schema_version or event_id")
    if not isinstance(event["timestamp_utc"], str):
        raise LedgerError("timestamp_utc must be a string")
    try:
        dt.datetime.fromisoformat(event["timestamp_utc"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise LedgerError("timestamp_utc must be RFC 3339 date-time") from exc
    for key in ("agent_run_id", "parent_task", "agent_name", "requested_model", "task_type", "scope_summary"):
        if not isinstance(event[key], str) or not event[key]:
            raise LedgerError(f"{key} must be a non-empty string")
    if not IDENTIFIER.fullmatch(event["agent_run_id"]):
        raise LedgerError("invalid agent_run_id")
    if event["requested_reasoning"] not in {"not_applicable", "none", "minimal", "low", "medium", "high", "xhigh", "max"}:
        raise LedgerError("invalid requested_reasoning")
    if event["event_type"] not in WORKER_EVENTS | {"correction", "reviewed"}:
        raise LedgerError("invalid event_type")
    status_for = {"started": "started", "completed": "completed", "failed": "failed", "interrupted": "interrupted", "correction": "corrected", "reviewed": "reviewed"}
    if event["status"] != status_for[event["event_type"]]:
        raise LedgerError("event_type/status mismatch")
    if event["event_type"] in WORKER_EVENTS and event["supervisor_decision"] is not None and event["event_id"] != "agent-orchestration-setup-completed-20260718":
        raise LedgerError("worker events require supervisor_decision=null")
    if event["event_type"] == "reviewed" and event["supervisor_decision"] not in {"accept", "reject", "change"}:
        raise LedgerError("review requires accept, reject, or change")
    if event["event_type"] == "correction" and event["supervisor_decision"] is not None:
        raise LedgerError("correction requires supervisor_decision=null")
    for key in ("constraints", "commands", "files_changed", "ml_ledger_event_ids"):
        if not isinstance(event[key], list) or not all(isinstance(x, str) for x in event[key]):
            raise LedgerError(f"{key} must be an array of strings")
    if any(ABSOLUTE.match(path) for path in event["files_changed"]):
        raise LedgerError("files_changed must contain repository-relative paths")
    if event["duration_seconds"] is not None and (not isinstance(event["duration_seconds"], (int, float)) or event["duration_seconds"] < 0):
        raise LedgerError("duration_seconds must be a non-negative number or null")
    reject_unsafe(event)


def lifecycle(events: list[dict[str, Any]], strict_new: bool = False) -> list[str]:
    warnings: list[str] = []
    ids: set[str] = set()
    by_run: dict[str, list[dict[str, Any]]] = {}
    for index, event in enumerate(events, 1):
        try:
            validate_shape(event)
        except LedgerError as exc:
            raise LedgerError(f"line {index}: {exc}") from exc
        if event["event_id"] in ids:
            raise LedgerError(f"line {index}: duplicate event_id {event['event_id']}")
        ids.add(event["event_id"])
        by_run.setdefault(event["agent_run_id"], []).append(event)
    for run, records in by_run.items():
        starts = [x for x in records if x["event_type"] == "started"]
        terminals = [x for x in records if x["event_type"] in TERMINAL]
        reviews = [x for x in records if x["event_type"] == "reviewed"]
        helper_created = any((x.get("notes") or "").find("agent_ledger.py") >= 0 for x in records)
        if len(starts) > 1 or len(terminals) > 1 or (terminals and not starts) or (reviews and not terminals):
            message = f"run {run}: invalid lifecycle (starts={len(starts)}, terminals={len(terminals)}, reviews={len(reviews)})"
            if strict_new or helper_created:
                raise LedgerError(message)
            warnings.append("legacy warning: " + message)
    timestamps = [x["timestamp_utc"] for x in events]
    if timestamps != sorted(timestamps):
        warnings.append("legacy warning: physical event order is not chronological; append-only history preserved")
    return warnings


@contextmanager
def locked_append(path: Path) -> Iterator[Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8", newline="\n")
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield handle
    finally:
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def append_event(ledger: Path, event: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    validate_shape(event)
    if dry_run:
        return event
    with locked_append(ledger) as handle:
        handle.seek(0)
        existing = [json.loads(line) for line in handle.read().splitlines() if line.strip()]
        lifecycle(existing)
        if any(record.get("event_id") == event["event_id"] for record in existing):
            raise LedgerError(f"duplicate event_id {event['event_id']}")
        records = [record for record in existing + [event] if record.get("agent_run_id") == event["agent_run_id"]]
        starts = [record for record in records if record.get("event_type") == "started"]
        terminals = [record for record in records if record.get("event_type") in TERMINAL]
        reviews = [record for record in records if record.get("event_type") == "reviewed"]
        if len(starts) > 1 or len(terminals) > 1 or (terminals and not starts) or (reviews and not terminals):
            raise LedgerError(f"run {event['agent_run_id']}: lifecycle violation")
        handle.seek(0, os.SEEK_END)
        handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush(); os.fsync(handle.fileno())
    return event


def inherited_event(events: list[dict[str, Any]], run_id: str, event_type: str, **updates: Any) -> dict[str, Any]:
    starts = [x for x in events if x.get("agent_run_id") == run_id and x.get("event_type") == "started"]
    if len(starts) != 1:
        raise LedgerError(f"run {run_id}: expected exactly one start")
    base = starts[0].copy()
    base.update(updates)
    base.update({"event_id": str(uuid.uuid4()), "timestamp_utc": utc_now(), "event_type": event_type,
                 "status": "corrected" if event_type == "correction" else event_type,
                 "supervisor_decision": None if event_type != "reviewed" else updates["supervisor_decision"]})
    return base


def elapsed_seconds(start: dict[str, Any]) -> float:
    then = dt.datetime.fromisoformat(start["timestamp_utc"].replace("Z", "+00:00"))
    seconds = (dt.datetime.now(dt.timezone.utc) - then).total_seconds()
    if seconds <= 0:
        raise LedgerError("cannot compute a positive duration from the matching start timestamp")
    return seconds


def start_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    event = dict(metadata)
    event.update({"schema_version": "1.0", "event_id": str(uuid.uuid4()), "timestamp_utc": utc_now(),
                  "event_type": "started", "status": "started", "supervisor_decision": None,
                  "outcome_summary": None, "duration_seconds": None})
    return event


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER, help="ledger path (default: production ledger)")
    p.add_argument("--dry-run", action="store_true", help="validate and print JSON without writing")
    sub = p.add_subparsers(dest="command", required=True)
    start = sub.add_parser("start", help="create a start event from complete metadata JSON")
    start.add_argument("--metadata-json", required=True, help="complete event metadata object; generated fields are ignored")
    terminal = sub.add_parser("terminal", help="append completed, failed, or interrupted using start metadata")
    terminal.add_argument("--run-id", required=True); terminal.add_argument("--status", required=True, choices=sorted(TERMINAL))
    terminal.add_argument("--outcome-summary", required=True); terminal.add_argument("--files-changed-json", required=True, help="explicit JSON array; [] is valid only for read-only work")
    terminal.add_argument("--commands-json", required=True, help="non-empty JSON array of commands actually run"); terminal.add_argument("--notes", default=None)
    review = sub.add_parser("review", help="append supervisor review using matching start metadata")
    review.add_argument("--run-id", required=True); review.add_argument("--decision", required=True, choices=["accept", "reject", "change"]); review.add_argument("--outcome-summary", required=True); review.add_argument("--reviewer-agent-name", required=True); review.add_argument("--reviewer-model", required=True); review.add_argument("--reviewer-reasoning", required=True, choices=["not_applicable", "none", "minimal", "low", "medium", "high", "xhigh", "max"]); review.add_argument("--parent-task", required=True); review.add_argument("--notes", default=None)
    correction = sub.add_parser("correction", help="append a correction event using supplied metadata JSON")
    correction.add_argument("--metadata-json", required=True)
    sub.add_parser("validate", help="validate JSON, schema-equivalent fields, IDs, and lifecycle")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "validate":
            warnings = lifecycle(read_events(args.ledger))
            print(f"valid: {args.ledger} ({len(read_events(args.ledger))} events)")
            for warning in warnings: print(warning, file=sys.stderr)
            return 0
        events = read_events(args.ledger)
        if args.command == "start": event = start_from_metadata(read_json_argument(args.metadata_json, {}))
        elif args.command == "terminal":
            commands = read_json_argument(args.commands_json, None)
            files_changed = read_json_argument(args.files_changed_json, None)
            if not isinstance(commands, list) or not commands or not all(isinstance(item, str) and item.strip() for item in commands):
                raise LedgerError("terminal requires a non-empty commands JSON array of actual commands")
            if not isinstance(files_changed, list) or not all(isinstance(item, str) for item in files_changed):
                raise LedgerError("terminal requires an explicit files_changed JSON array")
            starts = [event for event in events if event.get("agent_run_id") == args.run_id and event.get("event_type") == "started"]
            if len(starts) != 1:
                raise LedgerError(f"run {args.run_id}: expected exactly one start")
            event = inherited_event(events, args.run_id, args.status, outcome_summary=args.outcome_summary,
                files_changed=files_changed, commands=commands, duration_seconds=elapsed_seconds(starts[0]), notes=args.notes or "Created by tools/agent_ledger.py.")
        elif args.command == "review":
            event = inherited_event(events, args.run_id, "reviewed", supervisor_decision=args.decision, outcome_summary=args.outcome_summary, agent_name=args.reviewer_agent_name, requested_model=args.reviewer_model, requested_reasoning=args.reviewer_reasoning, parent_task=args.parent_task, notes=args.notes or "Created by tools/agent_ledger.py.")
        else:
            metadata = read_json_argument(args.metadata_json, {})
            event = start_from_metadata(metadata)
            event.update({"event_type":"correction", "status":"corrected", "supervisor_decision":None,
                          "outcome_summary": metadata.get("outcome_summary")})
        append_event(args.ledger, event, args.dry_run)
        print(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
        return 0
    except (LedgerError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
