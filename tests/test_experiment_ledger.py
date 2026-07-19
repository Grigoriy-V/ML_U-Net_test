import copy
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import experiment_ledger as ledger_helper


def metadata() -> dict:
    return {
        "schema_version": "1.0",
        "experiment_id": "test-experiment",
        "event_type": "smoke_test",
        "status": "completed",
        "git_commit": None,
        "config_path": None,
        "config_sha256": None,
        "dataset_fingerprint": None,
        "checkpoint_path": None,
        "checkpoint_step": None,
        "checkpoint_sha256": None,
        "exact_command": "python test.py",
        "runtime": {
            "device": "cpu",
            "gpu": None,
            "dtype": None,
            "batch": 1,
            "effective_batch": 1,
            "duration_seconds": 0.1,
        },
        "metrics": {"passed": True},
        "artifacts": {},
        "decision": "continue",
        "decision_reason": "test",
        "notes": "fixture",
    }


class ExperimentLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.ledger = self.root / "ledger.jsonl"

    def tearDown(self):
        self.temporary.cleanup()

    def test_production_history_validates_without_mutation(self):
        production = ROOT / "reports" / "experiment_ledger.jsonl"
        before = production.read_bytes()
        events = ledger_helper.read_events(production)
        self.assertEqual(len(events), 30)
        self.assertEqual(production.read_bytes(), before)
        self.assertEqual(
            hashlib.sha256(before).hexdigest(),
            "f06d29bd3eae2709781dd1031d68717b33fdf71c270dc3c60eb3ba4fec1e7cb6",
        )

    def test_append_preserves_prefix_and_adds_one_uuid_utc_line(self):
        prefix = (ROOT / "reports" / "experiment_ledger.jsonl").read_bytes()
        self.ledger.write_bytes(prefix)
        event = ledger_helper.append_event(metadata(), self.ledger)
        after = self.ledger.read_bytes()
        self.assertTrue(after.startswith(prefix))
        self.assertEqual(after.count(b"\n"), prefix.count(b"\n") + 1)
        self.assertEqual(len(ledger_helper.read_events(self.ledger)), 31)
        self.assertEqual(uuid.UUID(event["event_id"]).version, 4)
        parsed = dt.datetime.fromisoformat(
            event["timestamp_utc"].replace("Z", "+00:00")
        )
        self.assertEqual(parsed.utcoffset(), dt.timedelta(0))
        self.assertFalse((self.root / "ledger.jsonl.lock").exists())

    def test_rejects_caller_generated_fields_and_invalid_candidate(self):
        for key in ("event_id", "timestamp_utc"):
            value = metadata()
            value[key] = "caller-value"
            with self.assertRaises(ledger_helper.LedgerError):
                ledger_helper.append_event(value, self.ledger)
            self.assertFalse(self.ledger.exists())
        invalid = metadata()
        invalid["runtime"]["batch"] = 0
        with self.assertRaises(ledger_helper.LedgerError):
            ledger_helper.append_event(invalid, self.ledger)
        self.assertFalse(self.ledger.exists())
        self.assertFalse((self.root / "ledger.jsonl.lock").exists())

    def test_invalid_history_and_duplicate_ids_fail_without_mutation(self):
        self.ledger.write_text("{broken\n", encoding="utf-8")
        before = self.ledger.read_bytes()
        with self.assertRaises(ledger_helper.LedgerError):
            ledger_helper.append_event(metadata(), self.ledger)
        self.assertEqual(self.ledger.read_bytes(), before)
        self.assertFalse((self.root / "ledger.jsonl.lock").exists())

        production_events = ledger_helper.read_events(
            ROOT / "reports" / "experiment_ledger.jsonl"
        )
        duplicate = copy.deepcopy(production_events[0])
        self.ledger.write_text(
            "\n".join(json.dumps(item) for item in [duplicate, duplicate]) + "\n",
            encoding="utf-8",
        )
        before = self.ledger.read_bytes()
        with self.assertRaises(ledger_helper.LedgerError):
            ledger_helper.append_event(metadata(), self.ledger)
        self.assertEqual(self.ledger.read_bytes(), before)

    def test_legacy_ids_regressing_equal_and_seven_digit_timestamps(self):
        base = {
            **metadata(),
            "event_id": "descriptive-legacy-id",
            "timestamp_utc": "2026-07-18T15:03:08.4518049Z",
        }
        equal = {
            **metadata(),
            "event_id": "another.descriptive_legacy-id",
            "timestamp_utc": "2026-07-18T15:03:08.4518049Z",
        }
        regressing = {
            **metadata(),
            "event_id": "regressing-legacy-id",
            "timestamp_utc": "2026-07-17T00:00:00Z",
        }
        ledger_helper.validate_history([base, equal, regressing])

    def test_existing_lock_refuses_and_prewrite_failure_cleans_lock(self):
        lock = self.root / "ledger.jsonl.lock"
        lock.write_bytes(b"")
        with self.assertRaises(ledger_helper.LedgerError):
            ledger_helper.append_event(metadata(), self.ledger)
        self.assertTrue(lock.exists())
        lock.unlink()
        invalid = metadata()
        invalid["unexpected"] = True
        with self.assertRaises(ledger_helper.LedgerError):
            ledger_helper.append_event(invalid, self.ledger)
        self.assertFalse(lock.exists())
        self.assertFalse(self.ledger.exists())

    def test_uncertain_write_or_fsync_failure_preserves_lock(self):
        with mock.patch.object(
            ledger_helper.os, "write", side_effect=OSError("uncertain write")
        ):
            with self.assertRaises(OSError):
                ledger_helper.append_event(metadata(), self.ledger)
        self.assertTrue((self.root / "ledger.jsonl.lock").exists())

        second_ledger = self.root / "second.jsonl"
        with mock.patch.object(
            ledger_helper.os, "fsync", side_effect=OSError("uncertain fsync")
        ):
            with self.assertRaises(OSError):
                ledger_helper.append_event(metadata(), second_ledger)
        self.assertTrue((self.root / "second.jsonl.lock").exists())
        self.assertEqual(len(second_ledger.read_text().splitlines()), 1)

    def test_cli_validate_and_append_metadata_file(self):
        metadata_path = self.root / "metadata.json"
        metadata_path.write_text(json.dumps(metadata()), encoding="utf-8")
        append = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "experiment_ledger.py"),
                "--ledger",
                str(self.ledger),
                "append",
                "--metadata-file",
                str(metadata_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(append.returncode, 0, append.stderr)
        validate = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "experiment_ledger.py"),
                "--ledger",
                str(self.ledger),
                "validate",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(validate.returncode, 0, validate.stderr)
        self.assertIn("1 events", validate.stdout)


if __name__ == "__main__":
    unittest.main()
