# Generative experiment-ledger helper

## Outcome

Implemented a standard-library-only append helper for
`reports/experiment_ledger.jsonl` and routed the comparison evaluator through
its importable API. No training, evaluation, sampling, benchmark, dataset,
cache, checkpoint, or other material ML operation ran.

The production ledger remained byte-identical:

- events: `30`;
- SHA-256:
  `f06d29bd3eae2709781dd1031d68717b33fdf71c270dc3c60eb3ba4fec1e7cb6`.

No experiment-ledger event was appended for this implementation task.

## Contract

- Existing schema-v1 history accepts descriptive legacy event IDs,
  equal/regressing physical timestamps, and fractional seconds longer than six
  digits.
- New callers provide every schema field except `event_id` and
  `timestamp_utc`; attempts to provide either generated field are rejected.
- New events receive a UUIDv4 and system UTC timestamp inside an atomic
  `O_EXCL` sidecar lock.
- The complete history and candidate are validated before one compact UTF-8
  JSON line is written with `O_APPEND`, `os.write`, and `fsync`.
- Pre-write failures remove the helper-created lock and do not mutate the
  ledger. Uncertain write/fsync failures preserve the lock for human
  inspection.
- History and schema are never rewritten. Corrections remain new append-only
  events.
- `mini_diffusion/evaluate_comparison.py` has no direct experiment JSONL append
  and receives the generated event ID from the helper API.

## Commands and results

```powershell
python tools/experiment_ledger.py validate
python tests/test_agent_ledger.py -v
python tests/test_experiment_ledger.py -v
python tests/test_evaluate_comparison_ledger_integration.py -v
```

Results:

- production experiment-ledger validation: `30` events;
- existing agent-helper tests: `11/11`;
- new experiment-helper tests: `8/8`;
- evaluator integration guard: `1/1`.

Decision: ready for supervisor evidence review. No commit or push was made.
