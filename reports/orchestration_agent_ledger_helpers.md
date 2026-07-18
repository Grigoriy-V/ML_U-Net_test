# Validated agent-ledger helpers

**Status:** completed implementation and validation milestone; no ML operation was performed.

## Delivered

`tools/agent_ledger.py` is the sole project-local CLI for agent-execution events. It provides `start`, `terminal`, `review`, `correction`, and `validate`; `--dry-run` validates and prints without mutating the ledger. The helper generates UUID event IDs and UTC timestamps itself, validates required schema-equivalent fields and safe repository-relative paths, rejects secret-like values and worker decision misuse, and inherits static metadata from a matching start for terminal/review events.

Writes acquire a cross-platform advisory lock (`msvcrt` on Windows, `fcntl` on POSIX), validate before writing, append one UTF-8 JSON line only at EOF, flush, and fsync. Existing historical lifecycle/order irregularities are reported as legacy warnings; they are neither rewritten nor allowed to bypass strict lifecycle checks for the new event.

The `.gitignore` policy now ignores future `.codex/*` state by default while explicitly allowing the reproducible `.codex/config.toml` and `.codex/agents/*.toml` profiles. `.agents/` remains ignored because it is personal/local agent state rather than project configuration.

## Evidence

The targeted no-ML suite was run with:

```powershell
.\.venv\Scripts\python.exe -m unittest tests\test_agent_ledger.py -v
```

Result: `Ran 8 tests` and `OK`. Coverage includes completed/reviewed, failed/interrupted, dry-run non-mutation, invalid input non-mutation, absolute-path and worker-decision rejection, missing/duplicate lifecycle rejection, correction and EOF-byte preservation, concurrent writers, and production legacy-warning validation.

The production helper validation was run with:

```powershell
.\.venv\Scripts\python.exe tools\agent_ledger.py validate
```

It passed for the current production ledger and reported only preserved legacy warnings: two prior lifecycle anomalies and the already documented non-chronological physical event order.

The final no-ML regression command sequence also ran a Draft 2020-12 validation of all `81` pre-terminal production events (`0` errors), parsed all `4` tracked TOML profile/config files, ran `python tools/verify_public_repo.py` (passed: `2` ledgers, `7` public documents, `12` required evidence files, `141` tracked files), checked the `.codex` allowlist, and ran `git diff --check`. `git check-ignore -v` confirmed that synthetic future `.codex/future-private-state.toml` and `.agents/future-private-state.toml` are ignored, while the tracked allowlisted config and profile paths are not ignored.

Final regression checks and their exact results are recorded in the terminal event for this worker run. No experiment-ledger event was created because no material ML operation occurred.

## Operational rule

Repository workers must use this helper for all new agent-ledger records. If it cannot safely append, they must stop and report the failure; manual JSONL insertion or in-place editing is prohibited. Personal/private tasks outside the repository never write this ledger.

## Rework: safety and evidence corrections

The Windows lock is now acquired before any ledger-byte mutation. A forced Windows lock failure on an existing empty ledger leaves its bytes unchanged; the helper uses no sidecar lock file, so there is no stale-lock cleanup artifact. New terminals require explicit `--commands-json` with at least one actual command and explicit `--files-changed-json` (an explicit empty array is allowed for read-only work). Their duration is computed from the matching start timestamp and must be positive. Reviews now require an explicit reviewer identity and parent task instead of inheriting worker identity.

Historical terminal `c44ddd2a-bd53-4427-a893-32eab615fe44` remains unchanged despite its incomplete evidence; it is legacy evidence, not a template for new writes. The rework terminal records real commands and changed files. The misleading reviewed event `b2f06551-7147-43c2-bde6-5ebfd353044f` is corrected append-only to retain its real root-supervisor `change` decision while recording the correct reviewer identity.

## Independent implementation review — 2026-07-19

**Verdict: changes_required.** No ML operation occurred. The targeted suite passes, production helper validation passes with the three preserved legacy warnings, Draft 2020-12 schema validation reports `0` errors for `83` current events, all four tracked TOML files parse, the public verifier passes, and the `.codex` allowlist behaves as documented.

### Findings

- **Medium — Windows lock-acquisition failure can mutate a new empty ledger before the lock is held.** `locked_append` writes and fsyncs a newline when the file is empty before `msvcrt.locking` is acquired (`tools/agent_ledger.py:156-158`). If lock acquisition then fails, the command exits with an error but has changed the ledger, contrary to the claimed no-mutation-on-failure guarantee. Acquire a lock without pre-writing, or use a separate lock file/primitive, then append only after the lock succeeds. Add a Windows-targeted simulated lock-failure/non-mutation test.

- **Low — terminal evidence is supported but not enforced and is only partially covered.** The CLI accepts actual `--commands-json`, `--files-changed-json`, `--duration-seconds`, and required outcome, but defaults commands/files/duration to generic or empty values (`tools/agent_ledger.py:228-229,249-250`). This can lose the actual command, changed-file, and duration evidence required for a meaningful handoff. Require explicit terminal evidence (while allowing intentional empty file lists), and add CLI-level tests proving the recorded terminal event preserves all supplied evidence and that omissions fail clearly.

The review confirms schema-before-append (`tools/agent_ledger.py:178`), EOF append after validation (`tools/agent_ledger.py:181-196`), POSIX locking, per-run duplicate terminal/review protection, metadata inheritance, worker decision/path/secret rejection, and non-mutating dry-run/invalid-event unit coverage. The current tests use temporary ledgers except their read-only production legacy-warning check; they do not mutate the production ledger. No implementation files were altered by this review.

## Independent re-review — 2026-07-19

**Verdict: accept_with_limitations.** The three requested remediations are confirmed; no new finding remains in their approved scope. The only limitations are the explicitly retained, bounded historical warnings: one terminal without a start, one duplicate historical lifecycle, and the earlier non-chronological physical append order. They are reported by `validate`, are not rewritten, and do not permit invalid lifecycle events for new helper-created runs.

### Re-review evidence

- `locked_append` now takes the Windows or POSIX advisory lock before any ledger-byte write (`tools/agent_ledger.py:150-171`). The forced Windows empty-ledger lock-failure test proves byte identity and creates no sidecar file; four concurrent writers still produce an uncorrupted temporary ledger (`tests/test_agent_ledger.py:56-66`).
- Terminal CLI now requires a non-empty actual-command array and an explicit files-changed array; `[]` remains valid for genuinely read-only work. It derives a positive duration from the matching started timestamp (`tools/agent_ledger.py:216-222,236-256`). The rework terminal `c26fd16d-db81-46c2-92e4-7e73e1ce4028` records six actual checks, five changed paths, and duration `127.763728` seconds.
- Review CLI requires explicit reviewer agent/model/reasoning and parent task instead of inheriting the worker (`tools/agent_ledger.py:234-240,257-258`). Correction `edea3665-096d-43fa-bd55-6597fbce597b` accurately records the root-supervisor identity for reviewed event `b2f06551-7147-43c2-bde6-5ebfd353044f` while preserving the original record and its `change` decision append-only.
- Re-ran `./.venv/Scripts/python.exe -m unittest tests/test_agent_ledger.py -v`: `11` tests, `OK`. Tests cover the new forced lock failure, terminal evidence rejection/no mutation/positive duration, review terminal prerequisite and explicit identity, plus existing concurrency and production legacy-warning checks. Production ledger count was unchanged by the suite (`90` before and after).
- Production helper validation passed with only the three bounded legacy warnings; Draft 2020-12 validation found `0` errors for `90` events; four TOML files parsed; public verifier, `.codex` allowlist probes, and `git diff --check` passed.
