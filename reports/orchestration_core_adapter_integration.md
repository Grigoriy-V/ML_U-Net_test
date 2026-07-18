# Core v0.1 generative adapter integration

Core v0.1 is pinned at commit `14b7c2597d1a7e6c57a4ac8c15d3767338c0a27d`.
This adapter remains self-contained: it retains its project-local profiles,
helper and schema, uses no symlink, and has no runtime dependency on the Core
repository. Domain-specific generative rules and append-only ledgers were not
overwritten.

The integration is metadata-only. `orchestration.lock.json` is machine
verifiable: local mode validates all source hashes without a sibling Core;
`--core-root` also checks the recorded commit, version, Core hashes and
relationship semantics. `exact_copy` requires equal hashes; `adapted` and
`project_override` lock both sides but intentionally permit different hashes.
Before any hash is read, the validator rejects symlinks, Windows reparse
points, and linked/reparse parents in either checked root; unresolved link
status fails closed.
The no-ML regression records routing/profile
parsing, source ledger validation, an isolated lifecycle test, public evidence
verification, Core dry-run refusal against an unpinned target, and protected
artifact equality. No dataset, checkpoint, cache, sampling, training,
evaluation, benchmark, or experiment-ledger operation ran.

The next gated step is bootstrap of the separate sklearn adapter, then
umbrella/superchat verification.

## Independent acceptance review — 2026-07-19

### Verdict: changes required

The source adapter stays self-contained and the no-ML regression checks pass,
but `orchestration.lock.json` is not a verifiable Core pin. It records the
correct Core commit (`14b7c2597d1a7e6c57a4ac8c15d3767338c0a27d`) and version,
yet every `managed_files` value is descriptive prose rather than a digest or
an explicit expected relationship. It therefore cannot detect a change in any
claimed source-managed file, cannot establish which Core content was adopted,
and cannot distinguish intentional project-local divergence from drift.

Evidence: source helper SHA-256 values differ from the pinned Core helper and
schema (`tools/agent_ledger.py`: `99e9aff3...` vs `0269109b...`; ledger schema:
`91a381b8...` vs `d77284e9...`), while the lock stores no values with which to
validate either fact. The integration mode says self-contained and no runtime
dependency, which is correct, but that makes explicit source hashes and a
declared Core/source relationship essential rather than optional.

Passed review checks: the source 11-test helper suite passed; source helper
validation passed with only the three existing bounded legacy warnings; public
verification and `git diff --check` passed. Core `sync_core.py` dry-run was
non-mutating and `--apply` refused (exit 2). The source diff from
`2c617fc4ae9a4bf2f5394c11bee5d7c492b69c5c` contains only allowed review/log
ledger changes plus the untracked lock and this report; protected ML configs,
code, experiment ledger, checkpoints, outputs, datasets, and generated ML
artifacts were unchanged. No symlinks or runtime path to Core were found.

No ML operation, training, evaluation, sampling, benchmark, cache creation,
network operation, commit, or push occurred.

## Independent source-pin rework review — 2026-07-19

### Verdict: changes required

The new eight-entry machine-verifiable lock materially improves the pin: local
and explicit Core checks passed, exact-copy equality is checked, and the 14
targeted tests passed. However, `tools/validate_core_pin.py` accepts a managed
source file that is a symlink escaping the adapter root when the target bytes
match the recorded SHA-256. In an isolated temporary adapter containing the
eight managed paths, `AGENTS.md` was a symlink to the real source file outside
that temporary root; `validate(lock, temporary_root)` returned successfully.
`Path.is_file()` and hashing follow that symlink, while the validator never
checks `resolve()` containment or rejects reparse points.

This violates the stated self-contained adapter boundary and the required
symlink-escape rejection. Do not accept the pin until managed source and Core
paths are required to resolve inside their declared roots (or symlinks are
rejected outright), before hashing.

Other evidence: the 11 source helper tests and 3 pin tests passed; local pin
validation passed; explicit pinned-Core validation passed; source-as-non-Core
failed with `core hash mismatch`; source helper validation passed with only
the three pre-existing legacy warnings; public verification and `git diff
--check` passed. The Core remained at clean commit
`14b7c2597d1a7e6c57a4ac8c15d3767338c0a27d`.

## Final path-security re-review — 2026-07-19

### Verdict: accept

The reworked validator now rejects linked/reparse files and parent components
before hashing and verifies resolved containment. All 15 targeted tests passed
(11 agent-ledger and 4 core-pin tests). Independent fixtures reproduced the
prior external `AGENTS.md` symlink and it was rejected as `linked/reparse
file`; a linked `.codex` parent was rejected as `linked/reparse parent`; and a
Core-side managed-file symlink was rejected. Lexical traversal and absolute
paths were also rejected.

Valid local and explicit pinned-Core modes passed; a source/non-Core root
failed with `core hash mismatch`. Source ledger validation passed with only
three preserved legacy warnings; public verification, `git diff --check`,
Core sync dry-run, and unsafe apply refusal passed. The source diff from
`2c617fc4ae9a4bf2f5394c11bee5d7c492b69c5c` remains limited to orchestration
implementation/evidence and lifecycle records, with no experiment ledger, ML
code/config, checkpoints, outputs, datasets, or generated ML artifacts
changed. Core remained clean at `14b7c2597d1a7e6c57a4ac8c15d3767338c0a27d`.

No ML, network, commit, or push operation occurred.
