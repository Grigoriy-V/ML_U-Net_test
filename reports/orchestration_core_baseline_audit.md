# Orchestration Core baseline audit

**Status:** completed audit; no extraction or implementation was performed.

**Scope:** the project-local orchestration system only, before any Core repository or second ML repository exists.

**Evidence basis:** tracked files and the working tree inspected on 2026-07-18 UTC; commands and results are recorded below.

## 1. Git baseline and audit boundary

| Item | Baseline evidence |
| --- | --- |
| Repository / branch | Current repository, `main` |
| `HEAD` | `af6e371805deb72a815783810e789475c81e7b7b` (`docs: close portfolio evidence gaps`) |
| Pre-existing dirty files | `ML_PROJECT_ROADMAP.md`; `reports/agent_execution_ledger.jsonl` |
| Meaning of pre-existing changes | The roadmap migration plan and its worker/reviewer events existed before this audit. They are outside this worker's ownership and were preserved. |
| Untracked files before audit | None reported by `git status --short`. |
| Audit-owned changes | This report and this worker's append-only lifecycle events only. |

Tracked orchestration/validation inventory was obtained with `git ls-files` and `rg --files`; the relevant tracked files are `AGENTS.md`, `.codex/config.toml`, `.codex/agents/*.toml`, `docs/agent_orchestration.md`, `docs/ml_experiment_logging.md`, both ledger JSONL files and schemas, and `tools/verify_public_repo.py`. There is no existing standalone Core package, bootstrap tool, sync tool, or generic agent-ledger writer.

## 2. Inventory and portability disposition

| File or family | Current purpose | Current owner | Classification | Portability action |
| --- | --- | --- | --- | --- |
| `AGENTS.md` | Mandatory project policy; supervisor/worker roles; ML logging rules | Project supervisor | `core` plus project policy | `template` with an adapter section retained locally |
| `.codex/config.toml` | Thread/depth/runtime limits and interruption setting | Project orchestration | `core` | `template` |
| `.codex/agents/luna_clerk.toml` | Deterministic clerical profile, Luna `none` | Project orchestration | `core` | `copy` |
| `.codex/agents/terra_worker.toml` | Default bounded worker, Terra `low` | Project orchestration | `core` | `copy` |
| `.codex/agents/sol_specialist.toml` | Approved complex/high-risk profile, Sol `high` | Project orchestration | `core` | `copy` |
| `docs/agent_orchestration.md` | Lifecycle, routing, failures, human gate, ledger semantics | Project orchestration | `core` | `template` |
| `reports/agent_execution_ledger.schema.json` | Contract for worker and supervisor audit events | Project orchestration | `core` | `copy` |
| `reports/agent_execution_ledger.jsonl` | Project-specific append-only execution history | Project evidence | `project_evidence` | `exclude`; each project starts its own ledger |
| `docs/ml_experiment_logging.md` | Rules for material ML-operation ledger records | ML project process | `core` plus ML adapter guidance | `adapt` into a generic policy plus adapter-specific event contract |
| `reports/experiment_ledger.schema.json` | Diffusion-specific ML event schema | Generative ML adapter | `generative_adapter` | `adapt`; do not copy unchanged to classical ML |
| `reports/experiment_ledger.jsonl` | Historical diffusion experiment evidence | Generative ML evidence | `project_evidence` | `exclude` |
| `tools/verify_public_repo.py` | Public-evidence existence, Markdown-link, JSONL syntax, identity and forbidden-artifact checks | Generative repository readiness | `generative_adapter` | `adapt`; split generic orchestration checks from public-portfolio checks |
| `ML_PROJECT_ROADMAP.md` | Current project direction and migration authority | Project supervisor | `project_evidence` | `template` for a new project, not shared state |
| `PROJECT_LOG.md` and generative reports | Historical ML milestones and decisions | Generative ML adapter | `project_evidence` | `exclude` |
| `.gitignore`, `.venv/`, `.agents/`, datasets, outputs, evaluation | Local dependency/runtime/artifact boundaries | Local environment / generative ML | `local_only/not portable` | Tracked `.codex/config.toml` and `.codex/agents/*.toml` already copy with Git; exclude intentional local runtime/artifact state and define an explicit policy for future ignored configuration files. |

## 3. Current invariants

The following are confirmed by `AGENTS.md`, `.codex/config.toml`, `.codex/agents/*.toml`, and `docs/agent_orchestration.md`:

- A human approves direction and manually launches long training or evaluation. The supervisor specifies bounded work, reviews evidence, and makes the final decision; a worker performs the approved execution.
- Every worker task has one `started` event and exactly one terminal `completed`, `failed`, or `interrupted` event. Worker events set `supervisor_decision` to `null`; a later `reviewed` event belongs only to the supervisor.
- Luna is limited to deterministic clerical work at `gpt-5.6-luna` / `none`; Terra is the default bounded implementation and validation profile at `gpt-5.6-terra` / `low`; Sol requires explicit approval at `gpt-5.6-sol` / `high`.
- Project limits are `max_threads = 2`, `max_depth = 1`, `job_max_runtime_seconds = 1800`, and interruption messaging enabled.
- One write-heavy worker owns overlapping code, output, checkpoint-lineage, or artifact scope. Long training and evaluation remain semi-automatic and human-gated.
- Ledgers are append-only. Corrections are new events; prior JSONL lines are never rewritten. Actual material ML operations additionally require the experiment ledger, project log, and relevant report.
- Timestamps are programmatic UTC; paths are repository-relative; secrets, absolute local paths, and invented token/credit values are prohibited.

## 4. Dependency map

```text
AGENTS.md
  ├─> ML_PROJECT_ROADMAP.md (current project plan)
  ├─> .codex/config.toml + .codex/agents/*.toml (limits and profiles)
  ├─> docs/agent_orchestration.md (lifecycle and failure rules)
  ├─> reports/agent_execution_ledger.schema.json
  │     └─> reports/agent_execution_ledger.jsonl
  └─> reports/experiment_ledger.schema.json
        └─> reports/experiment_ledger.jsonl + PROJECT_LOG.md + generative reports

tools/verify_public_repo.py
  ├─> both JSONL ledgers (JSON syntax and unique IDs only)
  ├─> selected public documents and public-evidence files
  └─> Git tracked-file inventory
```

The orchestration rules are project-local and self-contained as files, but the present validation is not fully self-contained as a reusable Core: `tools/verify_public_repo.py` hard-codes generative portfolio documents and evidence, and neither ledger has a tracked generic writer/validator CLI. Full schema validation presently relies on the project virtual environment's `jsonschema` package, while TOML parsing relies on the active Python runtime.

## 5. Regression baseline: reproducible cheap checks

These are the required no-ML regression checks after extraction or adapter connection. They are intentionally document/configuration checks, not a unit/integration suite.

| Command | Result at baseline |
| --- | --- |
| `.\.venv\Scripts\python.exe -` with `tomllib` over `.codex/config.toml` and all `.codex/agents/*.toml` | Passed: four TOML files parsed. |
| `.\.venv\Scripts\python.exe -` with Draft 2020-12 validation of `reports/agent_execution_ledger.jsonl` | Passed: 72 events, 0 schema errors. |
| `.\.venv\Scripts\python.exe -` with Draft 2020-12 validation of `reports/experiment_ledger.jsonl` | Passed: 30 events, 0 schema errors. |
| `python tools/verify_public_repo.py` | Passed: 2 ledgers, 7 public documents, 12 required evidence files, 141 tracked files checked. |
| `git diff --check` | Passed; only line-ending warnings were emitted for pre-existing dirty files. |
| `git ls-files -s --` over the inventory paths and `Get-FileHash -Algorithm SHA256` | Completed; establishes tracked blob IDs and current content hashes for future comparison. |
| `rg -n '[A-Za-z]:[\\/]'` over policy/config/schema/verifier files | No unapproved absolute path found; the only matches were schema URLs. |
| `rg -n '(?i)(api[_-]?key|secret|password|token)'` over the same files | Policy-only references to forbidden secrets/tokens; no credential value found. |

`tools/verify_public_repo.py` does not perform JSON Schema validation: it verifies JSONL parsing and unique event IDs. The explicit Draft 2020-12 checks above remain required.

## 6. Portability blockers and risks

| Severity | State | Risk / evidence | Consequence and required treatment |
| --- | --- | --- | --- |
| High | Confirmed | Agent-ledger writes are manual append operations; no tracked writer validates an event before append. | A malformed, wrongly timestamped, absolute-path, or wrong-role event can enter history. Build a validated append helper before extraction. |
| High | Confirmed | `reports/experiment_ledger.schema.json` is titled for the mini diffusion project and requires checkpoint/cache/runtime fields. | It cannot be reused unchanged for a classical ML project. Keep it as a generative adapter and design a separate adapter schema. |
| High | Confirmed | `tools/verify_public_repo.py` hard-codes public portfolio documents and generative evidence paths. | Do not treat it as a Core validator; factor generic orchestration checks from repository-readiness checks. |
| Medium | Confirmed | `.gitignore` excludes `.codex/`, `.agents/`, environments, datasets, outputs, evaluation, and model binaries, while the current `.codex/config.toml` and `.codex/agents/*.toml` are already tracked. | A clone includes those existing tracked profiles. The portability risk is that future ignored files under `.codex/` or `.agents/` will not be automatically noticed or added without explicit packaging, force-add, or an ignore-policy change; local environments, datasets, outputs, and evaluation artifacts remain intentionally unported. |
| Medium | Confirmed | Luna has a documented session/reload caveat: an already-open launcher may remain on unsupported `minimal`; project policy says to fall back to Terra `low`. | Portability smoke must test effective profile availability and fallback, not merely parse TOML. |
| Medium | Confirmed | Verification commands and logging examples use Windows PowerShell paths and a project virtual environment. | Core documentation should express platform-neutral semantics and provide platform-specific command wrappers or documented alternatives. |
| Medium | Confirmed | Current roadmap contains a canonical local path, while ledgers prohibit absolute local paths. | A portable template must not copy canonical personal paths into operational ledgers or generated events. |
| Medium | Proposed | A multi-repository supervisor could mix a worker's workdir, Git diff, or ledger target. | Make repository/workdir/ledger target mandatory fields in future task specifications and validate one-repository scope before writes. |
| Medium | Proposed | Sync may overwrite adapter-local rules or leave a partial update when conflicts occur. | Require clean Git, dry-run, Core-owned-file allowlist, stop-on-conflict, validation before commit, and one adapter at a time. |
| Low | Confirmed | Current agent schema permits correction events but has no cross-event lifecycle validation in the schema itself. | The future helper must check `started`/terminal/reviewed linkage and preserve append-only correction semantics. |

No private-data leak was found in the audited policy/config/schema/verifier set. This is a narrow check, not a claim that all historical reports or local artifacts are free of private content.

## 7. Audit conclusion: proposed boundary

**Core candidate:** role lifecycle; task-specification contract; Luna/Terra/Sol routing and fallback; concurrency and human gates; agent-execution event schema, safe writer and lifecycle validation; UTC/path/privacy policy; supervisor review semantics; generic version/manifest and lesson-promotion documentation.

**Generative adapter candidate:** diffusion experiment schema and ledger, ML-operation logging details, checkpoint/cache/VAE/sampling/evaluation rules, `PROJECT_LOG.md`, generative reports, datasets/outputs, and public portfolio evidence verification.

This is a boundary recommendation only. No file has been moved, copied, generalized, or made into a Core repository.

## 8. Recommended next bounded task: validated logging helpers

**Goal:** design and implement only safe project-local helpers for agent-execution events before any Core extraction.

**In scope:** one command/interface for `started`, one for terminal `completed`/`failed`/`interrupted`, one for supervisor `reviewed`, and validation tooling. Inputs must include repository-relative scope, task/run identifiers, requested model/reasoning, constraints, actual commands, changed files, outcome, and actual ML-ledger IDs when applicable. Outputs are schema-valid UTF-8 append-only JSONL events with programmatic UTC timestamps and generated unique event IDs.

**Required stop conditions:** missing or non-relative path; invalid schema; duplicate event ID; incorrect worker/supervisor decision field; terminal event without a matching start; review without terminal evidence; existing lifecycle conflict; unavailable UTC; request to alter old JSONL lines; any request to log fabricated metrics, credits, tokens, or ML activity.

**Acceptance criteria:** invalid events cannot be appended through the normal helper; valid started/terminal/reviewed lifecycle passes schema and lifecycle checks; correction remains append-only; no experiment ledger is changed by a documentation-only smoke; baseline regression checks above pass; the implementation touches only explicitly approved Core-owned files; no automatic commit/push or ML command occurs.

## 9. Explicit non-actions

- No Core repository was created.
- No sklearn/classical ML repository was created.
- No configuration, code, schema, roadmap, `PROJECT_LOG.md`, public documentation, or experiment-ledger entry was changed.
- No unit/integration test suite, benchmark, dataset preparation, cache creation, training, sampling, evaluation, or other ML operation ran.
- No commit or push ran.
