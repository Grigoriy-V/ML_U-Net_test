# Agent orchestration

`AGENTS.md` is the mandatory project policy; `ML_PROJECT_ROADMAP.md` is the sole source of the current ML plan.

## Roles and lifecycle

The human approves direction and manually launches long training or evaluation commands. The supervisor reads evidence, writes a bounded worker task specification, reviews results, and makes the final `continue`, `stop`, `change`, or `freeze` decision. The worker performs only the approved implementation, investigation, validation, and logging work.

For every repository worker task: (1) supervisor dispatches scope, permitted files/artifacts and commands, stop conditions, reporting requirements, and acceptance criteria; (2) worker runs `python tools/agent_ledger.py start --metadata-json ...`; (3) worker executes approved work and records actual ML operations when applicable; (4) worker runs `terminal --status completed|failed|interrupted` exactly once; (5) only after explicit review, the supervisor runs `review --decision accept|reject|change`. Terminal and review commands inherit static metadata from the matching start; use `--dry-run` to preview and `validate` before a handoff. Worker lifecycle events always set `supervisor_decision` to `null`.

The helper generates UTC timestamps and event IDs, validates before append, locks the ledger, and writes only at EOF. Do not manually use `apply_patch`, in-place insertion, or a text editor to add a ledger record. If the helper cannot safely write, the worker stops, preserves the exact error in its report, and asks the supervisor to record a failure through the safe path; it must not bypass the protection. This ledger is for repository tasks only: personal or private tasks outside the repository never enter it.

## Routing and reasoning policy

| Profile | Model | Default reasoning | Allowed work |
| --- | --- | --- | --- |
| `luna_clerk` | `gpt-5.6-luna` | `none` | Deterministic extraction, status collection, formatting, reporting, and ledger work. No ML decisions or ML operations. |
| `terra_worker` | `gpt-5.6-terra` | `low` | Narrow implementation, targeted tests, standard diagnosis, and approved ML task execution. |
| `sol_specialist` | `gpt-5.6-sol` | `high` | Explicitly approved complex or high-risk work. |

Use the least capable safe profile. A worker may not escalate model/reasoning, change profile, or delegate without supervisor approval. Escalate Luna to Terra when judgment or code changes are required; escalate Terra to Sol only for material complexity, ambiguity, or risk.

The already-open built-in `luna_clerk` launcher can remain pinned to `minimal` independently of the project profile and fail before execution because Luna does not support that level. The project profile uses `none`; a new session or custom-profile reload may be needed before it takes effect. Fall back to `terra_worker` at `low` until then.

## Controls and failures

`.codex/config.toml` limits work to two threads, depth one, 1,800 seconds, and interruption messaging. Use one write-heavy worker for overlapping code, output directories, checkpoint lineages, or artifacts. Long training/evaluation is semi-automatic: worker prepares the command, human launches it, and worker resumes after human-reported completion. Do not autonomously launch a long run.

On validation failure, interruption, timeout, NaN/Inf, OOM, output collision, or scope conflict: stop, preserve evidence, append terminal `failed`/`interrupted`, and report the exact condition. Do not invent recovery work.

## Audit linkage

`reports/agent_execution_ledger.jsonl` is append-only and records `agent_run_id`, requested model/reasoning, scope, commands, changed files, outcomes, and known Git commits. A `correction` event identifies an earlier erroneous event in `notes` or `outcome_summary` and never rewrites it. Fix a validation mismatch by updating the schema and appending a correction event; never rewrite existing JSONL events. Luna profile invocations use requested reasoning `none`. A `reviewed` event is a supervisor-only record after explicit review. Terminal events list actual `ml_ledger_event_ids`. A terminal event committed in its own resulting commit cannot honestly know that commit hash, so `git_commit_after` may remain `null`. Keep paths repository-relative; do not record secrets, absolute local paths, or invented token/credit usage.

`reports/experiment_ledger.jsonl` remains the source of truth for actual ML operations and experiment decisions. The agent ledger records how bounded worker tasks were dispatched and concluded.

Event timestamps must be captured programmatically from system UTC at write time. Agents must never invent, round, backdate, or future-date them. If the exact time is unavailable, use a schema-supported null/unknown value or append a correction event; never fabricate a timestamp.
