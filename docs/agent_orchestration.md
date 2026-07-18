# Agent orchestration

`AGENTS.md` is the mandatory project policy; `ML_PROJECT_ROADMAP.md` is the sole source of the current ML plan.

## Roles and lifecycle

The human approves direction and manually launches long training or evaluation commands. The supervisor reads evidence, writes a bounded worker task specification, reviews results, and makes the final `continue`, `stop`, `change`, or `freeze` decision. The worker performs only the approved implementation, investigation, validation, and logging work.

For every worker task: (1) supervisor dispatches scope, permitted files/artifacts and commands, stop conditions, reporting requirements, and acceptance criteria; (2) worker appends `started`; (3) worker executes approved work and records actual ML operations when applicable; (4) worker appends exactly one terminal `completed`, `failed`, or `interrupted` event; (5) supervisor accepts, rejects, or changes the next action. The supervisor reads the agent ledger only for acceptance, anomalies/failures, or retrospectives.

## Routing and reasoning policy

| Profile | Model | Default reasoning | Allowed work |
| --- | --- | --- | --- |
| `luna_clerk` | `gpt-5.6-luna` | `minimal` | Deterministic extraction, status collection, formatting, reporting, and ledger work. No ML decisions or ML operations. |
| `terra_worker` | `gpt-5.6-terra` | `low` | Narrow implementation, targeted tests, standard diagnosis, and approved ML task execution. |
| `sol_specialist` | `gpt-5.6-sol` | `high` | Explicitly approved complex or high-risk work. |

Use the least capable safe profile. A worker may not escalate model/reasoning, change profile, or delegate without supervisor approval. Escalate Luna to Terra when judgment or code changes are required; escalate Terra to Sol only for material complexity, ambiguity, or risk.

The already-open session may not expose Luna even when its project profile exists. Fall back to `terra_worker` at `low`; a new session or Codex restart may be needed to load project custom agents.

## Controls and failures

`.codex/config.toml` limits work to two threads, depth one, 1,800 seconds, and interruption messaging. Use one write-heavy worker for overlapping code, output directories, checkpoint lineages, or artifacts. Long training/evaluation is semi-automatic: worker prepares the command, human launches it, and worker resumes after human-reported completion. Do not autonomously launch a long run.

On validation failure, interruption, timeout, NaN/Inf, OOM, output collision, or scope conflict: stop, preserve evidence, append terminal `failed`/`interrupted`, and report the exact condition. Do not invent recovery work.

## Audit linkage

`reports/agent_execution_ledger.jsonl` is append-only and records `agent_run_id`, requested model/reasoning, scope, commands, changed files, outcomes, and known Git commits. Terminal events list actual `ml_ledger_event_ids`. Keep paths repository-relative; do not record secrets, absolute local paths, or invented token/credit usage.

`reports/experiment_ledger.jsonl` remains the source of truth for actual ML operations and experiment decisions. The agent ledger records how bounded worker tasks were dispatched and concluded.
