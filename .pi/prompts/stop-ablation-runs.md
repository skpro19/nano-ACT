---
description: Stop every launched run in one ablation iteration without deleting artifacts
argument-hint: "<iteration-id-or-suffix>"
---

Stop all launched runs in the ablation iteration selected by `$ARGUMENTS`. This
command kills training only. It never deletes S3 artifacts, local run records, or
cached TensorBoard events, and it never uses the destructive purge path from
`vast-purge.md`. Only each saved watcher owns backup and rental removal after a
run becomes terminal.

Invocation is explicit authorization to stop the selected iteration's launched
runs. Do not argue, purge, reconcile by hand, resume training, rent replacements,
or run the sweep driver; the driver could advance other combos.

## Select the iteration

1. Require exactly one selector in `$ARGUMENTS`: a saved iteration ID, the short
   token `i<12 hex>` (the leading `i` is optional), or a local iteration
   directory. A suffix shorter than 12 hex is not accepted.
2. Resolve the selector with the driver's `locate` helper, which matches the
   trailing 12 hex of exactly one candidate under `.vast-train-local/ablations/`.
   Never guess and never search by config contents.
3. If the token matches no saved iteration, report the valid iteration IDs and
   stop. If the token matches several iterations, report **every** matching full
   iteration ID and stop: a shared suffix is a name clash and must not be
   resolved by guessing. Rerun only after the operator supplies a full ID or a
   directory that disambiguates it.
4. If `$ARGUMENTS` is empty, report the most recent valid iteration IDs and stop.

## Stop

Run exactly one helper for the selected selector, quoting it as a single
argument:

```bash
uv run --frozen python \
  .pi/prompts/scripts/vast-train/stop_ablation_runs.py \
  --resume "$SELECTOR" --wait-seconds 900
```

Keep stdout and stderr visible; do not redirect or suppress them. The helper:

- acquires the iteration's nonblocking driver lock, so an active driver blocks
  the stop instead of racing it;
- signals **every** nonterminal launched run before waiting on any single
  watcher, so no rental keeps training while another finishes its backup;
- proves each saved watcher is alive and sends TERM to the verified runner, then
  waits for the watcher report and independently confirmed rental removal;
- leaves terminal combos, unlaunched/provisional combos, other iterations,
  source configs, datasets, local records, S3 artifacts, and the shared
  TensorBoard dashboard untouched.

## Report and failures

- On success, report the iteration ID and one line per stopped run with its combo
  ID, instance ID, and reconciled status. Then report the iteration directory and
  `summary.txt` path.
- Report blockers rather than success when the lock is held, the watcher is
  missing, identity checks fail, or a wait budget expires. A `Blocked` line names
  the offending combo; rerun the same helper to finish verifying that watcher.
- Never print `instance.json`, `known_hosts`, `setup.env` values, credentials, or
  presigned URLs.
- A combo that never recorded training launch intent is not a run and is left
  for the driver or `abandon_provisional.py`; do not abandon it here.
- Do not stop the shared dashboard with `ablation_tensorboard.py stop`; use
  `/tb-s3` for dashboard recovery and stop it separately only when the operator
  explicitly asks.
