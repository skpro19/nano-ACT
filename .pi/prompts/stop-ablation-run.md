---
description: Stop one launched run from an ablation iteration without touching its sibling runs
argument-hint: "<combo>/<run-name>"
---

Stop the single launched run selected by `$ARGUMENTS`, a run name copied from the
iteration's shared TensorBoard dashboard (for example
`0000/2026-10-12_10-36-35_osc_pose_delta_k10_bs32_lr1e-05_beta0.01_wu0_st100000_img_av-eih_z1_ds_ph_robomimic_can_ph_original_actions_av_eih_s0_l1-ibeb087d354a2`).
This command kills training for exactly one rental and leaves every other run in
the same ablation iteration, all artifacts, and the shared dashboard running as
they are. It never uses the destructive purge path from `vast-purge.md`.

Invocation is explicit authorization to stop only the selected run. Do not
argue, purge, reconcile by hand, resume training, rent replacements, or run the
sweep driver; the driver could advance other combos.

## Parse the selector

1. Accept either `<combo>/<run-name>` (the dashboard form) or a bare
   `<run-name>`. The optional `<combo>` prefix is the four-digit combo ID.
2. Require the trailing iteration marker `-i<12 hex>` in the run name. It
   identifies the owning iteration; reject a selector without it.
3. Never guess a different run. The run name is matched exactly against the
   iteration's recorded `remote_run_name`, so a truncated or stale dashboard copy
   fails closed.

## Resolve and stop

Run exactly one helper with the selector quoted as a single argument:

```bash
uv run --frozen python \
  .pi/prompts/scripts/vast-train/stop_ablation_run.py \
  --run "$RUN_SELECTOR" --wait-seconds 900
```

Keep stdout and stderr visible; do not redirect or suppress them. The helper:

- extracts the iteration suffix, resolves it to exactly one saved iteration, and
  acquires that iteration's nonblocking driver lock, so an active driver blocks
  the stop instead of racing it;
- matches the run name (and the combo prefix when supplied) to exactly one
  launched combo;
- proves that combo's saved watcher is alive and sends TERM to its verified
  runner, then waits for the watcher report and independently confirmed removal;
- leaves the iteration's other combos, other iterations, source configs,
  datasets, local records, S3 artifacts, and the shared TensorBoard dashboard
  untouched.

## Report and failures

- On success, report the combo ID, instance ID, run name, and reconciled status,
  then the iteration directory and its `summary.txt` path. Confirm explicitly
  that the sibling runs were not touched.
- If the iteration suffix matches several iterations, report **every** matching
  full iteration ID and stop: a shared suffix is a name clash and must not be
  resolved by guessing.
- If the run name matches no combo (or the selector is missing the marker),
  report the blocker and the iteration's known `combo=run-name` pairs. Never
  substitute a different run.
- Never print `instance.json`, `known_hosts`, `setup.env` values, credentials, or
  presigned URLs.
- To stop every run in the iteration at once, use `/stop-ablation-runs` instead.
- Do not stop the shared dashboard with `ablation_tensorboard.py stop`; use
  `/tb-s3` for dashboard recovery and stop it separately only when the operator
  explicitly asks.
