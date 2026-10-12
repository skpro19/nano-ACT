---
description: Run the posterior-collapse audit tests for one saved sweep iteration or run
argument-hint: "<iteration-id | run-name> [test-ids...]"
---

Run the posterior-collapse audit suite for the saved sweep iteration `$1`, or for
one run within it, optionally restricted to specific tests `${@:2}` (default
`1 2 3 4`).

## Arguments

- `$1` — required. Either the short iteration token `i<12 hex>` of a sweep saved
  under `.vast-train-local/ablations/*/` (for example `i526a92dd6741`; the leading
  `i` is optional), or a TensorBoard run name. Iteration mode runs every run in
  the sweep; run mode runs only that one run. The TensorBoard run name is
  `<combo-id>/<run-name>` exactly as shown in the dashboard (for example
  `0000/2026-10-12_10-36-35_osc_pose_delta_k10_…-ibeb087d354a2`); the bare
  `<run-name>` is also accepted.
- `${@:2}` — optional space-separated test ids chosen from `1 2 3 4`. When
  omitted, all four are attempted.

Reject an argument that is neither an iteration token matching `i?[0-9a-f]{12}`
nor a run name ending in the iteration token `-i<12 hex>`, an unknown test id, or
a missing required argument; never guess.

## Checkpoint source

Checkpoints are read from the bucket, not from the local mirrors. The saved
iteration only records the sweep's runs; each run's checkpoint lives at
`s3://toy-act/checkpoints/act_v2/<run-name>/`, and the bucket is the source of
truth because local `checkpoints/act_v2` mirrors can be partial. The bucket,
prefix, region and AWS profile default to `toy-act`, `checkpoints/act_v2`,
`ap-south-1` and `toy-pickplace-backup`, and can be overridden with
`TOY_ACT_BUCKET`, `TOY_ACT_CHECKPOINT_PREFIX`, `TOY_ACT_AWS_REGION` and
`TOY_ACT_AWS_PROFILE`.

## What runs

`scripts/utils/posterior-collpase/run_suite.py` enumerates every run recorded by
the iteration (or selects the single run named by `$1`), resolves each run's own
latest checkpoint from the bucket (`--step` pins one step for every run
instead, and must exist in the bucket), downloads the selected checkpoint into
`checkpoints/act_v2/<run-name>/` when the local mirror is missing or stale,
derives the tagged output directory with `layout.py`, and dispatches each
requested test.

Both modes write the same per-run artifacts under
`debug/posterior-collapse/<iteration>/[<config-tag>/]<stem>/`, so a run name and
a full-iteration run produce identical output paths for the same run.

| Test | Behaviour |
|---|---|
| 1 | `run_test1.py`, then `plot_test1.py` |
| 2, 3 | no runner implemented yet — reported and skipped |
| 4 | `run_test4.py`, then `plot_test4.py`, only when the resolved checkpoint is the one that runner pins; otherwise reported and skipped |

Do not delete or overwrite an existing output directory. `run_test1.py` resumes
by skipping seed arrays that already exist and stops if the directory's
`checkpoint.sha256` names a different checkpoint.

## Workflow

1. Require `uv`, run from the repository root, and have AWS access to the
   checkpoint bucket (the default `toy-pickplace-backup` profile).
2. Validate `$1` and `${@:2}`; choose `--iteration "$1"` for an iteration token
   or `--run "$1"` for a run name.
3. Run the suite and stream the output. It exits non-zero when a run has no
   checkpoint in the bucket or a test fails.
4. Report a compact summary: per run, the config tag, the resolved checkpoint,
   and pass/skip/fail per requested test, plus the output directory under
   `debug/posterior-collapse/<iteration>/[<config-tag>/]<stem>/`.

## Reference implementation

```bash
# iteration token: every run in the sweep
uv run scripts/utils/posterior-collpase/run_suite.py --iteration "$1" --tests ${@:2}
# run name: only that run
uv run scripts/utils/posterior-collpase/run_suite.py --run "$1" --tests ${@:2}
```

- Pass exactly one of `--iteration` / `--run`, matching the form of `$1`.
- Omit the test ids to attempt all four.
- Add `--dry-run` to print the commands without executing them.
- Add `--step step_000100000` to pin one step for every run instead of each
  run's latest; a run without that step in the bucket is reported and skipped.
