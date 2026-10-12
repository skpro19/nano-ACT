---
description: Run the posterior-collapse audit tests for one saved sweep iteration
argument-hint: "<iteration-id> [test-ids...]"
---

Run the posterior-collapse audit suite for the saved sweep iteration `$1`,
optionally restricted to specific tests `${@:2}` (default `1 2 3 4`).

## Arguments

- `$1` — required. The short iteration token `i<12 hex>` of a sweep saved under
  `.vast-train-local/ablations/*/` (for example `i526a92dd6741`); the leading `i`
  is optional.
- `${@:2}` — optional space-separated test ids chosen from `1 2 3 4`. When
  omitted, all four are attempted.

Reject an iteration token that does not match `i?[0-9a-f]{12}`, an unknown test
id, or a missing required argument; never guess.

## What runs

`scripts/utils/posterior-collpase/run_suite.py` enumerates every run recorded by
the iteration, resolves each run's own latest checkpoint (`--step` pins one step
for every run instead), derives the tagged output directory with `layout.py`, and
dispatches each requested test:

| Test | Behaviour |
|---|---|
| 1 | `run_test1.py`, then `plot_test1.py` |
| 2, 3 | no runner implemented yet — reported and skipped |
| 4 | `run_test4.py`, then `plot_test4.py`, only when the resolved checkpoint is the one that runner pins; otherwise reported and skipped |

Do not delete or overwrite an existing output directory. `run_test1.py` resumes
by skipping seed arrays that already exist and stops if the directory's
`checkpoint.sha256` names a different checkpoint.

## Workflow

1. Require `uv` and run from the repository root.
2. Validate `$1` and `${@:2}`.
3. Run the suite and stream the output. It exits non-zero when a run has no
   local checkpoint or a test fails.
4. Report a compact summary: per run, the config tag, the resolved checkpoint,
   and pass/skip/fail per requested test, plus the output directory under
   `debug/posterior-collapse/<iteration>/[<config-tag>/]<stem>/`.

## Reference implementation

```bash
uv run scripts/utils/posterior-collpase/run_suite.py --iteration "$1" --tests ${@:2}
```

- Omit the test ids to attempt all four.
- Add `--dry-run` to print the commands without executing them.
- Add `--step step_000100000` to pin one step for every run instead of each
  run's latest; a run without it is reported and skipped.
