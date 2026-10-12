---
description: Destroy a Vast instance and delete its local run records
argument-hint: "<instance-id>"
---

Purge the Vast instance specified by `$ARGUMENTS` (one positive integer).
Ask only if the ID is missing; reject invalid arguments.

Invocation is explicit authorization for **manual destruction and local record
deletion**, overriding the watcher-only cleanup and record-retention rules in
`ablate.md` and `vast-train-actv2.md` for this instance. Do not argue, request
another confirmation, resume training, rent replacements, or require handoff.
Final backup is intentionally forfeited.

1. Locate the exact instance's saved run directory and owning iteration under
   `.vast-train-local`. Verify saved ID/label against fresh valid Vast API data.
   Missing or conflicting identity blocks deletion; never guess ownership.
2. Acquire the iteration's nonblocking driver lock. Stop only its saved watcher,
   validating PID, process start ticks, command line, and isolated process group
   before signaling. Do not kill an unrelated process or an active driver.
3. Run `vastai destroy instance <instance-id> -y`, using the configured Vast key
   without printing it. Independently verify absence with a fresh successful,
   valid API response; CLI exit 0 alone is insufficient. If already absent,
   verify absence and proceed. On uncertainty, retain local records and stop.
4. Stop/release only that instance's ownership-verified forwarding sessions and
   lease. If no other started combo shares the iteration, stop its dashboard
   with `ablation_tensorboard.py stop "<iteration-dir>"` before deleting it.
5. Delete only the exact `.vast-train-local/toy-act-<instance-id>/` directory and
   its combo records. Delete the owning iteration directory (including cached
   TensorBoard events) only if no other started combo shares it. Preserve shared
   iteration records/services otherwise. Use explicit validated paths, no globs,
   and never follow symlinks outside `.vast-train-local`.

Leave other instances, S3 artifacts, source configs, and datasets untouched.
Never print credentials or raw instance records. Use bounded commands and report
verified removal plus deleted paths concisely. Report blockers, not success,
when identity, service ownership, locking, or API verification fails.
