#!/usr/bin/env python3
"""Stop every launched run in one saved ablation iteration.

Each nonterminal launched combo is stopped through the watcher-owned
cancellation path in ``cancel_run.py``: the helper proves the saved watcher is
alive, sends TERM to the pinned runner, then waits for the watcher to back up,
report, and remove the rental. It never deletes local records or S3 artifacts,
never fabricates terminal markers, and never touches unlaunched combos. A short
``i<12 hex>`` suffix resolves to exactly one iteration.
"""

import argparse
import signal
import sys

from cancel_run import await_removal, request_cancel
from iteration import FINAL, Iteration, locate
from workflow_common import Blocked, lock, redact


def stop(*, iteration: Iteration, wait_seconds: int) -> int:
    active = [combo for combo in iteration.state['combos']
              if combo.get('training_launch_requested') and combo['status'] not in FINAL]
    if not active:
        print('No nonterminal launched runs to stop')
        return 0
    requested = []
    failures = []
    # Signal every run first so no rental keeps training while another finishes
    # its backup, then wait for each watcher's independently verified removal.
    for combo in active:
        instance_id = combo.get('instance_id')
        if instance_id is None:
            failures.append(f'{combo["id"]}: launched run has no recorded instance ID')
            continue
        try:
            if request_cancel(iteration=iteration, combo=combo, instance_id=instance_id):
                requested.append((combo, instance_id))
        except (Blocked, ValueError, KeyError, TypeError, OSError) as error:
            failures.append(f'{combo["id"]}: {redact(str(error))}')
    for combo, instance_id in requested:
        try:
            await_removal(iteration=iteration, combo=combo, instance_id=instance_id, wait_seconds=wait_seconds)
        except (Blocked, ValueError, KeyError, TypeError, OSError) as error:
            failures.append(f'{combo["id"]}: {redact(str(error))}')
    for failure in failures:
        print(f'Blocked: {failure}', file=sys.stderr)
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', required=True, help='explicit iteration ID, short token, or directory')
    parser.add_argument('--wait-seconds', type=int, default=900)
    args = parser.parse_args()
    if not 0 <= args.wait_seconds <= 3600:
        parser.error('Invalid wait budget')
    def interrupted(signum, frame, /) -> None:
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        directory = locate(identity=args.resume)
        with lock(directory / 'driver.lock'):
            return stop(iteration=Iteration(directory=directory), wait_seconds=args.wait_seconds)
    except KeyboardInterrupt:
        print('Stop interrupted; watchers and instances left untouched')
        return 130
    except (Blocked, ValueError, KeyError, TypeError, OSError) as error:
        print(f'Stop blocked: {redact(str(error))}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
