#!/usr/bin/env python3
"""Stop one launched run in a saved ablation iteration.

The selector is the TensorBoard run name, either ``<combo>/<run-name>`` or a
bare ``<run-name>``. The trailing ``-i<12 hex>`` marker identifies the owning
iteration, and the exact recorded run name identifies one combo. The run is
stopped through the watcher-owned cancellation path in ``cancel_run.py``, which
signals only that rental. Every other combo, all artifacts, and the shared
dashboard are left untouched.
"""

import argparse
import re
import signal
import sys

from cancel_run import cancel
from iteration import Iteration, locate
from workflow_common import Blocked, lock, redact

MARKER = re.compile(r'-i([0-9a-f]{12})$')


def split_selector(*, selector: str) -> tuple[str | None, str]:
    """Split the optional TensorBoard ``<combo>/`` prefix from the run name."""
    combo, separator, run_name = selector.partition('/')
    if not separator:
        return None, selector
    if not re.fullmatch(r'[0-9]{4}', combo) or not run_name or '/' in run_name:
        raise Blocked('Run selector must be <combo>/<run-name> or a bare run name')
    return combo, run_name


def select_combo(*, iteration: Iteration, combo_id: str | None, run_name: str) -> dict:
    matches = [combo for combo in iteration.state['combos']
               if combo.get('remote_run_name') == run_name
               and (combo_id is None or combo['id'] == combo_id)]
    if len(matches) != 1:
        known = ", ".join(f"{combo['id']}={combo.get('remote_run_name') or 'not handed off'}"
                          for combo in iteration.state['combos'])
        raise Blocked(f'Run name must match exactly one launched combo (matched {len(matches)}); known: {known}')
    return matches[0]


def stop(*, iteration: Iteration, combo_id: str | None, run_name: str, wait_seconds: int) -> None:
    combo = select_combo(iteration=iteration, combo_id=combo_id, run_name=run_name)
    instance_id = combo.get('instance_id')
    if instance_id is None:
        raise Blocked('Selected run has no recorded instance ID')
    print(f'Stopping combo {combo["id"]} instance {instance_id} ({run_name})')
    cancel(iteration=iteration, instance_id=instance_id, wait_seconds=wait_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='TensorBoard run name <combo>/<run-name> or <run-name>')
    parser.add_argument('--wait-seconds', type=int, default=900)
    args = parser.parse_args()
    if not 0 <= args.wait_seconds <= 3600:
        parser.error('Invalid wait budget')
    def interrupted(signum, frame, /) -> None:
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        combo_id, run_name = split_selector(selector=args.run)
        marker = MARKER.search(run_name)
        if marker is None:
            raise Blocked('Run name must end with the iteration marker -i<12 hex>')
        directory = locate(identity=f'i{marker.group(1)}')
        with lock(directory / 'driver.lock'):
            stop(iteration=Iteration(directory=directory), combo_id=combo_id, run_name=run_name,
                 wait_seconds=args.wait_seconds)
        return 0
    except KeyboardInterrupt:
        print('Stop interrupted; watcher and instance left untouched')
        return 130
    except (Blocked, ValueError, KeyError, TypeError, OSError) as error:
        print(f'Stop blocked: {redact(str(error))}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
