"""Plot stored paired reconstruction summaries without recomputing statistics."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha256(*, path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_plot(*, fig, output: Path, source: Path, code: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    output.with_suffix('.provenance.json').write_text(json.dumps({
        'input': str(source), 'input_sha256': sha256(path=source),
        'code_sha256': {str(path): sha256(path=path) for path in (code, Path(__file__))},
        'matplotlib_version': matplotlib.__version__, 'graph_sha256': sha256(path=output),
    }, indent=2) + '\n')
    print(f'PLOTTED {output}', flush=True)


def interval_points(*, axis, means, intervals, positions, color='steelblue') -> None:
    """Draw exact percentile limits, even when a mean lies outside its CI."""
    limits = np.asarray(intervals)
    axis.hlines(positions, limits[:, 0], limits[:, 1], color=color, lw=2)
    axis.plot(limits[:, 0], positions, '|', color=color)
    axis.plot(limits[:, 1], positions, '|', color=color)
    axis.plot(means, positions, 'o', color=color)


def reconstruction_values(*, result: dict) -> dict:
    """Own and intervention point estimates share the paired estimand."""
    items = list(result['paired'].values())
    own = np.array([item['own_l1'] for item in items])
    absolute = np.array([item['absolute'] for item in items])
    return {'own': own, 'intervention': own + absolute, 'absolute': absolute,
            'relative': np.array([item['relative'] for item in items]),
            'absolute_ci': np.array([item['absolute_95ci'] for item in items]),
            'relative_ci': np.array([item['relative_95ci'] for item in items])}


def plot_result(*, result: dict, title: str, output: Path, source: Path, code: Path) -> None:
    values = reconstruction_values(result=result)
    labels = [key.replace('_', ' ') + '\n' + item['classification']
              for key, item in result['paired'].items()]
    positions = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(16, max(5, len(labels) * .9)), layout='constrained')
    axes[0].plot(values['own'], positions, 'o', label='Own posterior', color='seagreen')
    axes[0].plot(values['intervention'], positions, 's', label='Intervention', color='darkorange')
    for index, (own, intervention) in enumerate(zip(values['own'], values['intervention'])):
        axes[0].annotate(f'{own:.4g} / {intervention:.4g}', (max(own, intervention), index),
                         xytext=(8, 8), textcoords='offset points', fontsize=8)
    axes[0].margins(x=.3)
    axes[0].set_title('Reconstruction point estimates\nNo marginal CIs computed')
    axes[0].set_xlabel('Normalized valid-step L1 (own / intervention)')
    axes[0].legend(fontsize=9)
    for axis, metric, scale, threshold, xlabel in (
            (axes[1], 'absolute', 1, .002, 'Intervention − own L1'),
            (axes[2], 'relative', 100, 2, 'Relative reconstruction effect (%)')):
        interval_points(axis=axis, means=values[metric] * scale,
                        intervals=values[f'{metric}_ci'] * scale, positions=positions)
        axis.axvspan(-threshold, threshold, color='crimson', alpha=.1, label=f'±{threshold:g} threshold')
        for boundary in (-threshold, threshold):
            axis.axvline(boundary, color='crimson', ls='--', lw=1)
        axis.axvline(0, color='black', lw=.7)
        axis.set_title(f'Paired {metric} effect\nStored bootstrap 95% CI')
        axis.set_xlabel(xlabel)
        axis.legend(fontsize=9)
    for axis in axes:
        axis.set_yticks(positions, labels if axis is axes[0] else [])
        axis.set_ylim(len(labels) - .5, -.5)
        axis.grid(axis='x', alpha=.2)
    fig.suptitle(f'{title}\n{result["targets"]} targets; {result["demonstrations"]} demonstrations; '
                 'equal-demo weights (draws/chunks/seeds averaged within demo)', fontsize=12)
    fig.supxlabel('Positive effect = worse reconstruction. Paired demo-bootstrap CIs conditional on fixed donors; '
                  'not a collapse diagnosis.\nMatched swaps use proprio, not images; held-out-candidate exposure is unverified.', fontsize=9)
    save_plot(fig=fig, output=output, source=source, code=code)


def main(*, test: int, code: Path) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--per-seed', action='store_true', help='Also plot stored by-seed summaries')
    args = parser.parse_args()
    root = args.checkpoint_dir / f'test-{test}'
    source = root / 'results.json'
    results = json.loads(source.read_text())
    for split, entry in results.items():
        title = f'Test {test}: paired reconstruction — {args.checkpoint_dir.parent.name} / {args.checkpoint_dir.name} / {split}'
        plot_result(result=entry['combined'], title=title + ' / combined seeds',
                    output=root / split / '01_paired_reconstruction_combined.png', source=source, code=code)
        if args.per_seed:
            for seed, result in entry['by_seed'].items():
                plot_result(result=result, title=title + f' / seed {seed}',
                            output=root / split / f'seed-{seed}' / '01_paired_reconstruction.png',
                            source=source, code=code)
