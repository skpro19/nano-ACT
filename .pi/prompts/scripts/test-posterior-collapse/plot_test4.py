"""Plot test-4 displacement: combined confidence summary and split/seed diagnostics."""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_reconstruction import interval_points, save_plot

KEYS = ('random_vs_zero', 'random_vs_random', 'posterior_mean_vs_zero')
LABELS = ('Prior draw vs zero', 'Distinct prior draws', 'Posterior mean vs zero')
COLORS = ('steelblue', 'darkorange', 'seagreen')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot_folder(folder):
    result = json.loads((folder / 'results.json').read_text())
    metadata = json.loads((folder / 'metadata.json').read_text())
    assert sha256(folder / 'arrays.npz') == metadata['arrays_sha256']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    with np.load(folder / 'arrays.npz') as arrays:
        for index, (key, label, color) in enumerate(zip(KEYS, LABELS, COLORS)):
            value = result['metrics'][key]
            mean, (low, high) = value['mean'], value['ci95']
            axes[0].errorbar(index, mean, yerr=[[mean - low], [high - mean]],
                             fmt='o', capsize=6, color=color, markersize=7)
            axes[0].annotate(f'{mean:.3g}\n{value["status"]}', (index, high),
                             xytext=(0, 12), textcoords='offset points', ha='center', fontsize=9)
            targets = arrays[key].mean(1, dtype=np.float64)
            demo_values = np.array([targets[arrays['demo'] == name].mean() for name in np.unique(arrays['demo'])])
            ordered = np.sort(demo_values)
            axes[1].step(ordered, np.arange(1, len(ordered) + 1) / len(ordered),
                         where='post', label=label, color=color)
    axes[0].set_xticks(range(3), ('Prior / zero', 'Prior / prior', 'Mean / zero'))
    axes[0].set_xlim(-.5, 2.5)
    axes[0].set_title('Equal-demo mean and bootstrap 95% CI')
    axes[0].set_ylabel('Valid-step normalized prediction L1 displacement')
    axes[0].set_yscale('log')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Per-demo mean normalized L1 displacement')
    axes[1].set_ylabel('Cumulative demonstration fraction')
    axes[1].set_ylim(0, 1.03)
    axes[1].set_title(f'Distribution across {result["demonstrations"]} demonstrations')
    for axis in axes:
        axis.grid(alpha=.2, which='both')
    axes[0].axhline(.002, color='crimson', ls='--', label='Threshold 0.002')
    axes[1].axvline(.002, color='crimson', ls='--', label='Threshold 0.002')
    axes[0].legend(loc='lower right', fontsize=9)
    axes[1].legend(loc='upper left', fontsize=9)
    fig.suptitle(f'Test 4: decoder latent sensitivity — {metadata["split"]}, seed {metadata["seed"]}\n'
                 f'{metadata["target_count"]} targets; 5 prior draws/target; all 10 pairs; k={metadata["k"]}', fontsize=12)
    output = folder / '01_decoder_latent_sensitivity.png'
    fig.savefig(output, dpi=160)
    plt.close(fig)
    (folder / 'plot_provenance.json').write_text(json.dumps({
        'command': 'uv run .pi/prompts/scripts/test-posterior-collapse/plot_test4.py',
        'code_sha256': sha256(Path(__file__)), 'matplotlib_version': matplotlib.__version__,
        'input_sha256': {name: sha256(folder / name) for name in ('arrays.npz', 'metadata.json', 'results.json')},
        'graph_sha256': sha256(output)}, indent=2) + '\n')
    print(f'PLOTTED {output}', flush=True)


def plot_combined(*, root: Path, checkpoint_dir: Path) -> None:
    source = root / 'results.json'
    for split, entry in json.loads(source.read_text()).items():
        result = entry['combined']
        items = [result['metrics'][key] for key in KEYS]
        fig, axis = plt.subplots(figsize=(10, 5), layout='constrained')
        interval_points(axis=axis, means=[item['mean'] for item in items],
                        intervals=[item['ci95'] for item in items], positions=np.arange(3))
        axis.set_yticks(range(3), [f'{label}\n{item["status"]}' for label, item in zip(LABELS, items)])
        axis.set_ylim(2.5, -.5)
        axis.axvline(.002, color='crimson', ls='--', label='Sensitivity threshold 0.002')
        axis.set_xlabel('Normalized valid-step prediction L1 displacement (not reconstruction error)')
        axis.set_title(f'Test 4: decoder latent sensitivity — {checkpoint_dir.parent.name} / {split}\n'
                       f'Combined seeds; {result["targets"]} targets; {result["demonstrations"]} demonstrations\n'
                       'Equal-demo mean and stored demo-bootstrap 95% CI; all 10 prior-draw pairs')
        axis.grid(axis='x', alpha=.2)
        axis.legend()
        save_plot(fig=fig, output=root / split / '02_decoder_sensitivity_combined.png',
                  source=source, code=Path(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.checkpoint_dir / 'test-4'
    plot_combined(root=root, checkpoint_dir=args.checkpoint_dir)
    for metadata in sorted(root.glob('*/seed-*/metadata.json')):
        plot_folder(metadata.parent)


if __name__ == '__main__':
    main()
