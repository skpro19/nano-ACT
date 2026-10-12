"""Plot test-4 displacement only, separately for each split/seed."""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import layout

CHECKPOINT = Path('checkpoints/act_v2/2026-10-12_06-37-34_k100_bs32_lr1e-05_beta10_wu0_st100000_img_av-eih_z1_ds_ph_av_eih_ph100_s0_l1-i126b6719b252/step_000022000.pt')
ITERATION = CHECKPOINT.parent.name.rsplit('-', 1)[1]
ROOT = layout.checkpoint_directory(
    output_root=layout.OUTPUT_ROOT, iteration=ITERATION, checkpoint_stem=CHECKPOINT.stem,
    tag=layout.tag_for_run(iteration=ITERATION, run_name=CHECKPOINT.parent.name)) / 'test-4'
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
    axes[0].set_ylim(1e-7, .06)
    axes[1].set_xscale('log')
    axes[1].set_xlim(1e-7, .06)
    axes[1].set_xlabel('Per-demo mean normalized L1 displacement')
    axes[1].set_ylabel('Cumulative demonstration fraction')
    axes[1].set_ylim(0, 1.03)
    axes[1].set_title('Distribution across 100 demonstrations')
    for axis in axes:
        axis.grid(alpha=.2, which='both')
    axes[0].axhline(.002, color='crimson', ls='--', label='Threshold 0.002')
    axes[1].axvline(.002, color='crimson', ls='--', label='Threshold 0.002')
    axes[0].legend(loc='lower right', fontsize=9)
    axes[1].legend(loc='upper left', fontsize=9)
    fig.suptitle(f'Test 4: decoder latent sensitivity — {metadata["split"]}, seed {metadata["seed"]}\n'
                 '200 targets; 5 independent prior draws/target; all 10 distinct pairs; k=100', fontsize=12)
    output = folder / '01_decoder_latent_sensitivity.png'
    fig.savefig(output, dpi=160)
    plt.close(fig)
    (folder / 'plot_provenance.json').write_text(json.dumps({
        'command': 'uv run scripts/utils/posterior-collpase/plot_test4.py',
        'code_sha256': sha256(Path(__file__)), 'matplotlib_version': matplotlib.__version__,
        'input_sha256': {name: sha256(folder / name) for name in ('arrays.npz', 'metadata.json', 'results.json')},
        'graph_sha256': sha256(output)}, indent=2) + '\n')
    print(f'PLOTTED {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for split in ('train', 'held-out-candidate'):
        for seed in (42, 123):
            plot_folder(ROOT / split / f'seed-{seed}')


if __name__ == '__main__':
    main()
