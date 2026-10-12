"""Visualise test 1 (posterior variation) results.

Reads the arrays.npz produced by run_test1.py and writes PNG diagnostics into
every <split>/seed-<n>/ sub-folder of a checkpoint output directory.
"""
from pathlib import Path
import argparse
import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SEEDS = (42, 123)
SPLITS = ('train', 'held-out-candidate')
THRESHOLD_SD = 0.02


def load_folder(folder):
    arrays = np.load(folder / 'arrays.npz')
    metadata = json.loads((folder / 'metadata.json').read_text())
    return arrays, metadata


def kl_to_prior(*, mu, logvar):
    return 0.5 * (mu ** 2 + np.exp(logvar) - logvar - 1).sum(1)


def plot_sd_per_dim(*, folder, arrays, title, seed=42):
    mu, sigma = arrays['mu'], arrays['sigma']
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for axis, value, name in ((axes[0], mu, 'mu'), (axes[1], sigma, 'sigma')):
        sd = np.asarray(value, dtype=np.float64).std(0)
        axis.bar(np.arange(len(sd)), sd, color='steelblue')
        axis.axhline(THRESHOLD_SD, color='crimson', ls='--', lw=1.2,
                     label=f'negligible threshold ({THRESHOLD_SD})')
        axis.axhline(1.0, color='black', ls=':', lw=1.2, label='prior SD (1.0)')
        axis.set_yscale('log')
        axis.set_ylabel(f'SD of {name}')
        axis.legend(loc='upper right', fontsize=8)
        axis.set_title(f'{name}: per-dimension SD across examples '
                       f'(max {sd.max():.3g}, median {np.median(sd):.3g})', fontsize=10)
    axes[-1].set_xlabel('latent dimension')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(folder / '01_sd_per_dim.png', dpi=140)
    plt.close(fig)


def plot_mu_histograms(*, folder, arrays, title):
    mu = np.asarray(arrays['mu'], dtype=np.float64)
    sd = mu.std(0)
    top = np.argsort(sd)[::-1][:4]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for axis, dim in zip(axes.ravel(), top):
        values = mu[:, dim]
        axis.hist(values, bins=40, density=True, color='steelblue', alpha=.8,
                  label='examples (this dim)')
        grid = np.linspace(values.min(), values.max(), 128)
        normal = np.exp(-.5 * ((grid - values.mean()) / (values.std() + 1e-12)) ** 2)
        normal /= normal.sum() * (grid[1] - grid[0])
        axis.plot(grid, normal, color='crimson', lw=1.2, label='fitted normal')
        axis.set_ylabel('density')
        axis.set_title(f'dim {dim}: sd={values.std():.3g} (prior sd=1)', fontsize=9)
        axis.legend(fontsize=7)
    fig.suptitle(f'mu values across examples are almost point masses\n{title}', fontsize=11)
    fig.tight_layout()
    fig.savefig(folder / '02_mu_histograms.png', dpi=140)
    plt.close(fig)


def plot_matched_pairs(*, folder, arrays, title):
    nn_proprio = np.asarray(arrays['nn_proprio_rms'], dtype=np.float64)
    nn_action = np.asarray(arrays['nn_action_rms'], dtype=np.float64)
    mu, sigma, source = arrays['mu'], arrays['sigma'], arrays['nn_source']
    mu_rms = np.sqrt(((mu - mu[source]) ** 2).mean(1))
    sigma_rms = np.sqrt(((sigma - sigma[source]) ** 2).mean(1))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
    for axis, values, name in ((axes[0], mu_rms, 'mu'), (axes[1], sigma_rms, 'sigma')):
        points = axis.scatter(nn_proprio, values, c=nn_action, s=10, cmap='viridis',
                              vmin=nn_action.min(), vmax=nn_action.max())
        axis.axhline(THRESHOLD_SD, color='crimson', ls='--', lw=1.2,
                     label=f'negligible threshold ({THRESHOLD_SD})')
        axis.set_yscale('log')
        axis.set_xlabel('nearest-observation proprio RMS')
        axis.set_title(f'posterior {name} RMS to nearest-observation neighbour '
                       f'(median {np.median(values):.3g})', fontsize=10)
        axis.legend(fontsize=8)
    axes[0].set_ylabel('posterior RMS difference')
    fig.colorbar(points, ax=axes, label='future-action RMS of the pair', fraction=.03)
    fig.suptitle(f'Observation-matched pairs: large action differences, tiny posterior differences\n{title}',
                 fontsize=11)
    fig.savefig(folder / '03_matched_pairs.png', dpi=140)
    plt.close(fig)


def plot_kl(*, folder, arrays, title):
    kl = kl_to_prior(mu=np.asarray(arrays['mu'], dtype=np.float64),
                     logvar=np.asarray(arrays['logvar'], dtype=np.float64))
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.hist(np.maximum(kl, 1e-12), bins=40, color='steelblue', alpha=.85)
    axis.set_xscale('log')
    axis.set_xlabel('KL(q(z|x,a) || N(0, I))  [nats]')
    axis.set_ylabel('examples')
    axis.set_title(f'KL to prior (mean {kl.mean():.3g}, max {kl.max():.3g} nats)\n{title}', fontsize=11)
    fig.tight_layout()
    fig.savefig(folder / '04_kl_to_prior.png', dpi=140)
    plt.close(fig)


def plot_per_demo_mean(*, folder, arrays, title):
    mu, demo = np.asarray(arrays['mu'], dtype=np.float64), arrays['demo']
    names = sorted(set(demo.tolist()), key=lambda name: int(name.split('_')[-1]))
    matrix = np.array([mu[demo == name].mean(0) for name in names])
    limit = np.abs(matrix).max()
    fig, axis = plt.subplots(figsize=(11, 8))
    image = axis.imshow(matrix, aspect='auto', cmap='coolwarm', vmin=-limit, vmax=limit)
    axis.set_xlabel('latent dimension')
    axis.set_ylabel('demonstration')
    axis.set_title(f'Per-demonstration mean mu (max |value| {limit:.3g}; prior sd = 1)\n{title}', fontsize=11)
    fig.colorbar(image, ax=axis, label='mean mu', fraction=.03)
    fig.tight_layout()
    fig.savefig(folder / '05_per_demo_mean_mu.png', dpi=140)
    plt.close(fig)


def plot_folder(*, folder, step):
    arrays, metadata = load_folder(folder)
    title = (f"{metadata['split']} / seed {metadata['seed']} | step {step} | "
             f"k={metadata['k']}, beta={metadata['beta']} | {metadata['posterior_examples']} examples")
    plot_sd_per_dim(folder=folder, arrays=arrays, title=title)
    plot_mu_histograms(folder=folder, arrays=arrays, title=title)
    plot_matched_pairs(folder=folder, arrays=arrays, title=title)
    plot_kl(folder=folder, arrays=arrays, title=title)
    plot_per_demo_mean(folder=folder, arrays=arrays, title=title)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=Path, required=True,
                        help='debug/posterior-collapse/<iteration>/<checkpoint>')
    args = parser.parse_args()
    step = args.checkpoint_dir.name
    folders = [args.checkpoint_dir / 'test-1' / split / f'seed-{seed}'
               for split in SPLITS for seed in SEEDS]
    for folder in folders:
        if not (folder / 'arrays.npz').exists():
            print(f'skip missing {folder}')
            continue
        plot_folder(folder=folder, step=step)
        print(f'plotted {folder}')


if __name__ == '__main__':
    main()
