"""Test 4 only: fixed-observation decoder latent sensitivity, not reconstruction.

Reuse gather/tensor/valid_l1 from the validated refined audit, and hashing/JSON
helpers from test 1. No donor selection, reconstruction, or tests 2/3 are run.
"""
from pathlib import Path
import argparse
import itertools
import json
import platform
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import h5py
import numpy as np
import torch

from scripts.dataset import images_to_tensor
from scripts.models.act_v2.model import ACTV2
from scripts.utils.analyze_z import normalization_from_checkpoint
from audit_helpers import tensor, valid_l1, verify_resume
from run_test1 import SEEDS, file_hash, save_json
from interventions import arguments, prepare, load_source, read_target
import layout

PAIRS = np.array(list(itertools.combinations(range(5), 2)))
METRICS = ('random_vs_zero', 'random_vs_random', 'posterior_mean_vs_zero')
THRESHOLD = .002


def summarize(*, arrays):
    """Average draws first, then chunks/seeds within demo; equal demo weights."""
    demos = np.concatenate([array['demo'] for array in arrays])
    unique = np.unique(demos)
    draws = np.random.default_rng(2026).integers(len(unique), size=(5000, len(unique)))
    result = {'demonstrations': len(unique), 'targets': len(demos), 'metrics': {}}
    for key in METRICS:
        targets = np.concatenate([array[key].mean(axis=1, dtype=np.float64) for array in arrays])
        means = np.array([targets[demos == name].mean() for name in unique])
        ci = np.quantile(means[draws].mean(1), [.025, .975])
        status = 'negligible' if ci[1] < THRESHOLD else 'meaningful' if ci[0] > THRESHOLD else 'uncertain'
        result['metrics'][key] = {'mean': float(means.mean()), 'ci95': ci.tolist(), 'status': status}
    return result


def report(*, result, label):
    lines = ['# Test 4 — decoder latent sensitivity', '', label, '',
             'Fixed observations; normalized valid-step prediction L1 displacement, **not reconstruction error**.',
             'Five independent N(0,I) draws per target; random–random uses all 10 unordered pairs (i<j).',
             'Average draws/pairs within targets, then chunks and seeds within demonstrations; equal demo weights.',
             '5000 demonstration bootstrap replicates (seed 2026), percentile 95% CI.',
             'Threshold 0.002: upper CI below = negligible; lower CI above = meaningful; otherwise uncertain.', '',
             '| Comparison | Mean normalized L1 | 95% CI | Classification |',
             '|---|---:|---|---|']
    for key, value in result['metrics'].items():
        lines.append(f"| {key} | {value['mean']:.8g} | [{value['ci95'][0]:.8g}, {value['ci95'][1]:.8g}] | {value['status']} |")
    lines += ['', f"{result['targets']} targets across {result['demonstrations']} demonstrations.", '',
              'Sensitivity to prior noise is separate from encoder collapse: a decoder may respond to random z even',
              'when posterior means barely vary. This test does not establish action-specific latent information',
              'or reconstruction utility. Held-out-candidate exposure outside the documented split is unverified.']
    return '\n'.join(lines) + '\n'


@torch.inference_mode()
def execute_seed(*, model, checkpoint, root, dataset, split, seed, folder, identity):
    started = time.time()
    source, meta, original = load_source(root=root, checkpoint=checkpoint, identity=identity,
                                         split=split, seed=seed, dataset=dataset)
    execution = {**identity, 'split': split, 'seed': seed,
                 'test1_arrays_sha256': file_hash(source / 'arrays.npz'),
                 'test1_metadata_sha256': file_hash(source / 'metadata.json')}
    if verify_resume(destination=folder, identity=execution):
        with np.load(folder / 'arrays.npz') as saved:
            return {key: saved[key].copy() for key in saved.files}
    selected = original['selected']
    demos = original['demo'][selected]
    timesteps = original['timestep'][selected]
    own_mu = original['mu'][selected]
    valid_steps = original['valid_steps'][selected]
    count = len(selected)
    k = checkpoint['config']['action_chunk_size']
    normalization = normalization_from_checkpoint(checkpoint=checkpoint)
    # CPU generator: each (target, draw, dimension) is an independently drawn value.
    generator = torch.Generator(device='cpu').manual_seed(seed)
    latents = torch.randn((count, 5, model.z_dims), generator=generator, dtype=torch.float32).numpy()
    assert all(np.all(np.any(latents[:, i] != latents[:, j], axis=1)) for i, j in PAIRS)
    predictions, masks, mus = [], [], []
    differences = {key: [] for key in METRICS}
    with h5py.File(dataset) as hdf5:
        for start in range(0, count, 8):
            positions = range(start, min(start + 8, count))
            rows = [read_target(hdf5=hdf5, demo=demos[p], timestep=timesteps[p],
                                config=checkpoint['config'], normalization=normalization) for p in positions]
            proprio, action, mask = [np.stack([row[i] for row in rows]) for i in range(3)]
            images = torch.stack([row[3] for row in rows]).cuda().float()
            prop = tensor(proprio)[:, None]
            mask_tensor = tensor(mask)
            mu, _ = model.posterior(proprio=prop, actions=tensor(action), mask=mask_tensor)
            np.testing.assert_allclose(mu[:, 0].cpu().numpy(), own_mu[start:start + len(rows)], rtol=1e-4, atol=2e-6)
            mus.append(mu[:, 0].cpu().numpy())
            img_tokens = model.image_encoder(images)
            prop_tokens = model.proprio_encoder(prop)

            def decode(z):
                return model.decode_from_tokens(img_tokens=img_tokens, proprio_tokens=prop_tokens, z=z)

            zero = decode(torch.zeros_like(mu))
            mean = decode(tensor(own_mu[start:start + len(rows)])[:, None])
            random = [decode(tensor(latents[start:start + len(rows), draw])[:, None]) for draw in range(5)]
            pred = torch.stack([zero, mean, *random], dim=1)
            assert pred.dtype == torch.float32 and torch.isfinite(pred).all()
            predictions.append(pred.cpu().numpy())
            masks.append(mask)
            differences['random_vs_zero'].append(torch.stack([
                valid_l1(left=value, right=zero, mask=mask_tensor) for value in random], 1).cpu().numpy())
            differences['random_vs_random'].append(torch.stack([
                valid_l1(left=random[i], right=random[j], mask=mask_tensor) for i, j in PAIRS], 1).cpu().numpy())
            differences['posterior_mean_vs_zero'].append(valid_l1(left=mean, right=zero, mask=mask_tensor)[:, None].cpu().numpy())
        predictions, masks = np.concatenate(predictions), np.concatenate(masks)
        differences = {key: np.concatenate(value) for key, value in differences.items()}
        np.testing.assert_array_equal(masks.sum(1), valid_steps)
        lengths = np.array([len(hdf5[f'data/{name}/next_obs/robot0_joint_pos']) for name in demos])
        np.testing.assert_array_equal(masks, np.arange(k)[None] < np.minimum(k, lengths - timesteps)[:, None])
        # Independent selection stream, includes padded and full chunks explicitly.
        checks = list(np.random.default_rng(seed + 9841).choice(count, size=min(3, count), replace=False))
        for eligible in (valid_steps < k, valid_steps == k):
            if eligible.any():
                checks.append(int(np.flatnonzero(eligible)[0]))
        validation = []
        for pos in sorted(set(checks)):
            proprio, action, mask, frames = read_target(hdf5=hdf5, demo=demos[pos], timestep=timesteps[pos],
                                                        config=checkpoint['config'], normalization=normalization)
            zs = np.concatenate([np.zeros_like(own_mu[pos])[None], own_mu[pos][None], latents[pos]])
            direct = np.stack([model.decode(proprio=tensor(proprio)[None, None], img=frames[None].cuda().float(),
                                           z=tensor(z)[None, None], use_z=True)[0].cpu().numpy() for z in zs])
            maximum = float(np.abs(direct - predictions[pos]).max())
            np.testing.assert_allclose(direct, predictions[pos], rtol=1e-4, atol=2e-5)
            validation.append({'target_position': int(pos), 'test1_index': int(selected[pos]), 'demo': str(demos[pos]),
                               'timestep': int(timesteps[pos]), 'valid_steps': int(valid_steps[pos]),
                               'all_7_direct_decodes_max_abs_difference': maximum})
    # Independent float64 slicing implementation; poisoned padding must not matter.
    for key in METRICS:
        recomputed, poisoned = [], []
        comparisons = [(i + 2, 0) for i in range(5)] if key == 'random_vs_zero' else [
            (i + 2, j + 2) for i, j in PAIRS] if key == 'random_vs_random' else [(1, 0)]
        for pos in range(count):
            row = predictions[pos].astype(np.float64)
            valid = int(valid_steps[pos])
            recomputed.append([np.abs(row[i, :valid] - row[j, :valid]).mean() for i, j in comparisons])
            altered = row.copy()
            altered[:, valid:] = np.arange(7)[:, None, None] * 1e6
            poisoned.append([np.abs(altered[i, :valid] - altered[j, :valid]).mean() for i, j in comparisons])
        np.testing.assert_allclose(recomputed, differences[key], rtol=2e-6, atol=1e-9)
        np.testing.assert_array_equal(recomputed, poisoned)
        altered = predictions.copy()
        for pos in range(count):
            altered[pos, :, int(valid_steps[pos]):] = np.arange(7)[:, None, None] * 1e6
        altered_tensor = tensor(altered)
        poisoned_helper = torch.stack([valid_l1(left=altered_tensor[:, i], right=altered_tensor[:, j],
                                               mask=tensor(masks)) for i, j in comparisons], 1).cpu().numpy()
        np.testing.assert_allclose(poisoned_helper, differences[key], rtol=2e-6, atol=1e-9)
    folder.mkdir(parents=True, exist_ok=True)
    arrays = dict(demo=demos, timestep=timesteps, selected=selected, valid_steps=valid_steps, mask=masks,
                  posterior_mu=own_mu, recomputed_mu=np.concatenate(mus), prior_z=latents,
                  prediction=predictions, random_pairs=PAIRS, **differences)
    np.savez_compressed(folder / 'arrays.npz', **arrays)
    result = summarize(arrays=[arrays])
    save_json(path=folder / 'results.json', value=result)
    (folder / 'summary.md').write_text(report(result=result, label=f'{split}, seed {seed}'))
    save_json(path=folder / 'metadata.json', value={
        'execution_identity': execution, 'split': split, 'seed': seed, 'k': k, 'strict_load': True,
        'eval': True, 'dtype': 'float32', 'draws': 5, 'random_pairing': PAIRS.tolist(),
        'prediction_axis': ['zero', 'posterior_mean', 'random_0', 'random_1', 'random_2', 'random_3', 'random_4'],
        'target_count': count, 'padded_targets': int((valid_steps < k).sum()),
        'normalization': meta['normalization'], 'test1_source': str(source),
        'test1_arrays_sha256': file_hash(source / 'arrays.npz'),
        'test1_metadata_sha256': file_hash(source / 'metadata.json'),
        'arrays_sha256': file_hash(folder / 'arrays.npz'), 'seconds': time.time() - started,
        'validation': {'direct_decode_spot_checks': validation, 'mask_matches_trajectory_length': True,
                       'all_metrics_float64_valid_slice_match': True, 'padding_poison_invariance': True,
                       'posterior_reencode_matches_test1': True},
        'statistics': {'replicates': 5000, 'bootstrap_seed': 2026, 'unit': 'demonstration',
                       'threshold': THRESHOLD, 'weighting': 'equal demonstration weights; draws then targets/seeds'},
        'helper_scope': 'gather/tensor/valid_l1 only from refined audit; no donors, reconstruction or tests 2/3'})
    print(f'COMPLETE {split} seed={seed} seconds={time.time() - started:.1f} validation={validation}', flush=True)
    return arrays


def main():
    args = arguments()
    checkpoint, model, root, destination, datasets, splits, identity = prepare(args=args, test=4)
    assert not model.training and all(p.dtype == torch.float32 for p in model.parameters())
    results = {}
    for split in splits:
        collected = []
        for seed in SEEDS:
            print(f'START {split} seed={seed}', flush=True)
            collected.append(execute_seed(model=model, checkpoint=checkpoint, root=root, dataset=datasets[split], split=split, seed=seed,
                                          folder=destination / split / f'seed-{seed}', identity=identity))
        results[split] = {'combined': summarize(arrays=collected),
                          'by_seed': {str(seed): summarize(arrays=[array]) for seed, array in zip(SEEDS, collected)}}
    save_json(path=destination / 'results.json', value=results)
    (destination / 'summary.md').write_text('\n'.join(report(result=entry['combined'], label=f'{split}, seeds 42 + 123')
                                                      for split, entry in results.items()))
    print('COMPLETE test 4 only', flush=True)


if __name__ == '__main__':
    main()
