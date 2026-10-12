"""Shared, fail-closed inputs and reconstruction interventions for tests 2–4."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

import h5py
import numpy as np
import torch

from run_test1 import (SEEDS, REPO, build_model, file_hash, gather, save_json,
                       resolve_train_dataset, resolve_held_out_dataset, build_splits, validate_bank)
from scripts.action_modes import resolve_checkpoint_representation
from scripts.dataset import images_to_tensor
from scripts.utils.analyze_z import normalization_from_checkpoint
from audit_helpers import tensor, valid_l1, paired, verify_resume
import layout


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--held-out-dataset', type=Path)
    parser.add_argument('--output-root', type=Path, default=layout.OUTPUT_ROOT)
    parser.add_argument('--tag')
    parser.add_argument('--sweep', type=Path)
    parser.add_argument('--train-commit', default='')
    return parser.parse_args()


def prepare(*, args, test):
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if not checkpoint['config']['use_z']:
        raise ValueError('posterior tests require use_z=True')
    representation = resolve_checkpoint_representation(checkpoint=checkpoint)
    tag = layout.resolve_tag(iteration=args.iteration, config=checkpoint['config'], sweep=args.sweep, tag=args.tag)
    root = layout.checkpoint_directory(output_root=args.output_root, iteration=args.iteration,
                                       checkpoint_stem=args.checkpoint.stem, tag=tag)
    train = resolve_train_dataset(checkpoint=checkpoint, override=args.dataset)
    held = resolve_held_out_dataset(train_dataset=train, override=args.held_out_dataset)
    splits, split_identity = build_splits(train_dataset=train, held_out_dataset=held)
    datasets = {'train': train, 'held-out-candidate': held}
    environment = subprocess.check_output(['uv', 'pip', 'freeze'], text=True)
    identity = {'checkpoint_path': str(args.checkpoint),
                'checkpoint_sha256': file_hash(args.checkpoint), 'config': checkpoint['config'],
                'normalization': {key: np.asarray(value).tolist() for key, value in checkpoint['normalization'].items()},
                'dataset_sha256': {str(path): file_hash(path) for path in (train, held) if path is not None},
                'splits': splits, 'split_provenance': split_identity, 'test': test,
                'code_sha256': {str(path): file_hash(path) for path in
                    [Path(__file__), Path(__file__).with_name(f'run_test{test}.py'),
                     Path(__file__).with_name('run_test1.py'), Path(__file__).with_name('audit_helpers.py'),
                     REPO / 'scripts/dataset.py', REPO / 'scripts/action_modes.py',
                     REPO / 'scripts/utils/analyze_z.py', *sorted((REPO / 'scripts/models').rglob('*.py'))]},
                'draws': 5, 'seeds': list(SEEDS), 'bootstrap_seed': 2026, 'replicates': 5000,
                'training_commit': args.train_commit,
                'runtime': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                            'h5py': h5py.__version__, 'cuda': torch.version.cuda,
                            'gpu': torch.cuda.get_device_name(), 'tf32': False},
                'environment': environment,
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    revision = identity.pop('git_head')
    # Canonicalize tuples and other JSON containers once, including per-seed identities.
    identity = json.loads(json.dumps(identity))
    source_provenance = root / 'test-1/provenance.json'
    saved_split = json.loads(source_provenance.read_text()) if source_provenance.exists() else None
    expected_split = {key: value for key, value in split_identity.items()
                      if key != 'dataset_sha256' or saved_split is not None and key in saved_split}
    if saved_split != expected_split:
        raise RuntimeError(f'Test 1 dataset/split provenance mismatch: {source_provenance}')
    destination = root / f'test-{test}'
    layout.assert_checkpoint(directory=destination, checkpoint_sha=identity['checkpoint_sha256'])
    if destination.exists() and any(destination.iterdir()):
        provenance = destination / 'provenance.json'
        if not provenance.exists() or {key: value for key, value in json.loads(provenance.read_text()).items() if key != 'git_head'} != identity:
            raise RuntimeError(f'Refusing incompatible existing output: {destination}')
    model = build_model(checkpoint=checkpoint, representation=representation)
    # Validate every prerequisite, including all legacy donors, before creating outputs.
    for split in splits:
        for seed in SEEDS:
            source, _, _ = load_source(root=root, checkpoint=checkpoint, identity=identity, split=split, seed=seed,
                                       dataset=datasets[split])
            validate_bank(model=model, checkpoint=checkpoint, checkpoint_sha=identity['checkpoint_sha256'],
                          split=split, dataset=datasets[split], names=splits[split], seed=seed, folder=source)
            # Tests 2–4 share this execution identity. Preflight every seed before
            # writing root provenance, even if a later seed is stale or damaged.
            execution = {**identity, 'split': split, 'seed': seed,
                         'test1_arrays_sha256': file_hash(source / 'arrays.npz'),
                         'test1_metadata_sha256': file_hash(source / 'metadata.json')}
            verify_resume(destination=destination / split / f'seed-{seed}', identity=execution)
    destination.mkdir(parents=True, exist_ok=True)
    save_json(path=destination / 'provenance.json', value={**identity, 'git_head': revision})
    (destination / 'checkpoint.sha256').write_text(identity['checkpoint_sha256'] + '\n')
    (destination / 'environment.txt').write_text(environment)
    return checkpoint, model, root, destination, datasets, splits, identity


def load_source(*, root, checkpoint, identity, split, seed, dataset):
    source = root / 'test-1' / split / f'seed-{seed}'
    meta = json.loads((source / 'metadata.json').read_text())
    expected = {'checkpoint_sha256': identity['checkpoint_sha256'], 'config': checkpoint['config'],
                'normalization': identity['normalization'], 'k': checkpoint['config']['action_chunk_size'],
                'split': split, 'seed': seed, 'dataset': str(dataset),
                'arrays_sha256': file_hash(source / 'arrays.npz')}
    for key, value in expected.items():
        if meta.get(key) != value:
            raise RuntimeError(f'Test 1 provenance mismatch ({key}): {source}')
    if 'dataset_sha256' in meta and meta['dataset_sha256'] != identity['dataset_sha256'][str(dataset)]:
        raise RuntimeError(f'Test 1 dataset hash mismatch: {source}')
    with np.load(source / 'arrays.npz') as stored:
        arrays = {key: stored[key].copy() for key in stored.files}
    for key, value in arrays.items():
        if value.dtype.kind not in 'US' and not np.isfinite(value).all():
            raise RuntimeError(f'Nonfinite test 1 array: {key}')
    selected = arrays['selected']
    if len(selected) != len(np.unique(selected)) or not len(selected):
        raise RuntimeError('Invalid test 1 target indices')
    if np.any(selected < 0) or np.any(selected >= len(arrays['demo'])):
        raise RuntimeError('Out-of-range test 1 target indices')
    if np.any(arrays['valid_steps'] < 1) or np.any(arrays['valid_steps'] > expected['k']):
        raise RuntimeError('Invalid test 1 valid-step counts')
    if sorted(np.unique(arrays['demo']).tolist()) != sorted(identity['splits'][split]):
        raise RuntimeError('Test 1 split membership mismatch')
    for kind in ('arbitrary', 'matched'):
        donors = arrays[f'{kind}_source']
        if np.any(donors < 0) or np.any(donors >= len(arrays['demo'])):
            raise RuntimeError('Out-of-range test 1 donor indices')
        if len(donors) != len(selected) or np.any(arrays['demo'][donors] == arrays['demo'][selected]):
            raise RuntimeError('Invalid other-demonstration donors')
    return source, meta, arrays


def read_target(*, hdf5, demo, timestep, config, normalization):
    proprio, action, mask = gather(demo=hdf5[f'data/{demo}'], timestep=int(timestep),
                                   k=config['action_chunk_size'], normalization=normalization,
                                   action_mode=config.get('action_mode', 'joint_position_aperture'))
    frames = images_to_tensor(images=[hdf5[f'data/{demo}/obs/{key}'][int(timestep)] for key in config['image_keys']])
    return proprio, action, mask, frames


def summarize_reconstruction(*, arrays):
    demos = np.concatenate([array['demo'] for array in arrays])
    errors = {key[6:]: np.concatenate([array[key] for array in arrays]).mean(1, dtype=np.float64)
              for key in arrays[0] if key.startswith('error_')}
    comparisons = {}
    for key, value in errors.items():
        if key.startswith('own_'):
            continue
        reference = errors['own_sample' if key.endswith('sample') else 'own_mean']
        comparisons[key] = paired(difference=value-reference, reference=reference, demos=demos)
    return {'paired': comparisons, 'targets': len(demos), 'demonstrations': len(np.unique(demos))}


def report_reconstruction(*, result, test, label):
    lines = [f'# Test {test} — paired reconstruction', '', label, '',
             'Normalized valid-step L1; intervention minus own (positive = worse).',
             'Mean and five shared-epsilon sampled comparisons are separate. Draws averaged first;',
             'chunks/seeds averaged within demonstration, then equal demonstration weights.',
             '5000 paired target-demonstration bootstrap replicates, seed 2026, percentile 95% CI.',
             'Equivalence: entire absolute CI within ±0.002 AND relative CI within ±2%.',
             'Meaningful harm: both lower limits exceed those thresholds; otherwise inconclusive/mixed.',
             'Intervals are conditional on fixed donor assignments; donor reuse/rematching uncertainty is excluded.',
             'Matched means nearest standardized proprio with contrasting actions, NOT image matching.',
             'No collapse diagnosis from reconstruction alone; consult test 1. Held-out-candidate exposure is unverified.', '',
             '| Intervention | Own L1 | Absolute effect [95% CI] | Relative effect [95% CI] | Classification |',
             '|---|---:|---|---|---|']
    for key, item in result['paired'].items():
        lines.append(f"| {key} | {item['own_l1']:.6g} | {item['absolute']:.6g} {item['absolute_95ci']} | "
                     f"{item['relative']:.6g} {item['relative_95ci']} | {item['classification']} |")
    return '\n'.join(lines) + '\n'


@torch.inference_mode()
def execute_reconstruction(*, model, checkpoint, root, dataset, split, seed, folder, identity, test):
    source, meta, original = load_source(root=root, checkpoint=checkpoint, identity=identity,
                                        split=split, seed=seed, dataset=dataset)
    execution = {**identity, 'split': split, 'seed': seed,
                 'test1_arrays_sha256': file_hash(source / 'arrays.npz'),
                 'test1_metadata_sha256': file_hash(source / 'metadata.json')}
    if verify_resume(destination=folder, identity=execution):
        with np.load(folder / 'arrays.npz') as saved:
            return {key: saved[key].copy() for key in saved.files}
    selected = original['selected']
    demos, times = original['demo'][selected], original['timestep'][selected]
    epsilon = torch.randn((len(selected), 5, model.z_dims), generator=torch.Generator().manual_seed(seed)).numpy()
    errors, masks = {}, []
    normalization = normalization_from_checkpoint(checkpoint=checkpoint)
    with h5py.File(dataset) as hdf5:
        for start in range(0, len(selected), 8):
            positions = np.arange(start, min(start + 8, len(selected)))
            indices = selected[positions]
            rows = [read_target(hdf5=hdf5, demo=demos[p], timestep=times[p],
                                config=checkpoint['config'], normalization=normalization) for p in positions]
            proprio, action, mask = [np.stack([row[i] for row in rows]) for i in range(3)]
            masks.append(mask)
            prop, target, valid = tensor(proprio)[:, None], tensor(action), tensor(mask)
            mu, lv = model.posterior(proprio=prop, actions=target, mask=valid)
            np.testing.assert_allclose(mu[:, 0].cpu(), original['mu'][indices], rtol=1e-4, atol=2e-6)
            np.testing.assert_allclose(lv[:, 0].cpu(), original['logvar'][indices], rtol=1e-4, atol=2e-6)
            images = torch.stack([row[3] for row in rows]).cuda().float()
            img_tokens, prop_tokens = model.image_encoder(images), model.proprio_encoder(prop)
            def decode(z):
                return model.decode_from_tokens(img_tokens=img_tokens, proprio_tokens=prop_tokens, z=tensor(z)[:, None])
            own_mu, own_sigma = original['mu'][indices], original['sigma'][indices]
            mean_latents = {'own_mean': own_mu}
            if test == 2:
                for kind in ('arbitrary', 'matched'):
                    mean_latents[f'{kind}_mean'] = original['mu'][original[f'{kind}_source'][positions]]
            else:
                mean_latents['zero_mean'] = np.zeros_like(own_mu)
            predictions = {key: [decode(z)] for key, z in mean_latents.items()}
            for draw in range(5):
                noise = epsilon[positions, draw]
                sampled = {'own_sample': own_mu + own_sigma * noise}
                if test == 2:
                    for kind in ('arbitrary', 'matched'):
                        donors = original[f'{kind}_source'][positions]
                        sampled[f'{kind}_sample'] = original['mu'][donors] + original['sigma'][donors] * noise
                else:
                    sampled.update(prior_sample=noise, zero_sample=np.zeros_like(own_mu))
                for key, z in sampled.items():
                    predictions.setdefault(key, []).append(decode(z))
            for key, values in predictions.items():
                scores = torch.stack([valid_l1(left=value, right=target, mask=valid) for value in values], 1).cpu().numpy()
                # Independent float64 valid slices and padding-poison check.
                for draw, value in enumerate(values):
                    prediction = value.cpu().numpy().astype(np.float64)
                    expected = np.array([np.abs(prediction[p, mask[p]] - action[p, mask[p]]).mean() for p in range(len(rows))])
                    np.testing.assert_allclose(scores[:, draw], expected, rtol=2e-6, atol=1e-7)
                    poisoned = value.clone()
                    poisoned[~torch.as_tensor(mask, device='cuda')] = 1e6
                    np.testing.assert_allclose(valid_l1(left=poisoned, right=target, mask=valid).cpu(), scores[:, draw], rtol=2e-6, atol=1e-7)
                errors.setdefault('error_' + key, []).append(scores)
            # Direct decoder spot check for every condition/draw on the first target of each batch.
            for key, values in predictions.items():
                for draw, value in enumerate(values):
                    if key.endswith('mean'):
                        z = mean_latents[key][0]
                    elif key == 'own_sample':
                        z = own_mu[0] + own_sigma[0] * epsilon[positions[0], draw]
                    elif key == 'prior_sample':
                        z = epsilon[positions[0], draw]
                    elif key == 'zero_sample':
                        z = np.zeros_like(own_mu[0])
                    else:
                        donor = original[key.replace('_sample', '_source')][positions[0]]
                        z = original['mu'][donor] + original['sigma'][donor] * epsilon[positions[0], draw]
                    direct = model.decode(proprio=prop[:1], img=images[:1], z=tensor(z)[None, None], use_z=True)
                    np.testing.assert_allclose(direct[0].cpu(), value[0].cpu(), rtol=1e-4, atol=2e-5)
    masks = np.concatenate(masks)
    np.testing.assert_array_equal(masks.sum(1), original['valid_steps'][selected])
    arrays = {'demo': demos, 'timestep': times, 'selected': selected, 'mask': masks,
              'valid_steps': masks.sum(1), 'epsilon': epsilon,
              **{key: np.concatenate(value) for key, value in errors.items()}}
    if test == 2:
        arrays.update({key: original[key] for key in original if key.startswith(('arbitrary_', 'matched_'))})
    for key, value in arrays.items():
        if value.dtype.kind not in 'US' and not np.isfinite(value).all():
            raise RuntimeError(f'Nonfinite output: {key}')
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / 'arrays.npz', **arrays)
    save_json(path=folder / 'metadata.json', value={'execution_identity': execution,
        'arrays_sha256': file_hash(folder / 'arrays.npz'), 'validation': {
            'posterior_reencode': True, 'float64_valid_slice_scores': True,
            'padding_poison_invariance': True, 'direct_decoder_all_conditions_spot_checks': True}})
    result = summarize_reconstruction(arrays=[arrays])
    save_json(path=folder / 'results.json', value=result)
    (folder / 'summary.md').write_text(report_reconstruction(result=result, test=test, label=f'{split}, seed {seed}'))
    print(f'COMPLETE test {test} {split} seed={seed}', flush=True)
    return arrays


def run_reconstruction(*, test):
    args = arguments()
    checkpoint, model, root, destination, datasets, splits, identity = prepare(args=args, test=test)
    results = {}
    for split in splits:
        collected = [execute_reconstruction(model=model, checkpoint=checkpoint, root=root,
            dataset=datasets[split], split=split, seed=seed, folder=destination / split / f'seed-{seed}',
            identity=identity, test=test) for seed in SEEDS]
        results[split] = {'combined': summarize_reconstruction(arrays=collected),
                         'by_seed': {str(seed): summarize_reconstruction(arrays=[array]) for seed, array in zip(SEEDS, collected)}}
    save_json(path=destination / 'results.json', value=results)
    (destination / 'summary.md').write_text('\n'.join(report_reconstruction(result=entry['combined'], test=test,
        label=f'{split}, seeds 42 + 123') for split, entry in results.items()))
