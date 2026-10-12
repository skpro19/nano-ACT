"""Test 1 — posterior variation audit for one ACT v2 checkpoint.

Reuses the validated posterior/matching helpers from the refined i526 audit so
normalization, masking, sampling and donor selection keep identical semantics.
Outputs land under debug/posterior-collapse/<iteration>/[<config_tag>/]<checkpoint>/test-1/,
where the config tag names the sweep's varying [grid] parameters (see layout.py).

Action representation (mode, action/proprio dimensions), training dataset and
normalization are read from the checkpoint, so the runner covers both the
original 8-D joint_position_aperture audit and the 7-D osc_pose_delta runs. When
the training dataset is the ph100 file, the matching full file supplies a
held-out-candidate split; any other dataset is reported as a single `train`
split, because every demonstration in it was trained on.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'debug' / 'posterior-tests-refined'))

import h5py
import numpy as np
import torch

from scripts.action_modes import resolve_checkpoint_representation, target_actions
from scripts.dataset import build_proprio
from scripts.models.act_v2.config import D_MODEL, N_HEAD, NUM_LAYERS, PROPRIO_DIMS, Z_DIMS
from scripts.models.act_v2.model import ACTV2
from scripts.utils.analyze_z import normalization_from_checkpoint
from run_audit import posterior_summary, tensor
import layout

# Historical ph100/full pair; still imported by run_test4.py and used as the
# default held-out relationship when a checkpoint records the ph100 dataset.
TRAIN = Path('datasets/can/ph/2026-09-29_02-01-42_agentview_robot0_eye_in_hand_ph100.hdf5')
FULL = Path('datasets/can/ph/2026-09-29_02-01-42_agentview_robot0_eye_in_hand.hdf5')
SEEDS = (42, 123)
POSTERIOR_PER_DEMO = 6
SELECTED_PER_DEMO = 2
LOWDIM = tuple(f'{group}/{key}' for group in ('obs', 'next_obs')
               for key in ('robot0_joint_pos', 'robot0_gripper_qpos'))
# Predeclared in methods.md of the refined audit; reused unchanged here.
NEGLIGIBLE_VARIATION_SD = 0.02
# Minimum future-action RMS for a "contrastive" observation-matched pair.
CONTRASTIVE_ACTION_RMS = 0.05
PH100_SUFFIX = '_ph100.hdf5'


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(*, path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def demo_names(hdf5):
    """Demonstration names in numeric order (demo_2 before demo_10)."""
    return sorted(hdf5['data'], key=lambda name: int(name.split('_')[-1]))


def trajectory_hash(demo):
    digest = hashlib.sha256()
    for key in LOWDIM:
        value = demo[key][:]
        digest.update(key.encode())
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def build_splits(*, train_dataset, held_out_dataset):
    """Train demos from the recorded dataset; optional lowdim-disjoint held-out complement."""
    with h5py.File(train_dataset) as train:
        train_names = demo_names(train)
        train_hashes = {name: trajectory_hash(train['data'][name]) for name in train_names}
    provenance = {'train_dataset': str(train_dataset), 'train_hashes': train_hashes,
                  'train_demos': len(train_names), 'hash_fields': LOWDIM}
    if held_out_dataset is None:
        provenance['status'] = ('training demonstrations only; every demo in the recorded '
                                'dataset was trained on, so no held-out split is claimed')
        return {'train': train_names}, provenance
    with h5py.File(held_out_dataset) as full:
        full_hashes = {name: trajectory_hash(full['data'][name]) for name in demo_names(full)}
    assert all(full_hashes.get(name) == value for name, value in train_hashes.items()), (
        'training demonstrations are not identical in the held-out file')
    held = [name for name in full_hashes if name not in train_hashes]
    assert not set(train_hashes.values()) & {full_hashes[name] for name in held}
    provenance.update({'held_out_dataset': str(held_out_dataset), 'full_hashes': full_hashes,
                       'held_out_candidate': held,
                       'status': 'candidate held-out; lowdim-disjoint; broader exposure unverified'})
    return {'train': train_names, 'held-out-candidate': held}, provenance


def gather(*, demo, timestep, k, normalization, action_mode):
    """One source action chunk in the checkpoint's own action representation."""
    proprio = build_proprio(joint_pos=demo['obs/robot0_joint_pos'][timestep],
                            gripper_qpos=demo['obs/robot0_gripper_qpos'][timestep])
    actions = target_actions(demo=demo, action_mode=action_mode, start=timestep, stop=timestep + k)
    valid = len(actions)
    padded = np.zeros((k, actions.shape[-1]), dtype=np.float32)
    padded[:valid] = actions
    return (normalization.normalize_proprio(value=proprio),
            normalization.normalize_action(value=padded), np.arange(k) < valid)


def sample_examples(*, hdf5, names, k, normalization, action_mode, seed):
    """6 starts per demo; the first 2 per demo become intervention targets (refined audit)."""
    rng = np.random.default_rng(seed)
    ids, selected = [], []
    proprios, actions, masks = [], [], []
    for name in names:
        length = len(hdf5[f'data/{name}/next_obs/robot0_joint_pos'])
        times = rng.choice(length, size=POSTERIOR_PER_DEMO, replace=False)
        selected.extend([len(ids), len(ids) + 1])
        for timestep in times:
            ids.append((name, int(timestep)))
            proprio, action, mask = gather(demo=hdf5[f'data/{name}'], timestep=int(timestep),
                                           k=k, normalization=normalization, action_mode=action_mode)
            proprios.append(proprio)
            actions.append(action)
            masks.append(mask)
    return (ids, np.array(selected), np.stack(proprios), np.stack(actions), np.stack(masks))


@torch.inference_mode()
def encode_posterior(*, model, proprios, actions, masks, batch=32):
    mus, logvars = [], []
    for start in range(0, len(proprios), batch):
        mu, logvar = model.posterior(proprio=tensor(proprios[start:start + batch])[:, None],
                                     actions=tensor(actions[start:start + batch]),
                                     mask=tensor(masks[start:start + batch]))
        mus.append(mu[:, 0].cpu().numpy())
        logvars.append(logvar[:, 0].cpu().numpy())
    return np.concatenate(mus), np.concatenate(logvars)


def action_rms(*, left, right, overlap):
    """Masked per-row action RMS; left/right (n,k,a), overlap (n,k) boolean."""
    difference = ((left - right) ** 2 * overlap[..., None]).sum((1, 2))
    return np.sqrt(difference / (np.maximum(overlap.sum(1), 1) * left.shape[-1]))


def observation_matching(*, proprios, actions, masks, demos, selected, seed):
    """Nearest-observation other-demo neighbour for all examples and contrastive donors for targets."""
    rng = np.random.default_rng(seed + 1000)
    count = len(demos)
    nn_source = np.empty(count, int)
    nn_proprio_rms, nn_action_rms = np.empty(count), np.empty(count)
    for index in range(count):
        other = np.flatnonzero(demos != demos[index])
        distance = np.sqrt(((proprios[other] - proprios[index]) ** 2).mean(1))
        source = int(other[np.argmin(distance)])
        nn_source[index] = source
        nn_proprio_rms[index] = distance.min()
        overlap = (masks[index] & masks[source])[None]
        nn_action_rms[index] = action_rms(left=actions[index][None], right=actions[source][None],
                                          overlap=overlap)[0]
    arbitrary, matched, candidate_count = [], [], []
    matched_proprio_rms, matched_action_rms = [], []
    for index in selected:
        other = np.flatnonzero(demos != demos[index])
        arbitrary.append(int(rng.choice(other)))
        overlap = masks[other] & masks[index]
        action_distance = action_rms(left=actions[other],
                                     right=np.broadcast_to(actions[index], actions[other].shape),
                                     overlap=overlap)
        eligible = (overlap.sum(1) >= masks[index].sum() / 2) & (action_distance >= CONTRASTIVE_ACTION_RMS)
        assert eligible.any(), f'No contrastive donor for example {index}'
        distance = np.sqrt(((proprios[other] - proprios[index]) ** 2).mean(1))
        choice = int(np.argmin(np.where(eligible, distance, np.inf)))
        matched.append(int(other[choice]))
        matched_proprio_rms.append(float(distance[choice]))
        matched_action_rms.append(float(action_distance[choice]))
        candidate_count.append(int(eligible.sum()))
    return {'nn_source': nn_source, 'nn_proprio_rms': nn_proprio_rms, 'nn_action_rms': nn_action_rms,
            'arbitrary_source': np.array(arbitrary), 'matched_source': np.array(matched),
            'matched_proprio_rms': np.array(matched_proprio_rms),
            'matched_action_rms': np.array(matched_action_rms),
            'matched_candidate_count': np.array(candidate_count)}


@torch.inference_mode()
def run_posterior(*, model, checkpoint, checkpoint_sha, split, dataset, names, seed,
                  action_mode, action_dims, destination):
    config = checkpoint['config']
    k = config['action_chunk_size']
    normalization = normalization_from_checkpoint(checkpoint=checkpoint)
    with h5py.File(dataset) as hdf5:
        ids, selected, proprios, actions, masks = sample_examples(
            hdf5=hdf5, names=names, k=k, normalization=normalization,
            action_mode=action_mode, seed=seed)
    mu, logvar = encode_posterior(model=model, proprios=proprios, actions=actions, masks=masks)
    sigma = np.exp(logvar / 2)
    demos = np.array([name for name, _ in ids])
    matching = observation_matching(proprios=proprios, actions=actions, masks=masks,
                                    demos=demos, selected=selected, seed=seed)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / 'arrays.npz', mu=mu, sigma=sigma, logvar=logvar,
                        demo=demos, timestep=np.array([t for _, t in ids]), selected=selected,
                        valid_steps=masks.sum(1), **matching)
    save_json(path=destination / 'metadata.json', value={
        'split': split, 'seed': seed, 'dataset': str(dataset), 'k': k,
        'action_mode': action_mode, 'action_dims': action_dims,
        'beta': config['beta'], 'lr': config['lr'], 'step': checkpoint['global_step'],
        'posterior_examples': len(ids), 'intervention_examples': len(selected),
        'posterior_padded': int((masks.sum(1) < k).sum()),
        'intervention_padded': int((masks[selected].sum(1) < k).sum()),
        'normalization': {key: np.asarray(value).tolist() for key, value in checkpoint['normalization'].items()},
        'config': config, 'strict_load': True, 'checkpoint_sha256': checkpoint_sha,
        'posterior_per_demo': POSTERIOR_PER_DEMO, 'selected_per_demo': SELECTED_PER_DEMO,
        'arrays_sha256': file_hash(destination / 'arrays.npz')})


def summarize_values(values):
    return {'mean': float(values.mean()),
            'quantiles_0_25_50_75_95_100': np.quantile(values, [0, .25, .5, .75, .95, 1]).tolist()}


def pair_rms(*, mu, sigma, source):
    return (np.sqrt(((mu - mu[source]) ** 2).mean(1)),
            np.sqrt(((sigma - sigma[source]) ** 2).mean(1)))


def summarize_split(*, files):
    arrays = [np.load(file) for file in files]

    def joined(key):
        return np.concatenate([array[key] for array in arrays])

    mu, sigma, logvar, demo = joined('mu'), joined('sigma'), joined('logvar'), joined('demo')
    posterior = posterior_summary(mu=mu, sigma=sigma, logvar=logvar, demos=demo)
    # Pair rows with their own seed's donors before pooling; donor indices are seed-local.
    nn_mu_rms = np.concatenate([pair_rms(mu=array['mu'], sigma=array['sigma'],
                                         source=array['nn_source'])[0] for array in arrays])
    nn_sigma_rms = np.concatenate([pair_rms(mu=array['mu'], sigma=array['sigma'],
                                            source=array['nn_source'])[1] for array in arrays])
    nn = {'mu_rms': summarize_values(nn_mu_rms), 'sigma_rms': summarize_values(nn_sigma_rms),
          'proprio_rms': summarize_values(joined('nn_proprio_rms')),
          'future_action_rms': summarize_values(joined('nn_action_rms'))}
    contrastive = joined('nn_action_rms') >= CONTRASTIVE_ACTION_RMS
    contrastive_summary = {
        'count': int(contrastive.sum()),
        'proprio_rms_mean': float(joined('nn_proprio_rms')[contrastive].mean()),
        'future_action_rms_mean': float(joined('nn_action_rms')[contrastive].mean()),
        'mu_rms': summarize_values(nn_mu_rms[contrastive]),
        'sigma_rms': summarize_values(nn_sigma_rms[contrastive])}
    return {'posterior': posterior, 'observation_matched_all': nn,
            'observation_matched_contrastive': contrastive_summary,
            'posterior_examples': len(demo), 'demonstrations': len(np.unique(demo)),
            'variation_threshold_sd': NEGLIGIBLE_VARIATION_SD,
            'files': [str(file) for file in files]}


def report(*, results, iteration, tag, checkpoint, checkpoint_sha, train_commit, action_mode, has_held_out):
    lines = ['# Test 1 — posterior variation', '',
             f'Iteration `{iteration}`, config tag `{tag or "none"}`, checkpoint `{checkpoint}` '
             f'(sha256 `{checkpoint_sha[:16]}…`, training commit `{train_commit[:8]}`, '
             f'action mode `{action_mode}`).', '',
             '**Question:** does the encoder produce example-specific posterior distributions (`mu`, `sigma`)?',
             f'Variation is negligible when the 95% bootstrap CI of the max per-dimension SD is below '
             f'{NEGLIGIBLE_VARIATION_SD} latent units, present when its lower limit exceeds that value, otherwise uncertain. '
             'Posterior variation alone does not establish action-specific information because the posterior also '
             'conditions on observations; observation-matched pairs are reported separately.', '',
             '| Split | Max mu SD [95% CI] | Max sigma SD [95% CI] | Variation | KL(q||prior) | Aggregate KL proxy |',
             '|---|---|---|---|---:|---:|']
    for split, result in results.items():
        post = result['combined']['posterior']
        lines.append(f"| {split} | {post['mu']['max_sd']:.6g} [{post['mu']['max_sd_95ci'][0]:.4g}, "
                     f"{post['mu']['max_sd_95ci'][1]:.4g}] | {post['sigma']['max_sd']:.6g} "
                     f"[{post['sigma']['max_sd_95ci'][0]:.4g}, {post['sigma']['max_sd_95ci'][1]:.4g}] | "
                     f"**{post['status']}** | {post['kl_to_prior_mean']:.4g} | "
                     f"{post['fitted_aggregate_kl_proxy_nats']:.4g} |")
    for split, result in results.items():
        combined = result['combined']
        post = combined['posterior']
        lines += ['', f'## {split}', '',
                  f"**Posterior variation: {post['status']}.** Max per-dim `mu` SD {post['mu']['max_sd']:.6g} "
                  f"(95% CI {post['mu']['max_sd_95ci']}); max per-dim `sigma` SD {post['sigma']['max_sd']:.6g} "
                  f"(95% CI {post['sigma']['max_sd_95ci']}).",
                  f"Mean KL to prior {post['kl_to_prior_mean']:.6g} nats; fitted aggregate Gaussian KL proxy "
                  f"{post['fitted_aggregate_kl_proxy_nats']:.6g} nats (diagnostic proxy, not mutual information).",
                  f"{combined['posterior_examples']} posterior examples across {combined['demonstrations']} demonstrations.", '',
                  '| Dimension | mean mu | SD mu | mean sigma | SD sigma |',
                  '|---:|---:|---:|---:|---:|']
        for dim in range(len(post['mu']['sd_per_dim'])):
            lines.append(f"| {dim} | {post['mu']['mean_per_dim'][dim]:.4g} | {post['mu']['sd_per_dim'][dim]:.4g} | "
                         f"{post['sigma']['mean_per_dim'][dim]:.4g} | {post['sigma']['sd_per_dim'][dim]:.4g} |")
        matched = combined['observation_matched_all']
        contrastive = combined['observation_matched_contrastive']
        lines += ['', '### Observation-matched pairs', '',
                  'Nearest other-demonstration neighbour by proprio RMS, so observations are as similar as available '
                  'among the sampled examples; future actions may still differ.', '',
                  f"- All matched pairs: proprio RMS mean {matched['proprio_rms']['mean']:.4g}, "
                  f"future-action RMS mean {matched['future_action_rms']['mean']:.4g}, posterior `mu` RMS mean "
                  f"{matched['mu_rms']['mean']:.4g}, `sigma` RMS mean {matched['sigma_rms']['mean']:.4g}.",
                  f"- Contrastive subset (future-action RMS >= {CONTRASTIVE_ACTION_RMS}): n={contrastive['count']}, "
                  f"proprio RMS mean {contrastive['proprio_rms_mean']:.4g}, future-action RMS mean "
                  f"{contrastive['future_action_rms_mean']:.4g}, posterior `mu` RMS mean "
                  f"{contrastive['mu_rms']['mean']:.4g}, `sigma` RMS mean {contrastive['sigma_rms']['mean']:.4g}."]
    notes = ['', '## Notes',
             '- Per-dimension tables above are the combined-seed summaries; `results.json` keeps per-seed detail.',
             '- Held-out-candidate is the full-file lowdim-disjoint complement, not certified unseen in all '
             'previous experiments.' if has_held_out else
             '- Only a `train` split is reported: the checkpoint trained on every demonstration in its recorded '
             'dataset, so no genuinely held-out split exists for this checkpoint.',
             '- This is a variation proxy, not a mutual-information or action-specificity measurement.']
    return '\n'.join(lines + notes) + '\n'


def resolve_train_dataset(*, checkpoint, override):
    recorded = (checkpoint.get('config') or {}).get('dataset')
    dataset = override or (Path(recorded) if recorded else None)
    if dataset is None:
        raise ValueError('no dataset recorded in the checkpoint and no --dataset given')
    if not dataset.exists():
        raise FileNotFoundError(f'training dataset not found: {dataset}')
    return dataset


def resolve_held_out_dataset(*, train_dataset, override):
    """Historical ph100 runs build the held-out candidate from the matching full file."""
    if override is not None:
        return override
    if train_dataset.name.endswith(PH100_SUFFIX):
        candidate = train_dataset.with_name(train_dataset.name[:-len(PH100_SUFFIX)] + '.hdf5')
        if candidate.exists():
            return candidate
    return None


def build_model(*, checkpoint, representation):
    config = checkpoint['config']
    model = ACTV2(
        d_model=D_MODEL, nhead=N_HEAD, num_layers=NUM_LAYERS, z_dims=Z_DIMS,
        proprio_dims=PROPRIO_DIMS, action_dims=representation['action_dims'],
        action_chunk_size=config['action_chunk_size'], use_z=True).cuda().eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    model.action_representation = representation
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=None,
                        help="training dataset; defaults to the checkpoint's recorded dataset")
    parser.add_argument('--held-out-dataset', type=Path, default=None,
                        help='optional full dataset used to build a held-out-candidate split')
    parser.add_argument('--output-root', type=Path, default=layout.OUTPUT_ROOT)
    parser.add_argument('--tag', default=None,
                        help='explicit config tag; overrides sweep derivation')
    parser.add_argument('--sweep', type=Path, default=None,
                        help='sweep spec used to derive the tag when the saved iteration is unavailable')
    parser.add_argument('--train-commit', default='')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    assert config['use_z']
    checkpoint_sha = file_hash(args.checkpoint)
    tag = layout.resolve_tag(iteration=args.iteration, config=config, sweep=args.sweep, tag=args.tag)
    destination = layout.checkpoint_directory(
        output_root=args.output_root, iteration=args.iteration,
        checkpoint_stem=args.checkpoint.stem, tag=tag) / 'test-1'
    layout.assert_checkpoint(directory=destination, checkpoint_sha=checkpoint_sha)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'checkpoint.sha256').write_text(f'{checkpoint_sha}  {args.checkpoint.name}\n')

    representation = resolve_checkpoint_representation(checkpoint=checkpoint)
    train_dataset = resolve_train_dataset(checkpoint=checkpoint, override=args.dataset)
    held_out_dataset = resolve_held_out_dataset(train_dataset=train_dataset,
                                                override=args.held_out_dataset)

    splits, provenance = build_splits(train_dataset=train_dataset, held_out_dataset=held_out_dataset)
    save_json(path=destination / 'provenance.json', value=provenance)
    save_json(path=destination / 'compatibility.json', value={
        'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'training_commit': args.train_commit,
        'act_v2_diff_vs_training_commit': subprocess.check_output(
            ['git', 'diff', args.train_commit, 'HEAD', '--', 'scripts/models/act_v2',
             'scripts/dataset.py', 'scripts/action_modes.py', 'scripts/utils/analyze_z.py'],
            text=True) if args.train_commit else None,
        'config_tag': tag,
        'scope': 'evaluation code subset only'})

    torch.manual_seed(42)
    model = build_model(checkpoint=checkpoint, representation=representation)
    dataset_for_split = {'train': train_dataset, 'held-out-candidate': held_out_dataset}

    results = {}
    for split, names in splits.items():
        files = []
        for seed in SEEDS:
            folder = destination / split / f'seed-{seed}'
            files.append(folder / 'arrays.npz')
            if folder.joinpath('arrays.npz').exists() and not args.force:
                print(f'SKIP existing {folder}', flush=True)
                continue
            print(f'START {split} seed={seed}', flush=True)
            run_posterior(model=model, checkpoint=checkpoint, checkpoint_sha=checkpoint_sha,
                          split=split, dataset=dataset_for_split[split], names=names, seed=seed,
                          action_mode=representation['mode'],
                          action_dims=representation['action_dims'], destination=folder)
        results[split] = {'combined': summarize_split(files=files),
                          'by_seed': {str(seed): summarize_split(files=[file])
                                      for seed, file in zip(SEEDS, files)}}
    save_json(path=destination / 'results.json', value=results)
    (destination / 'summary.md').write_text(report(
        results=results, iteration=args.iteration, tag=tag, checkpoint=args.checkpoint.name,
        checkpoint_sha=checkpoint_sha, train_commit=args.train_commit,
        action_mode=representation['mode'], has_held_out='held-out-candidate' in splits))
    print('COMPLETE test 1', flush=True)


if __name__ == '__main__':
    main()
