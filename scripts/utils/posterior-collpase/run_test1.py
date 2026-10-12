"""Test 1 — posterior variation audit for one ACT v2 checkpoint.

Reuses the validated posterior/matching helpers from the refined i526 audit so
normalization, masking, sampling and donor selection keep identical semantics.
Outputs land under debug/posterior-collapse/<iteration>/<checkpoint>/test-1/.
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

from scripts.models.act_v2.model import ACTV2
from scripts.utils.analyze_z import normalization_from_checkpoint
from run_audit import gather, posterior_summary, tensor

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


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(*, path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def trajectory_hash(demo):
    digest = hashlib.sha256()
    for key in LOWDIM:
        value = demo[key][:]
        digest.update(key.encode())
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def build_splits():
    """Training names from ph100; held-out candidate = full-file lowdim-disjoint complement."""
    with h5py.File(TRAIN) as train, h5py.File(FULL) as full:
        train_hashes = {name: trajectory_hash(train['data'][name]) for name in sorted(train['data'])}
        full_hashes = {name: trajectory_hash(full['data'][name]) for name in sorted(full['data'])}
        assert len(train_hashes) == 100 and len(full_hashes) == 200
        assert all(full_hashes.get(name) == value for name, value in train_hashes.items())
        held = sorted(set(full_hashes) - set(train_hashes))
        assert not set(train_hashes.values()) & {full_hashes[name] for name in held}
        return ({'train': sorted(train_hashes), 'held-out-candidate': held},
                {'train_hashes': train_hashes, 'full_hashes': full_hashes,
                 'held_out_candidate': held, 'hash_fields': LOWDIM,
                 'status': 'candidate held-out; lowdim-disjoint; broader exposure unverified'})


def sample_examples(*, hdf5, names, k, normalization, seed):
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
                                           k=k, normalization=normalization)
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
def run_posterior(*, model, checkpoint, checkpoint_sha, split, names, seed, destination):
    config = checkpoint['config']
    k = config['action_chunk_size']
    normalization = normalization_from_checkpoint(checkpoint=checkpoint)
    dataset = TRAIN if split == 'train' else FULL
    with h5py.File(dataset) as hdf5:
        ids, selected, proprios, actions, masks = sample_examples(
            hdf5=hdf5, names=names, k=k, normalization=normalization, seed=seed)
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
    nn_mu_rms, nn_sigma_rms = pair_rms(mu=mu, sigma=sigma, source=joined('nn_source'))
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


def report(*, results, iteration, checkpoint, checkpoint_sha, train_commit):
    lines = ['# Test 1 — posterior variation', '',
             f'Iteration `{iteration}`, checkpoint `{checkpoint}` '
             f'(sha256 `{checkpoint_sha[:16]}…`, training commit `{train_commit[:8]}`).', '',
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
    lines += ['', '## Notes',
              '- Per-dimension tables above are the combined-seed summaries; `results.json` keeps per-seed detail.',
              '- Held-out-candidate is the full-file lowdim-disjoint complement, not certified unseen in all '
              'previous experiments.',
              '- This is a variation proxy, not a mutual-information or action-specificity measurement.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=Path('debug/posterior-collapse'))
    parser.add_argument('--train-commit', default='')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    torch.set_num_threads(4)
    destination = args.output_root / args.iteration / args.checkpoint.stem / 'test-1'
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_sha = file_hash(args.checkpoint)
    (destination / 'checkpoint.sha256').write_text(f'{checkpoint_sha}  {args.checkpoint.name}\n')

    splits, provenance = build_splits()
    save_json(path=destination / 'provenance.json', value=provenance)
    save_json(path=destination / 'compatibility.json', value={
        'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'training_commit': args.train_commit,
        'act_v2_diff_vs_training_commit': subprocess.check_output(
            ['git', 'diff', args.train_commit, 'HEAD', '--', 'scripts/models/act_v2',
             'scripts/dataset.py', 'scripts/utils/analyze_z.py'], text=True) if args.train_commit else None,
        'scope': 'evaluation code subset only'})

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    assert config['use_z']
    torch.manual_seed(42)
    model = ACTV2(action_chunk_size=config['action_chunk_size'], use_z=True).cuda().eval()
    model.load_state_dict(checkpoint['model'], strict=True)

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
                          split=split, names=names, seed=seed, destination=folder)
        results[split] = {'combined': summarize_split(files=files),
                          'by_seed': {str(seed): summarize_split(files=[file])
                                      for seed, file in zip(SEEDS, files)}}
    save_json(path=destination / 'results.json', value=results)
    (destination / 'summary.md').write_text(report(
        results=results, iteration=args.iteration, checkpoint=args.checkpoint.name,
        checkpoint_sha=checkpoint_sha, train_commit=args.train_commit))
    print('COMPLETE test 1', flush=True)


if __name__ == '__main__':
    main()
