"""Statistical and runtime helpers shared by the posterior-collapse tests.

Retains the refined audit's masked losses, demo bootstrap and float64 moments,
without its historical checkpoint runner or debug-directory dependencies.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_resume(*, destination, identity):
    metadata_path = destination / 'metadata.json'
    arrays_path = destination / 'arrays.npz'
    if not metadata_path.exists():
        if arrays_path.exists():
            raise RuntimeError(f'Unverified orphan arrays: {destination}')
        return False
    metadata = json.loads(metadata_path.read_text())
    if metadata.get('execution_identity') != identity:
        raise RuntimeError(f'Stale/unverified resume artifact: {destination}; '
                           'preserve historical output explicitly and use a fresh output directory')
    if not arrays_path.exists():
        raise RuntimeError(f'Missing arrays: {destination}')
    if metadata.get('arrays_sha256') != file_hash(arrays_path):
        raise RuntimeError(f'Changed/unverified arrays: {destination}')
    return True


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float32, device='cuda')


def valid_l1(*, left, right, mask):
    return ((left-right).abs() * mask[..., None]).sum((1, 2)) / (mask.sum(1) * left.shape[-1])


def paired(*, difference, reference, demos):
    if not len(demos) or not np.isfinite(difference).all() or not np.isfinite(reference).all():
        raise ValueError('Paired effects require nonempty finite differences and references')
    unique = np.unique(demos)
    means = np.array([difference[demos == name].mean(dtype=np.float64) for name in unique])
    refs = np.array([reference[demos == name].mean(dtype=np.float64) for name in unique])
    draws = np.random.default_rng(2026).integers(len(unique), size=(5000, len(unique)))
    absolute = means[draws].mean(1)
    denominators = refs[draws].mean(1)
    if refs.mean() <= 0 or np.any(denominators <= 0):
        raise ValueError('Paired relative effects require positive own-L1 denominators in every bootstrap draw')
    relative = absolute / denominators
    if not np.isfinite(absolute).all() or not np.isfinite(relative).all():
        raise ValueError('Paired effects produced nonfinite bootstrap estimates')
    abs_ci, rel_ci = np.quantile(absolute, [.025, .975]), np.quantile(relative, [.025, .975])
    equivalent = np.max(np.abs(abs_ci)) < .002 and np.max(np.abs(rel_ci)) < .02
    harmful = abs_ci[0] > .002 and rel_ci[0] > .02
    return {'absolute': float(means.mean()), 'absolute_95ci': abs_ci.tolist(),
            'relative': float(means.mean()/refs.mean()), 'relative_95ci': rel_ci.tolist(),
            'own_l1': float(refs.mean()), 'classification': 'equivalent' if equivalent else 'meaningful harm' if harmful else 'inconclusive/mixed'}


def posterior_summary(*, mu, sigma, logvar, demos):
    # Float64 moments avoid catastrophic cancellation for sigma nearly equal to 1.
    mu, sigma, logvar = (np.asarray(value, dtype=np.float64) for value in (mu, sigma, logvar))
    unique = np.unique(demos)
    per_demo = {}
    for key, value in [('mu', mu), ('sigma', sigma)]:
        means = np.array([value[demos == name].mean(0) for name in unique])
        squares = np.array([(value[demos == name]**2).mean(0) for name in unique])
        sd = np.sqrt(np.maximum(0, squares.mean(0)-means.mean(0)**2))
        draws = np.random.default_rng(2026).integers(len(unique), size=(5000, len(unique)))
        # Bootstrap moments, preserving within-demo variation and avoiding huge example resamples.
        boot_sd = np.sqrt(np.maximum(0, squares[draws].mean(1)-means[draws].mean(1)**2))
        per_demo[key] = {'mean_per_dim': means.mean(0).tolist(), 'sd_per_dim': sd.tolist(),
                         'max_sd': float(sd.max()), 'max_sd_95ci': np.quantile(boot_sd.max(1), [.025, .975]).tolist()}
    intervals = [per_demo[key]['max_sd_95ci'] for key in ('mu', 'sigma')]
    status = 'negligible' if all(ci[1] < .02 for ci in intervals) else 'present' if any(ci[0] > .02 for ci in intervals) else 'uncertain'
    # Use original log variance, not rounded float32 sigma, for diagnostic KL.
    variance = np.exp(logvar)
    aggregate_var = variance.mean(0) + mu.var(0)
    relative_logvar = logvar - np.log(aggregate_var)
    bound = .5*(np.expm1(relative_logvar)-relative_logvar
                +(mu-mu.mean(0))**2/aggregate_var)
    return {**per_demo, 'status': status, 'kl_to_prior_mean': float((.5*(mu**2+np.expm1(logvar)-logvar)).sum(1).mean()),
            'fitted_aggregate_kl_proxy_nats': float(bound.sum(1).mean())}
