"""Packaged audit helpers work without any historical debug runner."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
HELPERS = REPO / '.pi/scripts/test-posterior-collapse'
sys.path.insert(0, str(HELPERS))
from audit_helpers import file_hash, paired, posterior_summary, verify_resume


def test_runners_import_without_debug_path(tmp_path):
    code = f"""
import sys
sys.path[:0] = [{str(HELPERS)!r}, {str(REPO)!r}]
import audit_helpers, run_test1, run_test2, run_test3, run_test4
assert not any('posterior-tests-refined' in path for path in sys.path)
assert 'run_audit' not in sys.modules
assert audit_helpers.__file__.startswith({str(HELPERS)!r})
"""
    subprocess.run([sys.executable, '-c', code], cwd=tmp_path, check=True)


def test_resume_rejects_orphans_missing_and_changed_arrays(tmp_path):
    arrays = tmp_path / 'arrays.npz'
    assert not verify_resume(destination=tmp_path, identity={})
    np.savez(arrays, values=np.ones(2))
    with pytest.raises(RuntimeError, match='orphan arrays'):
        verify_resume(destination=tmp_path, identity={})
    metadata = {'execution_identity': {}, 'arrays_sha256': file_hash(arrays)}
    (tmp_path / 'metadata.json').write_text(json.dumps(metadata))
    assert verify_resume(destination=tmp_path, identity={})
    arrays.write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='Changed/unverified arrays'):
        verify_resume(destination=tmp_path, identity={})
    arrays.unlink()
    with pytest.raises(RuntimeError, match='Missing arrays'):
        verify_resume(destination=tmp_path, identity={})


def test_paired_bootstrap_weights_demos_equally():
    demos = np.array(['a', 'a', 'a', 'b'])
    result = paired(difference=np.array([0., 0., 0., .02]),
                    reference=np.ones(4), demos=demos)
    assert result['absolute'] == pytest.approx(.01)
    assert result['relative'] == pytest.approx(.01)
    assert result['absolute_95ci'] == pytest.approx([0., .02])
    assert result['classification'] == 'inconclusive/mixed'
    equivalent = paired(difference=np.zeros(4), reference=np.ones(4), demos=demos)
    assert equivalent['classification'] == 'equivalent'
    harmful = paired(difference=np.full(4, .03), reference=np.ones(4), demos=demos)
    assert harmful['classification'] == 'meaningful harm'


@pytest.mark.parametrize('reference', [np.zeros(2), np.array([0., 1.])])
def test_paired_rejects_zero_bootstrap_denominators(reference):
    with pytest.raises(ValueError, match='positive own-L1 denominators'):
        paired(difference=np.zeros(2), reference=reference, demos=np.array(['a', 'b']))


@pytest.mark.parametrize('difference,reference', [
    (np.array([np.nan]), np.ones(1)),
    (np.zeros(1), np.array([np.inf])),
    (np.array([]), np.array([])),
])
def test_paired_rejects_nonfinite_or_empty_inputs(difference, reference):
    with pytest.raises(ValueError, match='nonempty finite'):
        paired(difference=difference, reference=reference, demos=np.array(['a'] * len(reference)))


def test_posterior_float64_moments_and_logvar_kl():
    mu = np.zeros((4, 2), dtype=np.float32)
    logvar = np.full((4, 2), 1e-8, dtype=np.float64)
    result = posterior_summary(mu=mu, sigma=np.ones_like(mu), logvar=logvar,
                               demos=np.array(['a', 'a', 'b', 'b']))
    assert result['status'] == 'negligible'
    assert result['sigma']['max_sd'] == 0.
    assert result['kl_to_prior_mean'] > 0.
    assert result['kl_to_prior_mean'] == pytest.approx(float(np.expm1(1e-8) - 1e-8), abs=1e-24)
