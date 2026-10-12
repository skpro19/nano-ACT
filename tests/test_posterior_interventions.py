"""Fail-closed posterior source banks and reconstruction masks."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / '.pi/scripts/test-posterior-collapse'))
import interventions
import run_test1
from audit_helpers import valid_l1, verify_resume


def test_json_canonical_identity_resumes(tmp_path):
    identity = json.loads(json.dumps({'split_provenance': {'hash_fields': ('actions', 'obs')}, 'test': 2}))
    np.savez(tmp_path / 'arrays.npz', values=np.ones(2))
    run_test1.save_json(path=tmp_path / 'metadata.json', value={
        'execution_identity': identity, 'arrays_sha256': run_test1.file_hash(tmp_path / 'arrays.npz')})
    assert verify_resume(destination=tmp_path, identity=identity)
    with pytest.raises(RuntimeError):
        verify_resume(destination=tmp_path, identity={**identity, 'test': 3})


@pytest.mark.parametrize('test', [2, 3, 4])
def test_prepare_canonicalizes_identity_and_ignores_revision(tmp_path, monkeypatch, test):
    from argparse import Namespace
    root = tmp_path / 'checkpoint-output'
    (root / 'test-1').mkdir(parents=True)
    split_provenance = {'hash_fields': ['actions', 'obs']}
    run_test1.save_json(path=root / 'test-1/provenance.json', value=split_provenance)
    checkpoint = {'config': {'use_z': True, 'tuple_setting': ('a', 'b')}, 'normalization': {}}
    args = Namespace(checkpoint=tmp_path / 'checkpoint.pt', iteration='iteration',
                     dataset=None, held_out_dataset=None, output_root=tmp_path, tag=None,
                     sweep=None, train_commit='training')
    monkeypatch.setattr(interventions.torch, 'load', lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(interventions.torch.cuda, 'get_device_name', lambda: 'test GPU')
    monkeypatch.setattr(interventions, 'resolve_checkpoint_representation', lambda **kwargs: {})
    monkeypatch.setattr(interventions.layout, 'resolve_tag', lambda **kwargs: 'tag')
    monkeypatch.setattr(interventions.layout, 'checkpoint_directory', lambda **kwargs: root)
    monkeypatch.setattr(interventions, 'resolve_train_dataset', lambda **kwargs: tmp_path / 'dataset')
    monkeypatch.setattr(interventions, 'resolve_held_out_dataset', lambda **kwargs: None)
    monkeypatch.setattr(interventions, 'build_splits', lambda **kwargs: ({'train': ['demo_0']}, split_provenance))
    monkeypatch.setattr(interventions, 'file_hash', lambda path: 'hash')
    monkeypatch.setattr(interventions, 'build_model', lambda **kwargs: None)
    monkeypatch.setattr(interventions, 'load_source', lambda **kwargs: (root, {}, {}))
    monkeypatch.setattr(interventions, 'validate_bank', lambda **kwargs: None)
    revision = ['before']
    monkeypatch.setattr(interventions.subprocess, 'check_output',
                        lambda command, **kwargs: revision[0] if command[0] == 'git' else 'environment')
    first = interventions.prepare(args=args, test=test)[-1]
    assert first['split_provenance']['hash_fields'] == ['actions', 'obs']
    assert first['config']['tuple_setting'] == ['a', 'b']
    assert 'git_head' not in first
    revision[0] = 'after-commit'
    assert interventions.prepare(args=args, test=test)[-1] == first
    destination = root / f'test-{test}'
    assert json.loads((destination / 'provenance.json').read_text())['git_head'] == 'after-commit'
    seed_folder = destination / 'train' / f'seed-{interventions.SEEDS[-1]}'
    seed_folder.mkdir(parents=True)
    np.savez(seed_folder / 'arrays.npz', values=np.ones(1))
    execution = {**first, 'split': 'train', 'seed': interventions.SEEDS[-1],
                 'test1_arrays_sha256': 'hash', 'test1_metadata_sha256': 'hash'}
    run_test1.save_json(path=seed_folder / 'metadata.json', value={
        'execution_identity': execution,
        'arrays_sha256': run_test1.file_hash(seed_folder / 'arrays.npz')})
    assert interventions.prepare(args=args, test=test)[-1] == first
    before = {path: path.read_bytes() for path in destination.iterdir() if path.is_file()}
    revision[0] = 'must-not-be-written'
    # A stale seed identity and changed arrays both fail before any root writes.
    metadata = json.loads((seed_folder / 'metadata.json').read_text())
    metadata['execution_identity']['seed'] = -1
    run_test1.save_json(path=seed_folder / 'metadata.json', value=metadata)
    with pytest.raises(RuntimeError, match='Stale/unverified'):
        interventions.prepare(args=args, test=test)
    assert all(path.read_bytes() == content for path, content in before.items())
    metadata['execution_identity'] = execution
    run_test1.save_json(path=seed_folder / 'metadata.json', value=metadata)
    np.savez(seed_folder / 'arrays.npz', values=np.zeros(1))
    with pytest.raises(RuntimeError, match='Changed/unverified'):
        interventions.prepare(args=args, test=test)
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize('hashed', [False, True])
def test_validation_checks_entire_bank_and_matching(tmp_path, hashed):
    dataset = tmp_path / 'dataset.hdf5'
    with h5py.File(dataset, 'w'):
        pass
    checkpoint = {'config': {'action_chunk_size': 2}, 'normalization': {}}
    ids = [('demo_0', 0), ('demo_1', 0)]
    selected = np.array([0])
    props, actions, masks = np.zeros((2, 1)), np.zeros((2, 2, 1)), np.ones((2, 2), bool)
    mu, lv = np.zeros((2, 1)), np.zeros((2, 1))
    matching = {'matched_source': np.array([1])}
    arrays = dict(demo=np.array(['demo_0', 'demo_1']), timestep=np.array([0, 0]),
                  selected=selected, valid_steps=np.array([2, 2]), mu=mu, logvar=lv,
                  sigma=np.ones((2, 1)), **matching)
    def save():
        np.savez(tmp_path / 'arrays.npz', **arrays)
        run_test1.save_json(path=tmp_path / 'metadata.json', value={
            'checkpoint_sha256': 'sha', 'config': checkpoint['config'], 'normalization': {},
            'split': 'train', 'seed': 42, 'dataset': str(dataset), 'k': 2,
            'arrays_sha256': run_test1.file_hash(tmp_path / 'arrays.npz'),
            **({'dataset_sha256': run_test1.file_hash(dataset)} if hashed else {})})
    def validate():
        run_test1.validate_bank(model=None, checkpoint=checkpoint, checkpoint_sha='sha',
                               split='train', dataset=dataset, names=['demo_0', 'demo_1'],
                               seed=42, folder=tmp_path)
    with patch.object(run_test1, 'normalization_from_checkpoint'), \
         patch.object(run_test1, 'sample_examples', return_value=(ids, selected, props, actions, masks)), \
         patch.object(run_test1, 'encode_posterior', return_value=(mu.copy(), lv.copy())) as encode, \
         patch.object(run_test1, 'observation_matching', return_value=matching):
        save()
        metadata_before = (tmp_path / 'metadata.json').read_bytes()
        validate()
        assert (tmp_path / 'metadata.json').read_bytes() == metadata_before
        # The current producer must reproduce all posteriors even with a dataset hash.
        encode.return_value = (mu + 1, lv.copy())
        with pytest.raises(AssertionError):
            validate()
        encode.return_value = (mu.copy(), lv.copy())
        assert encode.call_args.kwargs['proprios'].shape[0] == 2
        arrays['mu'] = np.array([[0.], [1.]])  # Nonselected donor must be checked.
        save()
        with pytest.raises(AssertionError):
            validate()
        arrays['mu'] = mu
        arrays['matched_source'] = np.array([0])
        save()
        with pytest.raises(AssertionError):
            validate()


def test_test1_resume_preserves_producer_compatibility(tmp_path, monkeypatch):
    destination = tmp_path / 'output' / 'test-1'
    destination.mkdir(parents=True)
    compatibility = b'{"code_sha256": {"original-producer": "original-hash"}}\n'
    (destination / 'compatibility.json').write_bytes(compatibility)
    checkpoint_path = tmp_path / 'checkpoint.pt'
    checkpoint_path.write_bytes(b'checkpoint')
    checkpoint = {'config': {'use_z': True}, 'normalization': {}}
    monkeypatch.setattr(sys, 'argv', ['run_test1.py', '--iteration', 'iteration',
                                    '--checkpoint', str(checkpoint_path)])
    monkeypatch.setattr(run_test1.torch, 'load', lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(run_test1.layout, 'resolve_tag', lambda **kwargs: 'tag')
    monkeypatch.setattr(run_test1.layout, 'checkpoint_directory', lambda **kwargs: destination.parent)
    monkeypatch.setattr(run_test1, 'resolve_checkpoint_representation', lambda **kwargs: {'mode': 'test'})
    monkeypatch.setattr(run_test1, 'resolve_train_dataset', lambda **kwargs: tmp_path / 'dataset')
    monkeypatch.setattr(run_test1, 'resolve_held_out_dataset', lambda **kwargs: None)
    monkeypatch.setattr(run_test1, 'build_splits', lambda **kwargs: ({}, {}))
    monkeypatch.setattr(run_test1, 'build_model', lambda **kwargs: None)
    monkeypatch.setattr(run_test1, 'report', lambda **kwargs: 'summary')
    run_test1.main()
    assert (destination / 'compatibility.json').read_bytes() == compatibility
    run_test1.main()
    assert (destination / 'compatibility.json').read_bytes() == compatibility


def test_dataset_hash_rejects_mutation(tmp_path):
    dataset = tmp_path / 'dataset'
    dataset.write_bytes(b'changed')
    np.savez(tmp_path / 'arrays.npz', values=np.ones(1))
    checkpoint = {'config': {'action_chunk_size': 1}, 'normalization': {}}
    run_test1.save_json(path=tmp_path / 'metadata.json', value={
        'checkpoint_sha256': 'sha', 'config': checkpoint['config'], 'normalization': {},
        'split': 'train', 'seed': 42, 'dataset': str(dataset), 'k': 1,
        'arrays_sha256': run_test1.file_hash(tmp_path / 'arrays.npz'), 'dataset_sha256': 'old'})
    with pytest.raises(RuntimeError, match='dataset hash mismatch'):
        run_test1.validate_bank(model=None, checkpoint=checkpoint, checkpoint_sha='sha',
                               split='train', dataset=dataset, names=[], seed=42, folder=tmp_path)


def test_valid_l1_ignores_padding():
    mask = torch.tensor([[True, False], [True, True]])
    prediction = torch.tensor([[[2.], [1e6]], [[2.], [4.]]])
    torch.testing.assert_close(valid_l1(left=prediction, right=torch.zeros_like(prediction), mask=mask),
                               torch.tensor([2., 3.]))
