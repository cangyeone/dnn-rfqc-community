"""Export completed DB/YP revision-2 bundles without observational data.

Run on the experiment host with --root EXPERIMENT --output NEW_DIRECTORY.
The original bundles remain byte-identical. Only bundles, hashes, aggregate
metrics and the frozen protocol are exported; no predictions, labels or RFs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import zipfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def export(root, output, release):
    protocol = json.loads((root/'protocol.json').read_text())
    assert protocol['version'] == 'dbyp_complete_20261009_v2'
    methods, seeds = protocol['methods'], protocol['training']['seeds']
    expected = {(m, s) for m in methods for s in seeds}
    assert len(expected) == 30 and len(methods) == 10 and len(seeds) == 3
    verification = json.loads((root/'summary/training_verification.json').read_text())
    assert verification['status'] == 'passed' and verification['n_runs'] == 30
    verified = {e['run']: e['completion_sha256'] for e in verification['receipts']}
    assert set(verified) == {f'{m}_seed{s}' for m, s in expected}
    assert {p.parent.name for p in (root/'runs').glob('*/completion.json')} == set(verified)
    protocol_hash = digest(root/'protocol.json')
    data_hash = digest(root/'cache/supervised/manifest.json')
    assert protocol_hash == verification['protocol_sha256']
    assert data_hash == protocol['data_manifest_sha256']
    manifest = json.loads((root/'cache/supervised/manifest.json').read_text())
    test_hashes = [digest(root/'cache/supervised/test'/name) for name in ['sample_ids.npy','labels.npy']]
    output.mkdir(parents=True, exist_ok=False)
    models = output/'models'; models.mkdir()
    entries, receipts = [], []
    checked = 0
    for method, seed in sorted(expected):
        source = root/'runs'/f'{method}_seed{seed}'
        done = json.loads((source/'completion.json').read_text())
        assert done['status'] == 'completed' and (done['method'], done['seed']) == (method, seed)
        assert digest(source/'completion.json') == verified[source.name]
        assert done['protocol_sha256'] == protocol_hash and done['dataset_manifest_sha256'] == data_hash
        assert [done['test_ids_sha256'], done['test_labels_sha256']] == test_hashes
        for name, sha in done['sha256'].items():
            assert digest(source/name) == sha, str(source/name)
            checked += 1
        bundle = json.loads((source/'bundle.json').read_text())
        assert bundle['method'] == method and bundle['seed'] == seed
        assert bundle['preprocessing']['fit_split'] == 'train'
        assert bundle['training']['phase_task_transfer'] is False
        files = {'bundle.json': digest(source/'bundle.json')}
        if (source/'weights.safetensors').exists():
            files['weights.safetensors'] = digest(source/'weights.safetensors')
            assert files['weights.safetensors'] == bundle['weights_sha256']
        archive = models/f'{method}-seed{seed}.zip'
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            for name in files:
                info = zipfile.ZipInfo(name, date_time=(2026,10,10,0,0,0))
                info.compress_type = zipfile.ZIP_DEFLATED
                with z.open(info, 'w') as dst, (source/name).open('rb') as src:
                    shutil.copyfileobj(src, dst)
        entry = dict(name=method, seed=seed,
                     url=f'https://github.com/cangyeone/dnn-rfqc-community/releases/download/{release}/{archive.name}',
                     sha256=digest(archive), size_bytes=archive.stat().st_size, files=files,
                     release=release, dataset_version=protocol['version'],
                     protocol_sha256=protocol_hash, dataset_manifest_sha256=data_hash)
        if method in ['reference_ag3','reference_multifilter']:
            entry['bundled_path'] = f'pretrained/{method}/seed{seed}'
        entries.append(entry)
        receipts.append(dict(method=method, seed=seed, source_completion_sha256=digest(source/'completion.json'),
                             bundle_files=files, threshold=bundle['threshold']))
        print(method, seed, archive.stat().st_size, flush=True)
    dump(output/'catalog_new.json', dict(release=release, models=entries))
    split_stations = {s: sum(e['split']==s for e in manifest['stations']) for s in ['train','val','test']}
    dump(output/'export_verification.json', dict(status='passed', release=release, dataset_version=protocol['version'],
         runs=len(receipts), source_files_verified=checked, source_bundles_byte_identical=True,
         observations_included=False, protocol_sha256=protocol_hash, dataset_manifest_sha256=data_hash,
         n_test=verification['n_test'], split_counts=manifest['split_counts'], split_stations=split_stations,
         discordant_records=manifest['discordant_records'], receipts=receipts))
    for name in ['method_summary.csv','seed_metrics.csv','single_multi_paired.csv','training_verification.json']:
        shutil.copy2(root/'summary'/name, output/name)
    shutil.copy2(root/'protocol.json', output/'protocol.json')
    with tarfile.open(output/'handoff.tar.gz','w:gz') as t:
        for p in sorted(output.iterdir()):
            if p.is_file() and p.name!='handoff.tar.gz': t.add(p, arcname=p.name)
        for entry in entries:
            if entry.get('bundled_path'):
                source = root/'runs'/f"{entry['name']}_seed{entry['seed']}"
                for name in entry['files']: t.add(source/name, arcname=entry['bundled_path']+'/'+name)
    print('Export complete:', output, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--release', default='v0.2.0')
    a = p.parse_args()
    export(a.root, a.output, a.release)
