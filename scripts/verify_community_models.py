"""Verify shipped Reference weights and optionally all release ZIPs; no downloads."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def verify(archives=None):
    package = ROOT/'src/rfqc_bench'
    catalog = json.loads((package/'model_zoo.json').read_text())
    entries = catalog['models']
    keys = {(m['name'], m['seed']) for m in entries}
    if len(keys) != len(entries) or len(entries) != 53:
        raise ValueError('Expected 53 unique registered model/seed bundles')
    expected = {('reference_ag1',20260929), ('reference_ag5',20260929)}
    expected |= {(name,seed) for name in ['reference_ag3','reference_multifilter']
                 for seed in [20260928,20260929,20260930]}
    bundled = set()
    for entry in entries:
        prefix = f'https://github.com/cangyeone/dnn-rfqc-community/releases/download/{catalog["release"]}/'
        if not entry['url'].startswith(prefix):
            raise ValueError('A model URL does not point to this community release')
        if entry.get('bundled_path'):
            bundled.add((entry['name'],entry['seed']))
            folder = (package/entry['bundled_path']).resolve()
            if not folder.is_relative_to(package.resolve()):raise ValueError('Invalid bundled path')
            for name, digest in entry['files'].items():
                if sha(folder/name) != digest:raise ValueError(f'Bundled checksum mismatch: {folder/name}')
            info = json.loads((folder/'bundle.json').read_text())
            if info['method'] != entry['name'] or info['seed'] != entry['seed']:
                raise ValueError('Bundle/catalog identity mismatch')
        if archives is not None:
            archive = Path(archives)/entry['url'].rsplit('/',1)[1]
            if archive.stat().st_size != entry['size_bytes'] or sha(archive) != entry['sha256']:
                raise ValueError(f'Archive checksum/size mismatch: {archive}')
            with zipfile.ZipFile(archive) as z:
                if sorted(z.namelist()) != sorted(entry['files']):raise ValueError(f'Unexpected members: {archive}')
                for name, digest in entry['files'].items():
                    if hashlib.sha256(z.read(name)).hexdigest() != digest:
                        raise ValueError(f'Archive member checksum mismatch: {archive}/{name}')
    if bundled != expected:raise ValueError('Missing or unexpected single/multi-filter bundled models')
    return dict(passed=True,configurations=len({e['name'] for e in entries}),registered_bundles=len(entries),
                bundled_verified=len(bundled),release_archives_verified=len(entries) if archives is not None else 0,
                all_urls_in_community_repository=True,observational_data_included=False)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives',type=Path)
    args=parser.parse_args()
    print(json.dumps(verify(args.archives),indent=2))
