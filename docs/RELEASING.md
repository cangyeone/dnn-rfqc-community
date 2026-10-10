# Community release procedure

The distribution name remains `rfqc-bench`; the repository is
`cangyeone/dnn-rfqc-community`. v0.1.3 is based on original RFQC Bench commit
`d44b0166b7b25e48c02c736e5803db9946e54a46`. Historical validation files describe
their original releases and are preserved as provenance, not new experiments.

v0.1.4 is a software-only update (thresholds, view selection, resampling and
portable installation). Keep the immutable 53 model ZIP URLs at v0.1.3; upload
the new wheel and source distribution without duplicating those archives. The
eight Reference bundles remain included, with identical weights and hashes.

v0.1.5 changes the plain `record` output to unique basenames, with full relative
paths preserved in CSV sidecars. Summary schema 3 adds `record_entries` without
changing the source-file meaning of `retained_files`. Weights and predictions
are unchanged; the same software-only release procedure applies.

## Current release: v0.2.0

v0.2.0 replaces 30 model/seed entries with completed corrected-DB/YP v2
weights. Six embedded AG3/multi-filter bundles are updated; two embedded
AG1/AG5 bundles and the other historical controls are unchanged. The catalog
explicitly marks each entry with its release and dataset version. Never overwrite
v0.1.3 assets. Upload only the 30 new ZIPs, wheel, sdist and checksums to v0.2.0.

`scripts/export_dbyp_models.py` verifies all 30 completion receipts and their
207 source-file hashes on the experiment host, then copies bundles byte-for-byte
to deterministic ZIPs. No RFs, labels, per-record predictions or private sample
lists enter the public export. `scripts/reproduce_dbyp_summary.py` checks the
published aggregate metrics and paired differences. With the 30 new ZIPs use
`verify_community_models.py --archives /path/to/zips --release v0.2.0`.

## Code and trained weights

1. Run `python -m pytest -q` and `python scripts/reproduce_comparisons.py --output /tmp/rfqc-comparison-check`.
2. Run `python scripts/verify_community_models.py --archives /path/to/verified/model-zips`.
   This hashes all 53 model archives and their internal files, plus the eight
   Reference bundles included in `src/rfqc_bench/pretrained/`.
3. Verify the wheel and source distribution contain those eight actual bundles.
   They are small enough to store in Git without LFS. No observational waveforms,
   labels, sample lists or private predictions belong in the distribution.
4. Run `python -m build` and `python -m twine check dist/*`. Test the installed
   wheel outside the source checkout; verify that Reference predictions work
   with network access disabled, and the HTTP extra works when installed.
5. Commit source and validation receipts. Push without force, tag the checked
   commit, and upload wheel, sdist and the new immutable ZIPs to the matching
   release of **this repository**. Check release asset SHA256 values against
   `src/rfqc_bench/model_zoo.json` before making the release public.
6. Test a non-bundled model download from the published community URL. Do not
   claim PyPI availability until an actual PyPI upload and installation succeed.

The 23 historical entries remain byte-identical mirrors of original v0.1.0
RF-trained weights; the 30 updated entries are byte-identical exports from the
completed DB/YP v2 fits. None are source-author pretrained models. Future weight
changes need a new version, new catalog hashes and a complete validation record.

`scripts/export_model_zoo.py` is retained as a historical maintainer tool for
the original benchmark archive. It is not needed for installation or screening;
normal users should use the released bundles. It writes new exports to a chosen
directory and must not overwrite the checked community catalog by default.

## Optional PyPI publication

The manual-only `.github/workflows/publish-pypi.yml` is not triggered by pushes
or tags. It requires ownership/trusted-publisher setup for `rfqc-bench` on PyPI,
owner `cangyeone`, repository `dnn-rfqc-community`, workflow `publish-pypi.yml`,
environment `pypi`. No credentials belong in Git. Installation from the GitHub
wheel and versioned Git URL is supported independently of PyPI publication.
