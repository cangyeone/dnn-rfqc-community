# Community release procedure

The distribution name remains `rfqc-bench`; the repository is
`cangyeone/dnn-rfqc-community`. v0.1.3 is based on original RFQC Bench commit
`d44b0166b7b25e48c02c736e5803db9946e54a46`. Historical validation files describe
their original releases and are preserved as provenance, not new experiments.

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
   commit, and upload wheel, sdist and all 53 immutable ZIPs to the matching
   release of **this repository**. Check release asset SHA256 values against
   `src/rfqc_bench/model_zoo.json` before making the release public.
6. Test a non-bundled model download from the published community URL. Do not
   claim PyPI availability until an actual PyPI upload and installation succeed.

The current archives are byte-identical mirrors of the original v0.1.0 RF-trained
weights, not newly exported source-author models. New DB/YP training was paused
pending data review and contributes no weights to this release. Do not substitute
those unfinished or unverified experiments. Future weight changes need a new
version, new catalog hashes and a complete validation record.

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
