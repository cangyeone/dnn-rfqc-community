# Paper revision runners

These scripts accompany the current DB/YP v2 aggregate results. Dependencies: NumPy, SciPy, pandas, scikit-learn, Matplotlib, pypdf, threadpoolctl; model timing also needs PyTorch/safetensors and the frozen RFQC runtime. Use a separate analysis environment when dependencies are absent; do not upgrade the original training environment. No observational data are shipped.

1. On the host with the frozen experiment, run `collect_dbyp_revision_20261013.py --root ... --output ...` to verify completed fits and collect a private input packet.
2. Run `recompute_dbyp_diagnostics.py --root ... --output ...` for [-2.5,20] s diagnostics; fitting and thresholds stay frozen. It imports the host's frozen runtime and H–κ core.
3. Put `benchmarking.py` alongside `time_dbyp_models.py`, select one idle GPU by UUID, then run `time_dbyp_models.py --root ... --output ...`. It executes30 sequential fresh processes.
4. Transfer the private packet/derived results to an analysis computer. Run `plot_dbyp_revision.py --packet ... --diagnostics ... --timing ... --paper <paper-checkout> --review ...`; it verifies predictions and produces all32 station figures/booklets.
5. `small_dbyp_review.py --root ... --export <external-stacks.npz>` exports the main-cohort manual reference; its plot mode saves all102 small-event station stacks. `finish_dbyp_figures.py` makes additional figures/tables. See each script's `--help`.
6. Compile the paper then run `verify_dbyp_revision.py --root <paper-checkout> --review ...` to check metrics, counts, figure references, compilation logs and the atlas.

Current manuscript sources and frozen data/architecture snapshot are maintained in the Overleaf project. Aggregate results can be inspected here without access to the observational packet. EQR screening deployment is documented separately in the repository README; these paper runners do not change screening defaults or trained weights.
