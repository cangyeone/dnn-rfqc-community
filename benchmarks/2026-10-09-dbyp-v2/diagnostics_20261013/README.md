# Current DB/YP v2 diagnostics and timing (2026-10-13)

Aggregate results only: no observational RF arrays, per-record labels, prediction IDs or private input packet are published here. The comparison is the unchanged October 9 cohort: 10 configurations × 3 seeds, 163889 records at 205 stations, test 27823 at32 stations. Seven literature neural models and FCM use AG3; reference models use AG3 or all seven filter views. The references consume waveforms only and start from scratch; no hand-descriptor or phase-task transfer experiment is part of this cohort.

All stack/coherence diagnostics were recalculated over [-2.5,20] s. Reduced candidate pools share30 label-blind draws per fraction; FCM recomputes its station-context features with frozen centres/scalers/threshold. H–κ uses matched per-RF slowness and original AG3 samples, including 100 half-retained-catalogue draws; boundary and multiple-peak flags remain. These conditional results do not establish true crustal properties.

The 102-station small-event application has no manual labels or matching baz/p. It supports retention and external-main-stack similarity only; no small-event classification accuracy, azimuth coverage improvement or H–κ is claimed.

Timing contains10 current bundles ×3 fresh-process rounds, sequential FP32 on one RTX5090 for neural methods. FCM times the complete test station pool on CPU and is a different workload. Raw durations, source hashes and round ranges are retained.

Reproducible runners are in `scripts/paper_revision_20261013`. Running the collection/diagnostic/plot scripts requires user-owned experiment data and frozen run folders; this repository does not silently download or invent those inputs. See the paper's source README and runner CLI help. The public library/API and model files are unchanged by this diagnostic update. Three timing rounds are not new training seeds.
