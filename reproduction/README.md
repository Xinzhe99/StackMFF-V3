# Reproducing the numerical tables

The bundle contains numerical inputs and Python producers for benchmark comparisons, ablations, robustness, model-selection sensitivity and real-stack evaluations. Experimental values are preserved. Current formatted table snapshots are in `expected_tables/`; previous snapshots are retained under `historical/`.

The current unified comparison includes 33 baseline settings and two proposed read-outs. ZMFF and ReDiffuse remain in the historical measurement archives but are excluded from the current complete-cohort table. `../benchmark/current/` provides the corresponding 35-row records and recomputed ranks.

Run from the repository root:

```bash
python reproduction/rebuild_tables.py
```

This runs the 13 numerical producers in a temporary copy, restores citation-backed venue labels and checks metric cells against the current snapshots. Caption wording and page placement are maintained with the manuscript and are not part of a numerical reproduction check. Tables derived directly from timing/deployment/downstream records are provided in `manual_tables/`, with available inputs in `measurement_records/`.

Individual producers remain available, for example:

```bash
python reproduction/work/paper/tables/make_tables.py
python reproduction/work/paper/tables/make_epoch_table.py
python reproduction/work/paper/tables/add_method_venues.py
```

Python, NumPy, SciPy and Matplotlib are required. Rebuilding numerical tables requires no GPU, MATLAB or image downloads. Re-running image fusion or training separately requires the source datasets, weights and implementation environments documented in the main README and third-party notices.

Roadmap Δ annotations subtract adjacent displayed four-decimal means so the printed arithmetic is consistent; all underlying values remain unrounded for plotting and seed-variation comparisons. The full-training row is distinct from the short-protocol cumulative study.
