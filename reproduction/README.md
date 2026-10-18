# Exact table-generation inputs

This folder mirrors the small part of the development `work/` tree actually read by the 13 active table generators. The generators and input CSVs are copied without changing experiment values. It includes the final training/validation epoch records, leave-one-out ablation inputs, robustness summaries, oracle results, color fidelity and real-stack evaluations. `expected_tables/` preserves the corresponding 17 current TeX table snapshots.

The main benchmark records in `../benchmark/` contain only the paper's method subset. Some historical input files here also contain development variants; the generators explicitly select their published rows. Alternate `*_finalws`/Hms checkpoints are not the final model.

Run each generator from the repository root, for example:

```bash
python reproduction/work/paper/tables/make_tables.py
python reproduction/work/paper/tables/make_epoch_table.py
python reproduction/work/paper/tables/make_color_table.py
```

Other generators in the same folder create the ablation, fusion, oracle, focus-label, no-reference range and robustness tables. `make_lfsf_table.py` explicitly separates values quoted from other publications from our re-evaluated image results; quoted values are not presented as our reruns.

Compare the generated table bodies against `expected_tables/`. Minor caption/float-placement differences do not change the data. Python, NumPy, SciPy and Matplotlib are required (Matplotlib is imported by the shared roadmap generator). No MATLAB, GPU or source image downloads are needed to rebuild these tables. This is a table-generation snapshot; rerunning training or baseline image generation additionally requires the datasets, model environments and original implementations described in the paper supplement.
