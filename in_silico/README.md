# In-silico simulation and noise validation

This directory contains the synthetic benchmark used in the ChronoClone paper: **517 clones from 12 ODE programs, 13 compartments and four sampled times**. It includes deterministic generation, the fixed-start noise operators, the complete saved summary grid, plotting notebooks, bounded reproducibility checks, and the historical production implementation.

The current paper figures and table numbering are linked in [manuscript_panel_map.csv](manuscript_panel_map.csv). [reproducibility.csv](reproducibility.csv) maps each scientific validation to its inputs and workflow. The paper-specific source remains [ChronoClone_paper](https://github.com/hiromitsuhari/ChronoClone_paper), including its [manuscript figure index](https://github.com/hiromitsuhari/ChronoClone_paper/blob/main/manuscript/FIGURES.md) and [Table S1](https://github.com/hiromitsuhari/ChronoClone_paper/blob/main/manuscript/supplementary_tables/Table_S1/README.md).

## What the experiment tests

The deterministic compartment model specifies proliferation, differentiation and loss, sampled at ODE coordinates 0, 4, 9 and 15 (displayed as t1–t4). Gaussian count variation is applied at t2–t4, followed by uniform redistribution among states. The complete t1 block is unchanged; redistribution preserves every affected clone-time total. The operators represent controlled count and allocation distortion, without assigning either exclusively to biological or technical noise.

The historical experiment covers 0%–100% on both axes in 2% increments, with 100 repeats at each of 2,601 grid points and master seed 20260729. Every estimator receives the same ordered realized eligible cohort. Eligibility requires the declared initial compartment, at least two consecutive observations beginning at t1, and at least three total cells. Generating labels are used only after fitting to assess recovery.

The four-method paper comparison is **ChronoClone, CLADES mouse preprocessing, K-means and Ward**. It uses full-eligible-cohort adjusted Rand index (ARI), recovered group number and fit success. ChronoClone's separate retention rule is silhouette ≥ −0.1, with no group-size filter and no refitting. Input eligibility, assignment retention and accuracy on the same retained subset have different denominators and are reported separately. Ribbons are empirical 2.5th–97.5th percentiles across simulation repeats.

**Cyclone abundance-only is a separate applicability audit.** Its adapter contains an abundance/correlation/Euclidean/Ward core with fixed K = 6 and its dynamic filter disabled. These state programs yield two intended total-abundance trajectories, and uniform redistribution leaves the abundance input unchanged. Its failures and estimable results are retained without interpreting this adapter as the complete published Cyclone workflow. Clone2vec is not benchmarked because the generator does not supply its native cellular-neighborhood input.

## Run the portable workflows

Use the exact dependencies in [environment.yml](environment.yml). From the package repository root, a reader can create an isolated environment and register a kernel:

```bash
micromamba create -n chronoclone-benchmark -f in_silico/environment.yml
micromamba activate chronoclone-benchmark
python -m ipykernel install --user --name chronoclone-benchmark --display-name "ChronoClone benchmark"
```

Execute the notebooks in a working copy; they retain visible outputs and write figure files:

```bash
python in_silico/run_notebook.py in_silico/notebooks/0001_simulation.ipynb --kernel chronoclone-benchmark
python in_silico/run_notebook.py in_silico/notebooks/0002_noise_generation.ipynb --kernel chronoclone-benchmark
python in_silico/run_notebook.py in_silico/notebooks/0006_benchmark_evidence_and_smoke.ipynb --kernel chronoclone-benchmark
python in_silico/run_notebook.py in_silico/notebooks/0005_t1_locked_four_method_noise_results.ipynb --kernel chronoclone-benchmark
python in_silico/run_notebook.py in_silico/notebooks/0003_t1_locked_method_noise_results.ipynb --kernel chronoclone-benchmark
```

- `0001` reconstructs the deterministic ODE programs and checks them against the locked reference.
- `0002` illustrates and validates the exact fixed-t1 count operators; it does not refit estimators.
- `0006` checks the released grid and source identities, then fits one clean condition using the historical implementation.
- `0005` recreates the four-method recovery surfaces, isolated-noise ARI curves, input eligibility, assignment retention, recovered group counts, fit-success curves, and same-retained-subset sensitivity plots from the saved summaries.
- `0003` retains the corresponding five-adapter display, including the separate abundance-only applicability evidence.

To reconstruct one configured count tensor independently of the saved results:

```bash
python in_silico/scripts/generate_counts.py --gaussian 20 --uniform 50 --repeat 0 --output in_silico/reproductions/counts_g20_u50_r0.npz
```

The generator writes the realized counts, eligible clone identities and a checksum/provenance record. Choose a fresh output stem; registered inputs and historical result files are protected. To run a single estimator comparison instead:

```bash
python in_silico/scripts/run_smoke.py --output in_silico/reproductions/clean_smoke.json
```

Optional noise/repeat arguments select one configured condition. These bounded workflows do not claim that the full historical grid has been rerun.

## Complete repeat-level evidence

Compact summary tables in [results/aggregate](results/aggregate) cover the complete grid and are sufficient for the result figures. The **1,300,500-row, 153,477,867-byte** repeat-level CSV is held once as a versioned asset in the paper repository because it exceeds ordinary Git's per-file limit. It includes the separate fifth adapter, input tensor hashes, seeds, fitting diagnostics and retention payloads. It does not store every full noisy count tensor; the exact generator reconstructs those from configuration and seeds.

Read [LARGE_ASSET.json](LARGE_ASSET.json) for the registered identity. The release currently remains a private draft, so authenticate GitHub CLI with access to the paper repository, then run:

```bash
gh auth login
python in_silico/scripts/fetch_repeat_evidence.py
```

This retrieves tag `manuscript-v5-20260914` from `hiromitsuhari/ChronoClone_paper`, verifies byte count and SHA-256, and atomically publishes `in_silico/results/aggregate/native_method_noise_runs.csv.gz`. An already matching file is reused; a different existing file is not overwritten. The browser's temporary `untagged-…` draft URL is not a public download. The full file supports independent repeat-level inspection and the portable Cyclone applicability diagnostic; summary plots do not require it. After fetching it, run:

```bash
python in_silico/run_notebook.py in_silico/notebooks/0004_cyclone_applicability_diagnostic.ipynb --kernel chronoclone-benchmark
```

This optional notebook streams the complete repeat aggregate, checks abundance-only applicability and method-failure behavior, and reproduces the clean-reference numerical failure.

## Historical implementation and current package

[frozen_runtime](frozen_runtime) contains the unchanged scientific cores, configuration, dependency records, production launchers, aggregators and tests. The benchmark calls only one function from the historical ChronoClone package: `normalize_clone_table`. Its [normalization module](frozen_runtime/normalization_runtime/chronoclone/normalization.py) is retained byte-for-byte from the historical `chronoclone-0.1.0` wheel, with a minimal import entry point and the original license. The complete wheel is excluded from distribution.

[The numerical-runtime record](frozen_runtime/provenance/numerical_runtime.json) identifies the original wheel by its historical checksum and separately records the shipped module, initializer and license checksums. The portable fit wrappers verify and import that exact numerical module; the count generator uses the unchanged noise cores and does not require the wheel. These files reproduce the benchmark's numerical dependency without substituting the maintained package's current implementation or distributing a replacement package.

Historical full-production launchers retain their original topology and runtime identity requirements. They are inspectable provenance, not a newly validated portable full-grid command. A new full-grid execution must establish an independent output directory and compatible runtime identity; never append its shards to the historical aggregate. The active [0004 applicability notebook](notebooks/0004_cyclone_applicability_diagnostic.ipynb) adapts the historical audit to this directory layout while preserving its numerical scope.

Current Figure 1A/1B artwork is linked to the paper repository instead of copying older conceptual artwork. Scientific figures generated here retain their own descriptive filenames; the panel map connects them to v5 numbering. All inputs in this directory are synthetic. These controlled results establish recovery and failure boundaries for the declared generator, not universal method superiority or biological generalization.
