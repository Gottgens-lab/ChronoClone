# Metaclone tutorial and density UMAP

Open [chronoclone_tutorial.ipynb](../../notebooks/chronoclone_tutorial.ipynb). The tutorial uses the included [Weinreb data](data/README.md) to identify metaclones and draw their distribution over Days 2, 4 and 6.

The input contains cell metadata and saved UMAP coordinates. It was already selected for progenitor origin and consecutive recovery, so the first filtering step checks eligibility within that cohort. Metaclone assignments are calculated in the tutorial. The [provenance record](data/provenance.json) documents the source workflow and the distinction between its saved result and the current execution.

## Run the tutorial

Run these commands from the repository root to create a separate environment with the [recorded dependencies](requirements.txt):

```bash
micromamba create -n chronoclone-tutorial -c conda-forge python=3.11.15 pip
micromamba run -n chronoclone-tutorial python -m pip install -r examples/weinreb/requirements.txt
micromamba run -n chronoclone-tutorial python -m pip install --no-deps -e .
micromamba run -n chronoclone-tutorial python -m ipykernel install --user --name chronoclone-tutorial
micromamba run -n chronoclone-tutorial python -m nbconvert --execute --inplace --ExecutePreprocessor.kernel_name=chronoclone-tutorial --ExecutePreprocessor.timeout=1800 notebooks/chronoclone_tutorial.ipynb
```

Select `chronoclone-tutorial` as the kernel for interactive execution. The notebook records the versions and command used for its saved run.

## Plot a density UMAP

After adding metaclone assignments to `adata.obs`, use:

```python
import chronoclone as cc
import matplotlib as mpl

with mpl.rc_context({"svg.fonttype": "none"}):
    fig, axes = cc.plot_metaclone_density_umap(
        adata,
        metaclone_key="metaclone",
        time_key="day",
        clone_key="LARRY_ID",
        time_order=[2, 4, 6],
        n_neighbors=100,
        min_vmax=5,
    )
    fig.savefig("density_umap.svg", bbox_inches="tight")
```

The function uses `adata.obsm["X_umap"]`; it does not fit an embedding or identify metaclones. Rows show days and columns show metaclones. Header `n` counts distinct clone IDs across all days.

Color reflects how many nearby cells share the plotted cell's metaclone and day. By default, neighbors are drawn from all input cells. A metaclone uses the same color scale across its displayed days, so its temporal pattern can be compared within a column. The gray background shows all cells; empty combinations retain their background. The plot uses neutral headers and no lineage-family labels or bars.

Use `time_order` and `metaclone_order` to choose the panel order. `clone_key=None` omits clone counts, and `neighbor_reference="assigned"` restricts neighbor calculations to cells with a metaclone assignment. The function returns a figure and a two-dimensional axes array. Foreground points and text remain editable in SVG; the dense background is rasterized.

## Outputs

- Threshold selection: [SVG](results/threshold_sweep.svg), [PDF](results/threshold_sweep.pdf), [PNG](results/threshold_sweep.png), and [table](results/threshold_sweep.csv).
- Density UMAP: [SVG](results/weinreb_metaclone_density_umap.svg), [PDF](results/weinreb_metaclone_density_umap.pdf), and [PNG](results/weinreb_metaclone_density_umap.png).
- [Clone assignments and support filters](results/clone_assignments.csv).
- [Execution settings and result summary](results/example_summary.json).

The input's [attribution and license](data/README.md) are separate from the package's code license.
