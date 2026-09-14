# Metaclone tutorial

Open [chronoclone_tutorial.ipynb](../../notebooks/chronoclone_tutorial.ipynb). The tutorial uses the included [Weinreb data](data/README.md) to identify metaclones and draw their distribution over Days 2, 4 and 6.

The identification input contains cell metadata and saved UMAP coordinates. Additional compact inputs provide the full annotation background and Day-2 expression for the original plots. It was already selected for progenitor origin and consecutive recovery, so the first filtering step checks eligibility within that cohort. Metaclone assignments are calculated in the tutorial. The [provenance record](data/provenance.json) documents the source workflow and the distinction between its saved result and the current execution.

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

The tutorial restores the plot types from `0010_ChronoClone_Weinreb2020.ipynb`. The silhouette views and each metaclone’s cell-state dynamics have separate cells and files. Its current deterministic fit is distinct from the historical results saved in the source notebook.

| Plot | Editable SVG | PNG |
|---|---|---|
| Cell-state annotations | [SVG](results/cell_state_umap.svg) | [PNG](results/cell_state_umap.png) |
| Terminal states | [SVG](results/terminal_state_umap.svg) | [PNG](results/terminal_state_umap.png) |
| Starting progenitor states | [SVG](results/progenitor_umap.svg) | [PNG](results/progenitor_umap.png) |
| Threshold sweep | [SVG](results/threshold_sweep.svg) | [PNG](results/threshold_sweep.png) |
| Silhouettes before filtering | [SVG](results/silhouette_before_filtering.svg) | [PNG](results/silhouette_before_filtering.png) |
| Silhouettes after filtering | [SVG](results/silhouette_after_filtering.svg) | [PNG](results/silhouette_after_filtering.png) |
| Consensus distance heatmap | [SVG](results/consensus_distance_heatmap.svg) | [PNG](results/consensus_distance_heatmap.png) |
| Clone counts | [SVG](results/metaclone_clone_counts.svg) | [PNG](results/metaclone_clone_counts.png) |
| Day-2 marker dotplot | [SVG](results/day2_metaclone_markers.svg) | [PNG](results/day2_metaclone_markers.png) |
| Density UMAP | [SVG](results/weinreb_metaclone_density_umap.svg) | [PNG](results/weinreb_metaclone_density_umap.png) |
| Cell states after continuity filtering | [SVG](results/filtered_cell_state_umap.svg) | [PNG](results/filtered_cell_state_umap.png) |

Individual metaclone dynamics:
- MC1: [SVG](results/metaclone_dynamics/metaclone_01_cell_state_dynamics.svg), [PNG](results/metaclone_dynamics/metaclone_01_cell_state_dynamics.png).
- MC2: [SVG](results/metaclone_dynamics/metaclone_02_cell_state_dynamics.svg), [PNG](results/metaclone_dynamics/metaclone_02_cell_state_dynamics.png).
- MC3: [SVG](results/metaclone_dynamics/metaclone_03_cell_state_dynamics.svg), [PNG](results/metaclone_dynamics/metaclone_03_cell_state_dynamics.png).
- MC4: [SVG](results/metaclone_dynamics/metaclone_04_cell_state_dynamics.svg), [PNG](results/metaclone_dynamics/metaclone_04_cell_state_dynamics.png).
- MC5: [SVG](results/metaclone_dynamics/metaclone_05_cell_state_dynamics.svg), [PNG](results/metaclone_dynamics/metaclone_05_cell_state_dynamics.png).
- MC6: [SVG](results/metaclone_dynamics/metaclone_06_cell_state_dynamics.svg), [PNG](results/metaclone_dynamics/metaclone_06_cell_state_dynamics.png).
- MC7: [SVG](results/metaclone_dynamics/metaclone_07_cell_state_dynamics.svg), [PNG](results/metaclone_dynamics/metaclone_07_cell_state_dynamics.png).
- MC8: [SVG](results/metaclone_dynamics/metaclone_08_cell_state_dynamics.svg), [PNG](results/metaclone_dynamics/metaclone_08_cell_state_dynamics.png).
- MC9: [SVG](results/metaclone_dynamics/metaclone_09_cell_state_dynamics.svg), [PNG](results/metaclone_dynamics/metaclone_09_cell_state_dynamics.png).
- MC10: [SVG](results/metaclone_dynamics/metaclone_10_cell_state_dynamics.svg), [PNG](results/metaclone_dynamics/metaclone_10_cell_state_dynamics.png).
- MC11: [SVG](results/metaclone_dynamics/metaclone_11_cell_state_dynamics.svg), [PNG](results/metaclone_dynamics/metaclone_11_cell_state_dynamics.png).
- MC12: [SVG](results/metaclone_dynamics/metaclone_12_cell_state_dynamics.svg), [PNG](results/metaclone_dynamics/metaclone_12_cell_state_dynamics.png).
- MC13: [SVG](results/metaclone_dynamics/metaclone_13_cell_state_dynamics.svg), [PNG](results/metaclone_dynamics/metaclone_13_cell_state_dynamics.png).
- MC14: [SVG](results/metaclone_dynamics/metaclone_14_cell_state_dynamics.svg), [PNG](results/metaclone_dynamics/metaclone_14_cell_state_dynamics.png).
- MC15: [SVG](results/metaclone_dynamics/metaclone_15_cell_state_dynamics.svg), [PNG](results/metaclone_dynamics/metaclone_15_cell_state_dynamics.png).
- MC16: [SVG](results/metaclone_dynamics/metaclone_16_cell_state_dynamics.svg), [PNG](results/metaclone_dynamics/metaclone_16_cell_state_dynamics.png).
- MC17: [SVG](results/metaclone_dynamics/metaclone_17_cell_state_dynamics.svg), [PNG](results/metaclone_dynamics/metaclone_17_cell_state_dynamics.png).
- MC18: [SVG](results/metaclone_dynamics/metaclone_18_cell_state_dynamics.svg), [PNG](results/metaclone_dynamics/metaclone_18_cell_state_dynamics.png).

The consensus heatmap embeds its dense matrix at 600 dpi; text and cluster boundaries remain editable. The density UMAP rasterizes its gray background and retains editable foreground points and text. The other figures retain vector marks and text.

Tables: [clone assignments](results/clone_assignments.csv), [threshold sweep](results/threshold_sweep.csv), [Day-2 differential expression](results/day2_differential_expression.csv.gz), and [filtered marker genes](results/day2_marker_genes.csv.gz). [Run settings and summary](results/example_summary.json) describe the identification fit.

The input's [attribution and license](data/README.md) are separate from the package's code license.
