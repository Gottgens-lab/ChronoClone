# ChronoClone

ChronoClone identifies metaclones: groups of clones with similar cell-state trajectories over time. It builds clone-by-time features, combines clustering results across resolutions, and displays the resulting groups on a density UMAP.

Start with the [tutorial](notebooks/chronoclone_tutorial.ipynb). It uses the included public Weinreb dataset and takes you from cell metadata to metaclone identification and visualization. The saved UMAP coordinates and compact Day-2 expression input are included, so no additional data download is needed. The notebook includes the annotation UMAPs, silhouette plots, consensus heatmap, clone counts, cell-state dynamics, marker dotplot, and density UMAP from the supplied workflow.

## Get started

```bash
git clone https://github.com/hiromitsuhari/ChronoClone.git
cd ChronoClone
python -m pip install -e .
```

For a separate environment with the tutorial's recorded dependencies and Jupyter kernel, follow the [example guide](examples/weinreb/README.md#run-the-tutorial).

## Workflow

1. Select clones with a defined starting state and consecutive observations.
2. Count each clone's cells by time and state, then normalize within each clone and time point.
3. Build a consensus across Leiden clustering resolutions.
4. Select an agglomerative threshold using silhouette scores and apply explicit support filters.
5. Transfer the labels to cells and plot a density UMAP across days and metaclones.

The tutorial shows every step with visible results. See the [data description](examples/weinreb/data/README.md) for the input columns and attribution, and the [example guide](examples/weinreb/README.md) for plotting options and saved outputs.

## Core functions

| Function | Purpose |
|---|---|
| `filter_lineage_continuity` | Select clones with the required starting state and consecutive recovery. |
| `build_clone_table` | Count cells by clone, time and state. |
| `normalize_clone_table` | Convert counts to state proportions within each clone and time point. |
| `build_consensus_matrix` | Combine co-clustering frequencies across resolutions. |
| `sweep_thresholds` | Select a threshold and return metaclone labels and silhouette scores. |
| `transfer_metaclone_labels` | Map clone labels back to individual cells. |
| `plot_metaclone_density_umap` | Draw the days-by-metaclones density UMAP. |

## License and data

Package code is [MIT licensed](LICENSE). The included Weinreb data retain their [source attribution and CC BY 4.0 terms](examples/weinreb/data/README.md). The [paper repository](https://github.com/hiromitsuhari/ChronoClone_paper) contains the manuscript materials.
