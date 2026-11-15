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

Releases are tagged. Version 0.2.0 adds several features per time point (see below) and changes no existing function; to install the earlier version, use `git checkout v0.1.0` after cloning, or `python -m pip install git+https://github.com/hiromitsuhari/ChronoClone.git@v0.1.0`. See the [changelog](CHANGELOG.md).

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
| `build_multi_feature_table` | Build the clone table from several features per time point, with different features allowed at different time points (0.2.0). |
| `balance_feature_groups` | Scale groups of features so that each adds a set share of the clone-to-clone distance (0.2.0). |

## Several features per time point

From 0.2.0, a clone can be described by several features at each time point, for example its cell types together with
binned measurements of every cell, and the set of features can differ between time points. `build_multi_feature_table`
replaces step 2 of the workflow: for every clone, time point and feature it gives the shares of the clone's cells in
each category. Features whose categories have an order (bins of a measurement, size classes) can be listed in
`ordered_features`, so that neighbouring categories count as close. When some features belong together (several
measurements of one property), `balance_feature_groups` gives every group the same share of the distance before
clustering. The rest of the workflow is unchanged.

```python
import chronoclone as cc

features_by_time = {0: ['marker_level', 'cell_size'],
                    1: ['cell_type', 'marker_level', 'cell_size'],
                    2: ['cell_type', 'marker_level', 'cell_size']}
categories = {'marker_level': ['low', 'mid', 'high'], 'cell_size': ['low', 'mid', 'high'],
              'cell_type': ['HSC', 'MPP', 'Mk', 'Ery']}
counts, shares = cc.build_multi_feature_table(adata, features_by_time, categories, clone_id_col='clone_id',
                                              time_col='day', ordered_features=['marker_level', 'cell_size'])
groups = {'cell_type': 'cell type', 'marker_level': 'marker', 'cell_size': 'size'}
balanced, report = cc.balance_feature_groups(shares, groups)
filtered, traj, consensus = cc.build_consensus_matrix(balanced, min_cells=1)
out = cc.sweep_thresholds(1 - consensus, filtered)
```

## Paper simulations and noise validation

The [in silico benchmark](in_silico/README.md) contains the paper’s deterministic simulation, fixed-start Gaussian and redistribution noise, complete validation summaries, executed notebooks, and editable figures. Start there to reproduce the synthetic experiment.

## License and data

Package code is [MIT licensed](LICENSE). The included Weinreb data retain their [source attribution and CC BY 4.0 terms](examples/weinreb/data/README.md). The [paper repository](https://github.com/hiromitsuhari/ChronoClone_paper) contains the manuscript materials.
