"""
visualization.py
----------------
Plotting functions for ChronoClone outputs.
Figures are not saved automatically; each function documents its display behavior.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import NearestNeighbors
from numbers import Integral, Real
from typing import List, Optional


def plot_threshold_sweep(
    thresholds: List[float],
    silhouette_scores: List[float],
    cluster_counts: List[int],
    optimal_threshold: float,
    best_score: float,
) -> None:
    """
    Two-panel plot: silhouette score and metaclone count vs distance threshold.

    Parameters
    ----------
    thresholds : list of float
    silhouette_scores : list of float
    cluster_counts : list of int
    optimal_threshold : float
    best_score : float
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(thresholds, silhouette_scores, 'o-', markersize=4)
    ax1.axvline(x=optimal_threshold, color='r', linestyle='--',
                label=f'Optimal: {optimal_threshold:.2f}')
    ax1.set_xlabel('Distance Threshold')
    ax1.set_ylabel('Average Silhouette Score')
    ax1.set_title('Cluster Cohesion vs Threshold')
    ax1.legend()

    ax2.plot(thresholds, cluster_counts, 'o-', markersize=4, color='steelblue')
    ax2.axvline(x=optimal_threshold, color='r', linestyle='--')
    ax2.set_xlabel('Distance Threshold')
    ax2.set_ylabel('Number of Metaclones')
    ax2.set_title('Metaclone Count vs Threshold')

    plt.tight_layout()
    plt.show()


def plot_threshold_and_silhouette(
    thresholds: List[float],
    silhouette_scores: List[float],
    cluster_counts: List[int],
    optimal_threshold: float,
    best_score: float,
    filtered_table: pd.DataFrame,
    final_labels: np.ndarray,
    sample_silhouette_values: np.ndarray,
) -> None:
    """
    Combined 2-panel figure: threshold sweep (left) and silhouette knife plot (right).

    This matches the full visualization from the original pipeline:
    - Left: silhouette score vs distance threshold with optimal marker
    - Right: per-clone silhouette knife plot coloured by cluster

    Parameters
    ----------
    thresholds : list of float
    silhouette_scores : list of float
    cluster_counts : list of int
    optimal_threshold : float
    best_score : float
    filtered_table : pd.DataFrame
        Clone-level feature matrix (used for axis sizing).
    final_labels : np.ndarray
        Cluster label per clone at optimal threshold.
    sample_silhouette_values : np.ndarray
        Per-clone silhouette scores aligned to final_labels.

    Examples
    --------
    >>> cc.plot_threshold_and_silhouette(
    ...     thresholds, silhouette_scores, cluster_counts,
    ...     optimal_threshold, best_score,
    ...     filtered_table, final_labels, sample_silhouette_values
    ... )
    """
    n_clusters = len(np.unique(final_labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: threshold sweep
    ax1.plot(thresholds, silhouette_scores, 'o-', markersize=4)
    ax1.axvline(x=optimal_threshold, color='r', linestyle='--',
                label=f'Optimal: {optimal_threshold:.2f}')
    ax1.set_xlabel('Distance Threshold')
    ax1.set_ylabel('Average Silhouette Score')
    ax1.set_title('Cluster Cohesion vs Threshold')
    ax1.legend()

    # Right: silhouette knife plot
    y_lower = 10
    for i in range(n_clusters):
        ith_vals = np.sort(sample_silhouette_values[final_labels == i])
        y_upper = y_lower + len(ith_vals)
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax2.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_vals,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax2.text(-0.05, y_lower + 0.5 * len(ith_vals), str(i))
        y_lower = y_upper + 10

    ax2.axvline(x=best_score, color='red', linestyle='--',
                label=f'Avg: {best_score:.2f}')
    ax2.set_title(f'Silhouette Plot for {n_clusters} Clusters')
    ax2.set_xlabel('Silhouette Coefficient')
    ax2.set_ylabel('Cluster label')
    ax2.set_yticks([])
    ax2.set_xticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_silhouette(
    filtered_table: pd.DataFrame,
    final_labels: np.ndarray,
    best_score: float,
) -> None:
    """
    Standalone silhouette knife plot for all clones.

    Parameters
    ----------
    filtered_table : pd.DataFrame
        Feature matrix used for silhouette_samples().
    final_labels : np.ndarray
        Cluster label per clone.
    best_score : float
        Average silhouette score (plotted as vertical line).
    """
    n_clusters = len(np.unique(final_labels))
    sample_silhouette_values = silhouette_samples(filtered_table, final_labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    y_lower = 10

    for i in range(n_clusters):
        ith_vals = np.sort(sample_silhouette_values[final_labels == i])
        y_upper = y_lower + len(ith_vals)
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_vals,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * len(ith_vals), str(i))
        y_lower = y_upper + 10

    ax.axvline(x=best_score, color='red', linestyle='--',
               label=f'Avg: {best_score:.2f}')
    ax.set_title(f'Silhouette Plot — {n_clusters} Clusters')
    ax.set_xlabel('Silhouette Coefficient')
    ax.set_yticks([])
    ax.set_xticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_silhouette_filtered(
    results_df: pd.DataFrame,
    silhouette_threshold: float = -0.1,
) -> None:
    """
    Silhouette knife plot after filtering clones below a score threshold.
    Consistent colouring with the full plot (normalised by total cluster count).

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of build_results_df(). Must have 'Silhouette_Score'
        and 'Metaclone_Cluster' columns.
    silhouette_threshold : float
        Clones below this score are excluded. Default -0.1.
    """
    filtered_df = results_df[
        results_df['Silhouette_Score'] >= silhouette_threshold
    ].copy()
    filtered_avg_score = filtered_df['Silhouette_Score'].mean()

    print(f"Original clone count : {len(results_df)}")
    print(f"Filtered clone count : {len(filtered_df)}")
    print(f"New average silhouette: {filtered_avg_score:.3f}")

    n_clusters_total = results_df['Metaclone_Cluster'].nunique()
    present_clusters = sorted(filtered_df['Metaclone_Cluster'].unique())

    fig, ax = plt.subplots(figsize=(8, 8))
    y_lower = 10

    for i in present_clusters:
        ith_vals = np.sort(
            filtered_df[filtered_df['Metaclone_Cluster'] == i]['Silhouette_Score'].values
        )
        y_upper = y_lower + len(ith_vals)
        color = cm.nipy_spectral(float(i) / n_clusters_total)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_vals,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * len(ith_vals), str(i))
        y_lower = y_upper + 10

    ax.axvline(x=filtered_avg_score, color='red', linestyle='--',
               label=f'Avg: {filtered_avg_score:.2f}')
    ax.set_title(f'Silhouette Plot (Filtered >= {silhouette_threshold})')
    ax.set_xlabel('Silhouette Coefficient')
    ax.set_yticks([])
    ax.set_xticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_consensus_heatmap(
    consensus_distance: np.ndarray,
    final_labels: np.ndarray,
    linewidth: float = 0.3,
) -> None:
    """
    Heatmap of the consensus distance matrix sorted by cluster assignment.

    Parameters
    ----------
    consensus_distance : np.ndarray
    final_labels : np.ndarray
    linewidth : float
        Width of cluster boundary lines.
    """
    idx = np.argsort(final_labels)
    sorted_distance = consensus_distance[idx, :][:, idx]
    sorted_labels = final_labels[idx]

    plt.figure(figsize=(12, 10))
    sns.heatmap(sorted_distance, cmap='viridis_r',
                xticklabels=False, yticklabels=False,
                cbar_kws={'label': 'Distance'})

    for b in np.where(np.diff(sorted_labels) != 0)[0]:
        plt.axhline(y=b + 0.5, color='red', linewidth=linewidth)
        plt.axvline(x=b + 0.5, color='red', linewidth=linewidth)

    plt.title('Consensus Distance Matrix (Sorted by Metaclone)', fontsize=14)
    plt.tight_layout()
    # plt.savefig(f'consensus_heatmap_YYYYMMDD.png', dpi=300)
    plt.show()


def plot_cluster_distribution(
    final_labels: np.ndarray,
    color: str = 'steelblue',
) -> None:
    """
    Bar chart showing number of clones per metaclone cluster.

    Parameters
    ----------
    final_labels : np.ndarray
    color : str
    """
    cluster_sizes = pd.Series(final_labels).value_counts().sort_index()

    plt.figure(figsize=(10, 5))
    cluster_sizes.plot(kind='bar', color=color, edgecolor='white')
    plt.xlabel('Metaclone ID')
    plt.ylabel('Number of Clones')
    plt.title('Clone Distribution Across Metaclones')
    plt.tight_layout()
    # plt.savefig(f'cluster_distribution_YYYYMMDD.png', dpi=300)
    plt.show()


def plot_metaclone_dynamics(
    adata,
    metaclone_key: str = 'metaclone',
    time_key: str = 'day',
    feature_key: str = 'cell_type',
    normalize: bool = True,
    figsize_per_metaclone: tuple = (4, 3),
    max_metaclones: Optional[int] = None,
    palette: str = 'tab20',
) -> None:
    """
    Stacked bar chart showing cell type composition over time for each metaclone.

    Parameters
    ----------
    adata : AnnData
    metaclone_key : str
    time_key : str
    feature_key : str
    normalize : bool
        If True, plot proportions. If False, plot raw counts.
    figsize_per_metaclone : tuple
    max_metaclones : int, optional
    palette : str
    """
    obs = adata.obs.copy()
    obs = obs[obs[metaclone_key].notna()].copy()
    obs[metaclone_key] = obs[metaclone_key].astype(str)

    metaclones = sorted(obs[metaclone_key].unique(),
                        key=lambda x: int(x) if x.replace('-', '', 1).isdigit() else x)
    timepoints = sorted(obs[time_key].unique())
    features = sorted(obs[feature_key].unique())

    if max_metaclones is not None:
        metaclones = metaclones[:max_metaclones]

    cmap = plt.get_cmap(palette)
    colors = {f: cmap(i / max(len(features) - 1, 1)) for i, f in enumerate(features)}

    n_cols = min(4, len(metaclones))
    n_rows = int(np.ceil(len(metaclones) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(figsize_per_metaclone[0] * n_cols,
                                      figsize_per_metaclone[1] * n_rows))
    axes = np.array(axes).flatten()

    for ax_idx, mc in enumerate(metaclones):
        ax = axes[ax_idx]
        mc_data = obs[obs[metaclone_key] == mc]

        counts = (
            mc_data.groupby([time_key, feature_key])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=features, fill_value=0)
        )

        if normalize:
            row_sums = counts.sum(axis=1)
            counts = counts.div(row_sums, axis=0).fillna(0)

        bottom = np.zeros(len(counts))
        x = np.arange(len(counts))
        for feat in features:
            if feat in counts.columns:
                vals = counts[feat].values
                ax.bar(x, vals, bottom=bottom,
                       color=colors[feat], label=feat, width=0.7)
                bottom += vals

        ax.set_title(f'Metaclone {mc}', fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([str(t) for t in counts.index], fontsize=8)
        ax.set_xlabel(time_key, fontsize=8)
        ax.set_ylabel('Proportion' if normalize else 'Cell Count', fontsize=8)
        ax.set_ylim(0, 1.05 if normalize else None)
        ax.tick_params(axis='y', labelsize=7)

    for ax_idx in range(len(metaclones), len(axes)):
        axes[ax_idx].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[f]) for f in features]
    fig.legend(handles, features, title=feature_key,
               bbox_to_anchor=(1.01, 0.5), loc='center left',
               fontsize=8, title_fontsize=9)

    plt.suptitle('Metaclone Differentiation Dynamics', fontsize=13,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    # plt.savefig(f'metaclone_dynamics_YYYYMMDD.png', dpi=300, bbox_inches='tight')
    plt.show()


def _density_umap_order(values, requested, name):
    """Resolve scalar display labels without coercing their identities."""
    if requested is None:
        observed = values.dropna()
        if isinstance(values.dtype, pd.CategoricalDtype) and values.dtype.ordered:
            order = [value for value in values.cat.categories
                     if observed.eq(value).any()]
        else:
            order = list(pd.unique(observed))
    else:
        if isinstance(requested, (str, bytes)):
            raise ValueError(f'{name} must be a nonempty sequence of labels')
        try:
            order = list(requested)
        except TypeError as exc:
            raise ValueError(f'{name} must be a nonempty sequence of labels') from exc
    if not order:
        raise ValueError(f'{name} must contain at least one label')
    if any(not pd.api.types.is_scalar(value) or pd.isna(value) for value in order):
        raise ValueError(f'{name} must contain nonmissing scalar labels')
    if pd.Index(order).has_duplicates:
        raise ValueError(f'{name} must not contain duplicate labels')
    return order


def _density_umap_label(value, prefix):
    """Add a display prefix once, retaining the original selection value."""
    if isinstance(value, Real) and not isinstance(value, bool) and float(value).is_integer():
        text = str(int(value))
    else:
        text = str(value)
    if text.lower().startswith(prefix.lower()):
        return text
    return f'{prefix} {text}' if prefix == 'Day' else f'{prefix}{text}'


def plot_metaclone_density_umap(
    adata,
    *,
    metaclone_key='metaclone',
    time_key='day',
    clone_key='LARRY_ID',
    embedding_key='X_umap',
    time_order=None,
    metaclone_order=None,
    unassigned_values=(-1, '-1'),
    n_neighbors=100,
    neighbor_reference='all',
    min_vmax=5.0,
    cmap='viridis',
    point_size=50.0,
    background_size=3.5,
    alpha=0.8,
    figsize=None,
    show=False,
):
    """Plot a density UMAP grid for metaclones.

    Rows are times and columns are metaclones. For each foreground cell, color
    is the number of its nearest reference-cell neighbors that share both its
    metaclone and time. By default, the reference includes every input cell.
    The queried cell itself is eligible as a neighbor.
    This is local group occupancy, not a probability-density estimate or a
    measure of clone abundance. No embedding or metaclone partition is fitted.

    Parameters
    ----------
    adata : AnnData
        Supplies cell annotations in ``obs`` and a finite, two-dimensional
        embedding with one row per cell in ``obsm[embedding_key]``.
    metaclone_key, time_key : str
        Cell-annotation columns specifying the two grid dimensions.
    clone_key : str or None
        Column used for header counts of distinct, nonmissing clones across
        all times for each metaclone. ``None`` omits those counts.
    embedding_key : str
        Existing two-dimensional embedding; defaults to ``'X_umap'``.
    time_order, metaclone_order : sequence or None
        Explicit display orders may include absent groups, which retain only
        their background. Defaults use observed order, respecting an ordered
        categorical column's category order. Missing labels are not displayed.
    unassigned_values : sequence
        Sentinel metaclone values excluded from foreground groups, in addition
        to missing values. All cells still appear in the background. They also
        remain in the neighbor reference when ``neighbor_reference='all'``.
    n_neighbors : int
        Positive requested neighborhood size, capped at the reference size.
        A ball-tree neighbor search is fitted once on the reference, including
        metaclones or times omitted from the displayed grid.
    neighbor_reference : {'all', 'assigned'}
        ``'all'`` uses every input cell, including missing and sentinel
        metaclone labels. This matches the full-background reference in cell
        39 of the author notebook ``0010_ChronoClone_Weinreb2020.ipynb``.
        ``'assigned'`` uses only cells with nonmissing, nonsentinel metaclone
        labels. Unassigned cells can occupy neighbor slots in ``'all'`` mode
        but do not contribute to any foreground group's count.
    min_vmax : float
        Positive lower bound for each metaclone's color maximum. Its maximum
        otherwise equals the largest foreground count over displayed times.
        The same scale is used down each metaclone column, with minimum zero;
        scales can differ between metaclones.
    cmap : str or matplotlib.colors.Colormap
        Shared foreground colormap. Header text uses a single neutral color.
    point_size, background_size : float
        Scatter marker areas in points squared. Foreground size must be
        positive; background size may be zero.
    alpha : float
        Foreground opacity between zero and one.
    figsize : pair of float or None
        Figure size in inches. Defaults to 1.55 inches per metaclone and
        5.5 inches per three time rows.
    show : bool
        Call ``matplotlib.pyplot.show()`` when true. Defaults to false.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : numpy.ndarray
        Always a two-dimensional array shaped ``(n_times, n_metaclones)``.
        The caller controls saving; foreground points remain vector elements,
        while the dense background is rasterized. For editable SVG labels use
        ``matplotlib.rc_context({'svg.fonttype': 'none'})`` when saving.

    Notes
    -----
    This function does not modify the input object or global plotting settings.
    There are no lineage-family labels, family colors, group bars or colorbar.
    """
    if not hasattr(adata, 'obs') or not hasattr(adata, 'obsm'):
        raise TypeError('adata must provide obs annotations and obsm embeddings')
    obs = adata.obs
    required = [metaclone_key, time_key]
    if clone_key is not None:
        required.append(clone_key)
    for key in required:
        if key not in obs:
            raise KeyError(f'Annotation column not found: {key}')
    if embedding_key not in adata.obsm:
        raise KeyError(f'Embedding not found: {embedding_key}')
    try:
        coordinates = np.asarray(adata.obsm[embedding_key], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('Embedding must contain finite numeric coordinates') from exc
    if coordinates.shape != (len(obs), 2):
        raise ValueError('Embedding must have shape (number of cells, 2)')
    if not np.isfinite(coordinates).all():
        raise ValueError('Embedding coordinates must all be finite')
    if isinstance(n_neighbors, (bool, np.bool_)) or not isinstance(n_neighbors, Integral) or n_neighbors <= 0:
        raise ValueError('n_neighbors must be a positive integer')
    if neighbor_reference not in ('all', 'assigned'):
        raise ValueError("neighbor_reference must be 'all' or 'assigned'")
    try:
        min_vmax = float(min_vmax)
        point_size = float(point_size)
        background_size = float(background_size)
        alpha = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError('Color scale, marker sizes and alpha must be numeric') from exc
    if not np.isfinite(min_vmax) or min_vmax <= 0:
        raise ValueError('min_vmax must be finite and positive')
    if not np.isfinite(point_size) or point_size <= 0:
        raise ValueError('point_size must be finite and positive')
    if not np.isfinite(background_size) or background_size < 0:
        raise ValueError('background_size must be finite and nonnegative')
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('alpha must be between zero and one')

    metaclones = obs[metaclone_key]
    times = obs[time_key]
    try:
        assigned_mask = (metaclones.notna() & ~metaclones.isin(unassigned_values)).to_numpy(dtype=bool)
    except TypeError as exc:
        raise ValueError('unassigned_values must be a sequence of sentinel labels') from exc
    assigned_indices = np.flatnonzero(assigned_mask)
    if len(assigned_indices) == 0:
        raise ValueError('No assigned metaclone cells are available for the density reference')
    assigned_metaclones = metaclones.iloc[assigned_indices]
    assigned_times = times.iloc[assigned_indices]
    display_times = _density_umap_order(times, time_order, 'time_order')
    display_metaclones = _density_umap_order(assigned_metaclones, metaclone_order, 'metaclone_order')
    assigned_coordinates = coordinates[assigned_indices]
    reference_indices = (np.arange(len(obs)) if neighbor_reference == 'all'
                         else assigned_indices)
    reference = coordinates[reference_indices]
    k = min(int(n_neighbors), len(reference))
    neighbors = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(reference)
    neighbor_indices = reference_indices[
        neighbors.kneighbors(assigned_coordinates, return_distance=False)
    ]

    groups = {}
    maxima = {}
    for mc in display_metaclones:
        mc_mask = assigned_metaclones.eq(mc).fillna(False).to_numpy(dtype=bool)
        vmax = min_vmax
        for day in display_times:
            mask = mc_mask & assigned_times.eq(day).fillna(False).to_numpy(dtype=bool)
            foreground = assigned_coordinates[mask]
            full_group_mask = np.zeros(len(obs), dtype=bool)
            full_group_mask[assigned_indices[mask]] = True
            density = full_group_mask[neighbor_indices[mask]].sum(axis=1)
            groups[(mc, day)] = (foreground, density)
            if len(density):
                vmax = max(vmax, float(density.max()))
        maxima[mc] = vmax

    n_rows, n_columns = len(display_times), len(display_metaclones)
    if figsize is None:
        figsize = (1.55 * n_columns, 5.5 * n_rows / 3)
    try:
        size = np.asarray(figsize, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('figsize must contain two finite positive dimensions') from exc
    if size.shape != (2,) or not np.isfinite(size).all() or (size <= 0).any():
        raise ValueError('figsize must contain two finite positive dimensions')
    fig, axes = plt.subplots(n_rows, n_columns, squeeze=False, figsize=tuple(size))
    low, high = coordinates.min(axis=0) - 1.0, coordinates.max(axis=0) + 1.0
    ink = '#26235F'
    for column, mc in enumerate(display_metaclones):
        for row, day in enumerate(display_times):
            ax = axes[row, column]
            ax.scatter(coordinates[:, 0], coordinates[:, 1], s=background_size,
                       color='lightgrey', linewidths=0, rasterized=True, zorder=0)
            foreground, density = groups[(mc, day)]
            if len(density):
                order = np.argsort(density, kind='stable')
                ax.scatter(foreground[order, 0], foreground[order, 1],
                           c=density[order], cmap=cmap, s=point_size, alpha=alpha,
                           edgecolors='black', linewidths=0.15,
                           vmin=0, vmax=maxima[mc], rasterized=False, zorder=2)
            ax.set_aspect('equal')
            ax.set_xlim(low[0], high[0])
            ax.set_ylim(low[1], high[1])
            ax.axis('off')
            if column == 0:
                ax.text(-0.08, 0.5, _density_umap_label(day, 'Day'),
                        transform=ax.transAxes, color=ink, fontsize=11,
                        fontweight='bold', ha='right', va='center')
            if row == 0:
                title = _density_umap_label(mc, 'MC')
                if clone_key is not None:
                    mask = metaclones.eq(mc).fillna(False)
                    title += f'\nn={obs.loc[mask, clone_key].nunique(dropna=True)}'
                ax.set_title(title, color=ink, fontsize=11, fontweight='bold', pad=10)
    left = min(0.28, max(0.03, 0.62 / size[0]))
    right = max(left + 0.1, 1 - 0.18 / size[0])
    top = max(0.45, 1 - 0.7 / size[1])
    bottom = min(0.15, 0.15 / size[1])
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom,
                        wspace=0.02, hspace=0.03)
    if show:
        plt.show()
    return fig, axes
