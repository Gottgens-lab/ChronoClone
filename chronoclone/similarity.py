"""
similarity.py
-------------
Clonal trajectory similarity scoring between treated and control conditions.

Computes a consensus co-clustering score for each treated clone against
its matched metaclone in the control, then standardises to a Z-score
relative to the control distribution per metaclone.

Design
------
- Metaclone classification is performed on the **control** only.
- Treated condition tables are built from **all clones** — no filtering.
  Clones with missing time points will naturally receive lower similarity
  scores, which is biologically meaningful.
- The only required filter is that clone LARRY_IDs exist in the reference
  AnnData (i.e. they were observed at all). Use filter_by_larry_ids().
- Scoring uses shared feature columns across conditions. Use
  get_shared_feature_columns() with ignore_times to exclude time points
  absent in treated conditions (e.g. Day0).
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ============================================================================
# Barcode utilities
# ============================================================================

def extract_barcode(larry_id: str) -> str:
    """Convert a LARRY_ID to a standard barcode_rep format."""
    parts = larry_id.rsplit('_', 1)
    if len(parts) == 2:
        return f"{parts[0]}_{parts[1][1:]}"
    return larry_id


def add_barcode_rep_column(
    tables: List[pd.DataFrame],
    barcode_col: str = 'barcode_rep',
) -> None:
    """Add a barcode_rep column (in-place) to each table if not already present."""
    for df in tables:
        if barcode_col not in df.columns:
            df[barcode_col] = [extract_barcode(idx) for idx in df.index]


def build_metaclone_dict(
    adata,
    barcode_col: str = 'barcode_rep',
    metaclone_col: str = 'metaclone',
) -> Dict[str, str]:
    """Build a {barcode_rep: metaclone} dictionary from a control AnnData object."""
    map_df = adata.obs[[barcode_col, metaclone_col]].drop_duplicates().dropna()
    return dict(zip(map_df[barcode_col], map_df[metaclone_col]))


def filter_by_larry_ids(
    normalized_table: pd.DataFrame,
    adata,
    larry_id_col: str = 'LARRY_ID',
) -> pd.DataFrame:
    """
    Retain only clones whose LARRY_ID exists in adata.

    This is the only filtering applied to treated condition tables.
    Clones with missing time points are kept — their missing data will
    naturally lower their similarity scores rather than excluding them.

    Parameters
    ----------
    normalized_table : pd.DataFrame
        Normalized clone table. Index must be LARRY_IDs.
    adata : AnnData
        Reference AnnData. LARRY_IDs are taken from adata.obs[larry_id_col].
    larry_id_col : str
        Column in adata.obs containing LARRY_IDs. Default 'LARRY_ID'.

    Returns
    -------
    pd.DataFrame
        Filtered normalized table.

    Examples
    --------
    >>> synthetic_table_a = cc.filter_by_larry_ids(synthetic_table_a, adata)
    >>> synthetic_table_b  = cc.filter_by_larry_ids(synthetic_table_b, adata)
    """
    larry_ids = adata.obs[larry_id_col].unique()
    mask = normalized_table.index.isin(larry_ids)
    n_before = len(normalized_table)
    n_after = mask.sum()
    if n_before != n_after:
        print(f"filter_by_larry_ids: {n_before} -> {n_after} clones "
              f"({n_before - n_after} dropped, not found in adata)")
    return normalized_table.loc[mask].copy()


# ============================================================================
# Feature column utilities
# ============================================================================

def get_feature_columns(
    table: pd.DataFrame,
    time_prefix: str = 'Day',
    include_times: Optional[List[str]] = None,
    ignore_times: Optional[List[str]] = None,
) -> List[str]:
    """
    Get feature columns from a normalized table.

    Parameters
    ----------
    table : pd.DataFrame
    time_prefix : str
    include_times : list of str, optional
        Keep only columns for these time labels e.g. ['Day4', 'Day9', 'Day15'].
    ignore_times : list of str, optional
        Drop columns for these time labels e.g. ['Day0'].

    Returns
    -------
    list of str
    """
    cols = [c for c in table.columns if c.startswith(time_prefix) and '_' in c]
    if include_times is not None:
        cols = [c for c in cols if c.split('_')[0] in set(include_times)]
    if ignore_times is not None:
        cols = [c for c in cols if c.split('_')[0] not in set(ignore_times)]
    return cols


def get_shared_feature_columns(
    tables: List[pd.DataFrame],
    time_prefix: str = 'Day',
    include_times: Optional[List[str]] = None,
    ignore_times: Optional[List[str]] = None,
) -> List[str]:
    """
    Get the intersection of feature columns present in all tables.

    Use this to ensure all conditions use the same feature space when scoring.
    Pass ignore_times=['Day0'] to exclude time points absent in treated conditions.

    Parameters
    ----------
    tables : list of pd.DataFrame
    time_prefix : str
    include_times : list of str, optional
    ignore_times : list of str, optional

    Returns
    -------
    list of str (sorted)
    """
    feature_sets = [
        set(get_feature_columns(t, time_prefix, include_times, ignore_times))
        for t in tables
    ]
    if not feature_sets:
        return []
    shared = set.intersection(*feature_sets)
    return sorted(shared, key=lambda x: (x.split('_')[0], x))


# ============================================================================
# Pattern helpers
# ============================================================================

def get_present_days(row: pd.Series, all_features: List[str]) -> set:
    """Return the set of days for which a clone has non-zero values."""
    present = set()
    days = sorted(set(c.split('_')[0] for c in all_features if '_' in c))
    for day in days:
        day_cols = [c for c in all_features if c.startswith(f"{day}_")]
        if day_cols and row[day_cols].sum() > 0:
            present.add(day)
    return present


def get_day_pattern_key(days_set: set) -> tuple:
    """Return a sorted tuple from a set of day strings."""
    return tuple(sorted(days_set))


def get_features_for_days(days_set: set, all_features: List[str]) -> List[str]:
    """Return feature columns corresponding to the given days."""
    cols = []
    for day in sorted(days_set):
        cols.extend([c for c in all_features if c.startswith(f"{day}_")])
    return cols


# ============================================================================
# Core scoring
# ============================================================================

def build_pattern_groups(
    treated_df: pd.DataFrame,
    ctrl_metaclone_dict: Dict[str, str],
    all_features: List[str],
) -> Dict[tuple, List[dict]]:
    """
    Group treated clones by their observed day pattern.

    Only clones with a control metaclone mapping are retained — because
    similarity scores require a reference metaclone to compare against.
    Clones missing from ctrl_metaclone_dict were not observed in the control
    and cannot be assigned a metaclone, so they are excluded from scoring.
    """
    patterns = defaultdict(list)
    for idx, t_row in treated_df.iterrows():
        bc = t_row['barcode_rep']
        mc = ctrl_metaclone_dict.get(bc)
        if mc is None:
            continue
        t_days = get_present_days(t_row, all_features)
        if not t_days:
            continue
        pattern_key = get_day_pattern_key(t_days)
        patterns[pattern_key].append({
            'barcode_rep': bc,
            'metaclone': mc,
            'treated_idx': idx,
        })
    return patterns


def calculate_pattern_consensus(
    pattern_key: tuple,
    clone_info_list: List[dict],
    treated_df: pd.DataFrame,
    ctrl_df: pd.DataFrame,
    ctrl_metaclone_dict: Dict[str, str],
    all_features: List[str],
    resolutions: Optional[List[float]] = None,
    n_neighbors: int = 15,
) -> Optional[Tuple[np.ndarray, pd.Index]]:
    """Compute treated-vs-control consensus co-clustering for one day pattern."""
    if resolutions is None:
        resolutions = [round(x * 0.1, 1) for x in range(11, 41)]

    feature_cols = get_features_for_days(set(pattern_key), all_features)
    if not feature_cols:
        return None

    t_indices = [info['treated_idx'] for info in clone_info_list]
    t_matrix = treated_df.loc[t_indices, feature_cols].fillna(0)

    relevant_metaclones = {info['metaclone'] for info in clone_info_list}
    ctrl_subset_mask = ctrl_df['barcode_rep'].map(ctrl_metaclone_dict).isin(relevant_metaclones)
    c_matrix = ctrl_df.loc[ctrl_subset_mask, feature_cols].fillna(0)

    if c_matrix.empty:
        return None

    combined = pd.concat([t_matrix, c_matrix], axis=0)
    adata = sc.AnnData(X=combined.values.astype(float))
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')

    n_treated = len(t_matrix)
    n_ctrl = len(c_matrix)
    consensus = np.zeros((n_treated, n_ctrl))

    for res in resolutions:
        sc.tl.leiden(adata, resolution=res, key_added='leiden')
        labels = adata.obs['leiden'].values
        t_labels = labels[:n_treated]
        c_labels = labels[n_treated:]
        for i in range(n_treated):
            consensus[i, c_labels == t_labels[i]] += 1

    consensus /= len(resolutions)
    return consensus, c_matrix.index


def compute_similarity_scores(
    treated_df: pd.DataFrame,
    ctrl_df: pd.DataFrame,
    ctrl_metaclone_dict: Dict[str, str],
    all_features: List[str],
    condition_name: str,
    resolutions: Optional[List[float]] = None,
    n_neighbors: int = 15,
) -> pd.DataFrame:
    """
    Compute pattern-consensus trajectory similarity scores for one condition.

    Parameters
    ----------
    treated_df : pd.DataFrame
        Normalized clone table for the treated condition. No filtering applied;
        all clones in the table are scored. Must have a 'barcode_rep' column.
    ctrl_df : pd.DataFrame
        Normalized clone table for the control condition.
    ctrl_metaclone_dict : dict
        {barcode_rep: metaclone} from control AnnData.
    all_features : list of str
        Shared feature columns to use (obtained via get_shared_feature_columns).
    condition_name : str
    resolutions : list of float, optional
    n_neighbors : int

    Returns
    -------
    pd.DataFrame with columns:
        barcode_rep, metaclone, day_pattern, n_days, pattern_similarity, comparison
    """
    print(f"Processing: {condition_name}")
    patterns = build_pattern_groups(treated_df, ctrl_metaclone_dict, all_features)
    all_scores = []

    for pattern_key, clone_info_list in patterns.items():
        result = calculate_pattern_consensus(
            pattern_key, clone_info_list,
            treated_df, ctrl_df,
            ctrl_metaclone_dict, all_features,
            resolutions, n_neighbors,
        )
        if result is None:
            continue

        consensus, ctrl_indices = result
        ctrl_barcodes = ctrl_df.loc[ctrl_indices, 'barcode_rep']
        ctrl_metaclones = ctrl_barcodes.map(ctrl_metaclone_dict).values

        for i, info in enumerate(clone_info_list):
            target_mc = info['metaclone']
            mc_mask = (ctrl_metaclones == target_mc)
            if mc_mask.sum() > 0:
                sim_score = consensus[i, mc_mask].mean()
                all_scores.append({
                    'barcode_rep': info['barcode_rep'],
                    'metaclone': target_mc,
                    'day_pattern': ','.join(pattern_key),
                    'n_days': len(pattern_key),
                    'pattern_similarity': sim_score,
                    'comparison': condition_name,
                })

    return pd.DataFrame(all_scores)


def compute_similarity_scores_multi(
    condition_tables: Dict[str, pd.DataFrame],
    ctrl_key: str,
    ctrl_metaclone_dict: Dict[str, str],
    all_features: List[str],
    resolutions: Optional[List[float]] = None,
    n_neighbors: int = 15,
    min_ctrl_n: int = 2,
    min_ctrl_std: float = 1e-6,
) -> pd.DataFrame:
    """
    Compute similarity scores for multiple conditions against one control table.

    Combines the raw pattern-similarity computation with Z-score
    standardisation in a single step.  The returned ``similarity_score``
    column is the **absolute** Z-score (direction does not matter).

    Parameters
    ----------
    condition_tables : dict
        {condition_label: normalized_table}.
        Synthetic example: {'SyntheticReference': reference_table,
                            'SyntheticA': comparison_table}
    ctrl_key : str
        Key in condition_tables to use as control reference.
    ctrl_metaclone_dict : dict
    all_features : list of str
    resolutions : list of float, optional
    n_neighbors : int
    min_ctrl_n : int, default 2
        Minimum number of control observations per metaclone to compute
        a valid z-score.  Metaclones with fewer are dropped.
    min_ctrl_std : float, default 1e-6
        Minimum standard deviation for the control distribution.  Guards
        against division by near-zero.

    Returns
    -------
    pd.DataFrame
        Concatenated results across all conditions with an additional
        ``similarity_score`` column (|z-score|).
    """
    ctrl_df = condition_tables[ctrl_key]
    all_results = []
    for condition_name, treated_df in condition_tables.items():
        result = compute_similarity_scores(
            treated_df=treated_df,
            ctrl_df=ctrl_df,
            ctrl_metaclone_dict=ctrl_metaclone_dict,
            all_features=all_features,
            condition_name=condition_name,
            resolutions=resolutions,
            n_neighbors=n_neighbors,
        )
        all_results.append(result)
    traj_scores = pd.concat(all_results, ignore_index=True)

    # --- Z-score standardisation per metaclone using control distribution ---
    control_stats = {}
    for mc in traj_scores['metaclone'].unique():
        if pd.isna(mc):
            continue
        ctrl_data = traj_scores[
            (traj_scores['metaclone'] == mc) &
            (traj_scores['comparison'] == ctrl_key)
        ]['pattern_similarity']
        if len(ctrl_data) < min_ctrl_n or ctrl_data.std() < min_ctrl_std:
            continue
        control_stats[mc] = {'mean': ctrl_data.mean(), 'std': ctrl_data.std()}

    def _similarity_score(row):
        mc = row['metaclone']
        if mc in control_stats:
            z = (row['pattern_similarity'] - control_stats[mc]['mean']) / control_stats[mc]['std']
            return np.abs(z)
        return np.nan

    traj_scores['similarity_score'] = traj_scores.apply(_similarity_score, axis=1)
    return traj_scores.dropna(subset=['similarity_score'])


# ============================================================================
# Z-score standardisation (deprecated — now integrated into
# compute_similarity_scores_multi)
# ============================================================================

def compute_zscore(
    traj_scores: pd.DataFrame,
    control_label: str = 'Control_Trajectory',
    min_ctrl_n: int = 2,
    min_ctrl_std: float = 1e-6,
) -> pd.DataFrame:
    """Standardise trajectory similarity scores per metaclone using control.

    .. deprecated::
        Use :func:`compute_similarity_scores_multi` instead, which now
        returns ``similarity_score`` (|z-score|) directly.
    """
    import warnings
    warnings.warn(
        "compute_zscore is deprecated. "
        "compute_similarity_scores_multi now returns 'similarity_score' directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    control_stats = {}

    for mc in traj_scores['metaclone'].unique():
        if pd.isna(mc):
            continue
        ctrl_data = traj_scores[
            (traj_scores['metaclone'] == mc) &
            (traj_scores['comparison'] == control_label)
        ]['pattern_similarity']

        if len(ctrl_data) < min_ctrl_n or ctrl_data.std() < min_ctrl_std:
            continue
        control_stats[mc] = {'mean': ctrl_data.mean(), 'std': ctrl_data.std()}

    def _zscore(row):
        mc = row['metaclone']
        if mc in control_stats:
            return (row['pattern_similarity'] - control_stats[mc]['mean']) / control_stats[mc]['std']
        return np.nan

    df = traj_scores.copy()
    df['z_score'] = df.apply(_zscore, axis=1)
    return df.dropna(subset=['z_score'])


# ============================================================================
# Visualization
# ============================================================================

def plot_similarity_score(
    df: pd.DataFrame,
    condition_map: Optional[Dict[str, str]] = None,
    condition_order: Optional[List[str]] = None,
    custom_colors: Optional[Dict[str, str]] = None,
    threshold: float = 1.282,
    x_offset: float = 0.27,
    line_height: float = 0.2,
    bracket_spacing: float = 2.0,
    figsize: tuple = (20, 6),
    y_min: float = 0,
    y_max: float = 10,
) -> None:
    """Boxplot + stripplot of trajectory similarity scores with significance brackets.

    Plots the ``similarity_score`` column (|z-score|), so all values are
    positive and the threshold line is drawn at a positive value (default
    1.282, corresponding to the 90th-percentile of the standard normal).

    By default, comparison labels are preserved and displayed in observed order.
    ``condition_map`` can rename or select labels, and ``condition_order`` can
    override their order. The first displayed condition is the reference for
    significance comparisons; supply an explicit order to choose that reference.
    Colors are assigned from a categorical palette unless supplied by the caller.
    """
    if condition_map is None:
        condition_map = {label: label for label in df['comparison'].dropna().unique()}
    if condition_order is None:
        condition_order = list(df['comparison'].map(condition_map).dropna().unique())
    if not condition_order:
        raise ValueError('condition_order must contain at least one condition')
    if custom_colors is None:
        custom_colors = dict(zip(
            condition_order, sns.color_palette('tab10', n_colors=len(condition_order))))

    df_plot = df.copy()
    df_plot['Condition'] = df_plot['comparison'].map(condition_map)
    df_plot = df_plot.dropna(subset=['similarity_score', 'Condition'])
    df_plot = df_plot.sort_values('metaclone')
    metaclones = sorted(df_plot['metaclone'].unique())

    plt.figure(figsize=figsize)
    ax = sns.boxplot(
        data=df_plot, x='metaclone', y='similarity_score', hue='Condition',
        hue_order=condition_order, palette=custom_colors,
        dodge=True, showfliers=False,
        boxprops={'alpha': 0.35}, whiskerprops={'alpha': 0.25},
        capprops={'alpha': 0.25}, medianprops={'alpha': 0.25},
    )
    sns.stripplot(
        data=df_plot, x='metaclone', y='similarity_score', hue='Condition',
        hue_order=condition_order, palette=custom_colors,
        dodge=True, size=7, alpha=0.85, jitter=0.15,
        edgecolor='none', ax=ax,
    )

    def _sig_text(p):
        if p < 0.001: return '***'
        if p < 0.01:  return '**'
        if p < 0.05:  return '*'
        return f'p={p:.2f}'

    ctrl_name = condition_order[0]
    cond_names = condition_order[1:]

    for i, mc in enumerate(metaclones):
        mc_data = df_plot[df_plot['metaclone'] == mc]
        ctrl = mc_data[mc_data['Condition'] == ctrl_name]['similarity_score'].values
        local_max = mc_data['similarity_score'].max()
        if np.isnan(local_max):
            local_max = 0

        for j, cond in enumerate(cond_names):
            treated = mc_data[mc_data['Condition'] == cond]['similarity_score'].values
            if len(ctrl) == 0 or len(treated) == 0:
                continue
            _, p_val = mannwhitneyu(ctrl, treated, alternative='two-sided')
            text = _sig_text(p_val)
            f_weight = 'bold' if p_val < 0.05 else 'normal'
            f_size = 14 if p_val < 0.05 else 10
            x1 = i - x_offset
            x2 = i + (j * x_offset)
            y = local_max + 0.5 + (j * bracket_spacing)
            plt.plot([x1, x1, x2, x2],
                     [y, y + line_height, y + line_height, y],
                     lw=1.2, c='#333333')
            plt.text((x1 + x2) * 0.5, y + line_height + 0.05, text,
                     ha='center', va='bottom', color='black',
                     fontsize=f_size, fontweight=f_weight)
            local_max = y + line_height

    handles, labels = ax.get_legend_handles_labels()
    unique_labels = {}
    final_handles, final_labels_list = [], []
    for h, l in zip(handles, labels):
        if l in custom_colors and l not in unique_labels:
            unique_labels[l] = True
            final_handles.append(h)
            final_labels_list.append(l)

    sig_handles = [
        Line2D([0], [0], color='w', marker='s', markerfacecolor='w', label='***  p < 0.001'),
        Line2D([0], [0], color='w', marker='s', markerfacecolor='w', label='**   p < 0.01'),
        Line2D([0], [0], color='w', marker='s', markerfacecolor='w', label='*    p < 0.05'),
    ]

    plt.subplots_adjust(bottom=0.3)
    first_legend = plt.legend(
        final_handles, final_labels_list, title='Condition',
        fontsize=15, title_fontsize=17,
        loc='upper left', bbox_to_anchor=(0.0, -0.15),
        frameon=True, ncol=3,
    )
    plt.gca().add_artist(first_legend)
    plt.legend(
        handles=sig_handles, title='Significance',
        fontsize=13, title_fontsize=15,
        loc='upper left', bbox_to_anchor=(0.4, -0.15),
        frameon=True, ncol=3,
    )

    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    plt.text(
        x=len(metaclones) - 0.5, y=threshold + 0.3,
        s=f'Threshold ({threshold})',
        color='red', fontsize=12, ha='right', va='bottom',
    )
    plt.title('Clonal Trajectory Similarity Score with Statistical Significance',
              fontsize=24, pad=30, fontweight='bold')
    plt.xlabel('Metaclone ID', fontsize=20, labelpad=15)
    plt.ylabel('Similarity Score (|Z-Score|)', fontsize=20, labelpad=15)
    plt.xlim(-0.7, len(metaclones) - 0.3)
    plt.ylim(y_min, y_max)
    plt.grid(axis='y', alpha=0.15, linestyle='--')
    plt.xticks(fontsize=18, fontweight='bold')
    plt.yticks(fontsize=14)
    plt.show()


# Backward-compatible alias
def plot_similarity_zscore(*args, **kwargs):
    """Deprecated. Use :func:`plot_similarity_score` instead."""
    import warnings
    warnings.warn(
        "plot_similarity_zscore is deprecated. Use plot_similarity_score instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return plot_similarity_score(*args, **kwargs)
