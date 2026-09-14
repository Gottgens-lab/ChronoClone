"""
preprocessing.py
----------------
Functions for preparing clone-level feature tables from AnnData or DataFrames.
Designed to be agnostic to feature type (RNA, proteome, imaging, etc.).
"""

import pandas as pd
import anndata as ad
from typing import Optional, List, Union


def rename_labels(
    adata: ad.AnnData,
    source_col: str,
    target_col: str,
    merge_map: Optional[dict] = None,
    merge_as: Optional[str] = None,
    merge_values: Optional[List[str]] = None,
) -> ad.AnnData:
    """
    Rename or merge categorical labels in adata.obs.

    Parameters
    ----------
    adata : AnnData
    source_col : str
        Source column in adata.obs (e.g. 'cell_type').
    target_col : str
        Name for new column.
    merge_map : dict, optional
        Full mapping dict {old_label: new_label}.
    merge_as : str, optional
        Single label to assign to all values in merge_values (e.g. 'Start').
    merge_values : list of str, optional
        Values in source_col to rename to merge_as (e.g. ['HSC', 'HSPC']).

    Returns
    -------
    AnnData with target_col added to obs.
    """
    adata.obs[target_col] = adata.obs[source_col].astype(str)

    if merge_map:
        adata.obs[target_col] = adata.obs[target_col].replace(merge_map)

    if merge_as and merge_values:
        adata.obs.loc[adata.obs[source_col].isin(merge_values), target_col] = merge_as

    return adata


def filter_lineage_continuity(
    adata: ad.AnnData,
    condition_key: str = 'condition',
    condition_val: str = 'C',
    day_key: str = 'day',
    cell_type_key: str = 'cellType',
    start_cell_type: Union[str, List[str]] = 'Start',
    barcode_key: str = 'LARRY_ID',
    min_persistence: int = 2,
) -> ad.AnnData:
    """
    Filter AnnData for a specific condition and retain only lineage barcodes
    that demonstrate strict consecutive temporal continuity from the first
    time point.

    A clone is retained if:
    1. It belongs to the specified condition.
    2. At t0, ALL its cells are start_cell_type (no contamination).
    3. It is observed consecutively across >= min_persistence time points
       from t0 with no gaps.

    Parameters
    ----------
    adata : AnnData
        Full annotated data matrix.
    condition_key : str
        Column in adata.obs for experimental condition.
    condition_val : str
        Value to filter on (e.g. 'C' for control).
    day_key : str
        Column in adata.obs for time points.
    cell_type_key : str
        Column in adata.obs for cell type labels.
    start_cell_type : str or list of str
        Cell type(s) defining clone origin at t0.
        If a list is given, barcodes are valid if ALL their t0 cells
        belong to ANY of the listed types (e.g. ['HSC', 'HSPC']).
    barcode_key : str
        Column in adata.obs for clone barcodes.
    min_persistence : int
        Minimum number of consecutive time points from t0. Default 2.

    Returns
    -------
    AnnData
        Filtered AnnData with only qualifying clones.

    Examples
    --------
    >>> adata_ctrl = cc.filter_lineage_continuity(
    ...     adata,
    ...     condition_key='condition',
    ...     condition_val='C',
    ...     day_key='day',
    ...     cell_type_key='cell_type',
    ...     start_cell_type=['HSC', 'HSPC'],
    ...     barcode_key='LARRY_ID',
    ...     min_persistence=2
    ... )
    """
    # Normalise start_cell_type to a set for uniform handling
    if isinstance(start_cell_type, str):
        start_types = {start_cell_type}
    else:
        start_types = set(start_cell_type)

    print(f"Starting lineage filtration for Condition: '{condition_val}'...")
    print(f"Start cell type(s): {start_types}")

    # Step 1: Filter for condition
    adata_sub = adata[adata.obs[condition_key] == condition_val].copy()

    # Step 2: Auto-detect sorted time points
    time_points = sorted(adata_sub.obs[day_key].unique())
    t0 = time_points[0]
    print(f"Detected time points: {time_points}")

    # Step 3: Identify valid start barcodes at t0
    # Valid = has >= 1 start-type cell at t0 AND no non-start-type cells at t0
    mask_t0_start = (
        (adata_sub.obs[day_key] == t0) &
        (adata_sub.obs[cell_type_key].isin(start_types))
    )
    mask_t0_other = (
        (adata_sub.obs[day_key] == t0) &
        (~adata_sub.obs[cell_type_key].isin(start_types))
    )
    barcodes_start = set(adata_sub.obs.loc[mask_t0_start, barcode_key].dropna().unique())
    barcodes_contam = set(adata_sub.obs.loc[mask_t0_other, barcode_key].dropna().unique())

    valid_start_barcodes = (barcodes_start - barcodes_contam)
    valid_start_barcodes = {b for b in valid_start_barcodes if b != ''}
    print(f"Valid Start barcodes at Day {t0}: {len(valid_start_barcodes)}")

    # Step 4: Filter by strict consecutive temporal continuity
    bc_day_counts = (
        adata_sub.obs
        .groupby(barcode_key)[day_key]
        .apply(set)
        .to_dict()
    )

    barcodes_to_keep = set()
    required_initial_sequence = set(time_points[:min_persistence])

    for bc in valid_start_barcodes:
        if bc not in bc_day_counts:
            continue

        observed_days = bc_day_counts[bc]

        if not required_initial_sequence.issubset(observed_days):
            continue

        obs_sorted = sorted(list(observed_days))
        try:
            last_idx = time_points.index(obs_sorted[-1])
            expected_sequence = time_points[:last_idx + 1]
            if obs_sorted == expected_sequence and len(obs_sorted) >= min_persistence:
                barcodes_to_keep.add(bc)
        except ValueError:
            continue

    print(f"Barcodes passing continuity filter "
          f"(min {min_persistence} timepoints): {len(barcodes_to_keep)}")

    # Step 5: Final subset
    adata_final = adata_sub[
        adata_sub.obs[barcode_key].isin(barcodes_to_keep)
    ].copy()

    # Step 6: Verification report
    print("\n" + "=" * 40)
    print("VERIFICATION REPORT")
    print("=" * 40)
    print(f"Original Cells : {adata.n_obs}")
    print(f"Filtered Cells : {adata_final.n_obs} "
          f"({adata_final.n_obs / adata.n_obs * 100:.2f}%)")

    d0_counts = (
        adata_final.obs[adata_final.obs[day_key] == t0][cell_type_key]
        .value_counts()
    )
    print(f"\nDay {t0} Composition:\n{d0_counts}")

    d0_non_start = d0_counts[~d0_counts.index.isin(start_types)]
    if len(d0_non_start) == 0:
        print(f"SUCCESS: Only start cell type(s) present at Day {t0}.")
    else:
        print(f"WARNING: Non-start cells found at Day {t0}: {d0_non_start.to_dict()}")

    return adata_final


def build_clone_table(
    adata: ad.AnnData,
    clone_id_col: str = 'LARRY_ID',
    time_col: str = 'day',
    feature_col: str = 'cell_type',
    time_prefix: str = 'Day',
    feature_prefix: str = 'Cluster',
    times_order: Optional[List] = None,
    features_order: Optional[List[str]] = None,
    min_total_count: int = 0,
) -> pd.DataFrame:
    """
    Build a (clones x time-feature) count table from AnnData.

    Each column represents the count of cells of a given feature
    (e.g. cell type) at a given time point for each clone.

    Parameters
    ----------
    adata : AnnData
        Must contain clone_id_col, time_col, and feature_col in obs.
    clone_id_col : str
        Column identifying clones (e.g. 'LARRY_ID', 'clone_barcode').
    time_col : str
        Column for time points (e.g. 'day', 'timepoint', 'passage').
    feature_col : str
        Column for features to count (e.g. 'cell_type', 'cluster', 'state').
    time_prefix : str
        Prefix for time columns (e.g. 'Day' -> 'Day0_ClusterHSC').
    feature_prefix : str
        Prefix for feature columns.
    times_order : list, optional
        Ordered time labels for final column ordering.
    features_order : list of str, optional
        Ordered feature labels for final column ordering.
    min_total_count : int
        Minimum total cell count across all time points to retain a clone.
        Default is 0 (keep all clones observed in adata, including those
        appearing at only one time point).
        Set to 1 to drop clones with a single cell total across all time points.

    Returns
    -------
    pd.DataFrame
        Index = clone IDs, columns = '{time_prefix}{t}_{feature_prefix}{f}'
    """
    times = adata.obs[time_col].unique()
    results = []

    for t in times:
        t_data = adata[adata.obs[time_col] == t]
        grouped = (
            t_data.obs
            .groupby([clone_id_col, feature_col])
            .size()
            .unstack(fill_value=0)
        )
        grouped.columns = [
            f'{time_prefix}{t}_{feature_prefix}{col}' for col in grouped.columns
        ]
        results.append(grouped)

    final_table = pd.concat(results, axis=1).sort_index(axis=1).fillna(0)

    # Compute per-time totals and overall total
    for t in times:
        t_cols = [c for c in final_table.columns if c.startswith(f'{time_prefix}{t}_')]
        final_table[f'_Total_{t}'] = final_table[t_cols].sum(axis=1)

    total_cols = [f'_Total_{t}' for t in times]
    final_table['_grand_total'] = final_table[total_cols].sum(axis=1)

    # Apply count filter: > min_total_count (default 0 means keep everything)
    final_table = final_table[final_table['_grand_total'] > min_total_count]
    final_table = final_table.drop(columns=total_cols + ['_grand_total'])

    # Reorder columns if ordering is specified
    if times_order and features_order:
        ordered_cols = []
        for t in times_order:
            for f in features_order:
                col = f'{time_prefix}{t}_{feature_prefix}{f}'
                if col in final_table.columns:
                    ordered_cols.append(col)
        final_table = final_table[ordered_cols]

    return final_table
