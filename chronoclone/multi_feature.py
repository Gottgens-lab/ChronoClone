"""
multi_feature.py
----------------
Build the clone table from several per-cell categorical features at once,
with a different set of features allowed at each time point.

This is an addition to the one-feature workflow
(``build_clone_table`` + ``normalize_clone_table``); those functions are
not changed.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import anndata as ad


def _default_categories(obs: pd.DataFrame, feature: str, mask: pd.Series) -> List:
    """Category order used when none is given for a feature."""
    col = obs[feature]
    if isinstance(col.dtype, pd.CategoricalDtype):
        return list(col.cat.categories)
    values = list(pd.unique(col[mask].dropna()))
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=str)


def build_multi_feature_table(
    adata: Union[ad.AnnData, pd.DataFrame],
    features_by_time: Mapping[Any, Sequence[str]],
    categories: Optional[Mapping[str, Sequence]] = None,
    clone_id_col: str = 'LARRY_ID',
    time_col: str = 'day',
    time_prefix: str = 'Day',
    feature_weights: Optional[Mapping[str, float]] = None,
    clone_order: Optional[Sequence] = None,
    keep_unobserved_categories: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a (clones x time-feature-category) table from several per-cell
    categorical features, with its own set of features at each time point.

    For every clone, time point and feature used at that time point, the
    cells of the clone are counted in each category of the feature. The
    counts are then turned into shares within each (time point, feature)
    block, so every block of a clone sums to 1 and every feature has the
    same weight at a time point. A clone with no cells at a time point
    (or no cells with a value for the feature) gets an all-zero block,
    as in ``build_clone_table`` + ``normalize_clone_table``.

    With one feature at every time point this gives the same shares as
    ``build_clone_table`` followed by ``normalize_clone_table`` (only the
    column names differ). With several features it gives the same table as
    running those two functions one feature at a time (each on the time
    points where the feature is used) and putting the tables side by side.

    The share table can be passed straight to ``build_consensus_matrix``
    and its output to ``sweep_thresholds``.

    Parameters
    ----------
    adata : AnnData or pd.DataFrame
        Cells. Only ``.obs`` is used (a DataFrame is used as ``.obs``).
        Must contain ``clone_id_col``, ``time_col`` and one categorical
        column per feature named in ``features_by_time``.
    features_by_time : dict
        ``{time point: [features used at that time point]}``. Keys are
        values of ``obs[time_col]`` (same type, e.g. ``0`` not ``'0'``).
        The order of the keys sets the order of the time points in the
        table, and the order of each list sets the order of the features
        at that time point. Time points not in the mapping are ignored.
    categories : dict, optional
        ``{feature: [categories in order]}``. Sets the order of the
        category columns of each feature. A cell value at a mapped time
        point that is not in the list raises ``ValueError`` (nothing is
        dropped silently). For a feature not in the mapping the order is
        the column's categories if it is a pandas Categorical, otherwise
        the sorted observed values.
    clone_id_col : str
        Column in obs with clone IDs.
    time_col : str
        Column in obs with time points.
    time_prefix : str
        Prefix of the time part of column names (default ``'Day'``).
    feature_weights : dict, optional
        ``{feature: weight}``. Multiplies the share block of that feature
        at every time point by ``weight`` (default 1). Counts are not
        weighted. Note that ``build_consensus_matrix`` uses Euclidean
        distances, so a weight ``w`` scales the feature's contribution to
        squared distances by ``w**2``.
    clone_order : sequence, optional
        Order of the rows. Must contain every clone that has a cell at a
        mapped time point, each once, and nothing else. Default: those
        clones sorted.
    keep_unobserved_categories : bool
        If False (default), a category column is kept only when at least
        one cell has that category at that time point, as
        ``build_clone_table`` does for string columns. If True, every category in
        ``categories`` gets a column at every time point where the feature
        is used (all zeros if never observed).

    Returns
    -------
    count_table : pd.DataFrame
        Cell counts (int). Index = clone IDs (named ``clone_id_col``),
        columns = ``'{time_prefix}{t}_{feature}:{category}'``, ordered by
        time point, then feature, then category.
    share_table : pd.DataFrame
        Same shape, index and columns: counts divided by the block total
        within each (time point, feature) block, times the feature weight.

    Rows include every clone with at least one cell at a mapped time point,
    also when none of its cells there has a value for a feature (zero
    block). Cells with a missing value (NaN/None) for a feature are not
    counted for that feature.

    Examples
    --------
    >>> features_by_time = {
    ...     0: ['marker_level', 'cell_size'],
    ...     1: ['cell_type', 'marker_level', 'cell_size'],
    ...     2: ['cell_type', 'marker_level', 'cell_size'],
    ... }
    >>> bins = ['low', 'mid', 'high']
    >>> categories = {'marker_level': bins, 'cell_size': bins,
    ...               'cell_type': ['HSC', 'MPP', 'Mk', 'Ery']}
    >>> counts, shares = cc.build_multi_feature_table(
    ...     adata, features_by_time, categories,
    ...     clone_id_col='clone_id', time_col='day',
    ... )
    >>> filtered, traj, consensus = cc.build_consensus_matrix(
    ...     shares, adata_full=adata, clone_id_col='clone_id', min_cells=1)
    >>> out = cc.sweep_thresholds(1 - consensus, filtered)
    """
    obs = adata.obs if isinstance(adata, ad.AnnData) else adata
    if not isinstance(obs, pd.DataFrame):
        raise TypeError('adata must be an AnnData or a pandas DataFrame')

    categories = dict(categories or {})
    feature_weights = dict(feature_weights or {})

    # ---- check inputs ---------------------------------------------------
    if not features_by_time:
        raise ValueError('features_by_time is empty')
    times = list(features_by_time.keys())
    features_used: List[str] = []
    for t, feats in features_by_time.items():
        if isinstance(feats, str):
            raise TypeError(f'features for time point {t!r} must be a list, not a string')
        feats = list(feats)
        if not feats:
            raise ValueError(f'no features given for time point {t!r}')
        if len(set(feats)) != len(feats):
            raise ValueError(f'duplicate features at time point {t!r}: {feats}')
        for f in feats:
            if f not in features_used:
                features_used.append(f)

    missing_cols = [c for c in [clone_id_col, time_col] + features_used if c not in obs.columns]
    if missing_cols:
        raise KeyError(f'columns not in obs: {missing_cols}')

    for f, cats in categories.items():
        if len(set(cats)) != len(list(cats)):
            raise ValueError(f'duplicate categories for feature {f!r}: {list(cats)}')

    unknown_w = sorted(set(feature_weights) - set(features_used), key=str)
    if unknown_w:
        raise ValueError(f'feature_weights for features not in features_by_time: {unknown_w}')
    for f, w in feature_weights.items():
        if not np.isfinite(w) or w < 0:
            raise ValueError(f'weight of feature {f!r} must be a finite number >= 0, got {w!r}')

    time_masks: Dict[Any, pd.Series] = {}
    for t in times:
        m = (obs[time_col] == t) & obs[clone_id_col].notna()
        if not m.any():
            raise ValueError(
                f'no cells with {time_col} == {t!r} (dtype of obs[{time_col!r}] is '
                f'{obs[time_col].dtype}); check the keys of features_by_time'
            )
        time_masks[t] = m

    # ---- rows ------------------------------------------------------------
    in_times = obs[time_col].isin(times) & obs[clone_id_col].notna()
    found = pd.Index(pd.unique(obs.loc[in_times, clone_id_col]))
    if clone_order is None:
        clone_index = found.sort_values()
    else:
        clone_index = pd.Index(list(clone_order))
        if not clone_index.is_unique:
            raise ValueError('clone_order has duplicate clone IDs')
        extra = clone_index.difference(found)
        lacking = found.difference(clone_index)
        if len(extra) or len(lacking):
            raise ValueError(
                'clone_order must list exactly the clones with cells at the mapped time points; '
                f'{len(lacking)} missing (e.g. {list(lacking[:5])}), '
                f'{len(extra)} not found (e.g. {list(extra[:5])})'
            )

    # ---- one block per (time point, feature) ----------------------------
    count_blocks, share_blocks = [], []
    for t in times:
        mask_t = time_masks[t]
        for f in features_by_time[t]:
            if f in categories:
                cats = list(categories[f])
            else:
                used_mask = obs[time_col].isin([s for s in times if f in features_by_time[s]])
                cats = _default_categories(obs, f, used_mask)

            sub = obs.loc[mask_t & obs[f].notna(), [clone_id_col, f]]
            cat_set = set(cats)
            unexpected = [v for v in pd.unique(sub[f]) if v not in cat_set]
            if unexpected:
                raise ValueError(
                    f'feature {f!r} at time point {t!r} has values not in its categories: '
                    f'{sorted(map(str, unexpected))}'
                )

            counts = pd.crosstab(
                sub[clone_id_col].to_numpy(), sub[f].astype(object).to_numpy()
            ) if len(sub) else pd.DataFrame()
            counts = counts.reindex(index=clone_index, columns=cats, fill_value=0)

            if not keep_unobserved_categories:
                counts = counts.loc[:, counts.sum(axis=0).to_numpy() > 0]

            block = counts.to_numpy(dtype=float)
            row_sums = block.sum(axis=1, keepdims=True)
            with np.errstate(divide='ignore', invalid='ignore'):
                shares = np.where(row_sums > 0, block / row_sums, block)
            w = feature_weights.get(f, 1)
            if w != 1:
                shares = shares * w

            names = [f'{time_prefix}{t}_{f}:{c}' for c in counts.columns]
            count_blocks.append(pd.DataFrame(counts.to_numpy(dtype=np.int64),
                                             index=clone_index, columns=names))
            share_blocks.append(pd.DataFrame(shares, index=clone_index, columns=names))

    count_table = pd.concat(count_blocks, axis=1)
    share_table = pd.concat(share_blocks, axis=1)
    if not count_table.columns.is_unique:
        dup = count_table.columns[count_table.columns.duplicated()].unique().tolist()
        raise ValueError(f'duplicate column names (two time points give the same label?): {dup[:5]}')

    count_table.index.name = share_table.index.name = clone_id_col
    count_table.columns.name = share_table.columns.name = None
    return count_table, share_table
