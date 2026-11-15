"""
multi_feature.py
----------------
Build the clone table from several per-cell categorical features at once,
with a different set of features allowed at each time point.

This is an addition to the one-feature workflow
(``build_clone_table`` + ``normalize_clone_table``); those functions are
not changed.

``build_multi_feature_table`` makes the table; features whose categories have
an order (percentile bins, division classes) can be coded so that the order
counts in the distance (``ordered_features``). ``balance_feature_groups``
then scales groups of features (e.g. all shape features together) so that
each group adds the same share of the clone-to-clone distance.
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
    ordered_features: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a (clones x time-feature-category) table from several per-cell
    categorical features, with its own set of features at each time point.

    For every clone, time point and feature used at that time point, the
    cells of the clone are counted in each category of the feature. The
    counts are then turned into shares within each (time point, feature)
    block, so every block of a clone sums to 1. A clone with no cells at a
    time point (or no cells with a value for the feature) gets an all-zero
    block, as in ``build_clone_table`` + ``normalize_clone_table``.

    Ordered features
    ~~~~~~~~~~~~~~~~
    With plain shares every category is its own column, so two clones in
    neighbouring categories (e.g. 5-8 cells and 9-16 cells) are as far apart
    as two clones in the first and last category. That is right for cell
    types, which have no order, but not for percentile bins or division
    classes. For a feature in ``ordered_features`` the block holds instead,
    for every category, the share of the clone's cells in that category or
    a lower one (cumulative share; columns named ``'...:<={category}'``),
    minus 0.5. For two clones whose cells all fall in one category each, at
    positions ``i`` and ``j`` among the block's columns, the squared
    Euclidean distance over the block is ``|i - j|``; in general it is the
    sum over the category borders of the squared difference of the two
    clones' cumulative shares.

    A clone with no cells in the block (no cells at that time point, or none
    with a value for the feature) gets an all-zero block, as with plain
    shares. Because of the minus 0.5, this lies halfway: its squared distance
    to a clone whose cells all fall in one category is ``K / 4`` (``K`` =
    number of columns of the block), whichever the category. Without it, an
    empty block would be close to the top category and far from the bottom
    one. The last column is 0.5 for a clone with cells and 0 for a clone
    without.

    Without ordered features: with one feature at every time point this
    gives the same shares as ``build_clone_table`` followed by
    ``normalize_clone_table`` (only the column names differ). With several
    features it gives the same table as running those two functions one
    feature at a time (each on the time points where the feature is used)
    and putting the tables side by side.

    The share table can be passed straight to ``build_consensus_matrix``
    and its output to ``sweep_thresholds``. When some features belong
    together (e.g. several shape measures), ``balance_feature_groups`` can
    first give every group the same share of the distance.

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
        is used (all zeros if never observed). For an ordered feature such
        a column repeats the column before it (or is all zeros if it is the
        first), which gives that category border extra weight.
    ordered_features : sequence of str, optional
        Features whose categories are ordered; their blocks hold cumulative
        shares (see "Ordered features" above). The order is taken from
        ``categories``, or from the column if it is an ordered pandas
        Categorical; an ordered feature with neither raises ``ValueError``
        (a sorted order such as ``n1, n17+, n2`` would be wrong).

    Returns
    -------
    count_table : pd.DataFrame
        Cell counts (int). Index = clone IDs (named ``clone_id_col``),
        columns = ``'{time_prefix}{t}_{feature}:{category}'``, ordered by
        time point, then feature, then category. For an ordered feature the
        columns are ``'...:<={category}'`` and hold cumulative counts (cells
        in that category or a lower one).
    share_table : pd.DataFrame
        Same shape, index and columns: counts divided by the block total
        within each (time point, feature) block, times the feature weight.
        For an ordered feature: cumulative counts divided by the block total,
        minus 0.5 (all zeros for a clone without cells in the block), times
        the feature weight.

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
    ...     ordered_features=['marker_level', 'cell_size'],   # cell types have no order
    ... )
    >>> groups = {'cell_type': 'cell type', 'marker_level': 'marker', 'cell_size': 'size'}
    >>> balanced, report = cc.balance_feature_groups(shares, groups)
    >>> filtered, traj, consensus = cc.build_consensus_matrix(
    ...     balanced, adata_full=adata, clone_id_col='clone_id', min_cells=1)
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

    if isinstance(ordered_features, str):
        raise TypeError('ordered_features must be a list of feature names, not a string')
    ordered = list(ordered_features or [])
    unknown_o = sorted(set(ordered) - set(features_used), key=str)
    if unknown_o:
        raise ValueError(f'ordered_features not in features_by_time: {unknown_o}')
    no_order = [f for f in ordered if f not in categories
                and not (isinstance(obs[f].dtype, pd.CategoricalDtype) and obs[f].dtype.ordered)]
    if no_order:
        raise ValueError(
            f'ordered features need their category order, in categories or as an ordered '
            f'pandas Categorical column: {no_order}'
        )
    ordered_set = set(ordered)

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

            block = counts.to_numpy(dtype=np.int64)
            row_sums = block.sum(axis=1, keepdims=True)
            labels = list(counts.columns)
            if f in ordered_set:
                # cells in this category or a lower one; the last column is the row total
                block = np.cumsum(block, axis=1)
                labels = [f'<={c}' for c in labels]
            with np.errstate(divide='ignore', invalid='ignore'):
                shares = np.where(row_sums > 0, block / np.maximum(row_sums, 1), 0.0)
            if f in ordered_set:
                # centred, so that a clone without cells (all zeros) is equally far from every category
                shares = np.where(row_sums > 0, shares - 0.5, 0.0)
            w = feature_weights.get(f, 1)
            if w != 1:
                shares = shares * w

            names = [f'{time_prefix}{t}_{f}:{c}' for c in labels]
            count_blocks.append(pd.DataFrame(block, index=clone_index, columns=names))
            share_blocks.append(pd.DataFrame(shares, index=clone_index, columns=names))

    count_table = pd.concat(count_blocks, axis=1)
    share_table = pd.concat(share_blocks, axis=1)
    if not count_table.columns.is_unique:
        dup = count_table.columns[count_table.columns.duplicated()].unique().tolist()
        raise ValueError(f'duplicate column names (two time points give the same label?): {dup[:5]}')

    count_table.index.name = share_table.index.name = clone_id_col
    count_table.columns.name = share_table.columns.name = None
    return count_table, share_table


def _column_features(columns: Sequence, features: Sequence[str]) -> List[Optional[str]]:
    """
    Feature of each column named ``'{time}_{feature}:{category}'`` (the
    names made by ``build_multi_feature_table``). The time part is what comes
    before the first ``'_'``, as elsewhere in ChronoClone; when several
    features fit, the longest one is taken. ``None`` when none fits.
    """
    by_length = sorted(features, key=len, reverse=True)
    out: List[Optional[str]] = []
    for col in columns:
        col = str(col)
        rest = col.split('_', 1)[1] if '_' in col else ''
        out.append(next((f for f in by_length if rest.startswith(f'{f}:')), None))
    return out


def balance_feature_groups(
    table: pd.DataFrame,
    groups: Mapping[str, Any],
    group_weights: Optional[Mapping[Any, float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scale groups of features so that each group adds a set share (equal by
    default) of the clone-to-clone distance used for clustering.

    ``build_consensus_matrix`` builds its neighbour graph, and
    ``sweep_thresholds`` scores its silhouettes, from Euclidean distances
    between the rows of the table. A group of features adds to these
    distances in proportion to the number of its columns and how much they
    vary between clones, so a group of three shape features outweighs a
    single cell-count feature. This function measures each group's share of
    the mean squared distance between two clones, over all pairs of rows,
    and multiplies all columns of each group by one factor so that the
    shares become the ones asked for. The mean squared distance over all
    groups stays the same, and so do the relative weights of features within
    a group.

    The shares are measured on the rows of ``table``, so pass the clones that
    will be clustered: if ``build_consensus_matrix`` is to drop clones by
    ``min_cells``, drop them from the table first.

    Parameters
    ----------
    table : pd.DataFrame
        Clones x columns, e.g. the share table of
        ``build_multi_feature_table`` (any ``ordered_features`` coding
        already applied). Column names ``'{time}_{feature}:{category}'``,
        with no ``'_'`` in the time part.
    groups : dict
        ``{feature: group}``. Every column must belong to a feature listed
        here, and every feature listed must have a column.
    group_weights : dict, optional
        ``{group: weight}``; the share of the mean squared distance a group
        gets is its weight divided by the sum of the weights. Default: all
        groups equal. When given, it must name every group exactly.

    Returns
    -------
    balanced : pd.DataFrame
        ``table`` with every column multiplied by its group's factor.
    report : pd.DataFrame
        One row per group: ``features``, ``n_columns``, ``share_before``
        (% of the mean squared distance), ``target_share`` (%), ``factor``
        and ``share_after`` (%).

    Examples
    --------
    >>> groups = {'cell_type': 'cell type', 'marker_level': 'marker',
    ...           'cell_size': 'morphology', 'circularity': 'morphology'}
    >>> balanced, report = cc.balance_feature_groups(shares, groups)
    """
    if not isinstance(table, pd.DataFrame):
        raise TypeError('table must be a pandas DataFrame')
    if len(table) < 2:
        raise ValueError('table needs at least two rows (clones)')
    if not groups:
        raise ValueError('groups is empty')
    if not table.columns.is_unique:
        raise ValueError('table has duplicate column names')

    X = table.to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError('table has missing or infinite values')

    features = list(groups)
    col_feature = _column_features(table.columns, features)
    unmatched = [c for c, f in zip(table.columns, col_feature) if f is None]
    if unmatched:
        raise ValueError(f'{len(unmatched)} columns belong to no feature in groups, '
                         f'e.g. {unmatched[:5]}')
    no_columns = [f for f in features if f not in set(col_feature)]
    if no_columns:
        raise ValueError(f'features in groups with no column in table: {no_columns}')

    group_names = list(pd.unique(pd.Series([groups[f] for f in features], dtype=object)))
    if group_weights is None:
        weights = {g: 1.0 for g in group_names}
    else:
        weights = dict(group_weights)
        extra = [g for g in weights if g not in group_names]
        lacking = [g for g in group_names if g not in weights]
        if extra or lacking:
            raise ValueError(f'group_weights must name every group exactly; '
                             f'not groups: {extra}, missing: {lacking}')
        for g, w in weights.items():
            if not np.isfinite(w) or w <= 0:
                raise ValueError(f'weight of group {g!r} must be a finite number > 0, got {w!r}')

    # mean over pairs of rows of the squared difference in a column = 2 x its sample variance
    col_group = np.array([groups[f] for f in col_feature], dtype=object)
    col_msd = 2 * X.var(axis=0, ddof=1)
    total = col_msd.sum()
    weight_sum = sum(weights.values())

    rows, factors = [], {}
    for g in group_names:
        in_g = col_group == g
        before = col_msd[in_g].sum()
        if before <= 0:
            raise ValueError(f'group {g!r} has the same values for every clone; it cannot be scaled')
        target = total * weights[g] / weight_sum
        factors[g] = np.sqrt(target / before)
        rows.append({'group': g,
                     'features': ', '.join(f for f in features if groups[f] == g),
                     'n_columns': int(in_g.sum()),
                     'share_before': 100 * before / total,
                     'target_share': 100 * weights[g] / weight_sum,
                     'factor': factors[g]})

    scale = np.array([factors[g] for g in col_group], dtype=float)
    balanced = pd.DataFrame(X * scale, index=table.index, columns=table.columns)

    after = 2 * balanced.to_numpy().var(axis=0, ddof=1)
    for r in rows:
        r['share_after'] = 100 * after[col_group == r['group']].sum() / after.sum()
    report = pd.DataFrame(rows).set_index('group')
    return balanced, report
