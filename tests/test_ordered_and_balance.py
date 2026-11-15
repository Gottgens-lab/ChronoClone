"""
tests/test_ordered_and_balance.py
Tests for the ordered-feature coding of build_multi_feature_table
(ordered_features) and for balance_feature_groups.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist

import chronoclone as cc
from chronoclone.multi_feature import _column_features
from tests.test_multi_feature import _synthetic_cells, FEATURES_BY_DAY, ORDER

ORDERED = ['size', 'level']          # 'start' and 'shape' stay plain shares here


@pytest.fixture(scope='module')
def synthetic():
    return _synthetic_cells()


@pytest.fixture(scope='module')
def plain(synthetic):
    return cc.build_multi_feature_table(
        synthetic, FEATURES_BY_DAY, ORDER, clone_id_col='clone_id', time_col='timepoint')


@pytest.fixture(scope='module')
def ordered(synthetic):
    return cc.build_multi_feature_table(
        synthetic, FEATURES_BY_DAY, ORDER, clone_id_col='clone_id', time_col='timepoint',
        ordered_features=ORDERED)


def _block_of(col):
    """'Day1_size:<=n2' -> ('Day1', 'size')"""
    t, rest = col.split('_', 1)
    return t, rest.rsplit(':', 1)[0]


def _blocks(columns):
    """{(time, feature): [columns in order]}, in column order."""
    out = {}
    for c in columns:
        out.setdefault(_block_of(c), []).append(c)
    return out


def _cumulate_blocks(table, features, centre=False):
    """
    Cumulative sums within each (time, feature) block of the given features, names with '<='.
    centre=True (share tables): minus 0.5 for rows with cells in the block, rows without stay 0.
    """
    out = table.copy()
    for (t, f), cols in _blocks(table.columns).items():
        if f in features:
            cum = table[cols].cumsum(axis=1)
            if centre:
                has = (table[cols].sum(axis=1) > 0).to_numpy()[:, None]
                cum = pd.DataFrame(np.where(has, cum.to_numpy() - 0.5, 0.0), index=cum.index, columns=cols)
            out[cols] = cum
    out.columns = [c.replace(f'{t}_{f}:', f'{t}_{f}:<=', 1) if f in features else c
                   for c in table.columns for t, f in [_block_of(c)]]
    return out


# ---------------------------------------------------------------------------
# Ordered coding
# ---------------------------------------------------------------------------

class TestOrderedHandExample:
    @pytest.fixture
    def tables(self):
        obs = pd.DataFrame({
            'clone': ['A', 'A', 'B', 'C', 'C'],
            'day':   [0,   0,   0,   0,   1],
            'x':     ['a', 'c', 'b', None, 'a'],
        })
        return cc.build_multi_feature_table(
            obs, {0: ['x'], 1: ['x']}, {'x': ['a', 'b', 'c']},
            clone_id_col='clone', time_col='day', ordered_features=['x'])

    def test_counts(self, tables):
        counts, _ = tables
        expected = pd.DataFrame(
            [[1, 1, 2, 0],
             [0, 1, 1, 0],
             [0, 0, 0, 1]],
            index=pd.Index(['A', 'B', 'C'], name='clone'),
            columns=['Day0_x:<=a', 'Day0_x:<=b', 'Day0_x:<=c', 'Day1_x:<=a'],
        )
        pd.testing.assert_frame_equal(counts, expected)

    def test_shares(self, tables):
        _, shares = tables
        expected = pd.DataFrame(
            [[0.0, 0.0, 0.5, 0.0],          # cumulative 1/2, 1/2, 1 minus 0.5; no cells on day 1
             [-0.5, 0.5, 0.5, 0.0],
             [0.0, 0.0, 0.0, 0.5]],         # no value on day 0 -> all zeros
            index=pd.Index(['A', 'B', 'C'], name='clone'),
            columns=['Day0_x:<=a', 'Day0_x:<=b', 'Day0_x:<=c', 'Day1_x:<=a'],
        )
        pd.testing.assert_frame_equal(shares, expected)


def test_squared_distance_is_number_of_categories_apart():
    cats = ['P0-10', 'P10-25', 'P25-50', 'P50-75', 'P75-90', 'P90-100']
    obs = pd.DataFrame({'clone': [f'c{i}' for i in range(6)], 'day': 0, 'x': cats})
    _, shares = cc.build_multi_feature_table(
        obs, {0: ['x']}, {'x': cats}, clone_id_col='clone', time_col='day',
        ordered_features=['x'])
    X = shares.loc[[f'c{i}' for i in range(6)]].to_numpy()
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    pos = np.arange(6)
    np.testing.assert_allclose(d2, np.abs(pos[:, None] - pos[None, :]))

    # a clone without a value in the block is equally far (K/4 = 1.5) from every category
    obs_z = pd.concat([obs, pd.DataFrame({'clone': ['z'], 'day': [0], 'x': [None]})], ignore_index=True)
    _, sz = cc.build_multi_feature_table(
        obs_z, {0: ['x']}, {'x': cats}, clone_id_col='clone', time_col='day', ordered_features=['x'])
    assert (sz.loc['z'] == 0).all()
    d2z = ((sz.loc[[f'c{i}' for i in range(6)]].to_numpy() - sz.loc['z'].to_numpy()) ** 2).sum(axis=1)
    np.testing.assert_allclose(d2z, 1.5)

    # plain shares: every pair of different categories is the same distance apart
    _, plain = cc.build_multi_feature_table(
        obs, {0: ['x']}, {'x': cats}, clone_id_col='clone', time_col='day')
    P = plain.to_numpy()
    d2p = ((P[:, None, :] - P[None, :, :]) ** 2).sum(axis=2)
    assert set(np.round(d2p[~np.eye(6, dtype=bool)], 12)) == {2.0}


class TestOrderedMatchesCumulatedPlain:
    def test_shares(self, plain, ordered):
        expected = _cumulate_blocks(plain[1], ORDERED, centre=True)
        pd.testing.assert_index_equal(ordered[1].columns, expected.columns)
        pd.testing.assert_frame_equal(ordered[1], expected, check_exact=False, rtol=0, atol=1e-12)

    def test_counts(self, plain, ordered):
        expected = _cumulate_blocks(plain[0], ORDERED)
        pd.testing.assert_frame_equal(ordered[0], expected)

    def test_unordered_features_unchanged(self, plain, ordered):
        cols = [c for c in plain[1].columns if _block_of(c)[1] not in ORDERED]
        pd.testing.assert_frame_equal(ordered[1][cols], plain[1][cols])

    def test_last_column_is_half_or_zero(self, ordered):
        _, shares = ordered
        n_checked = 0
        for (t, f), cols in _blocks(shares.columns).items():
            if f in ORDERED:
                assert set(shares[cols[-1]].unique()) == {0.0, 0.5}     # the synthetic data has gaps
                n_checked += 1
        assert n_checked == 6        # size and level on days 1-3

    def test_weight_applies_to_cumulative_block(self, synthetic, ordered):
        _, weighted = cc.build_multi_feature_table(
            synthetic, FEATURES_BY_DAY, ORDER, clone_id_col='clone_id', time_col='timepoint',
            ordered_features=ORDERED, feature_weights={'size': 2.0})
        size_cols = [c for c in weighted.columns if _block_of(c)[1] == 'size']
        other = [c for c in weighted.columns if c not in size_cols]
        pd.testing.assert_frame_equal(weighted[size_cols], 2.0 * ordered[1][size_cols])
        pd.testing.assert_frame_equal(weighted[other], ordered[1][other])


class TestOrderedInputs:
    def test_ordered_categorical_needs_no_categories(self):
        obs = pd.DataFrame({'clone': ['A', 'B'], 'day': [0, 0],
                            'x': pd.Categorical(['hi', 'lo'], categories=['lo', 'hi'], ordered=True)})
        _, shares = cc.build_multi_feature_table(
            obs, {0: ['x']}, clone_id_col='clone', time_col='day', ordered_features=['x'])
        assert list(shares.columns) == ['Day0_x:<=lo', 'Day0_x:<=hi']
        np.testing.assert_array_equal(shares.to_numpy(), [[-0.5, 0.5], [0.5, 0.5]])

    def test_no_order_raises(self):
        obs = pd.DataFrame({'clone': ['A', 'B'], 'day': [0, 0], 'x': ['n2', 'n17+']})
        with pytest.raises(ValueError, match='category order'):
            cc.build_multi_feature_table(obs, {0: ['x']}, clone_id_col='clone',
                                         time_col='day', ordered_features=['x'])
        obs['x'] = pd.Categorical(obs['x'])              # unordered Categorical
        with pytest.raises(ValueError, match='category order'):
            cc.build_multi_feature_table(obs, {0: ['x']}, clone_id_col='clone',
                                         time_col='day', ordered_features=['x'])

    def test_unknown_ordered_feature_raises(self):
        obs = pd.DataFrame({'clone': ['A'], 'day': [0], 'x': ['a']})
        with pytest.raises(ValueError, match='not in features_by_time'):
            cc.build_multi_feature_table(obs, {0: ['x']}, {'x': ['a']}, clone_id_col='clone',
                                         time_col='day', ordered_features=['y'])

    def test_string_raises(self):
        obs = pd.DataFrame({'clone': ['A'], 'day': [0], 'x': ['a']})
        with pytest.raises(TypeError):
            cc.build_multi_feature_table(obs, {0: ['x']}, {'x': ['a']}, clone_id_col='clone',
                                         time_col='day', ordered_features='x')


# ---------------------------------------------------------------------------
# balance_feature_groups
# ---------------------------------------------------------------------------

GROUPS = {'start': 'founder', 'shape': 'shape', 'size': 'growth', 'level': 'growth'}


def _group_msd(table, groups):
    """Mean squared pairwise distance per group, by brute force."""
    feats = _column_features(table.columns, list(groups))
    out = {}
    for g in pd.unique(pd.Series(list(groups.values()))):
        cols = [c for c, f in zip(table.columns, feats) if groups[f] == g]
        out[g] = pdist(table[cols].to_numpy(), 'sqeuclidean').mean()
    return pd.Series(out)


class TestBalance:
    @pytest.fixture(scope='class')
    def balanced(self, ordered):
        return cc.balance_feature_groups(ordered[1], GROUPS)

    def test_equal_shares_by_brute_force(self, ordered, balanced):
        table, report = balanced
        msd = _group_msd(table, GROUPS)
        np.testing.assert_allclose(msd / msd.sum() * 100, 100 / 3, rtol=1e-9)
        np.testing.assert_allclose(report['share_after'], 100 / 3, rtol=1e-9)

    def test_report_before_matches_brute_force(self, ordered, balanced):
        _, report = balanced
        msd = _group_msd(ordered[1], GROUPS)
        np.testing.assert_allclose(report['share_before'], (msd / msd.sum() * 100)[report.index],
                                   rtol=1e-9)
        assert report.loc['growth', 'features'] == 'size, level'
        assert report['n_columns'].sum() == ordered[1].shape[1]

    def test_total_distance_kept(self, ordered, balanced):
        before = pdist(ordered[1].to_numpy(), 'sqeuclidean').mean()
        after = pdist(balanced[0].to_numpy(), 'sqeuclidean').mean()
        assert after == pytest.approx(before, rel=1e-9)

    def test_one_factor_per_group(self, ordered, balanced):
        table, report = balanced
        feats = _column_features(table.columns, list(GROUPS))
        for c, f in zip(table.columns, feats):
            np.testing.assert_allclose(table[c], ordered[1][c] * report.loc[GROUPS[f], 'factor'],
                                       rtol=1e-12)
        pd.testing.assert_index_equal(table.index, ordered[1].index)
        pd.testing.assert_index_equal(table.columns, ordered[1].columns)

    def test_group_weights(self, ordered):
        table, report = cc.balance_feature_groups(
            ordered[1], GROUPS, group_weights={'founder': 1, 'shape': 2, 'growth': 1})
        msd = _group_msd(table, GROUPS)
        np.testing.assert_allclose((msd / msd.sum() * 100)[['founder', 'shape', 'growth']],
                                   [25, 50, 25], rtol=1e-9)
        np.testing.assert_allclose(report['target_share'], report['share_after'], rtol=1e-9)

    def test_runs_through_consensus_and_sweep(self, synthetic, balanced):
        pytest.importorskip('leidenalg')
        table, _ = balanced
        filtered, traj, consensus = cc.build_consensus_matrix(
            table, min_cells=1, adata_full=synthetic, clone_id_col='clone_id',
            n_neighbors=5, resolutions=[0.5, 1.0, 1.5])
        out = cc.sweep_thresholds(1 - consensus, filtered, thresholds=[0.1, 0.3, 0.5, 0.7])
        assert len(out[4]) == len(filtered)


class TestBalanceErrors:
    def test_column_without_feature(self, ordered):
        with pytest.raises(ValueError, match='belong to no feature'):
            cc.balance_feature_groups(ordered[1], {k: v for k, v in GROUPS.items() if k != 'level'})

    def test_feature_without_column(self, ordered):
        with pytest.raises(ValueError, match='no column'):
            cc.balance_feature_groups(ordered[1], {**GROUPS, 'other': 'shape'})

    def test_group_weights_must_match(self, ordered):
        with pytest.raises(ValueError, match='every group'):
            cc.balance_feature_groups(ordered[1], GROUPS, group_weights={'founder': 1, 'shape': 1})
        with pytest.raises(ValueError, match='every group'):
            cc.balance_feature_groups(ordered[1], GROUPS,
                                      group_weights={'founder': 1, 'shape': 1, 'growth': 1, 'x': 1})
        with pytest.raises(ValueError, match='> 0'):
            cc.balance_feature_groups(ordered[1], GROUPS,
                                      group_weights={'founder': 1, 'shape': 0, 'growth': 1})

    def test_constant_group(self):
        table = pd.DataFrame({'Day0_a:x': [1.0, 1.0, 1.0], 'Day0_b:y': [0.0, 1.0, 0.5]})
        with pytest.raises(ValueError, match='same values'):
            cc.balance_feature_groups(table, {'a': 'A', 'b': 'B'})

    def test_bad_values_and_rows(self):
        table = pd.DataFrame({'Day0_a:x': [1.0, np.nan], 'Day0_b:y': [0.0, 1.0]})
        with pytest.raises(ValueError, match='missing or infinite'):
            cc.balance_feature_groups(table, {'a': 'A', 'b': 'B'})
        with pytest.raises(ValueError, match='two rows'):
            cc.balance_feature_groups(table.iloc[:1], {'a': 'A', 'b': 'B'})


def test_column_features_takes_longest_match():
    cols = ['Day0_level:a', 'Day0_max_level:a', 'Day1_max_level:<=P0-10', 'Day1_cell_size:P0-10', 'junk']
    assert _column_features(cols, ['level', 'max_level', 'max', 'cell_size']) == \
        ['level', 'max_level', 'max_level', 'cell_size', None]

