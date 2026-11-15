"""
tests/test_multi_feature.py
Tests for chronoclone.multi_feature.build_multi_feature_table
"""

import numpy as np
import pandas as pd
import anndata as ad
import pytest
import chronoclone as cc

from tests.conftest import CELL_TYPES, DAYS_CTRL


# ---------------------------------------------------------------------------
# Reference: one feature at a time with the existing
# build_clone_table + normalize_clone_table, then side by side.
# ---------------------------------------------------------------------------

def _per_feature_reference(adata, features_by_time, order, clone_col, time_col, clones):
    days = list(features_by_time)
    features = []
    for t in days:
        for f in features_by_time[t]:
            if f not in features:
                features.append(f)
    counts, shares = [], []
    for f in features:
        days_f = [t for t in days if f in features_by_time[t]]
        a = adata[(adata.obs[time_col].isin(days_f) & adata.obs[f].notna()).to_numpy()].copy()
        ct = cc.build_clone_table(a, clone_id_col=clone_col, time_col=time_col, feature_col=f,
                                  time_prefix='Day', feature_prefix=f'{f}:', times_order=days_f,
                                  features_order=order[f], min_total_count=0)
        nm = cc.normalize_clone_table(ct, times_order=[f'Day{t}' for t in days_f],
                                      time_prefix='Day', feature_prefix=f'{f}:')
        counts.append(ct)
        shares.append(nm)

    def assemble(parts):
        tab = pd.concat(parts, axis=1).reindex(clones).fillna(0)
        cols = [f'Day{t}_{f}:{c}' for t in days for f in features_by_time[t] for c in order[f]
                if f'Day{t}_{f}:{c}' in tab.columns]
        return tab[cols]

    count_table, share_table = assemble(counts), assemble(shares)
    count_table.index.name = share_table.index.name = clone_col
    return count_table, share_table


# ---------------------------------------------------------------------------
# Synthetic data: several features, different features per time point
# ---------------------------------------------------------------------------

START = ['lo', 'mid', 'hi']
SHAPE = ['s1', 's2', 's3', 's4']
SIZE = ['n1', 'n2', 'n3-4', 'n5+']
LEVEL = ['L1', 'L2', 'L3']
ORDER = {'start': START, 'shape': SHAPE, 'size': SIZE, 'level': LEVEL}
FEATURES_BY_DAY = {
    0: ['start', 'shape'],
    1: ['size', 'level', 'shape'],
    2: ['size', 'level', 'shape'],
    3: ['size', 'level', 'shape'],
}


def _synthetic_cells(n_clones=40, seed=1):
    """
    Two groups of clones with different patterns. Includes:
    gaps (clones with no cells on a day), missing values, a category never
    seen on a day ('n5+' on days 1-2, 'L3' on day 1), a clone whose day-2 cells
    all lack 'level', and cells on a day not in FEATURES_BY_DAY (day 9),
    including one clone seen only on day 9.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_clones):
        cid = f'pen_{k}'
        group = k % 2
        n = 1
        for day in [0, 1, 2, 3]:
            if day > 0 and rng.random() < 0.15:     # gap
                continue
            if day > 0:
                n = n * int(rng.integers(1, 3))
            n_cells = min(n, 6)
            for i in range(n_cells):
                r = {'clone_id': cid, 'timepoint': day,
                     'start': None, 'size': None, 'level': None,
                     'shape': rng.choice(SHAPE[:2] if group == 0 else SHAPE[2:])}
                if day == 0:
                    r['start'] = rng.choice(START[:2] if group == 0 else START[1:])
                else:
                    r['size'] = 'n1' if n == 1 else 'n2' if n == 2 else 'n3-4' if n <= 4 else 'n5+'
                    if day < 3 and r['size'] == 'n5+':
                        r['size'] = 'n3-4'
                    lv = LEVEL[:2] if day == 1 else (LEVEL[:2] if group == 0 else LEVEL[1:])
                    r['level'] = rng.choice(lv) if rng.random() > 0.1 else None
                rows.append(r)
    # a clone whose day-2 cells all lack 'level' (size and shape present)
    for i in range(2):
        rows.append({'clone_id': 'pen_nolevel', 'timepoint': 0, 'start': 'lo', 'size': None,
                     'level': None, 'shape': 's1'} if i == 0 else
                    {'clone_id': 'pen_nolevel', 'timepoint': 2, 'start': None, 'size': 'n1',
                     'level': None, 'shape': 's2'})
    # cells on a day that is not in FEATURES_BY_DAY
    rows.append({'clone_id': 'pen_0', 'timepoint': 9, 'start': None, 'size': 'n1',
                 'level': 'L3', 'shape': 's4'})
    rows.append({'clone_id': 'pen_day9_only', 'timepoint': 9, 'start': None, 'size': 'n1',
                 'level': 'L3', 'shape': 's4'})
    obs = pd.DataFrame(rows)
    obs.index = obs.index.astype(str)
    return ad.AnnData(obs=obs)


@pytest.fixture(scope='module')
def synthetic():
    return _synthetic_cells()


@pytest.fixture(scope='module')
def synthetic_tables(synthetic):
    return cc.build_multi_feature_table(
        synthetic, FEATURES_BY_DAY, ORDER, clone_id_col='clone_id', time_col='timepoint')


# ---------------------------------------------------------------------------
# A small example worked out by hand
# ---------------------------------------------------------------------------

class TestHandExample:
    @pytest.fixture
    def tables(self):
        obs = pd.DataFrame({
            'clone': ['A', 'A', 'B', 'A', 'A', 'A', 'B'],
            'day':   [0,   0,   0,   1,   1,   1,   5],
            'x':     ['p', 'q', 'q', 'p', 'p', 'q', 'p'],
            'y':     [None, None, None, 'u', 'v', 'v', 'u'],
        })
        return cc.build_multi_feature_table(
            obs, {0: ['x'], 1: ['x', 'y']}, {'x': ['p', 'q'], 'y': ['u', 'v', 'w']},
            clone_id_col='clone', time_col='day')

    def test_counts(self, tables):
        counts, _ = tables
        expected = pd.DataFrame(
            [[1, 1, 2, 1, 1, 2],
             [0, 1, 0, 0, 0, 0]],
            index=pd.Index(['A', 'B'], name='clone'),
            columns=['Day0_x:p', 'Day0_x:q', 'Day1_x:p', 'Day1_x:q', 'Day1_y:u', 'Day1_y:v'],
        )
        pd.testing.assert_frame_equal(counts, expected)

    def test_shares(self, tables):
        _, shares = tables
        expected = pd.DataFrame(
            [[1 / 2, 1 / 2, 2 / 3, 1 / 3, 1 / 3, 2 / 3],
             [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]],
            index=pd.Index(['A', 'B'], name='clone'),
            columns=['Day0_x:p', 'Day0_x:q', 'Day1_x:p', 'Day1_x:q', 'Day1_y:u', 'Day1_y:v'],
        )
        pd.testing.assert_frame_equal(shares, expected)

    def test_keep_unobserved_categories(self):
        obs = pd.DataFrame({'clone': ['A', 'B'], 'day': [0, 0], 'x': ['p', 'p']})
        counts, shares = cc.build_multi_feature_table(
            obs, {0: ['x']}, {'x': ['p', 'q']}, clone_id_col='clone', time_col='day',
            keep_unobserved_categories=True)
        assert list(counts.columns) == ['Day0_x:p', 'Day0_x:q']
        assert (counts['Day0_x:q'] == 0).all() and (shares['Day0_x:q'] == 0).all()


# ---------------------------------------------------------------------------
# Same result as the one-feature-at-a-time way
# ---------------------------------------------------------------------------

class TestMatchesPerFeatureReference:
    def test_synthetic_matches_reference_exactly(self, synthetic, synthetic_tables):
        counts, shares = synthetic_tables
        clones = sorted(set(synthetic.obs.loc[synthetic.obs.timepoint.isin(FEATURES_BY_DAY),
                                              'clone_id']))
        ref_counts, ref_shares = _per_feature_reference(
            synthetic, FEATURES_BY_DAY, ORDER, 'clone_id', 'timepoint', clones)
        # the synthetic data must exercise the awkward cases
        assert 'Day1_size:n5+' not in ref_shares.columns and 'Day3_size:n5+' in ref_shares.columns
        assert 'Day1_level:L3' not in ref_shares.columns
        assert 'pen_day9_only' not in ref_shares.index
        assert not any(c.startswith('Day9_') for c in ref_shares.columns)

        pd.testing.assert_frame_equal(shares, ref_shares, check_exact=True)
        pd.testing.assert_frame_equal(counts, ref_counts, check_exact=True, check_dtype=False)

    def test_one_feature_matches_build_and_normalize(self, adata_full):
        # one feature at every day = existing build_clone_table + normalize_clone_table.
        # The reference is built here, not taken from the shared fixtures, because
        # other tests add a 'barcode_rep' column to those tables in place.
        adata = adata_full[(adata_full.obs['condition'] == 'SyntheticReference').to_numpy()].copy()
        days = [d.replace('Day', '') for d in DAYS_CTRL]
        clone_table = cc.build_clone_table(
            adata, clone_id_col='LARRY_ID', time_col='day', feature_col='cell_type',
            times_order=days, features_order=CELL_TYPES)
        normalized = cc.normalize_clone_table(
            clone_table, times_order=DAYS_CTRL, features_order=CELL_TYPES)
        _, shares = cc.build_multi_feature_table(
            adata, {d: ['cell_type'] for d in days}, {'cell_type': CELL_TYPES},
            clone_id_col='LARRY_ID', time_col='day')
        renamed = normalized.rename(columns=lambda c: c.replace('_Cluster', '_cell_type:'))
        renamed = renamed.loc[shares.index]
        assert list(shares.columns) == list(renamed.columns)
        np.testing.assert_array_equal(shares.to_numpy(), renamed.to_numpy(dtype=float))


# ---------------------------------------------------------------------------
# Properties of the output
# ---------------------------------------------------------------------------

class TestOutput:
    def test_column_names_and_order(self, synthetic_tables):
        counts, shares = synthetic_tables
        assert list(counts.columns) == list(shares.columns)
        keys = []
        for c in shares.columns:
            day, rest = c.split('_', 1)
            feat, cat = rest.split(':', 1)
            t = int(day.replace('Day', ''))
            assert feat in FEATURES_BY_DAY[t]
            keys.append((list(FEATURES_BY_DAY).index(t), FEATURES_BY_DAY[t].index(feat),
                         ORDER[feat].index(cat)))
        assert keys == sorted(keys)

    def test_blocks_sum_to_one_or_zero(self, synthetic, synthetic_tables):
        counts, shares = synthetic_tables
        obs = synthetic.obs
        for t, feats in FEATURES_BY_DAY.items():
            for f in feats:
                cols = [c for c in shares.columns if c.startswith(f'Day{t}_{f}:')]
                s = shares[cols].sum(axis=1)
                has = obs[(obs.timepoint == t) & obs[f].notna()].clone_id.unique()
                assert np.allclose(s[s.index.isin(has)], 1)
                assert (s[~s.index.isin(has)] == 0).all()
                # every cell counted once
                assert counts[cols].to_numpy().sum() == ((obs.timepoint == t) & obs[f].notna()).sum()

    def test_zero_block_for_missing_values(self, synthetic_tables):
        counts, shares = synthetic_tables
        lvl = [c for c in shares.columns if c.startswith('Day2_level:')]
        size = [c for c in shares.columns if c.startswith('Day2_size:')]
        assert shares.loc['pen_nolevel', lvl].sum() == 0
        assert shares.loc['pen_nolevel', size].sum() == 1

    def test_types_and_index(self, synthetic_tables):
        counts, shares = synthetic_tables
        assert all(dt == np.int64 for dt in counts.dtypes)
        assert all(dt == np.float64 for dt in shares.dtypes)
        assert counts.index.name == shares.index.name == 'clone_id'
        assert list(shares.index) == sorted(shares.index)
        assert not shares.isnull().any().any()

    def test_clone_order(self, synthetic, synthetic_tables):
        _, shares = synthetic_tables
        order = list(reversed(shares.index))
        _, shares2 = cc.build_multi_feature_table(
            synthetic, FEATURES_BY_DAY, ORDER, clone_id_col='clone_id', time_col='timepoint',
            clone_order=order)
        assert list(shares2.index) == order
        pd.testing.assert_frame_equal(shares2.loc[shares.index], shares)

    def test_feature_weights(self, synthetic, synthetic_tables):
        counts, shares = synthetic_tables
        counts_w, shares_w = cc.build_multi_feature_table(
            synthetic, FEATURES_BY_DAY, ORDER, clone_id_col='clone_id', time_col='timepoint',
            feature_weights={'shape': 0.5, 'level': 2})
        pd.testing.assert_frame_equal(counts_w, counts)
        for c in shares.columns:
            feat = c.split('_', 1)[1].split(':')[0]
            w = {'shape': 0.5, 'level': 2}.get(feat, 1)
            np.testing.assert_array_equal(shares_w[c].to_numpy(), shares[c].to_numpy() * w)

    def test_default_categories(self):
        obs = pd.DataFrame({'clone': ['A', 'A', 'B'], 'day': [0, 0, 0],
                            'x': pd.Categorical(['q', 'p', 'q'], categories=['q', 'p', 'r'])})
        counts, _ = cc.build_multi_feature_table(obs, {0: ['x']}, clone_id_col='clone',
                                                 time_col='day')
        assert list(counts.columns) == ['Day0_x:q', 'Day0_x:p']    # Categorical order, 'r' unseen
        obs['x'] = obs['x'].astype(str)
        counts, _ = cc.build_multi_feature_table(obs, {0: ['x']}, clone_id_col='clone',
                                                 time_col='day')
        assert list(counts.columns) == ['Day0_x:p', 'Day0_x:q']    # sorted


class TestErrors:
    OBS = pd.DataFrame({'clone': ['A', 'B'], 'day': [0, 0], 'x': ['p', 'q']})

    def _run(self, *args, **kwargs):
        kwargs.setdefault('clone_id_col', 'clone')
        kwargs.setdefault('time_col', 'day')
        return cc.build_multi_feature_table(self.OBS, *args, **kwargs)

    def test_unknown_category(self):
        with pytest.raises(ValueError, match='not in its categories'):
            self._run({0: ['x']}, {'x': ['p']})

    def test_time_point_with_no_cells(self):
        with pytest.raises(ValueError, match='no cells'):
            self._run({'0': ['x']}, {'x': ['p', 'q']})     # str key, int column

    def test_missing_column(self):
        with pytest.raises(KeyError):
            self._run({0: ['z']})

    def test_bad_clone_order(self):
        with pytest.raises(ValueError, match='clone_order'):
            self._run({0: ['x']}, clone_order=['A'])

    def test_bad_weights(self):
        with pytest.raises(ValueError):
            self._run({0: ['x']}, feature_weights={'z': 1})
        with pytest.raises(ValueError):
            self._run({0: ['x']}, feature_weights={'x': -1})

    def test_features_as_string(self):
        with pytest.raises(TypeError):
            self._run({0: 'x'})


# ---------------------------------------------------------------------------
# Plugs into the existing clustering
# ---------------------------------------------------------------------------

def test_share_table_runs_through_consensus_and_sweep(synthetic, synthetic_tables):
    pytest.importorskip('leidenalg')
    _, shares = synthetic_tables
    filtered, traj, consensus = cc.build_consensus_matrix(
        shares, min_cells=2, adata_full=synthetic, clone_id_col='clone_id',
        n_neighbors=5, resolutions=[0.5, 1.0, 1.5])
    sizes = synthetic.obs.groupby('clone_id').size()
    assert set(filtered.index) == set(shares.index[shares.index.isin(sizes[sizes >= 2].index)])
    assert consensus.shape == (len(filtered), len(filtered))
    out = cc.sweep_thresholds(1 - consensus, filtered, thresholds=[0.1, 0.3, 0.5, 0.7])
    final_labels, sil = out[4], out[5]
    assert len(final_labels) == len(sil) == len(filtered)

