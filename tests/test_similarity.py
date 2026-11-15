"""
tests/test_similarity.py
Tests for chronoclone.similarity
"""

import numpy as np
import pandas as pd
import pytest
import chronoclone as cc

from tests.conftest import CELL_TYPES, DAYS_CTRL, DAYS_COMPARISON, SCORING_N_NEIGHBORS


# ============================================================
# extract_barcode / add_barcode_rep_column
# ============================================================

class TestExtractBarcode:
    def test_standard_format(self):
        result = cc.extract_barcode('ACGTACGT_A1')
        assert result == 'ACGTACGT_1'

    def test_no_underscore(self):
        # Falls back gracefully
        result = cc.extract_barcode('ACGTACGT')
        assert result == 'ACGTACGT'

    def test_add_barcode_rep_column_in_place(self, normalized_table_ctrl):
        df = normalized_table_ctrl.copy()
        if 'barcode_rep' in df.columns:
            df = df.drop(columns=['barcode_rep'])
        cc.add_barcode_rep_column([df])
        assert 'barcode_rep' in df.columns
        assert len(df['barcode_rep']) == len(df)

    def test_add_barcode_rep_column_idempotent(self, all_norm_tables):
        # Calling twice should not change the column
        ctrl, synthetic_a, synthetic_b = all_norm_tables
        before = ctrl['barcode_rep'].copy()
        cc.add_barcode_rep_column([ctrl, synthetic_a, synthetic_b])
        pd.testing.assert_series_equal(ctrl['barcode_rep'], before)


# ============================================================
# build_metaclone_dict
# ============================================================

class TestBuildMetacloneDict:
    def test_returns_dict(self, ctrl_metaclone_dict):
        assert isinstance(ctrl_metaclone_dict, dict)

    def test_keys_are_larry_ids(self, adata_ctrl, ctrl_metaclone_dict):
        expected = set(adata_ctrl.obs['LARRY_ID'].unique())
        assert set(ctrl_metaclone_dict.keys()).issubset(expected)

    def test_values_are_metaclone_labels(self, ctrl_metaclone_dict):
        assert all(v.startswith('MC') for v in ctrl_metaclone_dict.values())


# ============================================================
# get_shared_feature_columns
# ============================================================

class TestGetSharedFeatureColumns:
    def test_returns_list(self, all_norm_tables):
        result = cc.get_shared_feature_columns(all_norm_tables, ignore_times=['Day0'])
        assert isinstance(result, list)

    def test_no_day0_when_ignored(self, all_norm_tables):
        result = cc.get_shared_feature_columns(all_norm_tables, ignore_times=['Day0'])
        assert not any(c.startswith('Day0_') for c in result)

    def test_shared_subset_of_all(self, all_norm_tables):
        ctrl, synthetic_a, synthetic_b = all_norm_tables
        shared = set(cc.get_shared_feature_columns(all_norm_tables, ignore_times=['Day0']))
        for df in all_norm_tables:
            assert shared.issubset(set(df.columns))

    def test_empty_tables_returns_empty(self):
        result = cc.get_shared_feature_columns([])
        assert result == []

    def test_include_times_filter(self, all_norm_tables):
        result = cc.get_shared_feature_columns(
            all_norm_tables, include_times=['Day4'], ignore_times=None
        )
        assert all(c.startswith('Day4_') for c in result)


# ============================================================
# compute_similarity_scores
# ============================================================

class TestComputeSimilarityScores:
    @pytest.fixture(scope='class')
    def shared_features(self, all_norm_tables):
        return cc.get_shared_feature_columns(all_norm_tables, ignore_times=['Day0'])

    @pytest.fixture(scope='class')
    def scores_ctrl(self, all_norm_tables, ctrl_metaclone_dict, shared_features):
        ctrl, _, _ = all_norm_tables
        return cc.compute_similarity_scores(
            treated_df=ctrl,
            ctrl_df=ctrl,
            ctrl_metaclone_dict=ctrl_metaclone_dict,
            all_features=shared_features,
            condition_name='SyntheticReference_Trajectory',
            n_neighbors=2,
        )

    def test_returns_dataframe(self, scores_ctrl):
        assert isinstance(scores_ctrl, pd.DataFrame)

    def test_expected_columns(self, scores_ctrl):
        for col in ['barcode_rep', 'metaclone', 'day_pattern', 'n_days',
                    'pattern_similarity', 'comparison']:
            assert col in scores_ctrl.columns, f'Missing column: {col}'

    def test_similarity_between_0_and_1(self, scores_ctrl):
        assert (scores_ctrl['pattern_similarity'] >= 0).all()
        assert (scores_ctrl['pattern_similarity'] <= 1).all()

    def test_comparison_label(self, scores_ctrl):
        assert (scores_ctrl['comparison'] == 'SyntheticReference_Trajectory').all()


# ============================================================
# compute_similarity_scores_multi
# ============================================================

class TestComputeSimilarityScoresMulti:
    @pytest.fixture(scope='class')
    def traj_scores(self, all_norm_tables, ctrl_metaclone_dict):
        ctrl, synthetic_a, synthetic_b = all_norm_tables
        shared_features = cc.get_shared_feature_columns(
            [ctrl, synthetic_a, synthetic_b], ignore_times=['Day0']
        )
        condition_tables = {
            'SyntheticReference_Trajectory'   : ctrl,
            'SyntheticA_Trajectory': synthetic_a,
            'SyntheticB_Trajectory'   : synthetic_b,
        }
        return cc.compute_similarity_scores_multi(
            condition_tables=condition_tables,
            ctrl_key='SyntheticReference_Trajectory',
            ctrl_metaclone_dict=ctrl_metaclone_dict,
            all_features=shared_features,
            n_neighbors=SCORING_N_NEIGHBORS,
        )

    def test_returns_dataframe(self, traj_scores):
        assert isinstance(traj_scores, pd.DataFrame)

    def test_all_conditions_present(self, traj_scores):
        expected = {'SyntheticReference_Trajectory', 'SyntheticA_Trajectory', 'SyntheticB_Trajectory'}
        assert expected.issubset(set(traj_scores['comparison'].unique()))

    def test_scores_in_range(self, traj_scores):
        assert (traj_scores['pattern_similarity'] >= 0).all()
        assert (traj_scores['pattern_similarity'] <= 1).all()

    def test_no_nan_scores(self, traj_scores):
        assert not traj_scores['pattern_similarity'].isnull().any()


# ============================================================
# compute_zscore
# ============================================================

class TestComputeZscore:
    @pytest.fixture(scope='class')
    def df_z(self, all_norm_tables, ctrl_metaclone_dict):
        ctrl, synthetic_a, synthetic_b = all_norm_tables
        shared_features = cc.get_shared_feature_columns(
            [ctrl, synthetic_a, synthetic_b], ignore_times=['Day0']
        )
        condition_tables = {
            'SyntheticReference_Trajectory'   : ctrl,
            'SyntheticA_Trajectory': synthetic_a,
            'SyntheticB_Trajectory'   : synthetic_b,
        }
        traj_scores = cc.compute_similarity_scores_multi(
            condition_tables=condition_tables,
            ctrl_key='SyntheticReference_Trajectory',
            ctrl_metaclone_dict=ctrl_metaclone_dict,
            all_features=shared_features,
            n_neighbors=SCORING_N_NEIGHBORS,
        )
        return cc.compute_zscore(traj_scores, control_label='SyntheticReference_Trajectory')

    def test_returns_dataframe(self, df_z):
        assert isinstance(df_z, pd.DataFrame)

    def test_z_score_column_exists(self, df_z):
        assert 'z_score' in df_z.columns

    def test_no_nan_z_scores(self, df_z):
        assert not df_z['z_score'].isnull().any()

    def test_ctrl_mean_near_zero(self, df_z):
        # Control Z-scores should be centred near 0 per metaclone
        ctrl_z = df_z[df_z['comparison'] == 'SyntheticReference_Trajectory']['z_score']
        assert ctrl_z.mean() == pytest.approx(0.0, abs=1.5)

    def test_min_ctrl_n_drops_metaclone(self, all_norm_tables, ctrl_metaclone_dict):
        # If min_ctrl_n is very high, no metaclone should pass
        ctrl, synthetic_a, synthetic_b = all_norm_tables
        shared_features = cc.get_shared_feature_columns(
            [ctrl, synthetic_a, synthetic_b], ignore_times=['Day0']
        )
        condition_tables = {
            'SyntheticReference_Trajectory'   : ctrl,
            'SyntheticA_Trajectory': synthetic_a,
        }
        traj_scores = cc.compute_similarity_scores_multi(
            condition_tables=condition_tables,
            ctrl_key='SyntheticReference_Trajectory',
            ctrl_metaclone_dict=ctrl_metaclone_dict,
            all_features=shared_features,
            n_neighbors=SCORING_N_NEIGHBORS,
        )
        df_z_strict = cc.compute_zscore(
            traj_scores, control_label='SyntheticReference_Trajectory', min_ctrl_n=10000
        )
        assert len(df_z_strict) == 0



class TestGenericSimilarityPlotDefaults:
    @pytest.mark.parametrize('mapping,order,expected', [
        (None, None, ['SyntheticReference', 'SyntheticA', 'SyntheticB']),
        (None, ['SyntheticB', 'SyntheticReference', 'SyntheticA'],
         ['SyntheticB', 'SyntheticReference', 'SyntheticA']),
        ({'SyntheticReference': 'Reference', 'SyntheticA': 'Group A', 'SyntheticB': 'Group B'},
         None, ['Reference', 'Group A', 'Group B']),
    ])
    def test_defaults_preserve_arbitrary_synthetic_conditions(self, mapping, order, expected, monkeypatch):
        import matplotlib.pyplot as plt
        from matplotlib.legend import Legend
        # Values are manufactured solely to test label handling; no dataset is loaded.
        rows = [
            {'metaclone': mc, 'comparison': label, 'similarity_score': 0.2 + 0.4 * i + 0.1 * j}
            for mc in ['MC0', 'MC1']
            for i, label in enumerate(['SyntheticReference', 'SyntheticA', 'SyntheticB'])
            for j in range(4)
        ]
        scores = pd.DataFrame(rows)
        before = scores.copy(deep=True)
        shown = []
        monkeypatch.setattr(plt, 'show', lambda: shown.append(True))
        try:
            cc.plot_similarity_score(scores, condition_map=mapping, condition_order=order)
            ax = plt.gca()
            condition_legend, = [item for item in ax.artists
                                 if isinstance(item, Legend) and item.get_title().get_text() == 'Condition']
            assert [item.get_text() for item in condition_legend.get_texts()] == expected
            assert len(ax.patches) >= 6  # Both metaclones retain all three condition boxes.
            assert shown == [True]
            pd.testing.assert_frame_equal(scores, before)
        finally:
            plt.close('all')

    def test_empty_default_conditions_raise_clear_error(self):
        scores = pd.DataFrame(columns=['comparison', 'metaclone', 'similarity_score'])
        with pytest.raises(ValueError, match='at least one condition'):
            cc.plot_similarity_score(scores)
