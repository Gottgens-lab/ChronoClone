"""
tests/test_integration.py
End-to-end integration test: full pipeline from raw adata to z-score plot.
Uses only the generated synthetic fixtures from conftest.py.
"""

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for CI
import chronoclone as cc

from tests.conftest import CELL_TYPES, DAYS_CTRL, DAYS_COMPARISON


class TestFullPipeline:
    """
    Runs the complete two-part workflow:
      Part 1: metaclone classification on control
      Part 2: similarity scoring across all conditions
    Validates shape, types, and key invariants at each step.
    """

    def test_01_filter_produces_valid_adata(self, adata_ctrl):
        assert adata_ctrl.n_obs > 0
        assert 'LARRY_ID' in adata_ctrl.obs.columns
        assert 'BC03' not in adata_ctrl.obs['LARRY_ID'].values

    def test_02_clone_table_shape(self, clone_table_ctrl):
        n_time_points = 4
        n_features = len(CELL_TYPES)
        assert clone_table_ctrl.shape[1] == n_time_points * n_features

    def test_03_normalized_table_row_sums(self, normalized_table_ctrl):
        for day in DAYS_CTRL:
            day_cols = [c for c in normalized_table_ctrl.columns if c.startswith(f'{day}_')]
            row_sums = normalized_table_ctrl[day_cols].sum(axis=1)
            assert ((row_sums.round(6) == 0) | (row_sums.round(6) == 1)).all()

    def test_04_comparison_tables_have_all_clones(self, adata_full, normalized_table_synthetic_a, normalized_table_synthetic_b):
        # All unique LARRY_IDs in adata for 'SyntheticA' and 'SyntheticB' must appear in tables
        expected_a = set(adata_full.obs[adata_full.obs['condition'] == 'SyntheticA']['LARRY_ID'].unique())
        expected_b = set(adata_full.obs[adata_full.obs['condition'] == 'SyntheticB']['LARRY_ID'].unique())
        assert set(normalized_table_synthetic_a.index) == expected_a
        assert set(normalized_table_synthetic_b.index) == expected_b

    def test_05_shared_features_exclude_day0(self, all_norm_tables):
        shared = cc.get_shared_feature_columns(all_norm_tables, ignore_times=['Day0'])
        assert len(shared) > 0
        assert not any(c.startswith('Day0_') for c in shared)

    def test_06_similarity_scores_valid(self, all_norm_tables, ctrl_metaclone_dict):
        ctrl, synthetic_a, synthetic_b = all_norm_tables
        shared = cc.get_shared_feature_columns([ctrl, synthetic_a, synthetic_b], ignore_times=['Day0'])
        condition_tables = {
            'SyntheticReference_Trajectory'   : ctrl,
            'SyntheticA_Trajectory': synthetic_a,
            'SyntheticB_Trajectory'   : synthetic_b,
        }
        scores = cc.compute_similarity_scores_multi(
            condition_tables=condition_tables,
            ctrl_key='SyntheticReference_Trajectory',
            ctrl_metaclone_dict=ctrl_metaclone_dict,
            all_features=shared,
            n_neighbors=2,
        )
        assert isinstance(scores, pd.DataFrame)
        assert len(scores) > 0
        assert (scores['pattern_similarity'] >= 0).all()
        assert (scores['pattern_similarity'] <= 1).all()
        assert set(scores['comparison'].unique()) == {
            'SyntheticReference_Trajectory', 'SyntheticA_Trajectory', 'SyntheticB_Trajectory'
        }

    def test_07_zscore_output(self, all_norm_tables, ctrl_metaclone_dict):
        ctrl, synthetic_a, synthetic_b = all_norm_tables
        shared = cc.get_shared_feature_columns([ctrl, synthetic_a, synthetic_b], ignore_times=['Day0'])
        condition_tables = {
            'SyntheticReference_Trajectory'   : ctrl,
            'SyntheticA_Trajectory': synthetic_a,
            'SyntheticB_Trajectory'   : synthetic_b,
        }
        scores = cc.compute_similarity_scores_multi(
            condition_tables=condition_tables,
            ctrl_key='SyntheticReference_Trajectory',
            ctrl_metaclone_dict=ctrl_metaclone_dict,
            all_features=shared,
            n_neighbors=2,
        )
        df_z = cc.compute_zscore(scores, control_label='SyntheticReference_Trajectory')
        assert 'z_score' in df_z.columns
        assert not df_z['z_score'].isnull().any()

    def test_08_plot_runs_without_error(self, all_norm_tables, ctrl_metaclone_dict):
        import matplotlib.pyplot as plt
        ctrl, synthetic_a, synthetic_b = all_norm_tables
        shared = cc.get_shared_feature_columns([ctrl, synthetic_a, synthetic_b], ignore_times=['Day0'])
        condition_tables = {
            'SyntheticReference_Trajectory'   : ctrl,
            'SyntheticA_Trajectory': synthetic_a,
            'SyntheticB_Trajectory'   : synthetic_b,
        }
        scores = cc.compute_similarity_scores_multi(
            condition_tables=condition_tables,
            ctrl_key='SyntheticReference_Trajectory',
            ctrl_metaclone_dict=ctrl_metaclone_dict,
            all_features=shared,
            n_neighbors=2,
        )
        df_z = cc.compute_zscore(scores, control_label='SyntheticReference_Trajectory')
        # Should run without raising
        cc.plot_similarity_zscore(
            df_z,
            condition_map={
                'SyntheticReference_Trajectory'   : 'SyntheticReference',
                'SyntheticA_Trajectory': 'SyntheticA',
                'SyntheticB_Trajectory'   : 'SyntheticB',
            },
            condition_order=['SyntheticReference', 'SyntheticA', 'SyntheticB'],
        )
        plt.close('all')
