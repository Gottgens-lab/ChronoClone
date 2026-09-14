"""
tests/test_preprocessing.py
Tests for chronoclone.preprocessing
"""

import pandas as pd
import numpy as np
import pytest
import chronoclone as cc

from tests.conftest import CELL_TYPES, DAYS_CTRL


# ============================================================
# rename_labels
# ============================================================

class TestRenameLabels:
    def test_new_column_created(self, adata_full):
        adata = adata_full.copy()
        adata = cc.rename_labels(adata, 'cell_type', 'cellType',
                                  merge_as='Start', merge_values=['HSC', 'HSPC'])
        assert 'cellType' in adata.obs.columns

    def test_merged_values_replaced(self, adata_full):
        adata = adata_full.copy()
        adata = cc.rename_labels(adata, 'cell_type', 'cellType',
                                  merge_as='Start', merge_values=['HSC', 'HSPC'])
        remaining = set(adata.obs['cellType'].unique())
        assert 'HSC' not in remaining
        assert 'HSPC' not in remaining
        assert 'Start' in remaining

    def test_unmerged_values_preserved(self, adata_full):
        adata = adata_full.copy()
        adata = cc.rename_labels(adata, 'cell_type', 'cellType',
                                  merge_as='Start', merge_values=['HSC', 'HSPC'])
        remaining = set(adata.obs['cellType'].unique())
        assert remaining.issubset({'Start', 'CMP', 'MEP', 'GMP'})

    def test_merge_map(self, adata_full):
        adata = adata_full.copy()
        adata = cc.rename_labels(adata, 'cell_type', 'ct2',
                                  merge_map={'HSC': 'Stem', 'HSPC': 'Stem'})
        assert 'Stem' in adata.obs['ct2'].unique()
        assert 'HSC' not in adata.obs['ct2'].unique()

    def test_source_column_unchanged(self, adata_full):
        adata = adata_full.copy()
        original = adata.obs['cell_type'].copy()
        cc.rename_labels(adata, 'cell_type', 'cellType',
                         merge_as='Start', merge_values=['HSC', 'HSPC'])
        pd.testing.assert_series_equal(adata.obs['cell_type'], original)


# ============================================================
# filter_lineage_continuity
# ============================================================

class TestFilterLineageContinuity:
    def test_returns_anndata(self, adata_ctrl):
        import anndata as ad
        assert isinstance(adata_ctrl, ad.AnnData)

    def test_only_valid_clones_kept(self, adata_ctrl):
        # BC03 had contaminated Day0 (CMP) -> must be excluded
        assert 'BC03' not in adata_ctrl.obs['LARRY_ID'].values

    def test_valid_clones_kept(self, adata_ctrl):
        for clone in ['BC00', 'BC01', 'BC02']:
            assert clone in adata_ctrl.obs['LARRY_ID'].values

    def test_condition_isolated(self, adata_ctrl):
        # Only control condition should remain
        assert set(adata_ctrl.obs['condition'].unique()) == {'SyntheticReference'}

    def test_no_non_start_at_day0(self, adata_ctrl):
        day0 = adata_ctrl[adata_ctrl.obs['day'] == '0']
        assert set(day0.obs['cell_type'].unique()).issubset({'HSC', 'HSPC'})

    def test_all_clones_have_day0(self, adata_ctrl):
        day0_clones = set(adata_ctrl.obs[adata_ctrl.obs['day'] == '0']['LARRY_ID'].unique())
        all_clones  = set(adata_ctrl.obs['LARRY_ID'].unique())
        assert day0_clones == all_clones


# ============================================================
# build_clone_table
# ============================================================

class TestBuildCloneTable:
    def test_returns_dataframe(self, clone_table_ctrl):
        assert isinstance(clone_table_ctrl, pd.DataFrame)

    def test_index_are_larry_ids(self, adata_ctrl, clone_table_ctrl):
        expected = set(adata_ctrl.obs['LARRY_ID'].unique())
        assert set(clone_table_ctrl.index).issubset(expected)

    def test_column_naming(self, clone_table_ctrl):
        # All columns should match Day{t}_Cluster{f} pattern
        import re
        pattern = re.compile(r'^Day\w+_Cluster\w+$')
        assert all(pattern.match(c) for c in clone_table_ctrl.columns)

    def test_no_extra_columns(self, clone_table_ctrl):
        # No internal _Total_ or _grand_total columns leaked
        assert not any(c.startswith('_') for c in clone_table_ctrl.columns)

    def test_counts_non_negative(self, clone_table_ctrl):
        assert (clone_table_ctrl.values >= 0).all()

    def test_min_total_count_zero_keeps_all(self, adata_ctrl):
        # Default min_total_count=0 should keep every clone
        table = cc.build_clone_table(
            adata_ctrl, clone_id_col='LARRY_ID', time_col='day', feature_col='cell_type',
            min_total_count=0,
        )
        expected_clones = set(adata_ctrl.obs['LARRY_ID'].unique())
        assert set(table.index) == expected_clones

    def test_min_total_count_filters(self, adata_ctrl):
        # Very high threshold should drop all rows
        table = cc.build_clone_table(
            adata_ctrl, clone_id_col='LARRY_ID', time_col='day', feature_col='cell_type',
            min_total_count=10000,
        )
        assert len(table) == 0

    def test_comparison_all_clones_kept(self, adata_full):
        # Synthetic comparison condition with no filter: expect all 4 unique LARRY_IDs
        adata_sub = adata_full[adata_full.obs['condition'] == 'SyntheticA'].copy()
        table = cc.build_clone_table(
            adata_sub, clone_id_col='LARRY_ID', time_col='day', feature_col='cell_type',
            times_order=['4', '9', '15'], features_order=CELL_TYPES,
        )
        expected = set(adata_sub.obs['LARRY_ID'].unique())
        assert set(table.index) == expected
