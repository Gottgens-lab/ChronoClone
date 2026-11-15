"""
tests/test_normalization.py
Tests for chronoclone.normalization
"""

import numpy as np
import pandas as pd
import pytest
import chronoclone as cc

from tests.conftest import CELL_TYPES, DAYS_CTRL, DAYS_COMPARISON


class TestNormalizeCloneTable:
    def test_returns_dataframe(self, normalized_table_ctrl):
        assert isinstance(normalized_table_ctrl, pd.DataFrame)

    def test_columns_match_expected(self, adata_ctrl, normalized_table_ctrl):
        # One column per (day, cell type) seen in the data, ordered by DAYS_CTRL
        # then CELL_TYPES. Day0 cells are HSC/HSPC only, so Day0 has no
        # CMP/MEP/GMP columns.
        obs = adata_ctrl.obs
        observed = set(zip('Day' + obs['day'].astype(str), obs['cell_type'].astype(str)))
        expected = [f'{day}_Cluster{ct}' for day in DAYS_CTRL for ct in CELL_TYPES
                    if (day, ct) in observed]
        assert list(normalized_table_ctrl.columns) == expected
        day0 = {c for c in normalized_table_ctrl.columns if c.startswith('Day0_')}
        assert day0 and day0.issubset({'Day0_ClusterHSC', 'Day0_ClusterHSPC'})
        for day in DAYS_CTRL[1:]:
            assert any(c.startswith(f'{day}_') for c in normalized_table_ctrl.columns)

    def test_row_sums_at_each_day(self, normalized_table_ctrl):
        # Each clone at each time point should sum to 1 (or 0 if absent)
        for day in DAYS_CTRL:
            day_cols = [c for c in normalized_table_ctrl.columns if c.startswith(f'{day}_')]
            row_sums = normalized_table_ctrl[day_cols].sum(axis=1)
            # Values are either 0 (absent) or 1.0 (normalised)
            assert ((row_sums.round(6) == 0) | (row_sums.round(6) == 1)).all(), \
                f'Row sums at {day} are not 0 or 1'

    def test_values_between_0_and_1(self, normalized_table_ctrl):
        assert (normalized_table_ctrl.values >= 0).all()
        assert (normalized_table_ctrl.values <= 1.0 + 1e-9).all()

    def test_comparison_no_day0_columns(self, normalized_table_synthetic_a):
        # Synthetic comparison table built without Day0 should have no Day0 columns
        assert not any(c.startswith('Day0_') for c in normalized_table_synthetic_a.columns)

    def test_comparison_has_correct_days(self, normalized_table_synthetic_a):
        for day in DAYS_COMPARISON:
            assert any(c.startswith(f'{day}_') for c in normalized_table_synthetic_a.columns)

    def test_no_nan(self, normalized_table_ctrl):
        assert not normalized_table_ctrl.isnull().any().any()
