"""
Shared fixtures for ChronoClone tests.

All fixtures use deterministic random seeds and construct minimal but
synthetic data; no experimental dataset is loaded:
  - 4 clones (BC00–BC03), condition SyntheticReference (reference)
  - 4 clones (BC00–BC03), condition SyntheticA (comparison, no Day0)
  - 4 clones (BC00–BC03), condition SyntheticB (comparison, no Day0)
  - Only BC00–BC02 appear at Day0 as pure HSC/HSPC (valid start)
  - BC03 in control has a contaminated Day0 (CMP) -> should be filtered out
  - Synthetic comparison conditions have clones observed at varying time points
    (missing days are intentional test cases)
"""

import numpy as np
import pandas as pd
import anndata as ad
import pytest


CELL_TYPES = ['HSC', 'HSPC', 'CMP', 'MEP', 'GMP']
DAYS_CTRL    = ['Day0', 'Day4', 'Day9', 'Day15']
DAYS_COMPARISON = ['Day4', 'Day9', 'Day15']
CONDITIONS   = ['SyntheticReference', 'SyntheticA', 'SyntheticB']


def _make_cells(clone_id, day, condition, cell_types, n_cells=4, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'LARRY_ID'  : clone_id,
        'day'       : str(day),
        'condition' : condition,
        'cell_type' : rng.choice(cell_types, size=n_cells),
        'barcode_rep': [f'{clone_id}_{condition}{i}' for i in range(n_cells)],
    })


@pytest.fixture(scope='session')
def adata_full():
    """
    Synthetic AnnData with a reference and two comparison conditions.
    The reference has Day0; comparisons start at Day4.
    """
    records = []

    # --- Control: BC00-BC02 pure HSC/HSPC at Day0; BC03 contaminated ---
    for clone in ['BC00', 'BC01', 'BC02']:
        records.append(_make_cells(clone, 0,  'SyntheticReference', ['HSC', 'HSPC'], seed=int(clone[2:])*10))
        records.append(_make_cells(clone, 4,  'SyntheticReference', CELL_TYPES,      seed=int(clone[2:])*10+1))
        records.append(_make_cells(clone, 9,  'SyntheticReference', CELL_TYPES,      seed=int(clone[2:])*10+2))
        records.append(_make_cells(clone, 15, 'SyntheticReference', CELL_TYPES,      seed=int(clone[2:])*10+3))

    # BC03 control: Day0 has CMP (contaminated) -> must be filtered
    records.append(_make_cells('BC03', 0,  'SyntheticReference', ['CMP'],      seed=30))
    records.append(_make_cells('BC03', 4,  'SyntheticReference', CELL_TYPES,   seed=31))
    records.append(_make_cells('BC03', 9,  'SyntheticReference', CELL_TYPES,   seed=32))
    records.append(_make_cells('BC03', 15, 'SyntheticReference', CELL_TYPES,   seed=33))

    # --- Synthetic comparison A: BC00-BC03, no Day0; BC03 only Day4 (missing Day9, Day15) ---
    for clone in ['BC00', 'BC01', 'BC02']:
        records.append(_make_cells(clone, 4,  'SyntheticA', CELL_TYPES, seed=int(clone[2:])*20))
        records.append(_make_cells(clone, 9,  'SyntheticA', CELL_TYPES, seed=int(clone[2:])*20+1))
        records.append(_make_cells(clone, 15, 'SyntheticA', CELL_TYPES, seed=int(clone[2:])*20+2))
    records.append(_make_cells('BC03', 4, 'SyntheticA', CELL_TYPES, seed=60))  # Day4 only

    # --- Synthetic comparison B: BC00-BC03, no Day0; BC03 only Day4 ---
    for clone in ['BC00', 'BC01', 'BC02']:
        records.append(_make_cells(clone, 4,  'SyntheticB', CELL_TYPES, seed=int(clone[2:])*30))
        records.append(_make_cells(clone, 9,  'SyntheticB', CELL_TYPES, seed=int(clone[2:])*30+1))
        records.append(_make_cells(clone, 15, 'SyntheticB', CELL_TYPES, seed=int(clone[2:])*30+2))
    records.append(_make_cells('BC03', 4, 'SyntheticB', CELL_TYPES, seed=90))  # Day4 only

    obs = pd.concat(records, ignore_index=True)
    X   = np.random.default_rng(42).random((len(obs), 5))
    return ad.AnnData(X=X, obs=obs)


@pytest.fixture(scope='session')
def adata_ctrl(adata_full):
    import chronoclone as cc
    return cc.filter_lineage_continuity(
        adata_full,
        condition_key='condition', condition_val='SyntheticReference',
        day_key='day', cell_type_key='cell_type',
        start_cell_type=['HSC', 'HSPC'],
        barcode_key='LARRY_ID', min_persistence=2,
    )


@pytest.fixture(scope='session')
def clone_table_ctrl(adata_ctrl):
    import chronoclone as cc
    return cc.build_clone_table(
        adata_ctrl,
        clone_id_col='LARRY_ID', time_col='day', feature_col='cell_type',
        times_order=['0', '4', '9', '15'], features_order=CELL_TYPES,
    )


@pytest.fixture(scope='session')
def normalized_table_ctrl(clone_table_ctrl):
    import chronoclone as cc
    return cc.normalize_clone_table(
        clone_table_ctrl, times_order=DAYS_CTRL, features_order=CELL_TYPES,
    )


@pytest.fixture(scope='session')
def normalized_table_synthetic_a(adata_full):
    import chronoclone as cc
    adata_sub = adata_full[adata_full.obs['condition'] == 'SyntheticA'].copy()
    ct = cc.build_clone_table(
        adata_sub,
        clone_id_col='LARRY_ID', time_col='day', feature_col='cell_type',
        times_order=['4', '9', '15'], features_order=CELL_TYPES,
    )
    return cc.normalize_clone_table(ct, times_order=DAYS_COMPARISON, features_order=CELL_TYPES)


@pytest.fixture(scope='session')
def normalized_table_synthetic_b(adata_full):
    import chronoclone as cc
    adata_sub = adata_full[adata_full.obs['condition'] == 'SyntheticB'].copy()
    ct = cc.build_clone_table(
        adata_sub,
        clone_id_col='LARRY_ID', time_col='day', feature_col='cell_type',
        times_order=['4', '9', '15'], features_order=CELL_TYPES,
    )
    return cc.normalize_clone_table(ct, times_order=DAYS_COMPARISON, features_order=CELL_TYPES)


@pytest.fixture(scope='session')
def all_norm_tables(normalized_table_ctrl, normalized_table_synthetic_a, normalized_table_synthetic_b):
    import chronoclone as cc
    tables = [normalized_table_ctrl, normalized_table_synthetic_a, normalized_table_synthetic_b]
    cc.add_barcode_rep_column(tables)
    return tables


@pytest.fixture(scope='session')
def ctrl_metaclone_dict(adata_ctrl, all_norm_tables):
    """Fake metaclone labels injected directly into adata_ctrl.obs."""
    larry_ids = adata_ctrl.obs['LARRY_ID'].unique()
    mc_map = {lid: f'MC{i % 2}' for i, lid in enumerate(larry_ids)}
    adata_ctrl.obs['metaclone'] = adata_ctrl.obs['LARRY_ID'].map(mc_map)
    # barcode_rep col must exist in adata_ctrl.obs for build_metaclone_dict
    import chronoclone as cc
    if 'barcode_rep' not in adata_ctrl.obs.columns:
        adata_ctrl.obs['barcode_rep'] = adata_ctrl.obs['LARRY_ID']
    return cc.build_metaclone_dict(adata_ctrl, barcode_col='LARRY_ID', metaclone_col='metaclone')
