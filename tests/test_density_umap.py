"""Scientific and layout contracts for metaclone density UMAP panels."""
import copy
import re
import xml.etree.ElementTree as ET

import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import numpy as np
import pandas as pd
import pytest

import chronoclone as cc


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


@pytest.fixture
def density_adata():
    # Interleaved unassigned cells distinguish all-cell from assigned-cell
    # neighbor references. Both reference modes retain every cell in the map.
    obs = pd.DataFrame({
        'metaclone': ['MC10', 'MC10', 'MC10', 'MC2', 'MC2', 'MC2', -1, None, '-1'],
        'day': ['Day2', 'Day2', 'Day10', 'Day10', 'Day10', 'Day2', 'Day2', 'Day2', 'Day2'],
        'LARRY_ID': ['F1', 'F1', 'F2', 'G1', None, 'G1', 'U1', 'U2', 'U3'],
    }, index=[f'cell_{i}' for i in range(9)])
    data = ad.AnnData(np.arange(18, dtype=float).reshape(9, 2), obs=obs)
    data.obsm['X_umap'] = np.array([
        [0.0, 0.0], [0.1, 0.01], [0.12, 0.015], [1.8, 0.2],
        [2.1, 0.3], [7.0, 0.5], [0.04, 0.003], [9.0, 1.0], [0.065, 0.004],
    ])
    # No lineage/state columns or color annotations are required by this plot.
    return data


def foreground(ax):
    return [c for c in ax.collections
            if isinstance(c, PathCollection) and c.get_array() is not None
            and len(c.get_offsets())]


def occupancy_by_coordinate(adata, mc, day, k, neighbor_reference='all'):
    """Independent small-data oracle: direct squared Euclidean distances."""
    obs = adata.obs
    selected = (obs.metaclone.notna() & ~obs.metaclone.isin([-1, '-1'])
                if neighbor_reference == 'assigned' else np.ones(len(obs), dtype=bool))
    coords = adata.obsm['X_umap'][selected]
    reference = obs.loc[selected]
    members = ((reference.metaclone == mc) & (reference.day == day)).to_numpy()
    expected = {}
    for i in np.flatnonzero(members):
        distances = np.sum((coords - coords[i]) ** 2, axis=1)
        neighbors = np.argsort(distances, kind='stable')[:min(k, len(coords))]
        expected[tuple(coords[i])] = int(members[neighbors].sum())
    return expected


@pytest.mark.parametrize('k', [1, 2, 100])
@pytest.mark.parametrize('neighbor_reference', ['all', 'assigned'])
def test_knn_counts_both_metaclone_and_day_in_selected_reference(density_adata, k, neighbor_reference):
    # A requested panel subset must not change the reference neighbor pool.
    fig, axes = cc.plot_metaclone_density_umap(
        density_adata, n_neighbors=k, time_order=['Day2'], metaclone_order=['MC10'],
        neighbor_reference=neighbor_reference)
    assert axes.shape == (1, 1)
    collection, = foreground(axes[0, 0])
    actual = {tuple(xy): float(count) for xy, count in
              zip(collection.get_offsets(), collection.get_array())}
    assert actual == occupancy_by_coordinate(density_adata, 'MC10', 'Day2', k, neighbor_reference)
    reference_size = 9 if neighbor_reference == 'all' else 6
    assert all(1 <= value <= min(k, reference_size) for value in actual.values())
    assert collection.norm.vmin == 0
    assert collection.norm.vmax == 5


def test_default_reference_includes_unassigned_neighbor_slots(density_adata):
    fig, axes = cc.plot_metaclone_density_umap(
        density_adata, n_neighbors=2, time_order=['Day2'], metaclone_order=['MC10'])
    collection, = foreground(axes[0, 0])
    actual = {tuple(xy): float(count) for xy, count in
              zip(collection.get_offsets(), collection.get_array())}
    all_cells = occupancy_by_coordinate(density_adata, 'MC10', 'Day2', 2, 'all')
    assigned = occupancy_by_coordinate(density_adata, 'MC10', 'Day2', 2, 'assigned')
    assert actual == all_cells
    assert all_cells != assigned
    assert all_cells[(0.0, 0.0)] == 1
    assert assigned[(0.0, 0.0)] == 2


def test_per_metaclone_scale_is_shared_across_requested_days():
    obs = pd.DataFrame({
        'metaclone': ['MC10'] * 9 + ['MC2'] * 3,
        'day': ['Day2'] * 7 + ['Day10'] * 2 + ['Day2'] * 3,
        'LARRY_ID': [f'family_{i}' for i in range(12)],
    }, index=[f'cell_{i}' for i in range(12)])
    data = ad.AnnData(np.zeros((12, 1)), obs=obs)
    data.obsm['X_umap'] = np.column_stack([np.arange(12), np.arange(12) ** 2])
    fig, axes = cc.plot_metaclone_density_umap(
        data, n_neighbors=100, time_order=['Day10', 'Day2'],
        metaclone_order=['MC10', 'MC2'])
    for row in range(2):
        for collection in foreground(axes[row, 0]):
            assert collection.norm.vmin == 0
            assert collection.norm.vmax == 7
    low_support, = foreground(axes[1, 1])
    assert low_support.norm.vmax == 5
    np.testing.assert_array_equal(low_support.get_array(), np.full(3, 3))
    # An unrequested high-density day must not set the scale for this call.
    fig, subset = cc.plot_metaclone_density_umap(
        data, time_order=['Day10'], metaclone_order=['MC10'], min_vmax=0.5)
    assert foreground(subset[0, 0])[0].norm.vmax == 2


def test_order_empty_panels_counts_background_and_no_lineage_decorations(density_adata):
    density_adata.obs['cell_type'] = 'UNWANTED_LINEAGE'
    density_adata.obs['lineage_family'] = 'UNWANTED_FAMILY'
    density_adata.uns['metaclone_colors'] = ['red', 'blue']
    days = ['Day10', 'Day2', 'DayAbsent']
    mcs = ['MC10', 'MC2', 'MCabsent']
    fig, axes = cc.plot_metaclone_density_umap(
        density_adata, time_order=days, metaclone_order=mcs, n_neighbors=2)
    assert axes.shape == (3, 3)
    assert len(fig.axes) == 9  # No colorbar or auxiliary composition axes.
    assert not fig.legends
    for col, (mc, count) in enumerate(zip(mcs, [2, 1, 0])):
        title = axes[0, col].get_title()
        assert title.splitlines()[0] == mc
        assert re.search(r'\bn\s*=\s*' + str(count) + r'\b', title)
        assert title.count('MC') == 1
    for row, day in enumerate(days):
        row_labels = [text.get_text() for text in axes[row, 0].texts]
        assert day in row_labels
        assert not any('DayDay' in text or 'Day Day' in text for text in row_labels)
    for row, day in enumerate(days):
        for col, mc in enumerate(mcs):
            ax = axes[row, col]
            assert ax.get_legend() is None
            assert not ax.containers  # No stacked bars or errorbar containers.
            assert not any('UNWANTED' in text.get_text() for text in ax.texts)
            background = [c for c in ax.collections
                          if isinstance(c, PathCollection) and c.get_array() is None]
            assert len(background) == 1
            np.testing.assert_array_equal(background[0].get_offsets(), density_adata.obsm['X_umap'])
            assert ax.get_xlim() == axes[0, 0].get_xlim()
            assert ax.get_ylim() == axes[0, 0].get_ylim()
            expected = occupancy_by_coordinate(density_adata, mc, day, 2)
            if not expected:
                assert not foreground(ax)
    assert len({axes[0, col].title.get_color() for col in range(3)}) == 1
    # Count headers describe the MC across all days, not just displayed cells.
    fig, one_day = cc.plot_metaclone_density_umap(
        density_adata, time_order=['Day2'], metaclone_order=['MC10'])
    assert re.search(r'\bn\s*=\s*2\b', one_day[0, 0].get_title())


@pytest.mark.parametrize('times,mcs,shape', [
    (['Day2'], ['MC10'], (1, 1)),
    (['Day2'], ['MC2', 'MC10'], (1, 2)),
    (['Day10', 'Day2'], ['MC10'], (2, 1)),
])
def test_axes_keep_row_column_shape_and_clone_count_is_optional(density_adata, times, mcs, shape):
    without_clones = density_adata.copy()
    del without_clones.obs['LARRY_ID']
    fig, axes = cc.plot_metaclone_density_umap(
        without_clones, clone_key=None, time_order=times, metaclone_order=mcs)
    assert axes.shape == shape
    assert [axes[0, i].get_title() for i in range(shape[1])] == mcs


def test_default_order_uses_observations_and_ordered_categories(density_adata):
    fig, axes = cc.plot_metaclone_density_umap(density_adata, clone_key=None)
    assert [ax.get_title() for ax in axes[0]] == ['MC10', 'MC2']
    assert [ax.texts[0].get_text() for ax in axes[:, 0]] == ['Day2', 'Day10']
    categorical = density_adata.copy()
    categorical.obs['day'] = pd.Categorical(
        categorical.obs.day, categories=['Day10', 'Day2', 'unused'], ordered=True)
    fig, axes = cc.plot_metaclone_density_umap(categorical, clone_key=None)
    assert axes.shape[0] == 2
    assert [ax.texts[0].get_text() for ax in axes[:, 0]] == ['Day10', 'Day2']


@pytest.mark.parametrize('n', [1, 3])
def test_singleton_and_coincident_points_have_finite_density(n):
    data = ad.AnnData(np.zeros((n, 1)), obs=pd.DataFrame({
        'metaclone': [0] * n, 'day': [2] * n,
    }, index=[f'cell_{i}' for i in range(n)]))
    data.obsm['X_umap'] = np.zeros((n, 2))
    fig, axes = cc.plot_metaclone_density_umap(data, clone_key=None)
    collection, = foreground(axes[0, 0])
    np.testing.assert_array_equal(collection.get_array(), np.full(n, n))
    assert np.isfinite(axes[0, 0].get_xlim()).all()
    assert np.isfinite(axes[0, 0].get_ylim()).all()
    assert axes[0, 0].get_title() == 'MC0'
    assert axes[0, 0].texts[0].get_text() == 'Day 2'


def test_inputs_and_global_style_unchanged_and_svg_export(density_adata, tmp_path, monkeypatch):
    density_adata.uns['untouched'] = {'note': 'retain this annotation'}
    before = density_adata.copy()
    before_uns = copy.deepcopy(density_adata.uns)
    before_style = dict(matplotlib.rcParams)
    shown = []
    monkeypatch.setattr(plt, 'show', lambda: shown.append(True))
    fig, axes = cc.plot_metaclone_density_umap(density_adata)
    assert not shown
    pd.testing.assert_frame_equal(density_adata.obs, before.obs)
    np.testing.assert_array_equal(density_adata.X, before.X)
    np.testing.assert_array_equal(density_adata.obsm['X_umap'], before.obsm['X_umap'])
    assert density_adata.uns == before_uns
    assert dict(matplotlib.rcParams) == before_style
    out = tmp_path / 'density.svg'
    with matplotlib.rc_context({'svg.fonttype': 'none'}):
        fig.savefig(out, format='svg')
    root = ET.parse(out).getroot()
    assert root.tag.endswith('svg')
    text = ' '.join(''.join(element.itertext()) for element in root.iter()
                    if element.tag.endswith('text'))
    assert 'MC10' in text and 'MC2' in text and 'Day2' in text and 'n=2' in text
    assert any(element.tag.endswith('path') for element in root.iter())


def test_explicit_show_and_custom_unassigned_sentinel(density_adata, monkeypatch):
    expected = occupancy_by_coordinate(density_adata, 'MC10', 'Day2', 2, 'assigned')
    density_adata.obs['metaclone'] = density_adata.obs.metaclone.replace({-1: 'outside', '-1': 'outside'})
    shown = []
    monkeypatch.setattr(plt, 'show', lambda: shown.append(True))
    fig, axes = cc.plot_metaclone_density_umap(
        density_adata, time_order=['Day2'], metaclone_order=['MC10'],
        n_neighbors=2, neighbor_reference='assigned', unassigned_values=('outside',), show=True)
    collection, = foreground(axes[0, 0])
    actual = {tuple(xy): float(count) for xy, count in
              zip(collection.get_offsets(), collection.get_array())}
    assert actual == expected
    assert shown == [True]


@pytest.mark.parametrize('kwargs', [
    {'n_neighbors': 0}, {'n_neighbors': -1}, {'n_neighbors': 2.5}, {'n_neighbors': True},
    {'min_vmax': 0}, {'min_vmax': -1}, {'min_vmax': np.nan}, {'min_vmax': np.inf},
    {'time_order': []}, {'time_order': ['Day2', 'Day2']},
    {'metaclone_order': []}, {'metaclone_order': ['MC10', 'MC10']},
    {'neighbor_reference': 'unsupported'},
])
def test_invalid_parameters_fail_clearly(density_adata, kwargs):
    with pytest.raises(ValueError):
        cc.plot_metaclone_density_umap(density_adata, **kwargs)


@pytest.mark.parametrize('kwargs', [
    {'metaclone_key': 'absent'}, {'time_key': 'absent'},
    {'clone_key': 'absent'}, {'embedding_key': 'absent'},
])
def test_missing_required_keys_fail_clearly(density_adata, kwargs):
    with pytest.raises((KeyError, ValueError), match='absent'):
        cc.plot_metaclone_density_umap(density_adata, **kwargs)


@pytest.mark.parametrize('malformation', ['one_coordinate', 'nan', 'inf'])
def test_invalid_embeddings_rejected(density_adata, malformation):
    if malformation == 'one_coordinate':
        density_adata.obsm['X_umap'] = np.zeros((density_adata.n_obs, 1))
    else:
        density_adata.obsm['X_umap'][0, 0] = np.nan if malformation == 'nan' else np.inf
    with pytest.raises(ValueError):
        cc.plot_metaclone_density_umap(density_adata)


def test_no_assigned_cells_rejected(density_adata):
    density_adata.obs['metaclone'] = [None, -1, '-1'] * 3
    with pytest.raises(ValueError, match='assigned'):
        cc.plot_metaclone_density_umap(density_adata)
