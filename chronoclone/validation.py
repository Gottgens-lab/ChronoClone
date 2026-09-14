"""
validation.py
-------------
Simulation and validation helper functions to evaluate ChronoClone under technical noise.
"""

import os
import numpy as np
import pandas as pd
import anndata as ad
import chronoclone as cc
from sklearn.metrics import adjusted_rand_score

# Default LARRY settings for backward compatibility & convenience
DEFAULT_POP = [
    'HSC', 'HSPC', 'MEP', 'CMP', 'Mk_prog', 'MEMP',
    'GMP', 'GMDP', 'Mast_prog', 'Baso_prog', 'Neu_prog', 'DC_prog', 'Mono_prog'
]
DEFAULT_TIME_COLS = ['0', '4', '9', '15']
DEFAULT_DAYS_CTRL = ['Day0', 'Day4', 'Day9', 'Day15']


def get_dev_folder(pct: int) -> str:
    """
    Get the deviation noise folder name based on the noise percentage.

    Parameters
    ----------
    pct : int
        Noise percentage (0 to 100).

    Returns
    -------
    str
        The folder name (e.g. 'dev_0_19').
    """
    if pct <= 19:
        return "dev_0_19"
    elif pct <= 39:
        return "dev_20_39"
    elif pct <= 59:
        return "dev_40_59"
    elif pct <= 79:
        return "dev_60_79"
    else:
        return "dev_80_100"


def load_all_noise_data(
    pct: int,
    noise_type: str,
    all_noise_dir: str = 'simulated_data/All_Noise',
    pop: list = DEFAULT_POP,
    time_cols: list = DEFAULT_TIME_COLS,
) -> pd.DataFrame:
    """
    Load simulated data from All_Noise directory for a specific single noise percentage.

    Parameters
    ----------
    pct : int
        Noise percentage (0 to 100).
    noise_type : str
        Type of noise ('gaussian' or 'uniform').
    all_noise_dir : str, default 'simulated_data/All_Noise'
        Path to the All_Noise parent directory.
    pop : list of str
        Cell population labels.
    time_cols : list of str
        Time points (e.g. ['0', '4', '9', '15']).

    Returns
    -------
    pd.DataFrame
        Concatenated clone frequency table with correct cell type/day columns.
    """
    vec = ['Day' + j + '_Cluster' + i for i in pop for j in time_cols]
    df = []
    for meta_id in range(1, 13):
        mc_idx = meta_id - 1
        dev_folder = get_dev_folder(pct)
        filename = f"mc_{meta_id}_{noise_type}_{pct}.csv"
        filepath = os.path.join(all_noise_dir, f"MC{mc_idx}", dev_folder, filename)
        temp = pd.read_csv(filepath, header=None)
        temp.index = 'c' + temp.index.astype(str) + '_mc' + str(meta_id)
        df.append(temp)
    final_table = pd.concat(df)
    final_table.columns = vec
    return final_table


def load_combined_noise_data(
    g_pct: int,
    u_pct: int,
    combined_noise_dir: str = 'simulated_data/Combined_Noise',
    pop: list = DEFAULT_POP,
    time_cols: list = DEFAULT_TIME_COLS,
) -> pd.DataFrame:
    """
    Load combined noise data from Combined_Noise directory.

    Parameters
    ----------
    g_pct : int
        Gaussian (normal) noise percentage.
    u_pct : int
        Uniform noise percentage.
    combined_noise_dir : str, default 'simulated_data/Combined_Noise'
        Path to the Combined_Noise parent directory.
    pop : list of str
        Cell population labels.
    time_cols : list of str
        Time points (e.g. ['0', '4', '9', '15']).

    Returns
    -------
    pd.DataFrame
        Concatenated clone frequency table.
    """
    vec = ['Day' + j + '_Cluster' + i for i in pop for j in time_cols]
    df = []
    for meta_id in range(1, 13):
        mc_idx = meta_id - 1
        filename = f"MC{mc_idx}_Normal_{g_pct}_Uniform_{u_pct}.csv"
        filepath = os.path.join(combined_noise_dir, f"MC{mc_idx}", filename)
        temp = pd.read_csv(filepath, header=None)
        temp.index = 'c' + temp.index.astype(str) + '_mc' + str(meta_id)
        df.append(temp)
    final_table = pd.concat(df)
    final_table.columns = vec
    return final_table


def run_chronoclone_validation(
    final_table: pd.DataFrame,
    pop: list = DEFAULT_POP,
    time_cols: list = DEFAULT_TIME_COLS,
    days_ctrl: list = DEFAULT_DAYS_CTRL,
    min_cells: int = 3,
    resolutions = None,
) -> float:
    """
    Run the complete ChronoClone workflow on a clone frequency matrix and evaluate accuracy.

    This function expands the clone counts to single-cell format, runs
    metacloning, and compares predicted metaclone labels against
    ground-truth metaclone IDs using the Adjusted Rand Index (ARI).

    Parameters
    ----------
    final_table : pd.DataFrame
        Clone frequency matrix (index: clone IDs, columns: DayX_ClusterY).
    pop : list of str
        Cell population labels.
    time_cols : list of str
        Time point labels.
    days_ctrl : list of str
        Full timepoint column prefixes (e.g. ['Day0', 'Day4', 'Day9', 'Day15']).
    min_cells : int, default 3
        Minimum cell count threshold for filtering clones.
    resolutions : array-like, optional
        Leiden resolutions to sweep. Defaults to np.arange(1.1, 4.1, 0.3).

    Returns
    -------
    float
        Adjusted Rand Index (ARI) score, where 1.0 represents perfect recovery.
    """
    if resolutions is None:
        resolutions = np.arange(1.1, 4.1, 0.3)

    # 1. Vectorized expansion of clone count table to single-cell AnnData
    # Filter to only feature columns (DayX_ClusterY format)
    feature_cols = [col for col in final_table.columns if '_Cluster' in col]
    clones = final_table.index.values
    larry_ids = []
    days = []
    cell_types = []
    
    for col in feature_cols:
        parts = col.split('_Cluster')
        day_val = parts[0].replace('Day', '')
        ct_val = parts[1]
        
        counts = np.round(final_table[col].values).astype(int)
        counts = np.clip(counts, 0, None)
        
        nonzero_idx = np.where(counts > 0)[0]
        if len(nonzero_idx) == 0:
            continue
        
        repeated_clones = np.repeat(clones[nonzero_idx], counts[nonzero_idx])
        larry_ids.append(repeated_clones)
        days.append(np.repeat(day_val, len(repeated_clones)))
        cell_types.append(np.repeat(ct_val, len(repeated_clones)))
        
    if len(larry_ids) == 0:
        return 0.0
        
    obs = pd.DataFrame({
        'LARRY_ID': np.concatenate(larry_ids),
        'day': np.concatenate(days),
        'cell_type': np.concatenate(cell_types),
        'condition': 'C'
    })
    obs.index = obs.index.astype(str)
    
    adata_ctrl = ad.AnnData(X=np.zeros((len(obs), 1)), obs=obs)
    
    try:
        # Build clone table
        clone_table = cc.build_clone_table(
            adata_ctrl, 
            clone_id_col='LARRY_ID', 
            time_col='day', 
            feature_col='cell_type', 
            times_order=time_cols, 
            features_order=pop
        )
        
        # Normalize clone table
        norm_table = cc.normalize_clone_table(
            clone_table, 
            times_order=days_ctrl, 
            features_order=pop
        )
        
        # Build consensus matrix
        filtered_table, traj_adata, consensus = cc.build_consensus_matrix(
            norm_table, 
            min_cells=min_cells, 
            adata_full=adata_ctrl, 
            clone_id_col='LARRY_ID', 
            resolutions=resolutions, 
            n_neighbors=15
        )
        
        if len(filtered_table) <= 1:
            return 0.0
            
        consensus_distance = 1 - consensus
        
        # Sweep thresholds
        thresholds, sil_scores, counts, opt_thresh, final_labels, sample_sil = cc.sweep_thresholds(
            consensus_distance, filtered_table
        )
        
        final_labels = pd.Series(final_labels, index=filtered_table.index, name='metaclone')
        
        # Extract ground truth from indices
        ground_truth_all = pd.Series(
            [idx.split('_mc')[1] for idx in final_table.index], 
            index=final_table.index, 
            name='metaclone_gt'
        ).apply(lambda x: f'mc{x}')
        
        clone_gt = ground_truth_all.loc[filtered_table.index]
        clone_pred = final_labels.astype(str)
        
        # Calculate Adjusted Rand Index
        ari = adjusted_rand_score(clone_gt, clone_pred)
        return ari
        
    except Exception as e:
        print(f"ChronoClone execution error: {e}")
        return 0.0


def run_clades_validation(
    final_table: pd.DataFrame,
    min_cells: int = 3,
    # resolution: float = 0.25,
    resolution: float = 1,
    n_neighbors: int = 15,
    n_pcs: int = 50,
) -> float:
    """
    Run the CLADES-style metacloning workflow on a clone count table and return the Adjusted Rand Index (ARI).

    Steps:
    1. Use raw cell counts <- no normalization => mat = sc.AnnData(final_table.astype(np.float32))
    2. Remove clones with less than 3 total cells => sc.pp.filter_cells(mat, min_counts=min_cells)
    3. Remove features that have no cells in => sc.pp.filter_genes(mat, min_counts=1)
    4. sc.tl.pca(mat)
    5. sc.pp.neighbors(mat, n_neighbors=n_neighbors, n_pcs=n_pcs)
    6. sc.tl.umap(mat)
    7. sc.tl.leiden(mat, resolution=resolution)

    Parameters
    ----------
    final_table : pd.DataFrame
        Clone frequency matrix (index: clone IDs, columns: DayX_ClusterY).
    min_cells : int, default 3
        Minimum cell count threshold for filtering clones.
    resolution : float, default 1
        Leiden clustering resolution.
    n_neighbors : int, default 15
        Number of neighbors for the k-NN graph.
    n_pcs : int, default 50
        Number of principal components to use for neighbors search.

    Returns
    -------
    float
        Adjusted Rand Index (ARI) score, where 1.0 represents perfect recovery.
    """
    import scanpy as sc

    # 1. Use raw cell counts — filter to feature columns only
    feature_cols = [col for col in final_table.columns if '_Cluster' in col]
    raw_data = final_table[feature_cols]
    mat = ad.AnnData(X=raw_data.values.astype(np.float32))
    mat.obs_names = raw_data.index.astype(str)
    mat.var_names = raw_data.columns.astype(str)

    try:
        # 2. Remove clones with less than 3 total cells
        sc.pp.filter_cells(mat, min_counts=min_cells)

        # 3. Remove features that have no cells in
        sc.pp.filter_genes(mat, min_counts=1)

        # Check if we have enough clones left
        if mat.n_obs <= 1 or mat.n_vars <= 1:
            return 0.0

        # 4. Run PCA
        # If n_obs or n_vars is very small, adjust n_comps
        n_comps = min(n_pcs, mat.n_obs - 1, mat.n_vars - 1)
        if n_comps <= 1:
            return 0.0
        sc.tl.pca(mat, n_comps=n_comps)

        # 5. Run neighbors
        sc.pp.neighbors(mat, n_neighbors=n_neighbors, n_pcs=n_comps)

        # 6. Run UMAP
        sc.tl.umap(mat)

        # 7. Run Leiden clustering
        sc.tl.leiden(mat, resolution=resolution)

        # Extract predicted labels
        pred_labels = mat.obs['leiden'].astype(str)

        # Extract ground truth from indices of the filtered clones
        ground_truth_all = pd.Series(
            [idx.split('_mc')[1] for idx in final_table.index],
            index=final_table.index,
            name='metaclone_gt'
        ).apply(lambda x: f'mc{x}')

        clone_gt = ground_truth_all.loc[mat.obs_names]
        clone_pred = pred_labels

        # Calculate Adjusted Rand Index
        ari = adjusted_rand_score(clone_gt, clone_pred)
        return ari

    except Exception as e:
        print(f"CLADES execution error: {e}")
        return 0.0


def run_clades_norm_validation(
    final_table: pd.DataFrame,
    norm_table: pd.DataFrame,
    min_cells: int = 3,
    resolution: float = 1.0,
    n_neighbors: int = 15,
    n_pcs: int = 50,
) -> float:
    """
    Run CLADES Leiden pipeline on ChronoClone's normalized trajectory table.
    """
    import scanpy as sc
    import anndata as ad

    try:
        # Initialize AnnData using normalized table
        mat = ad.AnnData(X=norm_table.values.astype(np.float32))
        mat.obs_names = norm_table.index.astype(str)
        mat.var_names = norm_table.columns.astype(str)

        # Filter cells and genes
        sc.pp.filter_cells(mat, min_counts=min_cells)
        sc.pp.filter_genes(mat, min_counts=1)

        if mat.n_obs <= 1 or mat.n_vars <= 1:
            return 0.0

        n_comps = min(n_pcs, mat.n_obs - 1, mat.n_vars - 1)
        if n_comps <= 1:
            return 0.0

        sc.tl.pca(mat, n_comps=n_comps)
        sc.pp.neighbors(mat, n_neighbors=n_neighbors, n_pcs=n_comps)
        sc.tl.umap(mat)
        sc.tl.leiden(mat, resolution=resolution)

        pred_labels = mat.obs['leiden'].astype(str)
        ground_truth_all = pd.Series(
            [idx.split('_mc')[1] for idx in final_table.index],
            index=final_table.index,
            name='metaclone_gt'
        ).apply(lambda x: f'mc{x}')

        clone_gt = ground_truth_all.loc[mat.obs_names]
        ari = adjusted_rand_score(clone_gt, pred_labels)
        return ari
    except Exception as e:
        print(f"CLADES Norm execution error: {e}")
        return 0.0


def run_kmeans_raw_validation(
    final_table: pd.DataFrame,
    min_cells: int = 3,
    n_clusters: int = 12,
) -> float:
    """
    Run k-Means clustering on filtered raw clone counts.
    """
    from sklearn.cluster import KMeans

    try:
        # Use columns representing counts (exclude Day_Total etc. if present)
        features = [col for col in final_table.columns if 'Cluster' in col]
        raw_data = final_table[features].copy()

        # Filter clones and features
        clones_filter = raw_data.sum(axis=1) >= min_cells
        raw_filtered = raw_data.loc[clones_filter].copy()
        raw_filtered = raw_filtered.loc[:, raw_filtered.sum(axis=0) >= 1]

        if len(raw_filtered) <= 1:
            return 0.0

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        pred_labels = kmeans.fit_predict(raw_filtered)

        ground_truth_all = pd.Series(
            [idx.split('_mc')[1] for idx in final_table.index],
            index=final_table.index,
            name='metaclone_gt'
        ).apply(lambda x: f'mc{x}')

        clone_gt = ground_truth_all.loc[raw_filtered.index]
        ari = adjusted_rand_score(clone_gt, pred_labels.astype(str))
        return ari
    except Exception as e:
        print(f"k-Means Raw execution error: {e}")
        return 0.0


def run_kmeans_norm_validation(
    final_table: pd.DataFrame,
    norm_table: pd.DataFrame,
    min_cells: int = 3,
    n_clusters: int = 12,
) -> float:
    """
    Run k-Means clustering on filtered normalized trajectories.
    """
    from sklearn.cluster import KMeans

    try:
        features = [col for col in final_table.columns if 'Cluster' in col]
        raw_data = final_table[features].copy()

        clones_filter = raw_data.sum(axis=1) >= min_cells
        norm_filtered = norm_table.loc[clones_filter].copy()

        if len(norm_filtered) <= 1:
            return 0.0

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        pred_labels = kmeans.fit_predict(norm_filtered)

        ground_truth_all = pd.Series(
            [idx.split('_mc')[1] for idx in final_table.index],
            index=final_table.index,
            name='metaclone_gt'
        ).apply(lambda x: f'mc{x}')

        clone_gt = ground_truth_all.loc[norm_filtered.index]
        ari = adjusted_rand_score(clone_gt, pred_labels.astype(str))
        return ari
    except Exception as e:
        print(f"k-Means Norm execution error: {e}")
        return 0.0


def run_hierarchical_raw_validation(
    final_table: pd.DataFrame,
    min_cells: int = 3,
    n_clusters: int = 12,
) -> float:
    """
    Run Hierarchical (Agglomerative) clustering on filtered raw clone counts.
    """
    from sklearn.cluster import AgglomerativeClustering

    try:
        features = [col for col in final_table.columns if 'Cluster' in col]
        raw_data = final_table[features].copy()

        clones_filter = raw_data.sum(axis=1) >= min_cells
        raw_filtered = raw_data.loc[clones_filter].copy()
        raw_filtered = raw_filtered.loc[:, raw_filtered.sum(axis=0) >= 1]

        if len(raw_filtered) <= 1:
            return 0.0

        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        pred_labels = agg.fit_predict(raw_filtered)

        ground_truth_all = pd.Series(
            [idx.split('_mc')[1] for idx in final_table.index],
            index=final_table.index,
            name='metaclone_gt'
        ).apply(lambda x: f'mc{x}')

        clone_gt = ground_truth_all.loc[raw_filtered.index]
        ari = adjusted_rand_score(clone_gt, pred_labels.astype(str))
        return ari
    except Exception as e:
        print(f"Hierarchical Raw execution error: {e}")
        return 0.0


def run_hierarchical_norm_validation(
    final_table: pd.DataFrame,
    norm_table: pd.DataFrame,
    min_cells: int = 3,
    n_clusters: int = 12,
) -> float:
    """
    Run Hierarchical (Agglomerative) clustering on filtered normalized trajectories.
    """
    from sklearn.cluster import AgglomerativeClustering

    try:
        features = [col for col in final_table.columns if 'Cluster' in col]
        raw_data = final_table[features].copy()

        clones_filter = raw_data.sum(axis=1) >= min_cells
        norm_filtered = norm_table.loc[clones_filter].copy()

        if len(norm_filtered) <= 1:
            return 0.0

        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        pred_labels = agg.fit_predict(norm_filtered)

        ground_truth_all = pd.Series(
            [idx.split('_mc')[1] for idx in final_table.index],
            index=final_table.index,
            name='metaclone_gt'
        ).apply(lambda x: f'mc{x}')

        clone_gt = ground_truth_all.loc[norm_filtered.index]
        ari = adjusted_rand_score(clone_gt, pred_labels.astype(str))
        return ari
    except Exception as e:
        print(f"Hierarchical Norm execution error: {e}")
        return 0.0

