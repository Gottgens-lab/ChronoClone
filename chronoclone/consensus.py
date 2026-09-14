"""
consensus.py
------------
Build a consensus clustering matrix via multi-resolution Leiden sweeps.
Works on any numeric clone x feature matrix.
"""

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from typing import List, Optional, Tuple


def build_consensus_matrix(
    normalized_table: pd.DataFrame,
    min_cells: int = 3,
    adata_full: Optional[ad.AnnData] = None,
    clone_id_col: str = 'LARRY_ID',
    n_neighbors: int = 15,
    resolutions: Optional[List[float]] = None,
    random_state: int = 0,
) -> Tuple[pd.DataFrame, ad.AnnData, np.ndarray]:
    """
    Filter clones, run multi-resolution Leiden clustering, and build
    a consensus co-clustering matrix.

    The consensus matrix encodes how consistently two clones are
    assigned to the same cluster across all tested resolutions.
    A value of 1.0 means they always cluster together; 0.0 means never.

    Parameters
    ----------
    normalized_table : pd.DataFrame
        Output of normalize_clone_table(). Index = clone IDs.
    min_cells : int
        Minimum number of cells a clone must have to be included.
    adata_full : AnnData, optional
        Full dataset used to filter valid clones by cell count.
        Must have clone_id_col in obs.
    clone_id_col : str
        Column in adata_full.obs for clone IDs.
    n_neighbors : int
        Number of neighbors for sc.pp.neighbors().
    resolutions : list of float, optional
        Leiden resolutions to sweep. Defaults to 1.1–4.0 in 0.1 steps.
    random_state : int
        Seed passed to neighbor-graph construction and each Leiden run.
        Defaults to 0, matching their existing default behavior.

    Returns
    -------
    filtered_table : pd.DataFrame
    trajectory_adata : AnnData
    consensus_matrix : np.ndarray  (n_clones x n_clones, values in [0, 1])
    """
    if resolutions is None:
        resolutions = [x / 10 for x in range(11, 41)]

    # Filter by minimum cell count
    if adata_full is not None:
        clone_counts = adata_full.obs.groupby(clone_id_col).size()
        valid_clones = clone_counts[clone_counts >= min_cells].index
        filtered_table = normalized_table.loc[
            normalized_table.index.isin(valid_clones)
        ]
    else:
        filtered_table = normalized_table

    # Build AnnData for scanpy graph + Leiden
    trajectory_adata = ad.AnnData(X=filtered_table.values.astype(float))
    trajectory_adata.obs_names = filtered_table.index.astype(str)
    trajectory_adata.var_names = filtered_table.columns.astype(str)

    sc.pp.neighbors(trajectory_adata, n_neighbors=n_neighbors, use_rep='X',
                    random_state=random_state)

    for res in resolutions:
        sc.tl.leiden(
            trajectory_adata,
            resolution=res,
            key_added=f'leiden_res{res:.1f}',
            random_state=random_state,
        )

    # Build consensus co-clustering matrix
    n_clones = len(trajectory_adata.obs_names)
    consensus_matrix = np.zeros((n_clones, n_clones))

    for res in resolutions:
        labels_cat = trajectory_adata.obs[f'leiden_res{res:.1f}'].values
        # Handle pandas Categorical objects gracefully
        if hasattr(labels_cat, 'codes'):
            labels = labels_cat.codes
        else:
            labels = labels_cat
        # Create a boolean matrix where element (i,j) is True if labels[i] == labels[j]
        match_matrix = (labels[:, None] == labels[None, :]).astype(float)
        # Self-matches are already present on the diagonal of match_matrix.
        consensus_matrix += match_matrix

    consensus_matrix /= len(resolutions)

    return filtered_table, trajectory_adata, consensus_matrix
