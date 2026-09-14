"""
clustering.py
-------------
Agglomerative clustering on consensus distance matrix,
threshold selection via silhouette scoring, and label transfer.
"""

import numpy as np
import pandas as pd
import anndata as ad
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from typing import List, Optional, Tuple


def sweep_thresholds(
    consensus_distance: np.ndarray,
    filtered_table: pd.DataFrame,
    thresholds: Optional[List[float]] = None,
) -> Tuple[List[float], List[float], List[int], float, np.ndarray, np.ndarray]:
    """
    Sweep distance thresholds for AgglomerativeClustering and score
    each partitioning with the silhouette coefficient.

    Also returns the final cluster labels and per-sample silhouette values
    at the optimal threshold, ready for plot_threshold_and_silhouette().

    Parameters
    ----------
    consensus_distance : np.ndarray
        Precomputed distance matrix, i.e. 1 - consensus_matrix.
    filtered_table : pd.DataFrame
        Feature matrix used for silhouette scoring (Euclidean).
    thresholds : list of float, optional
        Defaults to 0.01-0.90 in 0.01 steps.

    Returns
    -------
    thresholds : list of float
    silhouette_scores : list of float
    cluster_counts : list of int
    optimal_threshold : float
    final_labels : np.ndarray
        Cluster labels at the optimal threshold.
    sample_silhouette_values : np.ndarray
        Per-clone silhouette scores at the optimal threshold.
    """
    if thresholds is None:
        thresholds = [i * 0.01 for i in range(1, 91)]

    silhouette_scores = []
    cluster_counts = []

    print('Starting threshold sweep...')

    # Cache for silhouette scores of unique labelings to avoid redundant calculations
    cache = {}

    for threshold in thresholds:
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric='precomputed',
            linkage='average'
        )
        labels = model.fit_predict(consensus_distance)
        n_clusters = len(np.unique(labels))

        if 1 < n_clusters < len(filtered_table):
            labels_tuple = tuple(labels)
            if labels_tuple in cache:
                score = cache[labels_tuple]
            else:
                score = silhouette_score(filtered_table, labels)
                cache[labels_tuple] = score
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(-1)

        cluster_counts.append(n_clusters)

    best_idx = int(np.argmax(silhouette_scores))
    optimal_threshold = thresholds[best_idx]
    best_score = silhouette_scores[best_idx]

    print(f"\nSelected threshold : {optimal_threshold:.2f}")
    print(f"Max silhouette score: {best_score:.3f}")
    print(f"Clusters            : {cluster_counts[best_idx]}")

    # Refit at optimal threshold to get final labels + per-sample scores
    final_model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=optimal_threshold,
        metric='precomputed',
        linkage='average'
    )
    final_labels = final_model.fit_predict(consensus_distance)
    sample_silhouette_values = silhouette_samples(filtered_table, final_labels)

    return (thresholds, silhouette_scores, cluster_counts,
            optimal_threshold, final_labels, sample_silhouette_values)


def fit_final_clustering(
    consensus_distance: np.ndarray,
    optimal_threshold: float,
) -> np.ndarray:
    """
    Fit final AgglomerativeClustering with the optimal distance threshold.

    Parameters
    ----------
    consensus_distance : np.ndarray
        Precomputed distance matrix.
    optimal_threshold : float
        Output of sweep_thresholds().

    Returns
    -------
    final_labels : np.ndarray
        Integer cluster label per clone.
    """
    np.fill_diagonal(consensus_distance, 0)

    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=optimal_threshold,
        metric='precomputed',
        linkage='average'
    )
    return model.fit_predict(consensus_distance)


def build_results_df(
    filtered_table: pd.DataFrame,
    final_labels: np.ndarray,
    clone_id_col: str = 'LARRY_ID',
) -> pd.DataFrame:
    """
    Build a per-clone DataFrame with silhouette scores and cluster labels.
    Used as input to plot_silhouette_filtered().

    Parameters
    ----------
    filtered_table : pd.DataFrame
        Feature matrix (clones x features).
    final_labels : np.ndarray
        Output of fit_final_clustering() or sweep_thresholds().
    clone_id_col : str
        Name to give the clone ID column in output.

    Returns
    -------
    pd.DataFrame
        Columns: clone_id_col, 'Metaclone_Cluster', 'Silhouette_Score'
    """
    sample_silhouette_values = silhouette_samples(filtered_table, final_labels)

    return pd.DataFrame({
        clone_id_col: filtered_table.index,
        'Metaclone_Cluster': final_labels,
        'Silhouette_Score': sample_silhouette_values,
    }).reset_index(drop=True)


def transfer_metaclone_labels(
    adata: ad.AnnData,
    trajectory_adata: ad.AnnData,
    final_labels: np.ndarray,
    clone_id_col: str = 'LARRY_ID',
    target_col: str = 'metaclone',
) -> ad.AnnData:
    """
    Transfer metaclone cluster labels from trajectory_adata back to
    the full single-cell AnnData object.

    Parameters
    ----------
    adata : AnnData
        Full dataset containing clone_id_col in obs.
    trajectory_adata : AnnData
        Clone-level AnnData (obs_names = clone IDs).
    final_labels : np.ndarray
        Cluster labels aligned to trajectory_adata.obs_names.
    clone_id_col : str
        Column in adata.obs containing clone IDs.
    target_col : str
        Name of new column to add to adata.obs.

    Returns
    -------
    AnnData with target_col added to obs.
    """
    trajectory_adata.obs[target_col] = pd.Categorical(final_labels)
    larry_to_metaclone = trajectory_adata.obs[target_col].to_dict()
    adata.obs[target_col] = adata.obs[clone_id_col].map(larry_to_metaclone)
    adata.obs[target_col] = pd.Categorical(adata.obs[target_col])
    return adata
