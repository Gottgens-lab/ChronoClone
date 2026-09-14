"""Consensus probabilities must be valid before conversion to distances."""
import numpy as np
import pandas as pd

import chronoclone.consensus as consensus


def test_consensus_is_empirical_coclustering_probability(monkeypatch):
    table = pd.DataFrame([[1, 0], [0.8, 0.2], [0.2, 0.8], [0, 1]],
                         index=['A', 'B', 'C', 'D'], columns=['early', 'late'])
    partitions = {
        0.4: ['a', 'a', 'b', 'c'],
        0.8: ['a', 'b', 'b', 'c'],
        1.2: ['x', 'x', 'x', 'y'],
    }
    calls = []
    neighbor_calls = []
    monkeypatch.setattr(consensus.sc.pp, 'neighbors',
                        lambda *args, **kwargs: neighbor_calls.append(kwargs))

    def label_partition(adata, *, resolution, key_added, random_state):
        calls.append((resolution, random_state))
        adata.obs[key_added] = pd.Categorical(partitions[resolution])

    monkeypatch.setattr(consensus.sc.tl, 'leiden', label_partition)
    filtered, trajectory, observed = consensus.build_consensus_matrix(
        table, resolutions=list(partitions), n_neighbors=2, random_state=37)
    expected = np.array([
        [1, 2/3, 1/3, 0],
        [2/3, 1, 2/3, 0],
        [1/3, 2/3, 1, 0],
        [0, 0, 0, 1],
    ])
    np.testing.assert_allclose(observed, expected)
    np.testing.assert_array_equal(np.diag(observed), np.ones(4))
    assert ((observed >= 0) & (observed <= 1)).all()
    np.testing.assert_array_equal(observed, observed.T)
    distance = 1 - observed
    np.testing.assert_array_equal(np.diag(distance), np.zeros(4))
    np.testing.assert_allclose(distance[0, 1], 1/3)
    assert calls == [(resolution, 37) for resolution in partitions]
    assert neighbor_calls == [{'n_neighbors': 2, 'use_rep': 'X', 'random_state': 37}]
    pd.testing.assert_frame_equal(filtered, table)
    assert list(trajectory.obs_names) == list(table.index)
