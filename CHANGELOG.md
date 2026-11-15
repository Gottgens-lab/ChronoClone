# Changelog

## 0.2.0 (2026-11-15)

### Added
- `build_multi_feature_table`: builds the clone table from several per-cell features at once, with a different set of
  features allowed at each time point. Every (time point, feature) block holds the shares of the clone's cells in each
  category. With one feature per time point it gives the same shares as `build_clone_table` + `normalize_clone_table`.
- `ordered_features` option of `build_multi_feature_table`: for features whose categories have an order (bins of a
  measurement, size classes), each block holds cumulative shares minus 0.5. Neighbouring categories are then close and
  distant ones far apart, and a clone without cells in a block lies halfway between all categories.
- `balance_feature_groups`: scales groups of features so that each group adds a set share (equal by default) of the
  mean squared clone-to-clone distance used by `build_consensus_matrix` and `sweep_thresholds`.

### Unchanged
- No existing function changed. The one-feature workflow gives the same results as in 0.1.0.

### Tests
- Fixed six tests that failed in 0.1.0 (test code and fixtures only): two expected columns for cell types that the
  fixture does not contain at Day0, three got an empty result from `compute_similarity_scores_multi` with the old
  fixture, and one depended on the order in which the tests ran. The suite now passes in full, per file, in reverse
  order and test by test.

## 0.1.0

- First version: clone table from one feature per time point, consensus over Leiden resolutions, threshold selection
  by silhouette, label transfer, density UMAP; paper simulations and noise validation.
