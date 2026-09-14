# Benchmark implementation attribution

The retained numerical module is extracted byte-for-byte from the historical MIT-licensed implementation used for this experiment; its original license is included. Current package code remains at the repository root.

The CLADES comparator implements the specified mouse preprocessing partition from CLADES/clonaltrans v1.2.0, commit `a0442270b48614cad86a582f48695cab54b480cf`. It does not run the downstream NeuralODE or Gillespie model. The Cyclone comparator is limited to its documented abundance/correlation/Ward core and is retained as an applicability diagnostic. Source provenance and dependency versions are recorded in `frozen_runtime/`. Upstream software retains its original terms; the repository license does not relicense third-party implementations.
