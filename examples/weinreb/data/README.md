# Weinreb data

`weinreb_cells.csv.gz` contains 9,714 cells from 784 clones observed on Days 2, 4 and 6. It supplies the metadata and saved UMAP coordinates used by [the tutorial](../../../notebooks/chronoclone_tutorial.ipynb). Metaclone assignments are not included.

| Column | Meaning |
|---|---|
| `cell_id` | Unique cell ID, in the original row order. |
| `LARRY_ID` | Clone barcode. |
| `day` | Experimental day. |
| `cell_type` | Working state annotation used to build clone trajectories. |
| `cell_type_weinreb` | Original broader annotation, retained as optional metadata. |
| `UMAP1`, `UMAP2` | Stored float32 UMAP coordinates. |

The cohort was already filtered for Day-2 progenitor origin and consecutive recovery. The tutorial checks that eligibility again and uses `cell_type` with starting states `prog_1` and `prog_2`. The broader annotation is not used in fitting or plotting.

The [provenance file](provenance.json) contains the CSV checksum, reading options and original category order. The tutorial restores these automatically. Missing values use an explicit null token, distinct from an empty string; neither becomes a clone ID. This supplied cohort has no missing or empty clone IDs. Cell order, selected metadata and float32 coordinates were checked against the registered source objects.

## Attribution

Experimental data: Weinreb C, Rodriguez-Fraticelli A, Camargo F and Klein AM, [GEO GSE140802](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE140802), in-vitro sample GSM4185642.

Processed data: Mingze Gao (2024), [CLADES dataset, version 2](https://doi.org/10.6084/m9.figshare.27908142.v2), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This extract retains that attribution and license, separately from the package's code license.

Source hashes, extraction details and the brief context for historical versus current metaclone assignments are in [provenance.json](provenance.json).
