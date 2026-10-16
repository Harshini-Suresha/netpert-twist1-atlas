# Archive: report-file benchmark data (simulated, not real)

Everything here was extracted verbatim from `NetPert_ network response functions for drug-target prioritisation.html` so no content is lost.

- `tables/` — the 12 simulated-benchmark tables (18.1 per-network AUROCs, 18.2 top-30, 18.3–18.7 accuracy/stability/sweep tables, §20 knockout comparators, methods/cross-check/benchmark/Wilcoxon tables). The §0 real top-20 table was skipped: it duplicates `results/geneRankings_real.tsv`.
- `tables/gene_explorer_40nets.tsv` — all 12,000 gene rows behind the §19 explorer (40 seeds × 300 genes; weights ÷1e5, betweenness ÷1e4, PPR ÷1e5 as rendered).
- `figures/sim_fig_01–25.png` + `captions.tsv` — the 25 embedded simulation figures.

These are simulation-benchmark outputs. Real-data results live in `results/` root (`geneRankings_real.tsv`, `summary_real.json`, `figures/01_*`–`08_*`).
