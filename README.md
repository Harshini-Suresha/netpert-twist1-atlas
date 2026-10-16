# NetPert Twist1 — real lab data run

Reproduction of [NetPert](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012195)
(Perrone et al., PLOS Comp. Biol. 2024) using **only real experimental/curated data**.
There are no Barabási–Albert / Erdős–Rényi / Watts–Strogatz simulations anywhere in
`results/` — every number there comes from the sources below.

## What was run

- **Driver:** `Twist1` (inducible Twist1 mouse mammary organoids, Shamir et al. dissemination model).
- **Responses:** `data/real/responses.csv` — Twist1 + 182 differentially expressed genes
  (official `netpert_v1` manuscript project file; DE list from Shamir et al. RNA-seq,
  cf. paper: 183 genome-wide-significant genes).
- **Network (real):**
  - PPI: IID `mouse_annotated_PPIs.txt.gz`, `symbol1`/`symbol2` columns
    (952,754 rows → 1,886,194 directed edges after de-duplication; experimental + orthologous).
  - Regulatory: TRRUST v2 mouse `trrust_rawdata.mouse.tsv` (7,057 TF→target pairs, 827 TFs).
  - Full real network: **17,191 nodes / 1,891,745 directed edges**.
  - In-network responses: **168/182**; 14 missing are genuinely absent from IID+TRRUST
    (`Gpr110, Lphn3, Odz3, E030013I19Rik, Slco1a5, Gbp8, Fut2, Btbd11, Sept8,
    3110079O15Rik, 2610528A11Rik, 2010111I01Rik, 1600029D21Rik, St5`) — the paper's
    "Gene U" unconnected-response case, not a bug.
- **Method:** `src/run_real_netpert.py`
  - Dense NetPert (`scipy.linalg.expm`, Romberg `nt=16`, `r=0.5`) on a **real
    Twist1-centered subnetwork**: driver + 168 in-network responses + driver 1-hop
    (205) + shortest-path mediators (≤4 steps) + top 1-hop connectors → **1,200 nodes /
    228,199 directed edges**.
  - Conservation identity `Σ_x G_bx(T−t)·G_xa(t) = G_ba(T)`: max rel. deviation **1.99e-15**.
  - Betweenness-centrality-subset (driver → responses) on the **full 17k-node network**
    and on the subnetwork; out-degree baseline.
  - Drug mapping: Drug Repurposing Hub 2020-03-24 (human targets → mouse orthologs via
    MGI `HOM_MouseHumanSequence.rpt`); compounds with >5 targets omitted per paper.
- **Wet-lab assays (real):** `data/real/fig1_dissemination.csv` parsed from
  EwaldLab `2019_Prkd1` `Georgess_Prkd1_2019_Data_Final.xlsx` Fig.1
  (Georgess et al., Cancer Res. 2019): real log2FC, IC50, and **dissemination % at 1 µM**
  for inhibitors of Twist1-upregulated genes.

## Results (real data)

Top-20 NetPert intermediates (full table: `results/geneRankings_real.tsv`):

| rank | gene | weight | drugs (sample) |
|---|---|---|---|
| 1 | Pdgfra | 2.13e-02 | crenolanib, cediranib, CP-673451, … |
| 2 | Htt | 1.80e-02 | — |
| 3 | Tp53 | 1.76e-02 | — |
| 4 | Ctnnb1 | 1.54e-02 | ICG-001, iCRT-14, … |
| 5 | Cdh1 | 1.51e-02 | — |
| 6 | Zranb1 | 1.43e-02 | — |
| 7 | Vim | 1.37e-02 | — |
| 8 | Ppia | 1.36e-02 | cyclosporin-A, … |
| 9 | Stat3 | 1.28e-02 | napabucasin, WP1066, … |
| 10 | Clu | 1.28e-02 | — |
| 11 | Stub1 | 1.26e-02 | — |
| 12 | Nfkb1 | 1.25e-02 | aspirin, parthenolide, … |
| 13 | Ep300 | 1.15e-02 | C646, SGC-CBP30, curcumin, … |
| 14 | Sox2 | 1.14e-02 | — |
| 15 | Col2a1 | 1.08e-02 | — |
| 16 | Rela | 1.08e-02 | bortezomib, parthenolide, triptolide, … |
| 17 | Kif23 | 1.07e-02 | — |
| 18 | Hif1a | 1.04e-02 | YC-1, acriflavine, … |
| 19 | Hdac6 | 1.02e-02 | belinostat, dacinostat, … |
| 20 | H4c1 | 1.02e-02 | — |

Assay correlation (honest null): 18 genes have both a subnetwork ranking and a real
dissemination readout. Spearman(NetPert weight, dissemination %) = **+0.31, p=0.21,
n=18**; BC-subset = −0.10, p=0.71. I.e. **no significant prediction of dissemination
inhibition in this small subset**. Important context: all 18 are *response* genes
(type R), whereas NetPert is designed to prioritize *intermediates* — the published
dissemination screen targeted DE genes themselves, so this test is underpowered and
mismatched by construction (e.g. Sema5a shows 0% dissemination under Gö-6976/Prkd1
inhibition yet ranks 1133 as a response, as expected). The value here is the
real-data ranking + drug list, not this correlation.

Figures in `results/figures/` grouped by folder (see `gallery.html` on the site, or `results/figures/gallery.json` manifest):
- `01_rankings/`: top-10/20/30/50/100 bars, BC + hub leaderboards, rank curves, class split, top-k overlap.
- `02_network/`: response coverage, the missing genes, composition, degree distributions, Twist1 degree context, driver-to-response distances, mediator frequency, edge provenance, response direction.
- `03_assays/`: dissemination bars, IC50 distribution, potency/effect, log2FC analyses, NetPert/BC/degree vs assay, assay table, per-gene panels.
- `04_spotlights/`: one card per top-ranked NetPert gene (ranks, drugs, assay status).
- `05_drugs/`: compounds-per-gene, top compounds, druggable enrichment, top druggable list, per-target compound charts.
- `06_compare/`: pairwise scatters, rank shifts, winners/losers, intermediates/responses-only, agreement + assay-rho summaries.
- `07_overview/`: summary table, pipeline schematic.
- `08_report/`: report-style analogues computed on real data — score concentration curve, method-agreement panels, edge-removal robustness, relaxation sweep, hub-bias panels, weight-vs-distance, assay waterfall, top-30 rank heatmap, path-type counts/weights/DIR-enrichment, scorecard table.
- `09_complex/`: multi-panel dashboards combining ranking, network, assay, drug, path and robustness views.
- Plus originals at `results/figures/` root.
- Regenerate: `python src/make_extra_figures.py`, `python src/make_100_figures.py`, `python src/make_report_style_figures.py` and `python src/make_complex_figures.py` (full-network panels need the IID download, skipped gracefully if absent).

## Reproduce

```bash
python -m pip install -r requirements.txt   # numpy scipy pandas networkx matplotlib scikit-learn
python src/run_real_netpert.py
```

- IID (~38 MB) auto-downloads to `data/real/` if absent.
- For drug mapping, place `repurposing_drugs_20200324.txt`
  (https://s3.amazonaws.com/data.clue.io/repurposing/downloads/repurposing_drugs_20200324.txt)
  and `HOM_MouseHumanSequence.rpt`
  (https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt)
  in `/tmp/real_data/` (or edit paths); otherwise that step is skipped gracefully.
- Runtime ≈ 3–6 min (full-network betweenness dominates).

## Files

- `src/run_real_netpert.py` — full real-data pipeline (network build → NetPert → BC → drugs → assays → figures).
- `data/real/responses.csv` — Twist1 + 182 DE genes (official manuscript file).
- `data/real/fig1_dissemination.csv` — real DE + IC50 + dissemination % (EwaldLab).
- `data/real/trrust_rawdata.mouse.tsv` — TRRUST v2 mouse (downloaded from grnpedia.org).
- `results/geneRankings_real.tsv` — rankings/weights/BC/drugs for all 1,200 subnetwork genes.
- `results/summary_real.json` — network sizes, conservation check, correlation stats, provenance.
- `results/figures/` — real-data plots.
- `results/archive/` — everything extracted from the report HTML so nothing is missed: 12 benchmark tables, all 25 embedded figures + captions, and the 12,000-row gene-explorer dataset (see its README; simulated, not real).

## Limitations

- Dense `expm` forces the 1,200-gene Twist1 neighborhood for NetPert; BC runs on the full
  17k network for comparison. A sparse-`expm_multiply` full-network NetPert is future work.
- 14/182 responses absent from IID+TRRUST (interactome incompleteness — real limitation).
- PPI treated as undirected (both directions = 1); TRRUST directed; regulation sign ignored (40% missing + context-dependent, per paper).
- Human→mouse drug mapping assumes conserved activity; promiscuous compounds (>5 targets) omitted.
- No multiple-testing correction on the assay correlation; n=18, descriptive only.

## Provenance

- Paper: Perrone et al. 2024, https://doi.org/10.1371/journal.pcbi.1012195
- Code ref: https://github.com/joelbaderlab/netpert_v1 (`bin/netpert.py`, manuscript branch)
- PPI: http://iid.ophid.utoronto.ca/static/download/mouse_annotated_PPIs.txt.gz
- TF: https://www.grnpedia.org/trrust/data/trrust_rawdata.mouse.tsv
- Assays: https://github.com/EwaldLab/2019_Prkd1/ (Georgess et al. 2019, PMID 31676574)
- Drugs: Drug Repurposing Hub 2020-03-24; homology: MGI HOM_MouseHumanSequence.rpt
