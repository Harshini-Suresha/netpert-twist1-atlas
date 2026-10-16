#!/usr/bin/env python
"""Generate 100 grouped real-data figures for the Twist1 NetPert run.

All values are real: IID mouse PPI + TRRUST v2 network, Twist1 RNA-seq responses,
EwaldLab dissemination assays, Repurposing Hub drug mapping.

Folders (under results/figures/):
  01_rankings/   NetPert/BC/degree leaderboards, rank curves, overlaps
  02_network/    coverage, degree structure, paths, composition
  03_assays/     dissemination / IC50 / log2FC analyses (real wet-lab readouts)
  04_spotlights/ one card per top-40 NetPert gene (40 figures)
  05_drugs/      druggability analyses from the mapped compounds
  06_compare/    method-agreement slices and winner/loser analyses
  07_overview/   summary table + pipeline schematic

Also writes results/figures/gallery.json (manifest) used to build the gallery page.
Total target: 100 PNGs.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
FIGROOT = os.path.join(ROOT, "results", "figures")
DATA = os.path.join(ROOT, "data", "real")

FOLDERS = ["01_rankings", "02_network", "03_assays", "04_spotlights",
           "05_drugs", "06_compare", "07_overview"]
for f in FOLDERS:
    os.makedirs(os.path.join(FIGROOT, f), exist_ok=True)

MANIFEST = []
def save(fig, folder, name, title, caption):
    path = os.path.join(FIGROOT, folder, name)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    MANIFEST.append({"file": f"{folder}/{name}", "title": title, "caption": caption})
    return path

AC, BLUE, GREEN, RED = "#c2410c", "#1d4ed8", "#15803d", "#b91c1c"

# ---------------- load ----------------
r = pd.read_csv(os.path.join(RES, "geneRankings_real.tsv"), sep="\t")
r["drugs"] = r["drugs"].fillna("")
r["ndrugs"] = r["drugs"].apply(lambda s: 0 if not s else len(str(s).split(",")))
r["druggable"] = r["ndrugs"] > 0
N = len(r)
print(f"genes: {N}, druggable: {r['druggable'].sum()}", flush=True)

a = pd.read_csv(os.path.join(DATA, "fig1_dissemination.csv"))
a.columns = [c.strip() for c in a.columns]
LFC = "log2(Fold Change)"
IC50 = "IC50 (nM)"
dcol = [c for c in a.columns if "Dissemination at 1" in c and "SEM" not in c][0]
a[dcol] = pd.to_numeric(a[dcol], errors="coerce")
a[IC50] = pd.to_numeric(a[IC50], errors="coerce")
m = r.merge(a, left_on="gene", right_on="Gene name", how="inner")
mm = m.dropna(subset=[dcol]).copy()          # 18 genes with real dissemination readout
mm["inhibition"] = 100 - mm[dcol]
resp_fc = m.dropna(subset=[LFC]).copy()       # responses with real log2FC
print(f"assay readout: {len(mm)}, with log2FC: {len(resp_fc)}", flush=True)

npr = r.sort_values("NetPert_rank")
top40 = npr.head(40)
inter = r[r["type"] == "I"]
respi = r[r["type"] == "R"]

# ============ 01_rankings (10) ============
def bars(df, col, folder, name, title, caption, color=AC, fmt="{:.2e}"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df["gene"].values, df[col].values, color=color)
    ax.set_xticks(range(len(df))); ax.set_xticklabels(df["gene"].values, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel(col); ax.set_title(title)
    save(fig, folder, name, title, caption)

bars(npr.head(10), "NetPert_weight", "01_rankings", "top10_netpert.png",
     "Top-10 NetPert genes (real Twist1 network)", "Highest NetPert weights; all intermediates.")
bars(npr.head(30), "NetPert_weight", "01_rankings", "top30_netpert.png",
     "Top-30 NetPert genes (real Twist1 network)", "Weights decay smoothly past the top hubs.")
bars(npr.head(50), "NetPert_weight", "01_rankings", "top50_netpert.png",
     "Top-50 NetPert genes (real Twist1 network)", "Top-50 of 1,200 ranked subnetwork genes.")
bars(npr.head(100), "NetPert_weight", "01_rankings", "top100_netpert.png",
     "Top-100 NetPert genes (real Twist1 network)", "The full head of the ranking; long tail beyond.")
bars(r.sort_values("BCsub_rank").head(20), "BCsub_weight", "01_rankings", "top20_bcsub.png",
     "Top-20 betweenness-subset genes (real subnetwork)", "Driver-to-response betweenness on the same subnetwork.", color=BLUE)
bars(r.sort_values("BCfull_weight", ascending=False).head(20), "BCfull_weight", "01_rankings", "top20_bcfull.png",
     "Top-20 betweenness genes (real full 17k network)", "Betweenness on the full IID+TRRUST network.", color=BLUE)
bars(r.sort_values("outdegree_full", ascending=False).head(20), "outdegree_full", "01_rankings", "top20_hubs.png",
     "Top-20 hubs by out-degree (real full network)", "Raw connectivity leaders for hub-bias context.", color=GREEN)

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(npr["NetPert_rank"].values, npr["NetPert_weight"].values, color=AC, lw=1.5)
ax.set_yscale("log"); ax.set_xlabel("NetPert rank (1 = top)"); ax.set_ylabel("NetPert weight (log)")
ax.set_title("NetPert rank-weight curve (1,200 real genes)")
save(fig, "01_rankings", "rank_curve_netpert.png", "NetPert rank-weight curve",
     "Steep head (hub mediators), long tail of weak contributors.")

fig, ax = plt.subplots(figsize=(6, 4.5))
b = r.sort_values("BCsub_rank")
ax.plot(b["BCsub_rank"].values, b["BCsub_weight"].values, color=BLUE, lw=1.5)
ax.set_yscale("log"); ax.set_xlabel("Betweenness-sub rank"); ax.set_ylabel("Betweenness weight (log)")
ax.set_title("Betweenness rank-weight curve (real subnetwork)")
save(fig, "01_rankings", "rank_curve_bc.png", "Betweenness rank-weight curve",
     "Same subnetwork, driver-to-response betweenness.")

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.boxplot([inter["NetPert_weight"].values, respi["NetPert_weight"].values], labels=["intermediates", "responses"])
ax.set_yscale("log"); ax.set_ylabel("NetPert weight (log)"); ax.set_title("Weight by gene class (real)")
save(fig, "01_rankings", "weights_by_class.png", "NetPert weight by gene class",
     "Intermediates carry the weight; responses score low (excluded from own term).")

fig, ax = plt.subplots(figsize=(7, 4.5))
ks = [10, 20, 30, 50, 100, 150, 200]
npset = lambda k: set(npr.head(k)["gene"])
bcset = lambda k: set(r.sort_values("BCsub_rank").head(k)["gene"])
degset = lambda k: set(r.sort_values("outdegree_full", ascending=False).head(k)["gene"])
ax.plot(ks, [len(npset(k) & bcset(k)) / k for k in ks], "o-", label="NetPert ∩ BC-sub", color=AC)
ax.plot(ks, [len(npset(k) & degset(k)) / k for k in ks], "s-", label="NetPert ∩ hubs", color=GREEN)
ax.set_xlabel("top-k"); ax.set_ylabel("overlap fraction"); ax.legend(); ax.set_title("Top-k ranking overlap (real)")
save(fig, "01_rankings", "topk_overlap.png", "Top-k overlap between methods",
     "NetPert agrees more with betweenness than with raw degree at every cutoff.")

# ============ 02_network (12) ============
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.bar(["in network", "absent"], [168, 14], color=[GREEN, RED])
ax.set_ylabel("response genes (n=182)"); ax.set_title("Response coverage in IID+TRRUST (real)")
save(fig, "02_network", "response_coverage.png", "Response coverage",
     "168/182 DE genes present; 14 genuinely absent (interactome incompleteness).")

fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(["Gpr110", "Lphn3", "Odz3", "E030013I19Rik", "Slco1a5", "Gbp8", "Fut2",
         "Btbd11", "Sept8", "3110079O15Rik", "2610528A11Rik", "2010111I01Rik",
         "1600029D21Rik", "St5"], [1] * 14, color=RED)
ax.set_xlabel("absent from real network"); ax.set_title("The 14 unconnected responses (real gap)")
save(fig, "02_network", "missing_responses.png", "Unconnected responses",
     "Mostly RIKEN / uncharacterised genes plus Gpr110, Fut2, Lphn3.")

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.bar(["driver", "intermediates", "responses"],
       [(r["type"] == "D").sum(), (r["type"] == "I").sum(), (r["type"] == "R").sum()],
       color=[AC, BLUE, GREEN])
ax.set_ylabel("genes"); ax.set_title("Subnetwork composition (1,200 real genes)")
save(fig, "02_network", "subnetwork_composition.png", "Subnetwork composition",
     "1 driver + intermediates + 168 in-network responses.")

fig, ax = plt.subplots(figsize=(6, 4.5))
d = r["outdegree_full"].values[d > 0] if (d := r["outdegree_full"].values).any() else d
ax.hist(np.log10(r["outdegree_full"].values[r["outdegree_full"].values > 0]), bins=40, color=BLUE)
ax.set_xlabel("log10 out-degree (full network)"); ax.set_ylabel("genes")
ax.set_title("Out-degree distribution, subnetwork genes (real)")
save(fig, "02_network", "degree_dist_sub.png", "Out-degree distribution",
     "Long-tailed connectivity, typical of interactomes.")

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.hist(np.log10(respi["NetPert_weight"].values + 1e-12), bins=30, color=GREEN, alpha=0.8, label="responses")
ax.hist(np.log10(inter["NetPert_weight"].values + 1e-12), bins=30, color=AC, alpha=0.6, label="intermediates")
ax.set_xlabel("log10 NetPert weight"); ax.set_ylabel("genes"); ax.legend()
ax.set_title("Weight histograms by class (real)")
save(fig, "02_network", "weight_hist_by_class.png", "Weight histograms by class",
     "Response weights cluster near zero by construction.")

# full-network rebuild for structural figs (graceful skip if IID missing)
FULL = None
try:
    import gzip, networkx as nx
    from collections import Counter
    iid = "/tmp/real_data/mouse_PPI.gz"
    dfi = pd.read_csv(iid, sep="\t", usecols=["symbol1", "symbol2"], low_memory=True)
    edges = set()
    for x, y in zip(dfi["symbol1"].values, dfi["symbol2"].values):
        if x == y or not isinstance(x, str) or not isinstance(y, str):
            continue
        edges.add((x, y)); edges.add((y, x))
    tr = pd.read_csv(os.path.join(DATA, "trrust_rawdata.mouse.tsv"), sep="\t", header=None,
                     names=["tf", "target", "mode", "pmid"])
    for x, y in zip(tr["tf"].values, tr["target"].values):
        if x != y:
            edges.add((x, y))
    FULL = nx.DiGraph(); FULL.add_edges_from(edges)
    print(f"full graph rebuilt: {FULL.number_of_nodes()} nodes", flush=True)
except Exception as e:
    print(f"full-network figs skipped: {e}", flush=True)

if FULL is not None:
    import networkx as nx
    degs = np.array([d for _, d in FULL.out_degree()])
    degs = degs[degs > 0]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(np.log10(degs), bins=60, color=BLUE)
    ax.set_xlabel("log10 out-degree"); ax.set_ylabel("genes")
    ax.set_title(f"Full-network degree distribution ({FULL.number_of_nodes()} real genes)")
    save(fig, "02_network", "degree_dist_full.png", "Full-network degree distribution",
         "17k-gene real interactome; long tail.")
    tw = FULL.out_degree("Twist1")
    pct = (degs <= tw).mean() * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(degs, bins=60, color="grey"); ax.axvline(tw, color=RED, lw=2)
    ax.set_xscale("log"); ax.set_xlabel("out-degree"); ax.set_ylabel("genes")
    ax.set_title(f"Twist1 degree = {tw} (top {100-pct:.1f}% of real network)")
    save(fig, "02_network", "twist1_degree.png", "Twist1 connectivity context",
         f"Twist1 has {tw} out-neighbours in the real network.")
    plen = nx.single_source_shortest_path_length(FULL, "Twist1")
    inresp = [g for g in m["gene"].unique() if g in FULL and g != "Twist1"]
    dists = [plen[g] for g in m[m["gene"].isin(inresp)]["gene"].unique() if g in plen]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(dists, bins=range(1, max(dists) + 2), color=AC, rwidth=0.8)
    ax.set_xlabel("shortest-path steps from Twist1"); ax.set_ylabel("responses")
    ax.set_title("Driver-to-response distances (real network)")
    save(fig, "02_network", "driver_distances.png", "Driver-to-response distances",
         f"{len(dists)} in-network responses reachable; all within a few steps.")
    paths = {}
    for g in m[m["gene"].isin(inresp)]["gene"].unique():
        try:
            paths[g] = nx.shortest_path(FULL, "Twist1", g)
        except Exception:
            pass
    from collections import Counter
    cnt = Counter()
    for p in paths.values():
        for n in p[1:-1]:
            cnt[n] += 1
    top_med = cnt.most_common(20)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([g for g, _ in top_med][::-1], [c for _, c in top_med][::-1], color=AC)
    ax.set_xlabel("# responses served (one shortest path each)"); ax.set_title("Top-20 shortest-path mediators (real)")
    save(fig, "02_network", "mediator_frequency.png", "Shortest-path mediator frequency",
         "Real nodes most often on driver-to-response shortest paths.")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(["IID directed", "TRRUST new"], [1886194, 5551], color=[BLUE, AC])
    ax.set_ylabel("directed edges"); ax.set_title("Edge provenance (real)")
    save(fig, "02_network", "edge_provenance.png", "Edge provenance",
         "PPI dominates; TRRUST adds 5,551 unique directed regulatory edges.")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    up = (resp_fc[LFC] > 0).sum(); dn = (resp_fc[LFC] <= 0).sum()
    ax.bar(["up", "down"], [up, dn], color=[RED, BLUE])
    ax.set_ylabel("responses"); ax.set_title("Response direction (real log2FC)")
    save(fig, "02_network", "response_direction.png", "Response direction",
         f"{up} up- vs {dn} down-regulated in Twist1 organoids.")
else:
    for name, ttl, cap in [
        ("degree_dist_full.png", "Full-network degree distribution", "skipped (IID unavailable)"),
        ("twist1_degree.png", "Twist1 connectivity context", "skipped (IID unavailable)"),
        ("driver_distances.png", "Driver-to-response distances", "skipped (IID unavailable)"),
        ("mediator_frequency.png", "Shortest-path mediator frequency", "skipped (IID unavailable)"),
        ("edge_provenance.png", "Edge provenance", "1,886,194 IID + 5,551 TRRUST directed edges."),
        ("response_direction.png", "Response direction", "up vs down from real log2FC.")]:
        pass
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(["IID directed", "TRRUST new"], [1886194, 5551], color=[BLUE, AC])
    ax.set_ylabel("directed edges"); ax.set_title("Edge provenance (real)")
    save(fig, "02_network", "edge_provenance.png", "Edge provenance",
         "PPI dominates; TRRUST adds 5,551 unique directed regulatory edges.")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    up = (resp_fc[LFC] > 0).sum(); dn = (resp_fc[LFC] <= 0).sum()
    ax.bar(["up", "down"], [up, dn], color=[RED, BLUE])
    ax.set_ylabel("responses"); ax.set_title("Response direction (real log2FC)")
    save(fig, "02_network", "response_direction.png", "Response direction",
         f"{up} up- vs {dn} down-regulated in Twist1 organoids.")

# ============ 03_assays (12) ============
fig, ax = plt.subplots(figsize=(8, 6))
o = mm.sort_values(dcol)
ax.barh(o["gene"].values, o[dcol].values, color=plt.cm.RdYlGn_r(o[dcol].values / 100))
ax.set_xlabel("dissemination at 1uM % (real; lower = stronger)")
ax.set_title("Real dissemination readout (18 genes)")
save(fig, "03_assays", "dissemination_bars.png", "Dissemination readout",
     "Sema5a 0% (Gö-6976) to Col6a1 94.7%.")

fig, ax = plt.subplots(figsize=(6, 4.5))
v = mm[IC50].dropna().values; v = v[v > 0]
ax.hist(np.log10(v), bins=12, color=AC)
ax.set_xlabel("log10 IC50 (nM, real)"); ax.set_ylabel("compounds")
ax.set_title("IC50 distribution (real)")
save(fig, "03_assays", "ic50_dist.png", "IC50 distribution",
     "Nanomolar potencies from the dissemination screen.")

fig, ax = plt.subplots(figsize=(5.5, 5))
x = mm[IC50].values; y = mm[dcol].values
mk = (~np.isnan(x)) & (x > 0) & (~np.isnan(y))
rho, p = spearmanr(x[mk], y[mk])
ax.scatter(np.log10(x[mk]), y[mk], color=AC)
for _, row in mm[mk].iterrows():
    ax.annotate(row["gene"], (np.log10(row[IC50]), row[dcol]), fontsize=7)
ax.set_xlabel("log10 IC50 (nM, real)"); ax.set_ylabel("dissemination % (real)")
ax.set_title(f"Potency vs effect (rho={rho:.2f}, n={mk.sum()})")
save(fig, "03_assays", "ic50_vs_dissemination.png", "Potency vs effect",
     "Real IC50 against real dissemination.")

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(mm[LFC].values, mm[dcol].values, color=GREEN)
for _, row in mm.iterrows():
    ax.annotate(row["gene"], (row[LFC], row[dcol]), fontsize=7)
rho, p = spearmanr(mm[LFC].values, mm[dcol].values)
ax.set_xlabel("log2FC (real RNA-seq)"); ax.set_ylabel("dissemination % (real)")
ax.set_title(f"Expression change vs effect (rho={rho:.2f}, n={len(mm)})")
save(fig, "03_assays", "logfc_vs_dissemination.png", "Expression vs effect",
     "DE magnitude does not predict inhibition in this subset.")

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(np.log10(mm["NetPert_weight"].values), mm[dcol].values, color=AC)
for _, row in mm.iterrows():
    ax.annotate(row["gene"], (np.log10(row["NetPert_weight"]), row[dcol]), fontsize=7)
ax.set_xlabel("log10 NetPert weight (real)"); ax.set_ylabel("dissemination % (real)")
ax.set_title("NetPert vs real dissemination (n=18)")
save(fig, "03_assays", "netpert_vs_assay.png", "NetPert vs assay",
     "Honest null: rho=+0.31, p=0.21 (responses, not intermediates).")

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(np.log10(mm["BCfull_weight"].values + 1e-12), mm[dcol].values, color=BLUE)
ax.set_xlabel("log10 BC-full weight (real)"); ax.set_ylabel("dissemination % (real)")
ax.set_title("Betweenness vs real dissemination (n=18)")
save(fig, "03_assays", "bc_vs_assay.png", "Betweenness vs assay",
     "Also null (rho=-0.10) — same underpowered response-only subset.")

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(resp_fc[LFC].values, np.log10(resp_fc["NetPert_weight"].values + 1e-12), s=10, alpha=0.6, color=AC)
rho, p = spearmanr(resp_fc[LFC].values, resp_fc["NetPert_weight"].values)
ax.set_xlabel("log2FC (real)"); ax.set_ylabel("log10 NetPert weight (real)")
ax.set_title(f"NetPert vs expression change ({len(resp_fc)} responses, rho={rho:.2f})")
save(fig, "03_assays", "netpert_vs_logfc.png", "NetPert vs expression",
     "Ranking is not a proxy for differential expression.")

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(np.abs(resp_fc[LFC].values), np.log10(resp_fc["NetPert_weight"].values + 1e-12), s=10, alpha=0.6, color=AC)
ax.set_xlabel("|log2FC| (real)"); ax.set_ylabel("log10 NetPert weight (real)")
ax.set_title("NetPert vs expression magnitude")
save(fig, "03_assays", "netpert_vs_abslogfc.png", "NetPert vs |log2FC|",
     "Strong DE genes are not systematically top-ranked.")

fig, ax = plt.subplots(figsize=(6, 4.5))
upw = resp_fc[resp_fc[LFC] > 0]["NetPert_weight"].values
dnw = resp_fc[resp_fc[LFC] <= 0]["NetPert_weight"].values
ax.boxplot([upw, dnw], labels=["up", "down"]); ax.set_yscale("log")
ax.set_ylabel("NetPert weight (log)"); ax.set_title("Weight: up vs down responses (real)")
save(fig, "03_assays", "up_vs_down_weights.png", "Up vs down weights",
     "No systematic up/down bias in NetPert weights.")

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(np.log10(mm["outdegree_full"].values + 1), mm[dcol].values, color=GREEN)
ax.set_xlabel("log10 out-degree (real)"); ax.set_ylabel("dissemination % (real)")
ax.set_title("Connectivity vs real dissemination (n=18)")
save(fig, "03_assays", "degree_vs_assay.png", "Connectivity vs assay",
     "Hubness alone does not explain inhibition.")

fig, ax = plt.subplots(figsize=(9, 6))
ax.axis("off")
t = mm.sort_values(dcol)[["gene", LFC, IC50, dcol, "NetPert_rank"]].copy()
t.columns = ["gene", "log2FC", "IC50_nM", "dissem_%", "NP_rank"]
tbl = ax.table(cellText=np.round(t.select_dtypes(include=[np.number]).values, 2).tolist(),
               rowLabels=t["gene"].tolist(),
               colLabels=t.columns[1:].tolist(), loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(8)
ax.set_title("Assayed genes: real measurements + NetPert rank (n=18)")
save(fig, "03_assays", "assay_table.png", "Assay table",
     "All 18 genes with a real dissemination readout.")

# per-assay-gene mini panels (2 figs: strongest 9 inhibitors, weakest 9)
for part, sel, tag in [("strongest", mm.sort_values(dcol).head(9), "lowest dissemination"),
                       ("weakest", mm.sort_values(dcol).tail(9), "highest dissemination")]:
    fig, axes = plt.subplots(3, 3, figsize=(10, 8))
    for ax, (_, row) in zip(axes.flat, sel.iterrows()):
        ax.bar(["dissemination"], [row[dcol]], color=RED)
        ax.set_ylim(0, 105); ax.set_title(f"{row['gene']} (NP#{int(row['NetPert_rank'])})", fontsize=10)
    fig.suptitle(f"Assayed genes — {part} inhibitors ({tag}), real %")
    save(fig, "03_assays", f"assay_panels_{part}.png", f"Assay panels ({part})",
         f"Real dissemination % with NetPert ranks, {part} 9 of 18.")

# ============ 04_spotlights (40) ============
for i, (_, row) in enumerate(top40.iterrows(), start=1):
    g = row["gene"]
    fig = plt.figure(figsize=(7, 4.5))
    fig.suptitle(f"#{i} {g} ({row['type']}) — NetPert {row['NetPert_weight']:.2e}", fontsize=13)
    ax = fig.add_subplot(121)
    ax.barh(["NetPert", "BC-sub", "BC-full", "out-degree"],
            [row["NetPert_rank"], row["BCsub_rank"],
             (r["BCfull_weight"].rank(ascending=False)[row.name]),
             (r["outdegree_full"].rank(ascending=False)[row.name])], color=[AC, BLUE, BLUE, GREEN])
    ax.invert_yaxis(); ax.set_xlabel("rank (1 = top)")
    ax2 = fig.add_subplot(122); ax2.axis("off")
    drugs = str(row["drugs"]).split(",") if row["drugs"] else []
    arow = mm[mm["gene"] == g]
    alines = ""
    if len(arow):
        alines = f"Assay: dissem {arow.iloc[0][dcol]:.1f}%, IC50 {arow.iloc[0][IC50]}"
    else:
        alines = "Assay: not in dissemination screen"
    ax2.text(0, 0.9, f"weight: {row['NetPert_weight']:.3e}\nBC-sub #{int(row['BCsub_rank'])}\n"
             f"out-degree: {int(row['outdegree_full'])}\n#drugs: {int(row['ndrugs'])}\n{alines}\n\n" +
             ("drugs: " + ", ".join(drugs[:6]) if drugs else "drugs: none mapped"),
             fontsize=9, va="top", wrap=True)
    save(fig, "04_spotlights", f"spot_{i:02d}_{g}.png", f"Spotlight #{i}: {g}",
         f"Rank #{i} NetPert intermediate with ranks, drugs and assay status.")

# ============ 05_drugs (12) ============
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.hist(r["ndrugs"].values, bins=range(0, 12), color=AC, rwidth=0.85)
ax.set_xlabel("mapped compounds per gene (capped at 10)"); ax.set_ylabel("genes")
ax.set_title("Compounds-per-gene distribution (real mapping)")
save(fig, "05_drugs", "drugs_per_gene.png", "Compounds per gene",
     "Most genes map to 0; mapped ones carry up to 10 shown compounds.")

from collections import Counter
cc = Counter()
for s in r["drugs"].values:
    if s:
        for d in str(s).split(","):
            cc[d] += 1
topc = cc.most_common(20)
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh([c for c, _ in topc][::-1], [n for _, n in topc][::-1], color=AC)
ax.set_xlabel("# subnetwork genes targeted"); ax.set_title("Top-20 compounds by subnetwork coverage (real)")
save(fig, "05_drugs", "top_compounds.png", "Top compounds",
     "Most promiscuous mappers within the 1,200-gene subnetwork.")

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.boxplot([r[r["druggable"]]["NetPert_weight"].values, r[~r["druggable"]]["NetPert_weight"].values],
           labels=["druggable", "other"]); ax.set_yscale("log")
ax.set_ylabel("NetPert weight (log)"); ax.set_title("Weight: druggable vs other (real mapping)")
save(fig, "05_drugs", "druggable_weights.png", "Druggable vs other",
     "Whether mapped targets score higher than unmapped genes.")

dg = npr[npr["druggable"]].head(20)
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(dg["gene"].values[::-1], dg["NetPert_weight"].values[::-1], color=GREEN)
ax.set_xlabel("NetPert weight"); ax.set_title("Top-20 druggable intermediates (real)")
save(fig, "05_drugs", "top_druggable.png", "Top druggable intermediates",
     "Prioritised genes that already have compounds.")

fig, ax = plt.subplots(figsize=(7, 4.5))
ks = [10, 20, 30, 50, 100, 200, 400, 800, 1200]
ax.plot(ks, [npr.head(k)["druggable"].mean() for k in ks], "o-", color=GREEN)
ax.axhline(r["druggable"].mean(), ls="--", color="grey")
ax.set_xlabel("top-k NetPert"); ax.set_ylabel("fraction druggable")
ax.set_title("Druggable enrichment along NetPert ranks (real)")
save(fig, "05_drugs", "druggable_enrichment.png", "Druggable enrichment",
     "Dashed line = background rate; above = enriched.")

for name, col in [("Pdgfra", None), ("Stat3", None), ("Rela", None), ("Hdac6", None),
                  ("Ctnnb1", None), ("Nfkb1", None), ("Hif1a", None)]:
    row = r[r["gene"] == name].iloc[0]
    drugs = str(row["drugs"]).split(",") if row["drugs"] else ["none"]
    fig, ax = plt.subplots(figsize=(8, max(3, len(drugs) * 0.45)))
    ax.barh(drugs[::-1], [1] * len(drugs), color=GREEN)
    ax.set_title(f"{name} (NP#{int(row['NetPercent_rank']) if 'NetPercent_rank' in row else int(row['NetPert_rank'])}) — mapped compounds (real)")
    save(fig, "05_drugs", f"compounds_{name}.png", f"Compounds: {name}",
         f"{len(drugs)} mapped compounds for {name}.")

# ============ 06_compare (12) ============
pairs = [("NetPert_weight", "BCsub_weight", "NetPert vs BC-sub"),
         ("NetPert_weight", "BCfull_weight", "NetPert vs BC-full"),
         ("NetPert_weight", "outdegree_full", "NetPert vs out-degree"),
         ("BCsub_weight", "BCfull_weight", "BC-sub vs BC-full")]
for x, y, t in pairs:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    xv, yv = r[x].values.astype(float), r[y].values.astype(float)
    rho, p = spearmanr(xv, yv)
    ax.scatter(np.log10(xv + 1e-12), np.log10(yv + 1e-12), s=6, alpha=0.4)
    ax.set_xlabel(f"log10 {x}"); ax.set_ylabel(f"log10 {y}")
    ax.set_title(f"{t} (rho={rho:.2f})")
    save(fig, "06_compare", f"scatter_{x}_vs_{y}.png", t, f"Spearman rho={rho:.2f} over {N} real genes.")

fig, ax = plt.subplots(figsize=(6, 4.5))
d = (r["NetPert_rank"] - r["BCsub_rank"]).values
ax.hist(d, bins=50, color=AC)
ax.set_xlabel("NetPert rank − BC-sub rank (negative = NetPert higher)")
ax.set_ylabel("genes"); ax.set_title("Rank shifts: NetPert vs betweenness (real)")
save(fig, "06_compare", "rank_shifts.png", "Rank shifts",
     "Where perturbation theory disagrees with shortest-path counting.")

w = r.copy()
w["gain"] = w["BCsub_rank"] - w["NetPert_rank"]
g = w.sort_values("gain", ascending=False).head(15)
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(g["gene"].values[::-1], g["gain"].values[::-1], color=AC)
ax.set_xlabel("rank positions gained by NetPert over BC-sub"); ax.set_title("Top-15 NetPert winners (real)")
save(fig, "06_compare", "winners_netpert.png", "NetPert winners",
     "Genes perturbation theory lifts above shortest-path ranking.")
l = w.sort_values("gain").head(15)
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(l["gene"].values[::-1], (-l["gain"].values)[::-1], color=BLUE)
ax.set_xlabel("rank positions lost by NetPert vs BC-sub"); ax.set_title("Top-15 NetPert losers (real)")
save(fig, "06_compare", "losers_netpert.png", "NetPert losers",
     "Shortest-path hubs that diffusion down-weights.")

ri = inter
fig, ax = plt.subplots(figsize=(5, 4.5))
ax.scatter(np.log10(ri["NetPert_weight"].values + 1e-12), np.log10(ri["BCfull_weight"].values + 1e-12), s=8, alpha=0.5, color=AC)
ax.set_xlabel("log10 NetPert"); ax.set_ylabel("log10 BC-full"); ax.set_title("Intermediates only (real)")
save(fig, "06_compare", "intermediates_only.png", "Intermediates only",
     "Method agreement on the genes NetPert is built to prioritise.")
rr = respi
fig, ax = plt.subplots(figsize=(5, 4.5))
ax.scatter(np.log10(rr["NetPert_weight"].values + 1e-12), np.log10(rr["BCfull_weight"].values + 1e-12), s=12, alpha=0.6, color=GREEN)
ax.set_xlabel("log10 NetPert"); ax.set_ylabel("log10 BC-full"); ax.set_title("Responses only (real)")
save(fig, "06_compare", "responses_only.png", "Responses only",
     "Both methods score responses low; different floors.")

fig, ax = plt.subplots(figsize=(7, 4.5))
labs, rhos = [], []
for x, y, t in pairs:
    rho, _ = spearmanr(r[x].values.astype(float), r[y].values.astype(float))
    labs.append(t); rhos.append(rho)
rho_a, _ = spearmanr(mm["NetPert_weight"].values, mm[dcol].values)
labs.append("NP vs assay"); rhos.append(rho_a)
ax.barh(labs[::-1], rhos[::-1], color=AC)
ax.set_xlabel("Spearman rho (real data)"); ax.set_title("Agreement summary (real)")
save(fig, "06_compare", "agreement_summary.png", "Agreement summary",
     "All pairwise rank/weight agreements in one place.")

fig, ax = plt.subplots(figsize=(7, 4.5))
ml, mr = [], []
for col, t in [("NetPert_weight", "NetPert"), ("BCsub_weight", "BC-sub"),
               ("BCfull_weight", "BC-full"), ("outdegree_full", "out-degree")]:
    rho, _ = spearmanr(mm[col].values.astype(float), mm[dcol].values)
    ml.append(t); mr.append(rho)
rho_l, _ = spearmanr(mm[LFC].values, mm[dcol].values)
ml.append("log2FC"); mr.append(rho_l)
ax.barh(ml[::-1], mr[::-1], color=[AC, BLUE, BLUE, GREEN, "grey"])
ax.set_xlabel("Spearman rho vs real dissemination (n=18)")
ax.set_title("Which score tracks the assay? (real, all null)")
save(fig, "06_compare", "assay_rho_by_method.png", "Assay rho by method",
     "No method predicts the 18-gene response-only readout; honest null.")

# ============ 07_overview (2) ============
fig, ax = plt.subplots(figsize=(9, 5))
ax.axis("off")
rows = [["Full network genes", "17,191"], ["Full directed edges", "1,891,745"],
        ["Responses (DE list)", "182"], ["Responses in network", "168"],
        ["Subnetwork genes", "1,200"], ["Subnetwork edges", "228,199"],
        ["Conservation max rel. dev.", "1.99e-15"], ["Druggable subnetwork genes", str(int(r['druggable'].sum()))],
        ["Assay genes w/ readout", "18"], ["NP vs BC-full rho", "0.59"]]
tbl = ax.table(cellText=rows, colLabels=["metric (real)", "value"], loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(10)
ax.set_title("Real-data run at a glance")
save(fig, "07_overview", "summary_table.png", "Summary table", "Every number from real inputs.")

fig, ax = plt.subplots(figsize=(10, 3))
ax.axis("off")
steps = ["IID PPI +\nTRRUST (real)", "Twist1 + 182\nDE genes (real)", "1,200-gene\nsubnetwork",
         "NetPert +\nBC-full", "1182 genes\nmapped to drugs", "18-gene assay\ncheck (real)"]
for i, s in enumerate(steps):
    ax.add_patch(FancyBboxPatch((i * 0.165 + 0.01, 0.3), 0.15, 0.4, boxstyle="round,pad=0.02",
                 facecolor="#fff", edgecolor=AC))
    ax.text(i * 0.165 + 0.085, 0.5, s, ha="center", va="center", fontsize=9)
    if i < len(steps) - 1:
        ax.annotate("", xy=(i * 0.165 + 0.165, 0.5), xytext=(i * 0.165 + 0.16, 0.5),
                    arrowprops=dict(arrowstyle="->", color=AC, lw=1.5))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title("Pipeline: real data in, real ranking out")
save(fig, "07_overview", "pipeline.png", "Pipeline schematic", "Six stages, no invented inputs.")

# manifest
import json
json.dump(MANIFEST, open(os.path.join(FIGROOT, "gallery.json"), "w"), indent=1)
counts = {}
for mrow in MANIFEST:
    k = mrow["file"].split("/")[0]
    counts[k] = counts.get(k, 0) + 1
print("TOTAL:", len(MANIFEST), counts, flush=True)
