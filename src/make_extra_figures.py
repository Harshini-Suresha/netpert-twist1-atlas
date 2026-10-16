#!/usr/bin/env python
"""Extra real-data figures for the Twist1 NetPert run.

Reads results/geneRankings_real.tsv + data/real/fig1_dissemination.csv.
All plotted values are real (IID+TRRUST network, RNA-seq responses, wet-lab assays).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "results", "figures")
DATA = os.path.join(ROOT, "data", "real")
os.makedirs(FIG, exist_ok=True)

r = pd.read_csv(os.path.join(RES, "geneRankings_real.tsv"), sep="\t")
print(f"genes: {len(r)}")

# 1. weight distribution (log scale), by type
plt.figure(figsize=(7, 4))
w = r["NetPert_weight"].values
wpos = w[w > 0]
plt.hist(np.log10(wpos), bins=50, color="#c2410c", alpha=0.85)
plt.xlabel("log10 NetPert weight (real Twist1 network)")
plt.ylabel("genes")
plt.title(f"NetPert weight distribution (n={len(r)}, {len(wpos)} with weight > 0)")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "weight_dist_real.png"), dpi=150)
plt.close()

# 2. NetPert vs full-network betweenness (log-log)
plt.figure(figsize=(5.5, 5))
x = r["NetPert_weight"].values
y = r["BCfull_weight"].values
m = (x > 0) & (y > 0)
rho, p = spearmanr(x, y)
plt.scatter(np.log10(x[m]), np.log10(y[m]), s=8, alpha=0.5)
for _, row in r.sort_values("NetPert_rank").head(8).iterrows():
    if row["NetPert_weight"] > 0 and row["BCfull_weight"] > 0:
        plt.annotate(row["gene"], (np.log10(row["NetPert_weight"]), np.log10(row["BCfull_weight"])), fontsize=7)
plt.xlabel("log10 NetPert weight (real)")
plt.ylabel("log10 betweenness, full 17k network (real)")
plt.title(f"NetPert vs betweenness (Spearman rho={rho:.2f}, n={len(r)})")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "netpert_vs_bc_real.png"), dpi=150)
plt.close()
print(f"netpert-vs-bc rho={rho:.3f} p={p:.3g}")

# 3. hub-bias check: full-network out-degree vs NetPert weight
plt.figure(figsize=(5.5, 5))
d = r["outdegree_full"].values
m2 = (x > 0) & (d > 0)
rho2, p2 = spearmanr(x[m2], d[m2])
plt.scatter(np.log10(d[m2]), np.log10(x[m2]), s=8, alpha=0.5)
plt.xlabel("log10 out-degree, full network (real)")
plt.ylabel("log10 NetPert weight (real)")
plt.title(f"Hub-bias check (Spearman rho={rho2:.2f}, n={m2.sum()})")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "degree_vs_netpert_real.png"), dpi=150)
plt.close()
print(f"degree-vs-netpert rho={rho2:.3f} p={p2:.3g}")

# 4. rank-rank: NetPert rank vs BC-sub rank
plt.figure(figsize=(5.5, 5))
plt.scatter(r["NetPert_rank"], r["BCsub_rank"], s=8, alpha=0.5)
rho3, p3 = spearmanr(r["NetPert_rank"], r["BCsub_rank"])
plt.xlabel("NetPert rank (1 = top, real)")
plt.ylabel("Betweenness-sub rank (1 = top, real)")
plt.title(f"Rank agreement (Spearman rho={rho3:.2f}, n={len(r)})")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "rank_comparison_real.png"), dpi=150)
plt.close()

# 5. assay inhibition bars (real dissemination % at 1uM, 18 genes)
a = pd.read_csv(os.path.join(DATA, "fig1_dissemination.csv"))
a.columns = [c.strip() for c in a.columns]
dcol = [c for c in a.columns if "Dissemination at 1" in c and "SEM" not in c][0]
m = r.merge(a, left_on="gene", right_on="Gene name", how="inner")
m[dcol] = pd.to_numeric(m[dcol], errors="coerce")
mm = m.dropna(subset=[dcol]).sort_values(dcol)
plt.figure(figsize=(8, 6))
colors = plt.cm.RdYlGn_r((mm[dcol].values / 100.0))
plt.barh(mm["gene"], mm[dcol].values, color=colors)
plt.xlabel("Dissemination at 1 uM % — real assay (lower = stronger inhibition)")
plt.title(f"Real dissemination readout for ranked genes (n={len(mm)})")
for i, (g, v, rk) in enumerate(zip(mm["gene"], mm[dcol], mm["NetPert_rank"])):
    plt.text(v + 1, i, f"NP#{rk}", fontsize=7, va="center")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "assay_inhibition_real.png"), dpi=150)
plt.close()

# 6. coverage: responses in-network vs missing; network sizes
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
ax[0].bar(["in network", "absent"], [168, 14], color=["#15803d", "#b91c1c"])
ax[0].set_ylabel("response genes (real DE list, n=182)")
ax[0].set_title("Response coverage in IID+TRRUST")
for i, v in enumerate([168, 14]):
    ax[0].text(i, v + 2, str(v), ha="center", fontsize=10)
ax[1].bar(["full network", "Twist1 subnetwork"], [17191, 1200], color=["#1d4ed8", "#c2410c"])
ax[1].set_ylabel("genes (real)")
ax[1].set_title("Network sizes")
for i, v in enumerate([17191, 1200]):
    ax[1].text(i, v + 300, str(v), ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "coverage_real.png"), dpi=150)
plt.close()

print("saved:", sorted(os.listdir(FIG)))
