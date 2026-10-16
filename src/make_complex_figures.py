#!/usr/bin/env python
"""Complex composite dashboards (multi-panel) from Twist1 data.
Saves to results/figures/09_complex/ + appends gallery.json manifest."""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = "/Users/harshinisuresha/Downloads/network pertubation"
RES = ROOT + "/results"
FIGD = ROOT + "/results/figures/09_complex"
DATA = ROOT + "/data/"
os.makedirs(FIGD, exist_ok=True)
MAN = []

def save(fig, name, title, caption):
    fig.tight_layout()
    fig.savefig(f"{FIGD}/{name}", dpi=100)
    plt.close(fig)
    MAN.append({"file": f"09_complex/{name}", "title": title, "caption": caption})

AC, BLUE, GREEN, RED, GREY, PUR = "#c2410c", "#1d4ed8", "#15803d", "#b91c1c", "#78716c", "#7c3aed"
plt.rcParams.update({"font.size": 8})

r = pd.read_csv(f"{RES}/geneRankings_real.tsv", sep="\t")
r["drugs"] = r["drugs"].fillna("")
w = r["NetPert_weight"].values
bc = r["BCfull_weight"].values
npr = r.sort_values("NetPert_rank")
sw = json.load(open("/tmp/real_swap.json"))

a = pd.read_csv(f"{DATA}/fig1_dissemination.csv")
a.columns = [c.strip() for c in a.columns]
a = a.rename(columns={"Unnamed: 5": "drug", "IC50 (nM)": "ic50",
                      "Dissemination at 1 µM": "dissem", "log2(Fold Change)": "log2fc"})
m = r.merge(a, left_on="gene", right_on="Gene name", how="inner")
m["dissem"] = pd.to_numeric(m["dissem"], errors="coerce")
m["ic50"] = pd.to_numeric(m["ic50"], errors="coerce")
mm = m.dropna(subset=["dissem"]).sort_values("dissem")

# 1. overview dashboard (2x3)
fig, ax = plt.subplots(2, 3, figsize=(12, 7))
fig.suptitle("Twist1 NetPert — overview dashboard", fontsize=12)
t20 = npr.head(20)
ax[0, 0].bar(t20["gene"].values, t20["NetPert_weight"].values, color=AC)
ax[0, 0].set_xticklabels(t20["gene"].values, rotation=75, ha="right", fontsize=6)
ax[0, 0].set_xticks(range(20)); ax[0, 0].set_title("Top-20 NetPert weights")
ax[0, 1].hist(np.log10(w[w > 0]), bins=40, color=AC)
ax[0, 1].set_title("Weight distribution (log10)"); ax[0, 1].set_xlabel("log10 weight")
xs = np.arange(1, len(w) + 1)
ax[0, 2].plot(xs, np.cumsum(np.sort(w)[::-1]), color=AC, label="NetPert")
ax[0, 2].plot(xs, np.cumsum(np.sort(bc)[::-1]) / bc.sum(), color=BLUE, label="Betweenness")
ax[0, 2].legend(fontsize=7); ax[0, 2].set_title("Concentration curves"); ax[0, 2].set_xlabel("top-k")
pairs = [("NP–BCsub", spearmanr(r["NetPert_weight"], r["BCsub_weight"])[0]),
         ("NP–BCfull", spearmanr(r["NetPert_weight"], r["BCfull_weight"])[0]),
         ("NP–degree", spearmanr(r["NetPert_weight"], r["outdegree_full"])[0])]
ax[1, 0].barh([p[0] for p in pairs], [p[1] for p in pairs], color=BLUE)
ax[1, 0].set_title("Method agreement (Spearman)")
lv = ["10%", "25%", "50%"]; kk = ["0.1", "0.25", "0.5"]
ax[1, 1].errorbar(lv, [sw["robustness"][k]["np_mean"] for k in kk],
                  yerr=[sw["robustness"][k]["np_sd"] for k in kk], fmt="o-", color=AC, label="NetPert")
ax[1, 1].errorbar(lv, [sw["robustness"][k]["bc_mean"] for k in kk],
                  yerr=[sw["robustness"][k]["bc_sd"] for k in kk], fmt="s-", color=BLUE, label="BC")
ax[1, 1].legend(fontsize=7); ax[1, 1].set_title("Edge-removal robustness")
swp = sw["sweep"]
ax[1, 2].plot(list(swp.keys()), [swp[k]["rho"] for k in swp], "o-", color=PUR)
ax[1, 2].set_title("Relaxation sweep (rho vs r=0.5)"); ax[1, 2].set_xlabel("r")
save(fig, "dashboard_overview.png", "Overview dashboard",
     "Ranking, distribution, concentration, agreement, robustness and sweep in one view.")

# 2. robustness deep-dive (2x2)
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("Robustness deep-dive (edge deletion, 3 repeats)", fontsize=12)
ax[0, 0].errorbar(lv, [sw["robustness"][k]["np_mean"] for k in kk],
                  yerr=[sw["robustness"][k]["np_sd"] for k in kk], fmt="o-", color=AC, label="NetPert")
ax[0, 0].errorbar(lv, [sw["robustness"][k]["bc_mean"] for k in kk],
                  yerr=[sw["robustness"][k]["bc_sd"] for k in kk], fmt="s-", color=BLUE, label="BC")
ax[0, 0].legend(); ax[0, 0].set_title("Spearman vs unperturbed"); ax[0, 0].set_xlabel("edges removed")
ax[0, 1].bar(lv, [sw["robustness"][k]["top20_overlap"] for k in kk], color=AC)
ax[0, 1].set_title("NetPert top-20 overlap"); ax[0, 1].set_ylabel("genes /20")
ax[1, 0].plot(list(swp.keys()), [swp[k]["rho"] for k in swp], "o-", color=PUR)
ax[1, 0].set_title("Relaxation: rho vs r=0.5"); ax[1, 0].set_xlabel("r")
ax[1, 1].plot(list(swp.keys()), [swp[k]["top20"] for k in swp], "s-", color=GREEN)
ax[1, 1].set_title("Relaxation: top-20 overlap"); ax[1, 1].set_xlabel("r"); ax[1, 1].set_ylabel("genes /20")
save(fig, "dashboard_robustness.png", "Robustness dashboard",
     "Edge-deletion stability plus relaxation sensitivity, with repeat intervals.")

# 3. assay dashboard (2x3)
fig, ax = plt.subplots(2, 3, figsize=(12, 7))
fig.suptitle("Assay dashboard — dissemination readouts", fontsize=12)
o = mm.sort_values("dissem")
ax[0, 0].barh(o["gene"].values, o["dissem"].values, color=plt.cm.RdYlGn_r(o["dissem"].values / 100))
ax[0, 0].set_title("Dissemination % at 1uM (lower = stronger)")
v = mm["ic50"].dropna().values; v = v[v > 0]
ax[0, 1].hist(np.log10(v), bins=10, color=AC)
ax[0, 1].set_title("IC50 distribution (log10 nM)")
ax[0, 2].scatter(np.log10(mm["NetPert_weight"].values), mm["dissem"].values, color=AC, s=20)
ax[0, 2].set_title("NetPert vs dissemination"); ax[0, 2].set_xlabel("log10 weight")
ax[1, 0].scatter(mm["log2fc"].values, mm["dissem"].values, color=GREEN, s=20)
ax[1, 0].set_title("log2FC vs dissemination"); ax[1, 0].set_xlabel("log2FC")
mk = mm["ic50"].notna() & (mm["ic50"] > 0)
ax[1, 1].scatter(np.log10(mm.loc[mk, "ic50"].values), mm.loc[mk, "dissem"].values, color=PUR, s=20)
ax[1, 1].set_title("IC50 vs dissemination"); ax[1, 1].set_xlabel("log10 IC50")
ax[1, 2].scatter(np.log10(mm["BCfull_weight"].values + 1e-12), mm["dissem"].values, color=BLUE, s=20)
ax[1, 2].set_title("Betweenness vs dissemination"); ax[1, 2].set_xlabel("log10 BC weight")
save(fig, "dashboard_assay.png", "Assay dashboard",
     "Every wet-lab readout crossed with every ranking.")

# 4. network dashboard (2x3)
fig, ax = plt.subplots(2, 3, figsize=(12, 7))
fig.suptitle("Network dashboard — IID+TRRUST structure", fontsize=12)
d = r["outdegree_full"].values; d = d[d > 0]
ax[0, 0].hist(np.log10(d), bins=40, color=BLUE)
ax[0, 0].set_title("Out-degree distribution (log10)")
ax[0, 1].bar(["IID directed", "TRRUST new"], [1886194, 5551], color=[BLUE, AC])
ax[0, 1].set_title("Edge provenance")
ct = r["type"].value_counts()
ax[0, 2].bar(["driver", "intermediates", "responses"],
             [(r["type"] == "D").sum(), (r["type"] == "I").sum(), (r["type"] == "R").sum()],
             color=[AC, BLUE, GREEN])
ax[0, 2].set_title("Subnetwork composition")
ax[1, 0].bar(["DIR", "DI", "IR", "I"], [154, 51, 838, 156], color=[AC, BLUE, GREEN, GREY])
ax[1, 0].set_title("Path-type counts")
ax[1, 1].scatter(np.log10(r["outdegree_full"].values + 1), np.log10(w + 1e-12), s=6, alpha=0.4, color=AC)
ax[1, 1].set_title("Hub-bias scatter"); ax[1, 1].set_xlabel("log10 degree"); ax[1, 1].set_ylabel("log10 weight")
ax[1, 2].bar(["in network", "absent"], [168, 14], color=[GREEN, RED])
ax[1, 2].set_title("Response coverage")
save(fig, "dashboard_network.png", "Network dashboard",
     "Degrees, provenance, composition, path types, hub bias and coverage together.")

# 5. methods dashboard (2x2)
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("Methods dashboard — head-to-head on ", fontsize=12)
ax[0, 0].scatter(np.log10(w + 1e-12), np.log10(bc + 1e-12), s=6, alpha=0.4, color=BLUE)
ax[0, 0].set_title("NetPert vs betweenness"); ax[0, 0].set_xlabel("log10 NP"); ax[0, 0].set_ylabel("log10 BC")
t30 = npr.head(30)
mat = np.column_stack([t30["NetPert_rank"].values, t30["BCsub_rank"].values,
                       t30["BCfull_weight"].rank(ascending=False).values,
                       t30["outdegree_full"].rank(ascending=False).values])
im = ax[0, 1].imshow(np.log10(mat), aspect="auto", cmap="YlOrRd")
ax[0, 1].set_xticks(range(4)); ax[0, 1].set_xticklabels(["NP", "BCs", "BCf", "deg"], fontsize=7)
ax[0, 1].set_title("Top-30 rank heatmap")
ax[1, 0].bar(["NP–BCs", "NP–BCf", "NP–deg"],
             [spearmanr(r["NetPert_weight"], r["BCsub_weight"])[0],
              spearmanr(r["NetPert_weight"], r["BCfull_weight"])[0],
              spearmanr(r["NetPert_weight"], r["outdegree_full"])[0]], color=AC)
ax[1, 0].set_title("Pairwise Spearman rho")
ks = [10, 20, 30, 50, 100, 200]
npset = lambda k: set(npr.head(k)["gene"])
bcset = lambda k: set(r.sort_values("BCsub_rank").head(k)["gene"])
ax[1, 1].plot(ks, [len(npset(k) & bcset(k)) / k for k in ks], "o-", color=AC)
ax[1, 1].set_title("Top-k overlap NP ∩ BC"); ax[1, 1].set_xlabel("k"); ax[1, 1].set_ylabel("fraction")
save(fig, "dashboard_methods.png", "Methods dashboard",
     "Scatters, heatmap, correlations and overlaps between all rankings.")

# 6. drugs dashboard (2x2)
from collections import Counter
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("Drugs dashboard — Repurposing Hub mapping", fontsize=12)
ndr = r["drugs"].apply(lambda s_: 0 if not s_ else len(str(s_).split(",")))
ax[0, 0].hist(ndr.values, bins=range(0, 12), color=AC, rwidth=0.85)
ax[0, 0].set_title("Compounds per gene")
cc = Counter()
for s_ in r["drugs"].values:
    if s_:
        for x in str(s_).split(","):
            cc[x] += 1
topc = cc.most_common(12)
ax[0, 1].barh([c for c, _ in topc][::-1], [n for _, n in topc][::-1], color=GREEN)
ax[0, 1].set_title("Top compounds by coverage"); ax[0, 1].tick_params(labelsize=6)
ax[1, 0].plot([10, 20, 30, 50, 100, 200, 400, 800, 1200],
              [npr.head(k)["drugs"].ne("").mean() for k in [10, 20, 30, 50, 100, 200, 400, 800, 1200]],
              "o-", color=GREEN)
ax[1, 0].axhline((r["drugs"] != "").mean(), ls="--", color=GREY)
ax[1, 0].set_title("Druggable fraction along ranks"); ax[1, 0].set_xlabel("top-k")
dg = npr[npr["drugs"] != ""].head(12)
ax[1, 1].barh(dg["gene"].values[::-1], dg["NetPert_weight"].values[::-1], color=AC)
ax[1, 1].tick_params(labelsize=6); ax[1, 1].set_title("Top druggable intermediates")
save(fig, "dashboard_drugs.png", "Drugs dashboard",
     "Coverage, top compounds, enrichment along ranks and top druggable genes.")

# 7. paths dashboard (2x2)
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("Paths dashboard — DIR structure on edges", fontsize=12)
ax[0, 0].bar(["DIR", "DI", "IR", "I"], [154, 51, 838, 156], color=[AC, BLUE, GREEN, GREY])
ax[0, 0].set_title("Path-type counts")
ax[0, 1].bar(["DIR", "DI", "IR", "I"], [3.800987e-03, 2.393529e-04, 1.248262e-04, 2.6542565e-06],
            color=[AC, BLUE, GREEN, GREY])
ax[0, 1].set_yscale("log"); ax[0, 1].set_title("Median weight by path type")
ax[1, 0].plot([10, 20, 50, 100, 200], [10, 20, 50, 100, 149], "o-", color=AC)
ax[1, 0].set_title("DIR count in top-k"); ax[1, 0].set_xlabel("k"); ax[1, 0].set_ylabel("DIR genes")
ax[1, 1].bar(["drop TRRUST"], [0.945], color=PUR)
ax[1, 1].set_ylim(0.8, 1.0); ax[1, 1].set_title("Ablation rho (keep 531 edges?)")
save(fig, "dashboard_paths.png", "Paths dashboard",
     "DIR counts, weights, enrichment and the TRRUST ablation in one view.")

man = json.load(open(ROOT + "/results/figures/gallery.json"))
man.extend(MAN)
json.dump(man, open(ROOT + "/results/figures/gallery.json", "w"), indent=1)
from collections import Counter as C
print("TOTAL:", len(man), dict(C(m["file"].split("/")[0] for m in man)))
