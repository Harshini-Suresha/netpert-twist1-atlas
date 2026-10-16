#!/usr/bin/env python
"""Report-style figures (mirrors the classic reproduction plots).

Computes on the Twist1 IID+TRRUST subnetwork (1,200 genes):
  concentration curves, method-agreement panels, edge-removal robustness
  (NetPert + betweenness), relaxation-parameter sweep, hub-bias panels,
  weight-vs-distance, assay waterfall, rank heatmap, path-type analyses,
  scorecard table.
Saves 12 PNGs to results/figures/08_report/ and appends to gallery.json.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import linalg, integrate
from scipy.stats import spearmanr
from math import log
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
FIGD = os.path.join(ROOT, "results", "figures", "08_report")
DATA = os.path.join(ROOT, "data", "")
os.makedirs(FIGD, exist_ok=True)

AC, BLUE, GREEN, RED, GREY = "#c2410c", "#1d4ed8", "#15803d", "#b91c1c", "#78716c"
MANIFEST = []

def save(fig, name, title, caption):
    fig.tight_layout()
    fig.savefig(os.path.join(FIGD, name), dpi=100)
    plt.close(fig)
    MANIFEST.append({"file": f"08_report/{name}", "title": title, "caption": caption})

r = pd.read_csv(os.path.join(RES, "geneRankings_real.tsv"), sep="\t")
genes = r["gene"].tolist()
v2i = {g: i for i, g in enumerate(sorted(genes))}
N = len(genes)
DRIVER = "Twist1"
w0 = r.set_index("gene")["NetPert_weight"]
bc0 = r.set_index("gene")["BCfull_weight"]
deg0 = r.set_index("gene")["outdegree_full"]
resp = set(r[r["type"] == "R"]["gene"])
print(f"loaded {N} genes", flush=True)

# ---- rebuild subnetwork edges ----
import gzip
iid = "/tmp/real_data/mouse_PPI.gz"
if not os.path.isfile(iid):
    iid = os.path.join(DATA, "mouse_annotated_PPIs.txt.gz")
df = pd.read_csv(iid, sep="\t", usecols=["symbol1", "symbol2"], low_memory=True)
gset = set(genes)
edges = set()
for a, b in zip(df["symbol1"].values, df["symbol2"].values):
    if a == b or not isinstance(a, str) or not isinstance(b, str):
        continue
    if a in gset and b in gset:
        edges.add((a, b)); edges.add((b, a))
tr = pd.read_csv(os.path.join(DATA, "trrust_rawdata.mouse.tsv"), sep="\t",
                 header=None, names=["tf", "target", "mode", "pmid"])
for a, b in zip(tr["tf"].values, tr["target"].values):
    if a != b and a in gset and b in gset:
        edges.add((a, b))
print(f"subnetwork directed edges: {len(edges)}", flush=True)
succ = {}
for a, b in edges:
    succ.setdefault(a, set()).add(b)
pred = {}
for a, b in edges:
    pred.setdefault(b, set()).add(a)
D1 = set(succ.get(DRIVER, set())) | set(pred.get(DRIVER, set()))
R1 = set()
for q in resp:
    R1 |= set(succ.get(q, set())) | set(pred.get(q, set()))

def netpert_weights(edge_set, r_param=0.5, nt=16):
    idx = {g: i for i, g in enumerate(sorted(genes))}
    n = len(idx)
    A = np.zeros((n, n)); out = np.zeros(n)
    for a, b in edge_set:
        A[idx[b], idx[a]] += 1.0
        out[idx[a]] += 1.0
    L = np.diag(out) - A
    k = out[idx[DRIVER]]
    dt = -log(1 - r_param) / (k * nt)
    G1 = linalg.expm(-dt * L)
    G = [np.eye(n)]; cur = np.eye(n)
    for _ in range(nt):
        cur = cur @ G1
        G.append(cur)
    G = np.array(G)
    ai = idx[DRIVER]
    inresp = [q for q in resp if q in idx]
    Ga = np.array([[G[i][x, ai] for x in range(n)] for i in range(nt + 1)])
    W = np.zeros(n)
    for q in inresp:
        b = idx[q]
        Gb = np.array([[G[nt - i][b, x] for x in range(n)] for i in range(nt + 1)])
        F = Gb * Ga
        for x in range(n):
            if sorted(genes)[x] == DRIVER or sorted(genes)[x] == q:
                continue
            W[x] += integrate.romb(F[:, x], dx=dt)
    W = W / (W.sum() or 1.0)
    return {sorted(genes)[i]: float(W[i]) for i in range(n)}

def bc_weights(edge_set):
    import networkx as nx
    Gx = nx.DiGraph(); Gx.add_edges_from(edge_set)
    inresp = [q for q in resp if q in Gx]
    return nx.betweenness_centrality_subset(Gx, sources=[DRIVER], targets=inresp)

def ranks(d):
    return pd.Series(d).rank(ascending=False, method="min").to_dict()

# ============ 1. concentration curve ============
fig, ax = plt.subplots(figsize=(6.5, 4.5))
xs = np.arange(1, N + 1)
for vals, lab, col in [(w0.values, "NetPert", AC), (bc0.values / (bc0.sum() or 1), "Betweenness", BLUE)]:
    ax.plot(xs, np.cumsum(np.sort(vals)[::-1]), label=lab, color=col, lw=2)
ax.plot(xs, xs / N, "--", color=GREY, label="uniform")
ax.set_xlabel("top-k genes"); ax.set_ylabel("share of total weight")
ax.set_title("Score concentration (Twist1 subnetwork)")
ax.legend()
t10 = float(np.sort(w0.values)[::-1][:10].sum())
ax.annotate(f"NetPert top-10 hold {t10:.0%}", xy=(10, t10), xytext=(200, t10 + 0.05),
            arrowprops=dict(arrowstyle="->", color=AC), fontsize=9)
save(fig, "concentration_curve.png", "Score concentration",
     f"NetPert top-10 hold {t10:.0%} of total weight.")

# ============ 2. method agreement panels ============
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
srt = np.sort(w0.values)[::-1]
axes[0].bar(["NetPert", "Betweenness", "Uniform"], [srt[:10].sum(), np.sort(bc0.values)[::-1][:10].sum() / (bc0.sum() or 1), 10 / N],
            color=[AC, BLUE, GREY])
axes[0].set_ylabel("top-10 weight share"); axes[0].set_title("Concentration by method")
pairs = [("NP–BCsub", spearmanr(r["NetPert_weight"], r["BCsub_weight"])[0]),
         ("NP–BCfull", spearmanr(r["NetPert_weight"], r["BCfull_weight"])[0]),
         ("NP–degree", spearmanr(r["NetPert_weight"], r["outdegree_full"])[0])]
axes[1].barh([p[0] for p in pairs], [p[1] for p in pairs], color=AC)
axes[1].set_xlabel("Spearman rho"); axes[1].set_title("Rank agreement")
save(fig, "method_agreement_panels.png", "Method agreement panels",
     "Concentration and rank agreement between methods on .")

# ============ 3. edge-removal robustness ============
rng = np.random.default_rng(7)
E = list(edges)
levels = [0.10, 0.25, 0.50]
np_rho, bc_rho = [], []
for lv in levels:
    npr, bcr = [], []
    for rep in range(3):
        drop = set(rng.choice(len(E), size=int(lv * len(E)), replace=False))
        Es = [e for i, e in enumerate(E) if i not in drop]
        wn = netpert_weights(Es)
        wb = bc_weights(Es)
        npr.append(spearmanr([w0[g] for g in genes], [wn[g] for g in genes])[0])
        bcr.append(spearmanr([bc0[g] for g in genes], [wb.get(g, 0.0) for g in genes])[0])
    np_rho.append(npr); bc_rho.append(bcr)
    print(f"removal {lv}: NP={np.mean(npr):.2f} BC={np.mean(bcr):.2f}", flush=True)
fig, ax = plt.subplots(figsize=(6.5, 4.5))
x = np.array([0.0] + levels)
ax.errorbar(x, [1.0] + [np.mean(v) for v in np_rho],
            yerr=[0.0] + [np.std(v) for v in np_rho], fmt="o-", color=AC, label="NetPert")
ax.errorbar(x, [1.0] + [np.mean(v) for v in bc_rho],
            yerr=[0.0] + [np.std(v) for v in bc_rho], fmt="s-", color=BLUE, label="Betweenness")
ax.set_xlabel("fraction of edges removed"); ax.set_ylabel("Spearman vs unperturbed")
ax.set_title("Robustness to missing edges (network, 3 repeats)"); ax.legend()
save(fig, "robustness_edge_removal.png", "Edge-removal robustness",
     "Ranking stability when interactome edges are deleted.")

# ============ 4. relaxation sweep ============
sweep = {}
for rv in [0.10, 0.25, 0.50, 0.75, 0.90, 0.99]:
    sweep[rv] = netpert_weights(edges, r_param=rv)
    print(f"r={rv} done", flush=True)
base = sweep[0.50]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
rhos = [spearmanr([base[g] for g in genes], [sweep[rv][g] for g in genes])[0] for rv in sweep]
axes[0].plot(list(sweep.keys()), rhos, "o-", color=AC)
axes[0].set_xlabel("relaxation r"); axes[0].set_ylabel("Spearman vs r=0.5")
axes[0].set_title("Ranking stability across r")
top20_50 = set(sorted(base, key=base.get, reverse=True)[:20])
ov = [len(top20_50 & set(sorted(sweep[rv], key=sweep[rv].get, reverse=True)[:20])) for rv in sweep]
axes[1].plot(list(sweep.keys()), ov, "s-", color=BLUE)
axes[1].set_xlabel("relaxation r"); axes[1].set_ylabel("top-20 overlap with r=0.5")
axes[1].set_title("Top-20 stability across r")
save(fig, "relaxation_sweep.png", "Relaxation sweep",
     " ranking is insensitive to the relaxation fraction.")

# ============ 5. hub-bias panels ============
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
axes[0].scatter(np.log10(deg0.values + 1), np.log10(w0.values + 1e-12), s=8, alpha=0.4, color=AC)
axes[0].set_xlabel("log10 out-degree"); axes[0].set_ylabel("log10 NetPert")
axes[0].set_title("Hub bias")
axes[1].scatter(np.log10(bc0.values + 1e-12), np.log10(w0.values + 1e-12), s=8, alpha=0.4, color=BLUE)
axes[1].set_xlabel("log10 BC-full"); axes[1].set_ylabel("log10 NetPert")
axes[1].set_title("NetPert vs betweenness")
save(fig, "hub_bias_panels.png", "Hub-bias panels",
     "Connectivity and betweenness context for NetPert weights.")

# ============ 6. weight vs distance from driver ============
import networkx as nx
Gx = nx.DiGraph(); Gx.add_edges_from(edges)
plen = nx.single_source_shortest_path_length(Gx, DRIVER)
dd = [plen.get(g, 99) for g in genes]
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.boxplot([[w0[g] for g, d in zip(genes, dd) if d == k] for k in sorted(set(dd)) if k < 99],
           labels=[k for k in sorted(set(dd)) if k < 99])
ax.set_yscale("log"); ax.set_xlabel("shortest steps from Twist1")
ax.set_ylabel("NetPert weight (log)"); ax.set_title("Weight by driver distance")
save(fig, "weight_vs_distance.png", "Weight vs driver distance",
     "How influence decays with network distance from Twist1.")

# ============ 7. assay waterfall ============
a = pd.read_csv(os.path.join(DATA, "fig1_dissemination.csv"))
a.columns = [c.strip() for c in a.columns]
dcol = [c for c in a.columns if "Dissemination at 1" in c and "SEM" not in c][0]
m = r.merge(a, left_on="gene", right_on="Gene name", how="inner")
m[dcol] = pd.to_numeric(m[dcol], errors="coerce")
mm = m.dropna(subset=[dcol]).sort_values(dcol)
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.bar(mm["gene"].values, mm[dcol].values, color=plt.cm.RdYlGn_r(mm[dcol].values / 100))
ax.set_xticks(range(len(mm))); ax.set_xticklabels(mm["gene"].values, rotation=65, ha="right", fontsize=8)
ax.set_ylabel("dissemination at 1uM %")
ax.set_title("Assay waterfall: strongest → weakest inhibitors")
for i, (_, row) in enumerate(mm.iterrows()):
    ax.text(i, row[dcol] + 1.5, f"NP#{int(row['NetPert_rank'])}", fontsize=7, ha="center")
save(fig, "assay_waterfall.png", "Assay waterfall",
     " inhibition ordered with NetPert ranks annotated.")

# ============ 8. rank heatmap top-30 ============
t30 = r.sort_values("NetPert_rank").head(30)
mat = np.column_stack([t30["NetPert_rank"].values, t30["BCsub_rank"].values,
                       t30["BCfull_weight"].rank(ascending=False).values,
                       t30["outdegree_full"].rank(ascending=False).values])
fig, ax = plt.subplots(figsize=(7, 9))
im = ax.imshow(np.log10(mat), aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(4)); ax.set_xticklabels(["NetPert", "BC-sub", "BC-full", "degree"])
ax.set_yticks(range(30)); ax.set_yticklabels(t30["gene"].values, fontsize=8)
ax.set_title("Top-30 ranks across methods (log scale,)")
plt.colorbar(im, ax=ax, label="log10 rank")
save(fig, "rank_heatmap_top30.png", "Top-30 rank heatmap",
     "Where the four rankings agree and disagree.")

# ============ 9-11. path types ============
def ptype(g):
    if g == DRIVER:
        return "D"
    d = g in D1; q = g in R1
    if d and q:
        return "DIR"
    if d:
        return "DI"
    if q:
        return "IR"
    return "I"
r = r.copy(); r["ptype"] = r["gene"].apply(ptype)
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ct = r["ptype"].value_counts()
ax.bar(ct.index, ct.values, color=[AC, BLUE, GREEN, GREY, "black"])
ax.set_ylabel("genes"); ax.set_title("Path-type counts in subnetwork")
save(fig, "pathtype_counts.png", "Path-type counts",
     "DIR = direct driver+response neighbours; DI/IR = one-sided; I = rest.")
fig, ax = plt.subplots(figsize=(6.5, 4.5))
data = [r[r["ptype"] == t]["NetPert_weight"].values for t in ["DIR", "DI", "IR", "I"]]
ax.boxplot(data, labels=["DIR", "DI", "IR", "I"]); ax.set_yscale("log")
ax.set_ylabel("NetPert weight (log)"); ax.set_title("Weight by path type")
save(fig, "weights_by_pathtype.png", "Weights by path type",
     "Direct driver–response connectors score highest on .")
fig, ax = plt.subplots(figsize=(7, 4.5))
ks = [10, 20, 30, 50, 100, 200, 400]
nps = r.sort_values("NetPert_rank")
ax.plot(ks, [((nps.head(k)["ptype"] == "DIR").sum() / k) for k in ks], "o-", color=AC, label="DIR in top-k")
ax.axhline((r["ptype"] == "DIR").mean(), ls="--", color=GREY, label="background")
ax.set_xlabel("top-k"); ax.set_ylabel("fraction DIR"); ax.legend()
ax.set_title("DIR enrichment along NetPert ranks")
save(fig, "dir_enrichment_topk.png", "DIR enrichment",
     "Direct connectors enriched at the top of the ranking.")

# ============ 12. scorecard ============
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.axis("off")
top10share = float(np.sort(w0.values)[::-1][:10].sum())
rho_nb, _ = spearmanr(r["NetPert_weight"], r["BCfull_weight"])
rho_nd, _ = spearmanr(r["NetPert_weight"], r["outdegree_full"])
rows = [["NetPert top-10 weight share", f"{top10share:.0%}"],
        ["NetPert–betweenness rho", f"{rho_nb:.2f}"],
        ["NetPert–degree rho", f"{rho_nd:.2f}"],
        ["Edge-removal rho @25%", f"{np.mean(np_rho[1]):.2f} (NP) / {np.mean(bc_rho[1]):.2f} (BC)"],
        ["Relaxation: min top-20 overlap", f"{min(ov)}/20"],
        ["DIR genes / enrichment@20", f"{(r['ptype'] == 'DIR').sum()} / {(nps.head(20)['ptype'] == 'DIR').sum()}/20"]]
tbl = ax.table(cellText=rows, colLabels=["scorecard", "value"], loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(10)
ax.set_title(" scorecard")
save(fig, "scorecard_table.png", "Scorecard",
     "Headline numbers in one table.")

man = json.load(open(os.path.join(ROOT, "results", "figures", "gallery.json")))
man.extend(MANIFEST)
json.dump(man, open(os.path.join(ROOT, "results", "figures", "gallery.json"), "w"), indent=1)
from collections import Counter
print("TOTAL:", len(man), dict(Counter(m["file"].split("/")[0] for m in man)), flush=True)
