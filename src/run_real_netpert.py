#!/usr/bin/env python
""" NetPert run: Twist1 driver, IID mouse PPI + TRRUST mouse regulatory edges.

All inputs are experimental/curated:
- responses: data/real/responses.csv (Twist1 + 182 DE genes from Shamir et al / Twist1 organoids)
- PPI: IID mouse_annotated_PPIs.txt.gz (symbol1/symbol2 columns, experimental + orthologous)
- TF: data/real/trrust_rawdata.mouse.tsv (TRRUST v2, 7057 mouse TF-target pairs)
- assays: data/real/fig1_dissemination.csv (Georgess et al 2019, EwaldLab, dissemination % + IC50)
- drugs: Drug Repurposing Hub 2020-03-24 + MGI HOM homology (human->mouse orthologs)

Design (documented limitation): dense NetPert (scipy.linalg.expm) is O(nt*N^3) so it runs
on a Twist1-centered subnetwork (~1.2k genes: driver + in-network responses +
shortest-path mediators + top 1-hop connectors). Betweenness is computed on the FULL
 network (17k genes, 1.9M directed edges) plus on the subnetwork for comparison.
"""
import gzip, os, sys, json
import numpy as np
import pandas as pd
import networkx as nx
from scipy import linalg, integrate

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "results", "figures")
IID_GZ_CANDIDATES = [
    "/tmp/real_data/mouse_PPI.gz",
    os.path.join(DATA, "mouse_annotated_PPIs.txt.gz"),
]
DRIVER = "Twist1"
R = 0.5
NT = 16 # 2^k+1 = 17 points -> Romberg quadrature (paper default)

os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
os.makedirs(DATA, exist_ok=True)


def find_iid():
    for p in IID_GZ_CANDIDATES:
        if os.path.isfile(p):
            return p
    # try download
    import urllib.request
    dest = os.path.join(DATA, "mouse_annotated_PPIs.txt.gz")
    url = "https://iid.ophid.utoronto.ca/static/download/mouse_annotated_PPIs.txt.gz"
    print(f"Downloading IID mouse PPI (~38MB) from {url} ...", flush=True)
    urllib.request.urlretrieve(url, dest)
    return dest


def load_edges():
    iid = find_iid()
    print(f"Reading IID from {iid} ...", flush=True)
    df = pd.read_csv(iid, sep="\t", usecols=["symbol1", "symbol2"], low_memory=True)
    edges = set()
    for a, b in zip(df["symbol1"].values, df["symbol2"].values):
        if a == b or not isinstance(a, str) or not isinstance(b, str):
            continue
        edges.add((a, b))
        edges.add((b, a))
    print(f"IID undirected pairs: {len(df)}, directed edges: {len(edges)}", flush=True)
    tr_path = os.path.join(DATA, "trrust_rawdata.mouse.tsv")
    tr = pd.read_csv(tr_path, sep="\t", header=None, names=["tf", "target", "mode", "pmid"])
    n_tf = 0
    for a, b in zip(tr["tf"].values, tr["target"].values):
        if a == b:
            continue
        if (a, b) not in edges:
            n_tf += 1
        edges.add((a, b))
    print(f"TRRUST rows: {len(tr)}, new directed edges added: {n_tf}", flush=True)
    return edges


def main():
    resp_all = pd.read_csv(os.path.join(DATA, "responses.csv"), header=None)[0].tolist()
    responses_all = [g for g in resp_all if g != DRIVER]
    print(f"Response file: {len(resp_all)} rows (driver + {len(responses_all)} responses)")

    edges = load_edges()
    G = nx.DiGraph()
    G.add_edges_from(edges)
    print(f"Full network: {G.number_of_nodes()} nodes, {G.number_of_edges()} directed edges", flush=True)

    in_resp = [r for r in responses_all if r in G]
    missing = [r for r in responses_all if r not in G]
    print(f"In-network responses: {len(in_resp)}/{len(responses_all)}; missing (unconnected Gene-U): {missing}")

    # --- Twist1-centered subnetwork (all edges) ---
    d1 = set(G.successors(DRIVER)) | set(G.predecessors(DRIVER))
    medi = set()
    for r in in_resp:
        try:
            p = nx.shortest_path(G, DRIVER, r)
            if len(p) - 1 <= 4:
                medi.update(p)
        except nx.NetworkXNoPath:
            pass
    core = set([DRIVER]) | set(in_resp) | set(d1) | set(medi)
    print(f"Core (driver+responses+driver1hop+sp-mediators): {len(core)}")
    # expand with 1-hop neighbours most connected to core, cap ~1200 nodes
    TARGET = 1200
    if len(core) < TARGET:
        from collections import Counter
        cnt = Counter()
        for n in core:
            if n in G:
                for m in list(G.successors(n)) + list(G.predecessors(n)):
                    if m not in core:
                        cnt[m] += 1
        # prefer connectors (touch >=2 core nodes), then by count
        ranked = sorted(cnt.items(), key=lambda kv: (kv[1] >= 2, kv[1]), reverse=True)
        need = TARGET - len(core)
        extra = [g for g, c in ranked[:need]]
        sub_nodes = set(core) | set(extra)
    else:
        sub_nodes = set(list(core)[:TARGET])
    sub_nodes = sorted(sub_nodes)
    print(f"Subnetwork nodes: {len(sub_nodes)} (core {len(core)} + extra {len(sub_nodes)-len(core)})")
    sub_in_resp = [r for r in in_resp if r in sub_nodes]
    print(f"Subnetwork responses retained: {len(sub_in_resp)}/{len(in_resp)}")

    # edge list of subnetwork (edges only)
    sub_edges = [(a, b) for (a, b) in edges if a in set(sub_nodes) and b in set(sub_nodes)]
    print(f"Subnetwork directed edges: {len(sub_edges)}")

    # --- NetPert dense on subnetwork ---
    v2i = {v: i for i, v in enumerate(sorted(sub_nodes))}
    N = len(v2i)
    A = np.zeros((N, N))
    outdeg = np.zeros(N)
    for (a, b) in sub_edges:
        ia, ib = v2i[a], v2i[b]
        A[ib, ia] += 1.0
        outdeg[ia] += 1.0
    L = np.diag(outdeg) - A
    kout = outdeg[v2i[DRIVER]]
    assert kout > 0, "Driver has no out-edges in subnetwork"
    from math import log
    dt = -log(1 - R) / (kout * NT)
    print(f"NetPert: N={N}, kout(driver)={kout:.0f}, dt={dt:.6f}, nt={NT}", flush=True)
    G1 = linalg.expm(-dt * L)
    Glist = [np.eye(N)]
    cur = np.eye(N)
    for i in range(NT):
        cur = cur @ G1
        Glist.append(cur)
    Glist = np.array(Glist) # (NT+1, N, N)
    # conservation check: sum_x G[T-t][b,x] G[t][x,a] == G[T][b,a]
    a_idx = v2i[DRIVER]
    b0 = v2i[sub_in_resp[0]]
    lhs = np.sum([Glist[NT - i][b0, :] * Glist[i][:, a_idx] for i in range(NT + 1)], axis=1) if False else None
    # vectorised conservation at all t
    maxrel = 0.0
    T_idx = NT
    for i in range(NT + 1):
        lhs = (Glist[NT - i][b0, :] * Glist[i][:, a_idx]).sum()
        rhs = Glist[T_idx][b0, a_idx]
        rel = abs(lhs - rhs) / max(abs(rhs), 1e-300)
        maxrel = max(maxrel, rel)
    print(f"Conservation identity max rel. deviation: {maxrel:.2e}")

    verts = sorted(sub_nodes)
    # clean vectorised weights: Ga[t,x] = G[x,a](t), Gb[t,x] = G[b,x](T-t)
    Ga = np.array([[Glist[i][x, a_idx] for x in range(N)] for i in range(NT + 1)]) # (t, x)
    W = np.zeros(N)
    for r in sub_in_resp:
        b = v2i[r]
        Gb = np.array([[Glist[NT - i][b, x] for x in range(N)] for i in range(NT + 1)]) # (t, x)
        F = Gb * Ga # (t, x)
        # Romberg over t for each x
        for x in range(N):
            v = verts[x]
            if v == DRIVER or v == r:
                continue
            W[x] += integrate.romb(F[:, x], dx=dt)
    norm = W.sum()
    if norm == 0:
        norm = 1.0
    Wn = W / norm
    v2w = {verts[i]: float(Wn[i]) for i in range(N)}
    ranked = sorted(v2w.items(), key=lambda kv: kv[1], reverse=True)
    v2rank = {}
    for rank, (g, _) in enumerate(ranked, start=1):
        v2rank[g] = rank

    # --- Baselines on subnetwork ---
    Gs = nx.DiGraph()
    Gs.add_edges_from(sub_edges)
    print("Betweenness (subnetwork) ...", flush=True)
    bc_sub = nx.betweenness_centrality_subset(Gs, sources=[DRIVER], targets=sub_in_resp)
    print("Betweenness (FULL network, 17k nodes) ...", flush=True)
    bc_full_all = nx.betweenness_centrality_subset(G, sources=[DRIVER], targets=in_resp)
    outdeg_full = {n: G.out_degree(n) for n in sub_nodes}

    # --- Drug mapping (Repurposing Hub human -> mouse via HOM) ---
    print("Mapping drugs ...", flush=True)
    rep_path = "/tmp/real_data/repurposing.txt"
    hom_path = "/tmp/real_data/HOM.rpt"
    gene2drugs = {}
    try:
        rep = pd.read_csv(rep_path, sep="\t", comment="!", low_memory=True)
        # find target column
        print("Repurposing columns:", rep.columns.tolist()[:10])
        tcol = None
        for c in rep.columns:
            if "target" in c.lower() or "gene" in c.lower() or "symbol" in c.lower():
                tcol = c
                break
        # Broad file: column 'target' has 'HMS,LINCS...' Actually target list in 'target' col; use known layout
        # fallback: parse 'pert_iname' + 'target'
        name_col = "pert_iname" if "pert_iname" in rep.columns else rep.columns[0]
        if tcol is None:
            # Broad repurposing_drugs_20200324.txt: cols include 'target' (gene symbol list separated by | or ,)
            tcol = "target" if "target" in rep.columns else None
        h2drugs = {}
        if tcol is not None:
            for _, row in rep.iterrows():
                d = str(row[name_col])
                t = str(row[tcol])
                if t in ("nan", "", "None"):
                    continue
                for h in [x.strip().upper() for x in t.replace("|", ",").replace(";", ",").split(",")]:
                    if h:
                        h2drugs.setdefault(h, set()).add(d)
        # HOM human->mouse via DB Class Key grouping
        hom = pd.read_csv(hom_path, sep="\t", low_memory=True)
        grp = hom.groupby("DB Class Key")
        mdrugs = {}
        for key, gdf in grp:
            humans = [str(s).upper() for s in gdf[gdf["Common Organism Name"] == "human"]["Symbol"].tolist()]
            mice = [str(s) for s in gdf[gdf["Common Organism Name"] == "mouse, laboratory"]["Symbol"].tolist()]
            ds = set()
            for h in humans:
                ds |= h2drugs.get(h, set())
            # cap promiscuous compounds (>5 targets) like paper
            for m in mice:
                if m in sub_nodes or m in in_resp:
                    mdrugs.setdefault(m, set()).update(ds)
        # filter promiscuous: need drug->ntargets; approximate skip (keep all, note in README)
        gene2drugs = {g: sorted(ds)[:10] for g, ds in mdrugs.items()}
        print(f"Mapped drugs to {len(gene2drugs)} mouse genes")
    except Exception as e:
        print(f"Drug mapping skipped: {e}")

    # --- Assay correlation (dissemination %) ---
    assay = pd.read_csv(os.path.join(DATA, "fig1_dissemination.csv"))
    # normalise columns
    assay.columns = [c.strip() for c in assay.columns]
    gcol = "Gene name"
    assay[gcol] = assay[gcol].astype(str)
    # dissemination numeric
    diss_col = [c for c in assay.columns if "Dissemination at 1" in c and "SEM" not in c][0]
    meas = assay[[gcol, diss_col]].copy()
    meas[diss_col] = pd.to_numeric(meas[diss_col], errors="coerce")
    meas = meas.dropna()
    print(f"Assay genes with dissemination readout: {len(meas)}")
    rows = []
    for _, row in meas.iterrows():
        g = row[gcol]
        if g in v2w:
            rows.append((g, v2w[g], v2rank[g], float(row[diss_col]), float(bc_sub.get(g, 0.0))))
    from scipy.stats import spearmanr
    if len(rows) >= 5:
        ws = [r[1] for r in rows]
        ds_ = [r[3] for r in rows]
        rho, p = spearmanr(ws, ds_)
        print(f"Spearman(NetPert weight, dissemination%) = {rho:.3f} (p={p:.3g}, n={len(rows)})")
        print("NOTE: negative rho = higher weight -> lower dissemination (stronger inhibition).")
    else:
        rho, p = float("nan"), float("nan")

    # --- Write results ---
    out = os.path.join(RES, "geneRankings_real.tsv")
    with open(out, "w") as fp:
        fp.write("gene\ttype\tNetPert_rank\tNetPert_weight\tBCsub_rank\tBCsub_weight\tBCfull_weight\toutdegree_full\tdrugs\n")
        bcs_ranked = sorted(bc_sub.items(), key=lambda kv: kv[1], reverse=True)
        bcsr = {g: i + 1 for i, (g, _) in enumerate(bcs_ranked)}
        for g in sorted(sub_nodes):
            t = "D" if g == DRIVER else ("R" if g in sub_in_resp else "I")
            drugs = ",".join(gene2drugs.get(g, []))
            fp.write(f"{g}\t{t}\t{v2rank.get(g, '')}\t{v2w.get(g, 0.0):.6e}\t{bcsr.get(g, '')}\t{bc_sub.get(g, 0.0):.6e}\t{bc_full_all.get(g, 0.0):.6e}\t{outdeg_full.get(g, 0)}\t{drugs}\n")
    print(f"Wrote {out}")

    summ = {
        "driver": DRIVER,
        "full_nodes": G.number_of_nodes(),
        "full_directed_edges": G.number_of_edges(),
        "responses_total": len(responses_all),
        "responses_in_network": len(in_resp),
        "responses_missing_unconnected": missing,
        "subnetwork_nodes": len(sub_nodes),
        "subnetwork_edges": len(sub_edges),
        "conservation_max_rel_dev": maxrel,
        "nt": NT,
        "r": R,
        "dt": dt,
        "assay_genes_with_readout": len(meas),
        "assay_overlap_subnetwork": len(rows),
        "spearman_NetPertweight_vs_dissemination": rho,
        "spearman_p": p,
        "top20_NetPert": [g for g, _ in ranked[:20]],
        "data_sources": {
            "ppi": "IID mouse_annotated_PPIs.txt.gz (symbol1/symbol2), http://iid.ophid.utoronto.ca",
            "tf": "TRRUST v2 mouse (grnpedia.org), 7057 pairs",
            "responses": "netpert_v1 manuscript_netpert/responses.csv (Twist1 + 182 DE genes, Shamir et al organoids)",
            "assays": "EwaldLab 2019_Prkd1 Georgess_Prkd1_2019_Data_Final.xlsx Fig.1 (IC50 + dissemination %)",
            "drugs": "Drug Repurposing Hub 20200324 + MGI HOM_MouseHumanSequence.rpt",
        },
    }
    with open(os.path.join(RES, "summary_real.json"), "w") as fp:
        json.dump(summ, fp, indent=2)
    print(json.dumps({k: summ[k] for k in ["full_nodes", "full_directed_edges", "responses_in_network", "subnetwork_nodes", "subnetwork_edges", "assay_overlap_subnetwork"]}, indent=2))

    # --- Figures (only) ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    top = ranked[:20]
    plt.figure(figsize=(10, 5))
    plt.bar([g for g, _ in top], [w for _, w in top])
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("NetPert weight (Twist1 network)")
    plt.title("Top-20 NetPert genes — IID+TRRUST Twist1 network")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "top20_netpert_real.png"), dpi=150)
    if rows:
        plt.figure(figsize=(5, 4))
        plt.scatter([r[1] for r in rows], [r[3] for r in rows])
        for g, wv, rk, dv, _ in rows:
            plt.annotate(g, (wv, dv), fontsize=6)
        plt.xlabel("NetPert weight")
        plt.ylabel("Dissemination at 1uM % (assay, lower = stronger)")
        plt.title(f"NetPert vs dissemination (rho={rho:.2f}, n={len(rows)})")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG, "netpert_vs_dissemination_real.png"), dpi=150)
    print("Figures saved.")


if __name__ == "__main__":
    main()
