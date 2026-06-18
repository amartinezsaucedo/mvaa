from __future__ import annotations

import glob
import pickle
from collections import Counter
from math import log
from typing import Any, Dict, List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from scipy.stats import spearmanr

from mvaa.alignment.alignment import MultiViewAlignments
from mvaa.alignment.infrastructure_filter import INFRA_FILTER
from mvaa.utils.decompositions import build_dict_from_directory
from mvaa.utils.graph import read_graphml

METRICS_DIR     = "mid_results/metrics"
MEMBERSHIPS_DIR = "mid_results/services"
STRUCTURAL_METRICS = [
    "CHM", "CHD", "IFN", "IRN", "OPN",
    "SMQ", "SCOH", "SCOP", "CMQ", "CCOH", "CCOP", "SERVICES",
]


def load_metrics(project: str, metrics_dir: str = METRICS_DIR) -> pd.DataFrame:
    dfs = []
    for csv_path in glob.glob(f"monoliths/{project}/{metrics_dir}/{project}_K*.csv"):
        k = int(csv_path.split("_K")[-1].replace(".csv", ""))
        df = pd.read_csv(
            csv_path, header=None,
            names=["ID", "RESOLUTION", "CHM", "CHD", "IFN", "IRN",
                   "OPN", "SMQ", "SCOH", "SCOP", "CMQ", "CCOH", "CCOP", "SERVICES"],
        )
        df["K"] = k
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No CSV found for '{project}' in {metrics_dir}")
    df = pd.concat(dfs).reset_index(drop=True)
    df["ID"] = df["ID"].str.strip()
    return df


def _normalize(d: Dict[str, float]) -> Dict[str, float]:
    s = sum(d.values())
    return {k: v / s for k, v in d.items()} if s > 0 else {k: 0.0 for k in d}


def normalized_entropy(dist: Dict[str, float], eps: float = 1e-12) -> float:
    k = len(dist)
    if k <= 1:
        return 0.0
    ent = -sum(p * log(max(p, eps)) for p in dist.values() if p > 0)
    return ent / log(k)


def induced_P_bc_given_cluster(
        impl_nodes: List[str],
        P_u_to_c: dict,
        G_design: nx.Graph,
        bc_list: List[str],
        *,
        missing_u: str = "skip",
        top_k: int = 1,
        alpha: float = 0.0,
) -> Dict[str, float]:
    scores = {bc: 0.0 for bc in bc_list}
    count  = 0
    for u in impl_nodes:
        dist_u = P_u_to_c.get(u)
        if dist_u is None:
            if missing_u == "uniform":
                for bc in bc_list:
                    scores[bc] += 1.0 / len(bc_list)
                count += 1
            continue
        pairs = sorted(dist_u, key=lambda x: x[1], reverse=True)[:top_k]
        Z     = sum(p for _, p in pairs)
        if Z <= 0:
            continue
        for c, p_cu in pairs:
            bp = G_design.nodes.get(c, {}).get("bc_probs", {})
            if not bp:
                continue
            w = p_cu / Z
            for bc in bc_list:
                scores[bc] += w * bp.get(bc, 0.0)
        count += 1
    if count == 0:
        return {bc: 1.0 / len(bc_list) for bc in bc_list}
    return _normalize({bc: scores[bc] / count for bc in bc_list})


def P_Cd_global_from_P_Cd_given_Ci(
        P_Cd_given_Ci: dict,
        membership: dict,
        impl_nodes: list,
        bc_list: list,
) -> Dict[str, float]:
    size: dict = {}
    total = 0
    for u in impl_nodes:
        if u in membership:
            ci = str(membership[u])
            size[ci] = size.get(ci, 0) + 1
            total    += 1
    if total == 0:
        return {bc: 0.0 for bc in bc_list}
    P = {bc: 0.0 for bc in bc_list}
    for ci, w in size.items():
        if ci not in P_Cd_given_Ci:
            continue
        p_ci = w / total
        for bc in bc_list:
            P[bc] += p_ci * P_Cd_given_Ci[ci].get(bc, 0.0)
    s = sum(P.values())
    return {bc: P[bc] / s for bc in bc_list} if s > 0 else P


def clusters_from_membership(membership: dict) -> Dict[str, List[str]]:
    clusters: dict = {}
    for u, cid in membership.items():
        clusters.setdefault(str(cid), []).append(str(u))
    return clusters


def entropy_metric_for_decomposition(
        membership: dict,
        G_design: nx.Graph,
        P_u_to_c: dict,
        bc_list: List[str],
        *,
        top_k: int = 1,
        missing_u: str = "skip",
        alpha: float = 0.0,
) -> Dict[str, Any]:
    impl_clusters = clusters_from_membership(membership)
    sizes         = {ci: len(nodes) for ci, nodes in impl_clusters.items()}
    total_size    = sum(sizes.values())

    P_Cd_given_Ci: dict = {}
    H_Ci:          dict = {}

    for ci, nodes in impl_clusters.items():
        dist = induced_P_bc_given_cluster(
            nodes, P_u_to_c, G_design, bc_list,
            missing_u=missing_u, top_k=top_k, alpha=alpha,
        )
        P_Cd_given_Ci[ci] = dist
        H_Ci[ci]          = normalized_entropy(dist)

    impl_nodes = list(membership.keys())

    results = {}
    for agg in ("size_weighted_mean", "mean"):
        if agg == "size_weighted_mean":
            penalty = (
                sum(H_Ci[ci] * sizes.get(ci, 0) for ci in H_Ci) / total_size
                if total_size else 0.0
            )
        else:
            penalty = sum(H_Ci.values()) / len(H_Ci) if H_Ci else 0.0
        results[f"penalty_{agg}"] = penalty

    P_Cd_global = P_Cd_global_from_P_Cd_given_Ci(
        P_Cd_given_Ci, membership, impl_nodes, bc_list
    )
    H_global = normalized_entropy(P_Cd_global)

    out = {
        "H_global":    H_global,
        "H_Ci":        H_Ci,
        "P_Cd_given_Ci": P_Cd_given_Ci,
        "n_clusters":  len(impl_clusters),
    }
    for key, penalty in results.items():
        out[key]                         = penalty
        out[key.replace("penalty", "score")] = 1.0 - penalty
        out[key.replace("penalty_", "penalty_rel_")] = (
            penalty / H_global if H_global > 1e-12 else 0.0
        )
    return out


def build_class_to_bc(G_design: nx.Graph, P_u_to_c: dict) -> Dict[str, str]:
    class_to_bc: dict = {}
    for u, dist_u in P_u_to_c.items():
        if not dist_u:
            continue
        best_c = max(dist_u, key=lambda x: x[1])[0]
        bp     = G_design.nodes.get(best_c, {}).get("bc_probs", {})
        if bp:
            class_to_bc[u] = max(bp, key=bp.get)
    return class_to_bc


def nmi_decomposition(
        membership: dict,
        class_to_bc: dict,
        *,
        exclude: set | None = None,
) -> float:
    exclude = exclude or set()
    pairs = [
        (str(cid), class_to_bc[u])
        for u, cid in membership.items()
        if u in class_to_bc and u not in exclude
    ]
    if not pairs:
        return 0.0
    n       = len(pairs)
    p_c     = Counter(p[0] for p in pairs)
    p_bc    = Counter(p[1] for p in pairs)
    p_joint = Counter(pairs)
    mi = sum(
        (cnt / n) * log((cnt / n) / ((p_c[c] / n) * (p_bc[bc] / n)) + 1e-12)
        for (c, bc), cnt in p_joint.items()
    )
    H_c  = -sum((v / n) * log(v / n + 1e-12) for v in p_c.values())
    H_bc = -sum((v / n) * log(v / n + 1e-12) for v in p_bc.values())
    return 2 * mi / (H_c + H_bc) if (H_c + H_bc) > 0 else 0.0

def spearman_analysis(
        metrics_df: pd.DataFrame,
        score_col: str = "NMI",
) -> pd.DataFrame:
    rows = []
    for metric in STRUCTURAL_METRICS:
        col = metrics_df[metric].astype(float)
        if col.nunique() <= 1:
            rows.append({"metric": metric, "rho": float("nan"), "p_value": float("nan")})
            continue
        rho, p = spearmanr(metrics_df[score_col], col)
        rows.append({"metric": metric, "rho": rho, "p_value": p})
    return pd.DataFrame(rows)

def analyze_project(
        project:   str,
        G_design:  nx.Graph,
        P_u_to_c:  dict,
        bc_list:   List[str],
        top_k:     int = 1,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  PROJECT: {project.upper()}")
    print(f"{'='*60}")

    metrics_df     = load_metrics(project)
    decompositions = build_dict_from_directory(f"monoliths/{project}/mid_results/services")
    class_to_bc    = build_class_to_bc(G_design, P_u_to_c)

    coverage = metrics_df["ID"].isin(decompositions).sum()
    print(f"CSV rows: {len(metrics_df)}  |  Unique memberships: {len(decompositions)}"
          f"  |  Successful match: {coverage}/{len(metrics_df)}")
    print(f"Classes with inferred BC: {len(class_to_bc)}")

    nmi_map:     dict = {}
    nmi_filtered_map:     dict = {}
    entropy_map: dict = {}
    infra_classes = INFRA_FILTER.get(project, set())

    for dec_id, payload in decompositions.items():
        mem = payload["membership"]
        nmi_map[dec_id]     = nmi_decomposition(mem, class_to_bc)
        nmi_filtered_map[dec_id] = nmi_decomposition(
            mem, class_to_bc, exclude=infra_classes
        )
        entropy_map[dec_id] = entropy_metric_for_decomposition(
            mem, G_design, P_u_to_c, bc_list, top_k=top_k
        )

    metrics_df = metrics_df.copy()
    metrics_df["NMI"]                      = metrics_df["ID"].map(nmi_map)
    metrics_df["NMI_filtered"] = metrics_df["ID"].map(nmi_filtered_map)
    metrics_df["penalty_swm"]              = metrics_df["ID"].map(
        lambda i: entropy_map.get(i, {}).get("penalty_size_weighted_mean"))
    metrics_df["penalty_mean"]             = metrics_df["ID"].map(
        lambda i: entropy_map.get(i, {}).get("penalty_mean"))
    metrics_df["score_swm"]                = metrics_df["ID"].map(
        lambda i: entropy_map.get(i, {}).get("score_size_weighted_mean"))
    metrics_df["penalty_rel_swm"]          = metrics_df["ID"].map(
        lambda i: entropy_map.get(i, {}).get("penalty_rel_size_weighted_mean"))
    metrics_df["H_global"]                 = metrics_df["ID"].map(
        lambda i: entropy_map.get(i, {}).get("H_global"))
    metrics_df["n_clusters"]               = metrics_df["ID"].map(
        lambda i: entropy_map.get(i, {}).get("n_clusters"))

    metrics_df = metrics_df.dropna(subset=["NMI"])

    for col in ["NMI", "penalty_swm", "penalty_mean", "score_swm", "penalty_rel_swm", "H_global"]:
        metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce")

    for col, label in [
        ("NMI",          "NMI"),
        ("penalty_swm",  "penalty (size_weighted_mean)"),
        ("penalty_mean", "penalty (mean)"),
    ]:
        vals = metrics_df[col].dropna()
        print(f"\n{label}:")
        print(f"  min={vals.min():.4f}  max={vals.max():.4f}  "
              f"std={vals.std():.4f}  range={vals.max()-vals.min():.4f}")

    print(f"\nTop 5 decompositions by NMI:")
    top5 = metrics_df.nlargest(5, "NMI")[
        ["ID", "K", "RESOLUTION", "NMI", "penalty_swm", "score_swm", "n_clusters"]
    ]
    print(top5.to_string(index=False))

    print(f"\nTop 5 decompositions by score_rel (size_weighted_mean):")
    top5_e = metrics_df.nlargest(5, "score_swm")[
        ["ID", "K", "RESOLUTION", "NMI", "penalty_swm", "score_swm", "n_clusters"]
    ]
    print(top5_e.to_string(index=False))

    print(f"\nSpearman NMI vs structural metrics:")
    corr_nmi = spearman_analysis(metrics_df, score_col="NMI")
    _print_spearman(corr_nmi)

    print(f"\nSpearman penalty_swm vs structural metrics:")
    corr_pen = spearman_analysis(metrics_df, score_col="penalty_swm")
    _print_spearman(corr_pen)

    print(f"\nSpearman penalty_mean vs structural metrics:")
    corr_pen_m = spearman_analysis(metrics_df, score_col="penalty_mean")
    _print_spearman(corr_pen_m)

    print(metrics_df["NMI_filtered"].describe())
    rho, p = spearmanr(
        metrics_df["NMI"].dropna(),
        metrics_df["NMI_filtered"].dropna()
    )
    print(f"Stability NMI vs NMI_filtered: ρ={rho:.3f}, p={p:.4f}")

    top_raw = metrics_df.nlargest(10, "NMI")[["ID", "NMI", "NMI_filtered", "n_clusters"]]
    top_raw["rank_raw"]      = top_raw["NMI"].rank(ascending=False)
    top_raw["rank_filtered"] = metrics_df["NMI_filtered"].rank(ascending=False)
    top_raw["rank_delta"]    = top_raw["rank_raw"] - top_raw["rank_filtered"]
    print(top_raw.to_string(index=False))

    metrics_df.to_csv(f"results/RQ1/metrics_{project}.csv", index=False)
    corr_nmi.to_csv(f"results/RQ2/spearman_nmi_{project}.csv", index=False)
    corr_pen.to_csv(f"results/RQ2/spearman_penalty_swm_{project}.csv", index=False)
    corr_pen_m.to_csv(f"results/RQ2/spearman_penalty_mean_{project}.csv", index=False)

    return {
        "project":    project,
        "metrics_df": metrics_df,
        "corr_nmi":   corr_nmi,
        "corr_pen":   corr_pen,
        "corr_pen_m": corr_pen_m,
        "nmi_map":    nmi_map,
        "entropy_map": entropy_map,
    }


def _print_spearman(corr_df: pd.DataFrame):
    print(f"  {'Metric':<10} {'rho':>8} {'p-value':>10}  sig")
    print(f"  {'-'*38}")
    for _, row in corr_df.iterrows():
        sig = "**" if row["p_value"] < 0.05 else ("*" if row["p_value"] < 0.10 else "")
        rho = f"{row['rho']:>8.3f}" if not pd.isna(row["rho"]) else "     nan"
        p   = f"{row['p_value']:>10.3f}" if not pd.isna(row["p_value"]) else "       nan"
        print(f"  {row['metric']:<10} {rho} {p}  {sig}")


def comparative_report(results: list[dict]):
    print(f"\n{'='*60}")
    print(f"  COMPARATIVE REPORT")
    print(f"{'='*60}")

    projects = [r["project"] for r in results]

    print("\nSpearman ρ (NMI) per project:")
    _print_comparative_spearman(results, corr_key="corr_nmi")

    print("\nSpearman ρ (penalty size_weighted_mean) per project:")
    _print_comparative_spearman(results, corr_key="corr_pen")

    print("\nSpearman ρ (penalty mean) per project:")
    _print_comparative_spearman(results, corr_key="corr_pen_m")

    print(f"\n{'Project':<15} {'NMI min':>9} {'NMI max':>9} {'NMI std':>9} "
          f"{'pen_swm min':>12} {'pen_swm max':>12} {'pen_swm std':>12}")
    print("-" * 82)
    for r in results:
        df   = r["metrics_df"]
        nmi  = df["NMI"].dropna()
        pen  = df["penalty_swm"].dropna()
        print(f"{r['project']:<15} {nmi.min():>9.4f} {nmi.max():>9.4f} {nmi.std():>9.4f} "
              f"{pen.min():>12.4f} {pen.max():>12.4f} {pen.std():>12.4f}")

    _plot_spearman_comparative(results, corr_key="corr_nmi",  title="NMI")
    _plot_spearman_comparative(results, corr_key="corr_pen",  title="penalty (size_weighted_mean)")
    _plot_distributions(results)

    combined = pd.concat(
        [r["metrics_df"].assign(project=r["project"]) for r in results],
        ignore_index=True,
    )
    combined.to_csv("results/RQ1/metrics_combined.csv", index=False)


def _print_comparative_spearman(results: list[dict], corr_key: str):
    corr_dfs = [r[corr_key].set_index("metric") for r in results]
    projects  = [r["project"] for r in results]
    header = f"  {'Metric':<10}" + "".join(f"  {p.upper()[:12]:>15}" for p in projects) + "  consistent"
    print(header)
    print(f"  {'-'*(12 + 17*len(projects) + 12)}")
    for metric in STRUCTURAL_METRICS:
        row_str = f"  {metric:<10}"
        rhos = []
        for corr_df in corr_dfs:
            if metric not in corr_df.index or pd.isna(corr_df.loc[metric, "rho"]):
                row_str += f"  {'nan':>15}"
                continue
            rho = corr_df.loc[metric, "rho"]
            p   = corr_df.loc[metric, "p_value"]
            sig = "**" if p < 0.05 else ("*" if p < 0.10 else "")
            row_str += f"  {rho:>8.3f}{sig:>3}{'':>4}"
            rhos.append(rho)
        consistent = "✓" if len(rhos) == 2 and rhos[0] * rhos[1] > 0 else "✗"
        print(row_str + f"  {consistent}")


def _plot_spearman_comparative(results: list[dict], corr_key: str, title: str):
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 6))
    if len(results) == 1:
        axes = [axes]
    colors = ["steelblue", "coral", "mediumseagreen"]

    for ax, r, color in zip(axes, results, colors):
        corr_df = r[corr_key].dropna(subset=["rho"])
        metrics  = corr_df["metric"].tolist()
        rhos     = corr_df["rho"].tolist()
        p_values = corr_df["p_value"].tolist()
        bar_colors = [
            "darkblue"       if p < 0.05 else
            "cornflowerblue" if p < 0.10 else
            color
            for p in p_values
        ]
        ax.barh(metrics, rhos, color=bar_colors)
        ax.axvline(0,    color="black", linewidth=0.8)
        ax.axvline( 0.5, color="gray",  linewidth=0.5, linestyle="--")
        ax.axvline(-0.5, color="gray",  linewidth=0.5, linestyle="--")
        ax.set_xlim(-1.1, 1.1)
        df   = r["metrics_df"]
        nmi  = df["NMI"].dropna()
        ax.set_title(
            f"{r['project'].upper()}\nn={len(df)}  range={nmi.max()-nmi.min():.3f}  std={nmi.std():.3f}",
            fontsize=11,
        )
        ax.set_xlabel("Spearman ρ")
        ax.legend(handles=[
            mpatches.Patch(color="darkblue",       label="p < 0.05"),
            mpatches.Patch(color="cornflowerblue", label="p < 0.10"),
            mpatches.Patch(color=color,            label="n.s."),
        ], fontsize=8)

    plt.suptitle(f"Spearman ρ: {title} vs Structural Metrics", fontsize=13)
    plt.tight_layout()
    fname = f"results/graphics/spearman_comparative_{title.split()[0]}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()


def _plot_distributions(results: list[dict]):

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, col, label in [
        (axes[0], "NMI",         "NMI (higher = better)"),
        (axes[1], "penalty_swm", "penalty swm (lower = better)"),
    ]:
        data   = [r["metrics_df"][col].dropna().values for r in results]
        labels = [r["project"] for r in results]
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        colors = ["steelblue", "coral", "mediumseagreen"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax.set_title(label)
        ax.set_ylabel(col)

    plt.suptitle("BC alignment metrics distribution", fontsize=13)
    plt.tight_layout()
    plt.savefig("results/graphics/distributions_comparative.png", dpi=150, bbox_inches="tight")
    plt.show()


def run_full_analysis(projects_config: list[dict]) -> list[dict]:
    results = []
    for cfg in projects_config:
        r = analyze_project(
            project  = cfg["project"],
            G_design = cfg["G_design"],
            P_u_to_c = cfg["P_u_to_c"],
            bc_list  = cfg["bc_list"],
            top_k    = cfg.get("top_k", 1),
        )
        results.append(r)

    if len(results) > 1:
        comparative_report(results)

    return results


PROJECTS_CONFIG = {
    "cargo":     ["booking", "handling", "routing", "tracking"],
    "jpetstore": ["catalog", "cart", "order", "account"],
    "daytrader": ["account", "portfolio", "trade", "quote"],
}

if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[3])
    import argparse
    parser = argparse.ArgumentParser(description="BC-alignment analysis across monolith systems")
    parser.add_argument("--apps", nargs="+", default=list(PROJECTS_CONFIG),
                        choices=list(PROJECTS_CONFIG), metavar="APP",
                        help="Systems to analyse (default: all three)")
    args = parser.parse_args()

    configs = []
    for project in args.apps:
        G_design = read_graphml(f"monoliths/{project}/vista_disenio_{project}.graphml")
        with open(f"monoliths/{project}/alignment_results.pkl", "rb") as f:
            results = pickle.load(f)
        configs.append({
            "project":  project,
            "G_design": G_design,
            "P_u_to_c": results.P_I_D,
            "bc_list":  PROJECTS_CONFIG[project],
            "top_k":    1,
        })
    run_full_analysis(configs)
