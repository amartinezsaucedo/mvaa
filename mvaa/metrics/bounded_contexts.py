from __future__ import annotations

import pickle
import json
from math import log
from statistics import stdev, mean
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Callable, Optional, Any
import networkx as nx

from mvaa.alignment.alignment import MultiViewAlignments
from mvaa.utils.graph import read_graphml

ImplNode = str
DesignNode = str
ImplClusterId = str
DesignClusterId = str

AlignmentDist = Dict[ImplNode, List[Tuple[DesignNode, float]]]
DesignDecomposition = Dict[DesignClusterId, List[DesignNode]]
ImplDecomposition = Dict[ImplClusterId, List[ImplNode]]


def clusters_from_graph(G: nx.Graph, *, cluster_attr: str = "service") -> Dict[str, List[str]]:
    clusters: Dict[str, List[str]] = {}
    for n, data in G.nodes(data=True):
        if cluster_attr not in data or data[cluster_attr] is None:
            continue
        cid = str(data[cluster_attr])
        clusters.setdefault(cid, []).append(str(n))
    return clusters


def clusters_from_membership(
        membership: Dict[str, Any],
        *,
        impl_node_mapper: Optional[Callable[[str], str]] = None,
) -> ImplDecomposition:
    clusters: Dict[str, List[str]] = {}
    for raw_u, raw_cid in membership.items():
        u = impl_node_mapper(raw_u) if impl_node_mapper else str(raw_u)
        cid = str(raw_cid)
        clusters.setdefault(cid, []).append(u)
    return clusters


def _normalize(d: Dict[str, float]) -> Dict[str, float]:
    s = sum(d.values())
    if s <= 0.0:
        return {k: 0.0 for k in d.keys()}
    return {k: v / s for k, v in d.items()}


def normalized_entropy(dist: Dict[str, float], *, eps: float = 1e-12) -> float:
    k = len(dist)
    if k <= 1:
        return 0.0
    ent = 0.0
    for p in dist.values():
        if p <= 0.0:
            continue
        ent -= p * log(max(p, eps))
    return ent / log(k)


def bca_metrics_for_all_decompositions(
        decompositions: Dict[str, Dict[str, Any]],
        bc_list: List[str],
        *,
        G_design: nx.Graph,
        P_u_to_c: AlignmentDist,
        top_k: int,
        alpha: float,
        design_cluster_attr: str = "service",
        impl_node_mapper: Optional[Callable[[str], str]] = None,
        missing_u: str = "skip",
        aggregate: str = "size_weighted_mean",
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for dec_id, payload in decompositions.items():
        membership = payload["membership"]
        out[dec_id] = bca_metric_for_decomposition(
            membership=membership,
            G_design=G_design,
            P_u_to_c=P_u_to_c,
            top_k=top_k,
            bc_list=bc_list,
            design_cluster_attr=design_cluster_attr,
            impl_node_mapper=impl_node_mapper,
            missing_u=missing_u,
            aggregate=aggregate,
            alpha=alpha
        )
    return out

def _smooth_and_normalize(dist: Dict[str, float], *, alpha: float) -> Dict[str, float]:
    if alpha <= 0.0:
        return _normalize(dist)
    k = len(dist)
    if k == 0:
        return dist
    return {cid: (p + alpha) / (1.0 + alpha * k) for cid, p in dist.items()}


def induced_P_bc_given_cluster(
        impl_nodes: List[ImplNode],
        P_u_to_c: AlignmentDist,
        G_design: nx.Graph,
        bc_list: List[str],
        *,
        missing_u: str = "skip",
        top_k: Optional[int] = 1,
        min_prob_threshold: float = 0.0
) -> Dict[str, float]:
    scores = {bc: 0.0 for bc in bc_list}
    k_bc = len(bc_list)
    count = 0

    for u in impl_nodes:
        dist_u = P_u_to_c.get(u)
        if dist_u is None:
            if missing_u == "uniform":
                for bc in bc_list:
                    scores[bc] += 1.0 / k_bc
                count += 1
            continue

        pairs = sorted(dist_u, key=lambda x: x[1], reverse=True)
        if top_k is not None:
            pairs = pairs[:top_k]
        if min_prob_threshold > 0:
            pairs = [(c, p) for c, p in pairs if p >= min_prob_threshold]
        if not pairs:
            continue

        Z = sum(p for _, p in pairs)
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
        return {bc: 1.0 / k_bc for bc in bc_list}

    return _normalize({bc: scores[bc] / count for bc in bc_list})


def P_Cd_global_from_P_Cd_given_Ci(P_Cd_given_Ci: dict, membership: dict, impl_nodes: list, bc_list: list):
    size = {}
    total = 0
    for u in impl_nodes:
        if u in membership:
            ci = str(membership[u])
            size[ci] = size.get(ci, 0) + 1
            total += 1
    if total == 0:
        return {bc: 0.0 for bc in bc_list}

    P = {bc: 0.0 for bc in bc_list}
    for ci, w in size.items():
        if ci not in P_Cd_given_Ci:
            continue
        p_ci = w / total
        for bc in bc_list:
            P[bc] += p_ci * float(P_Cd_given_Ci[ci].get(bc, 0.0))

    s = sum(P.values())
    if s > 0:
        for bc in P:
            P[bc] /= s
    return P



def bca_metric_for_decomposition(
        *,
        membership: Dict[str, Any],
        G_design: nx.Graph,
        P_u_to_c: AlignmentDist,
        top_k: int,
        bc_list: List[str],
        design_cluster_attr: str = "service",
        impl_node_mapper: Optional[Callable[[str], str]] = None,
        missing_u: str = "skip",
        aggregate: str = "size_weighted_mean",
        alpha: float = 0.0,
) -> Dict[str, Any]:
    impl_clusters: ImplDecomposition = clusters_from_membership(membership, impl_node_mapper=impl_node_mapper)
    design_clusters: DesignDecomposition = clusters_from_graph(G_design, cluster_attr=design_cluster_attr)

    P_Cd_given_Ci: Dict[str, Dict[str, float]] = {}
    H_Ci: Dict[str, float] = {}

    sizes = {ci: len(nodes) for ci, nodes in impl_clusters.items()}
    total_size = sum(sizes.values())

    for ci, nodes in impl_clusters.items():
        dist = induced_P_bc_given_cluster(
            nodes,
            P_u_to_c,
            G_design,
            bc_list,
            missing_u=missing_u,
            top_k=top_k
        )
        P_Cd_given_Ci[ci] = dist
        H_Ci[ci] = normalized_entropy(dist)


    if aggregate == "sum":
        penalty = sum(H_Ci.values())
    elif aggregate == "mean":
        penalty = (sum(H_Ci.values()) / len(H_Ci)) if H_Ci else 0.0
    elif aggregate == "size_weighted_mean":
        penalty = (sum(H_Ci[ci] * sizes.get(ci, 0) for ci in H_Ci.keys()) / total_size) if total_size else 0.0
    else:
        raise ValueError(f"Unknown aggregate: {aggregate}")
    impl_nodes = [
        (impl_node_mapper(raw_u) if impl_node_mapper else str(raw_u))
        for raw_u in membership.keys()
    ]

    P_Cd_global = P_Cd_global_from_P_Cd_given_Ci(P_Cd_given_Ci, membership, impl_nodes, bc_list)
    H_global = normalized_entropy(P_Cd_global)
    penalty_rel = (penalty / H_global) if H_global > 1e-12 else 0.0   # lower = better
    score_rel = 1.0 - penalty_rel                                     # higher = better

    covered = sum(
        1 for raw_u in membership.keys()
        if (impl_node_mapper(raw_u) if impl_node_mapper else raw_u) in P_u_to_c
    )

    return {
        "penalty": penalty,                 # expected H(Cd|Ci) (normalized by log|Cd|)
        "penalty_rel": penalty_rel,         # H(Cd|Ci) / H(Cd)
        "score_rel": score_rel,             # 1 - penalty_rel
        "H_global": H_global,               # H(Cd)
        "P_Cd_global": P_Cd_global,         # global distribution
        "H_Ci": H_Ci,
        "P_Cd_given_Ci": P_Cd_given_Ci,
        "coverage_nodes_with_P": covered,
        "n_impl_nodes": len(membership),
        "n_impl_clusters": len(impl_clusters),
        "n_design_contexts": len(design_clusters),
    }

def P_bc_given_u(
        u: str,
        *,
        P_u_to_c: AlignmentDist,
        G_design: nx.Graph,
        bc_list: List[str],
        top_k: Optional[int] = 3,
        bc_probs_attr: str = "bc_probs",
) -> Dict[str, float]:
    scores = {bc: 0.0 for bc in bc_list}

    dist_u = P_u_to_c.get(u)
    if not dist_u:
        return scores

    pairs = dist_u
    if top_k is not None:
        pairs = sorted(dist_u, key=lambda x: x[1], reverse=True)[:top_k]

    Z = sum(p for _, p in pairs)
    if Z <= 0.0:
        return scores

    for c, p_cu in pairs:
        d = G_design.nodes.get(c)
        if not d:
            continue
        bc_probs = d.get(bc_probs_attr)
        if not isinstance(bc_probs, dict):
            continue

        w = p_cu / Z
        for bc in bc_list:
            scores[bc] += w * float(bc_probs.get(bc, 0.0))

    s = sum(scores.values())
    if s > 0:
        for bc in scores:
            scores[bc] /= s

    return scores


def print_top3_bc_per_u(
        *,
        P_u_to_c: AlignmentDist,
        G_design: nx.Graph,
        bc_list: List[str],
        impl_nodes: List[str],
        top_k: int = 3,
        max_nodes: int = 20,
):
    print("=== Top-3 P(BC | u) ===")
    for i, u in enumerate(impl_nodes[:max_nodes]):
        dist = P_bc_given_u(
            u,
            P_u_to_c=P_u_to_c,
            G_design=G_design,
            bc_list=bc_list,
            top_k=top_k,
        )
        ranked = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3]

        print(f"\n{u}")
        for bc, p in ranked:
            print(f"  {bc:10s} {p:.3f}")


def nmi_decomposition(membership, class_to_bc, impl_nodes):
    pairs = []
    for u in impl_nodes:
        u_str = str(u)
        if u_str not in membership:
            continue
        bctx = class_to_bc.get(u_str.split(".")[-1])
        if bctx is None:
            continue
        pairs.append((str(membership[u_str]), bctx[1]))  # (cluster, BC)

    if not pairs:
        return 0.0

    n = len(pairs)
    clusters = [p[0] for p in pairs]
    bcs = [p[1] for p in pairs]

    p_c = Counter(clusters)
    p_bc = Counter(bcs)
    p_joint = Counter(pairs)

    mi = 0.0
    for (c, bctx), cnt in p_joint.items():
        p_ij = cnt / n
        p_i = p_c[c] / n
        p_j = p_bc[bctx] / n
        mi += p_ij * log(p_ij / (p_i * p_j) + 1e-12)

    H_c  = -sum((v/n) * log(v/n + 1e-12) for v in p_c.values())
    H_bc = -sum((v/n) * log(v/n + 1e-12) for v in p_bc.values())

    nmi = 2 * mi / (H_c + H_bc) if (H_c + H_bc) > 0 else 0.0
    return nmi

def print_results(G_impl, results, decompositions, scores):
    impl_nodes = list(G_impl.nodes())
    print(f"Total classes in G_impl: {len(impl_nodes)}")
    print(f"With P_I_D:       {sum(1 for u in impl_nodes if u in results.P_I_D)}")
    print(f"With P_I_D_via_T: {sum(1 for u in impl_nodes if u in results.P_I_D_via_T)}")
    print(f"Con ambas:       {sum(1 for u in impl_nodes if u in results.P_I_D and u in results.P_I_D_via_T)}")

    all_decomp_nodes = set()
    for payload in decompositions.values():
        all_decomp_nodes.update(payload["membership"].keys())
    print(f"\nNodes in decompositions: {len(all_decomp_nodes)}")
    print(f"With P_I_D:       {sum(1 for u in all_decomp_nodes if u in results.P_I_D)}")
    print(f"With P_I_D_via_T: {sum(1 for u in all_decomp_nodes if u in results.P_I_D_via_T)}")

    vals = [v["penalty"] for v in scores.values()]
    print(f"\npenalty: std={stdev(vals):.4f} min={min(vals):.4f} max={max(vals):.4f}")

    class_to_bc = {}
    for u, dist_u in P_u_to_c.items():
        if not dist_u:
            continue
        best_c = max(dist_u, key=lambda x: x[1])[0]
        bp = G_design.nodes.get(best_c, {}).get("bc_probs", {})
        if bp:
            dominant_bc = max(bp, key=bp.get)
            class_to_bc[u.split(".")[-1]] = (best_c.split("#")[0], dominant_bc)

    impl_nodes = list(G_impl.nodes())
    nmi_scores = {}
    for dec_id, payload in decompositions.items():
        nmi_scores[dec_id] = nmi_decomposition(
            payload["membership"], class_to_bc, impl_nodes
        )

    for dec_id, nmi in sorted(nmi_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{dec_id}: NMI={nmi:.4f}  (penalty={scores[dec_id]['penalty']:.4f})")

    print(f"{'Decomposition':<20} {'k':>4} {'NMI':>8} {'penalty':>10} {'score_rel':>10} {'pure_clusters':>14}")
    print("-" * 70)
    for dec_id, nmi in sorted(nmi_scores.items(), key=lambda x: x[1], reverse=True):
        res = scores[dec_id]
        k = res["n_impl_clusters"]
        print(f"{dec_id:<20} {k:>4} {nmi:>8.4f} {res['penalty']:>10.4f} "
              f"{res['score_rel']:>10.4f} {sum(1 for h in res['H_Ci'].values() if h < 0.5):>14}")

    print(f"\nNMI:    std={stdev(nmi_scores.values()):.4f}  "
          f"range={max(nmi_scores.values())-min(nmi_scores.values()):.4f}")
    print(f"penalty: std={stdev(v['penalty'] for v in scores.values()):.4f}  "
          f"range={max(v['penalty'] for v in scores.values())-min(v['penalty'] for v in scores.values()):.4f}")

    bc_groups = defaultdict(list)
    for cls, (concept, bctx) in class_to_bc.items():
        bc_groups[bctx].append(cls)

    for bctx, classes in sorted(bc_groups.items()):
        print(f"\n{bctx} ({len(classes)} classes):")
        for c in classes:
            print(f"  {c}")

    print(f"\nWithout inferred BC: {90 - len(class_to_bc)} classes")

    for dec_id, res in sorted(scores.items(), key=lambda kv: kv[1]["penalty"]):
        n_clusters = res["n_impl_clusters"]
        h_vals = list(res["H_Ci"].values())
        import statistics
        pure = sum(1 for h in h_vals if h < 0.5)
        mixed = sum(1 for h in h_vals if h > 0.8)
        print(f"dec={dec_id} k={n_clusters:2d} penalty={res['penalty']:.4f} "
              f"H_mean={statistics.mean(h_vals):.3f} "
              f"pure_clusters(<0.5)={pure} mixed_clusters(>0.8)={mixed}")


    entropies = []
    for u, dist_u in P_u_to_c.items():
        if not dist_u:
            continue
        Z = sum(p for _, p in dist_u)
        if Z <= 0:
            continue
        h = -sum((p/Z) * log(p/Z + 1e-12) for _, p in dist_u if p > 0)
        h_norm = h / log(len(dist_u)) if len(dist_u) > 1 else 0
        entropies.append((u.split(".")[-1], h_norm))

    entropies.sort(key=lambda x: x[1], reverse=True)
    print("Classes with flattest P_u_to_c (worst signal):")
    for name, h in entropies[:10]:
        print(f"  {name}: H_norm={h:.3f}")
    print(f"\nMean H_norm: {sum(h for _,h in entropies)/len(entropies):.3f}")

    scores = bca_metrics_for_all_decompositions(
        decompositions,
        G_design=G_design,
        P_u_to_c=P_u_to_c,
        top_k=1,
        bc_list=bc["bc_list"],
        alpha=0.0,
        missing_u="skip",
        aggregate="size_weighted_mean",
    )


    ranking = sorted(scores.items(), key=lambda kv: kv[1]["penalty_rel"], reverse=True)

    top1_via_T = Counter()
    for u, dist_u in results.P_I_D_via_T.items():
        if dist_u:
            best = max(dist_u, key=lambda x: x[1])
            top1_via_T[best[0]] += 1

    print("Top1 via T:")
    for concept, count in top1_via_T.most_common(10):
        bp = G_design.nodes[concept].get("bc_probs", {})
        dominant_bc = max(bp, key=bp.get) if bp else "?"
        print(f"  {concept} ({dominant_bc}): {count} classes")



    top1_concepts = Counter()
    for u, dist_u in P_u_to_c.items():
        best = max(dist_u, key=lambda x: x[1])[0]
        top1_concepts[best] += 1

    print("Top-1 concept distribution:")
    for concept, count in top1_concepts.most_common():
        bp = G_design.nodes[concept].get("bc_probs", {})
        dominant_bc = max(bp.values()) if bp else 0
        print(f"  {concept}: {count} classes → dominant_bc={dominant_bc:.3f}")

    vals = [(k, v["penalty"], v["penalty_rel"]) for k, v in scores.items()]
    penalties = [v[1] for v in vals]

    print(f"penalty   std={stdev(penalties):.4f}  min={min(penalties):.4f}  max={max(penalties):.4f}")

    key = sorted(scores.items(), key=lambda kv: kv[1]["penalty_rel"])[0][0]
    print("Best decomposition P_Cd_given_Ci:")
    for ci, dist in scores[key]["P_Cd_given_Ci"].items():
        dominant = max(dist.values())
        print(f"  cluster {ci}: dominant={dominant:.3f}  {dist}")

    print(f"Ranking top-1: {ranking[0]}")
    print(decompositions.get(ranking[0][0])["membership"])
    print({n: data["service"] for n, data in G_design.nodes(data=True)})
    #print(P_u_to_c)

    vals = [(k, v["H_global"], v["penalty_rel"]) for k, v in scores.items()]
    vals_sorted = sorted(vals, key=lambda t: t[1])  # by H_global
    print("H_global min/max:", vals_sorted[0], vals_sorted[-1])
    print("std H_global:", (sum((x[1]-sum(v[1] for v in vals)/len(vals))**2 for x in vals)/len(vals))**0.5)

    key = ranking[0][0]
    m = decompositions[key]["membership"]

    impl_nodes = list(G_impl.nodes())

    covered = sum(1 for u in impl_nodes if u in m)
    print("impl_nodes:", len(impl_nodes), "covered_by_membership:", covered)

    print("n_impl_clusters (per scores):", scores[key]["n_impl_clusters"])
    print("n_clusters (per membership):", len(set(m[u] for u in impl_nodes if u in m)))

    def signature(membership, impl_nodes):
        sizes = {}
        for u in impl_nodes:
            if u in membership:
                ci = membership[u]
                sizes[ci] = sizes.get(ci, 0) + 1
        return tuple(sorted(sizes.values()))

    keys = list(decompositions.keys())[:5]
    for i in range(len(keys)-1):
        a, b = keys[i], keys[i+1]
        sa = signature(decompositions[a]["membership"], impl_nodes)
        sb = signature(decompositions[b]["membership"], impl_nodes)
        print(a, b, "same_size_signature?", sa == sb)


    def diff_count(m1, m2, impl_nodes):
        return sum(1 for u in impl_nodes if u in m1 and u in m2 and m1[u] != m2[u])

    a, b = keys[0], keys[1]
    print("different assignments:", diff_count(decompositions[a]["membership"], decompositions[b]["membership"], impl_nodes))

    print_top3_bc_per_u(
        P_u_to_c=P_u_to_c,
        G_design=G_design,
        bc_list=bc["bc_list"],
        impl_nodes=impl_nodes,
        top_k=3,
        max_nodes=300,
    )


if __name__ == "__main__":
    import os
    import argparse
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[2])
    parser = argparse.ArgumentParser(description="Bounded context alignment metrics")
    parser.add_argument("--app", default="jpetstore",
                        help="Monolith system to analyse")
    args = parser.parse_args()
    project = args.app
    G_design = read_graphml(
        f"monoliths/{project}/vista_disenio_{project}.graphml"
    )
    G_impl = read_graphml(
        f"monoliths/{project}/vista_implementacion_{project}_c.graphml"
    )
    G_data = read_graphml(
        f"monoliths/{project}/vista_datos_{project}.graphml"
    )
    with open(f"monoliths/{project}/alignment_results.pkl", "rb") as f:
        results = pickle.load(f)

    with open(f"monoliths/{project}/decompositions_{project}.pkl", "rb") as f:
        decompositions = pickle.load(f)

    with open(f"monoliths/{project}/bc.json", "rb") as f:
        bc = json.load(f)

    P_u_to_c = results.P_I_D # u -> [(concept, prob), ...]

    scores = bca_metrics_for_all_decompositions(
        decompositions, G_design=G_design, P_u_to_c=P_u_to_c,
        bc_list=bc["bc_list"],
        top_k=1, alpha=0.0, missing_u="skip", aggregate="size_weighted_mean",
    )

    print_results(G_impl, results, decompositions, scores)
