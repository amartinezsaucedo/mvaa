from __future__ import annotations

from typing import Dict, List, Optional, Any
from collections import defaultdict
import math
import json
import networkx as nx

from mvaa.utils.graph import export_graph


def _entropy_bits(dist: Dict[str, float], eps: float = 1e-12) -> float:
    h = 0.0
    for p in dist.values():
        p = max(eps, float(p))
        h -= p * math.log(p, 2)
    return h



def _normalize(dist: Dict[str, float]) -> Dict[str, float]:
    s = float(sum(dist.values()))
    if s <= 0:
        return dist
    return {k: v / s for k, v in dist.items()}


def _argmax_key(d: Dict[str, float]) -> Optional[str]:
    if not d:
        return None
    return max(d.items(), key=lambda kv: kv[1])[0]


def _edge_effective_weight(
        edata: Dict[str, Any],
        alpha_srl_direct: float = 1.0,
        alpha_llm: float = 0.6,
        use_freq: bool = False,
) -> float:
    base = float(edata.get("weight", 1.0))
    freq = int(edata.get("freq", 1))
    rel_types = set(edata.get("relation_types", []) or [])

    if "srl-direct" in rel_types:
        a = alpha_srl_direct
    elif "llm" in rel_types:
        a = alpha_llm
    else:
        a = 0.7

    wf = math.log(1.0 + max(0, freq)) if use_freq else 1.0
    return max(0.0, base) * wf * a


def _make_prior(
        bc_list: List[str],
        prior: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    if prior is None:
        return {bc: 1.0 / len(bc_list) for bc in bc_list}
    out = {bc: float(prior.get(bc, 0.0)) for bc in bc_list}
    out = _normalize(out)
    if sum(out.values()) <= 0:
        return {bc: 1.0 / len(bc_list) for bc in bc_list}
    return out


def _make_soft_seed_dist(
        bc_list: List[str],
        bc: str,
        strength: float = 0.95,
) -> Dict[str, float]:
    k = len(bc_list)
    strength = max(0.0, min(1.0, strength))
    rest = (1.0 - strength) / (k - 1) if k > 1 else 0.0
    return {b: (strength if b == bc else rest) for b in bc_list}


def infer_canonical_bcs_soft(
        G: nx.DiGraph,
        bc_list: List[str],
        seeds: Dict[str, str],                # node -> bc label
        *,
        iters: int = 50,
        undirected_propagation: bool = True,
        alpha_srl_direct: float = 1.0,
        alpha_llm: float = 0.25,
        use_freq_in_prop: bool = False,
        seed_strength: float = 0.95,          # how "hard" the seed prior is
        seed_anchor: float = 0.90,            # how strongly to re-anchor seeds each iteration
        teleport: float = 0.10,               # global prior mixing; higher => less degenerate
        prior: Optional[Dict[str, float]] = None,  # base prior over BCs (None => uniform)
        # preserve some of previous state
        inertia: float = 0.0,                 # 0..1, mix in previous P[n] to reduce oscillation
        out_prob_attr: str = "bc_probs",
        out_label_attr: str = "bc",
) -> Dict[str, Dict[str, float]]:
    nodes = list(G.nodes())
    base_prior = _make_prior(bc_list, prior)

    # build seed priors once
    seed_prior: Dict[str, Dict[str, float]] = {}
    for n, bc in seeds.items():
        if n in G:
            seed_prior[n] = _make_soft_seed_dist(bc_list, bc, strength=seed_strength)

    def neighbors(n: str):
        if undirected_propagation:
            for nb in G.successors(n):
                yield nb, G[n][nb]
            for nb in G.predecessors(n):
                yield nb, G[nb][n]
        else:
            for nb in G.successors(n):
                yield nb, G[n][nb]

    # init: start from base prior everywhere; seeds start from their seed prior
    P: Dict[str, Dict[str, float]] = {}
    for n in nodes:
        if n in seed_prior:
            P[n] = dict(seed_prior[n])
        else:
            P[n] = dict(base_prior)

    # iterate
    for _ in range(iters):
        newP: Dict[str, Dict[str, float]] = {}

        for n in nodes:
            # 1) neighbor aggregation
            acc = {bc: 0.0 for bc in bc_list}
            wsum = 0.0

            for nb, edata in neighbors(n):
                w_eff = _edge_effective_weight(
                    edata,
                    alpha_srl_direct=alpha_srl_direct,
                    alpha_llm=alpha_llm,
                    use_freq=use_freq_in_prop,
                )
                if w_eff <= 0:
                    continue
                wsum += w_eff
                pnb = P.get(nb)
                if not pnb:
                    continue
                for bc in bc_list:
                    acc[bc] += w_eff * pnb[bc]

            if wsum > 0:
                acc = {bc: acc[bc] / wsum for bc in bc_list}
                acc = _normalize(acc)
            else:
                acc = dict(P[n])  # no neighbors => keep

            # 2) global teleport prior (prevents collapse)
            mixed = {
                bc: (1.0 - teleport) * acc[bc] + teleport * base_prior[bc]
                for bc in bc_list
            }
            mixed = _normalize(mixed)

            # 3) optional inertia
            if inertia > 0.0:
                mixed = {
                    bc: (1.0 - inertia) * mixed[bc] + inertia * P[n][bc]
                    for bc in bc_list
                }
                mixed = _normalize(mixed)

            # 4) seed re-anchoring (only for seeded nodes)
            if n in seed_prior and seed_anchor > 0.0:
                sp = seed_prior[n]
                mixed = {
                    bc: (1.0 - seed_anchor) * mixed[bc] + seed_anchor * sp[bc]
                    for bc in bc_list
                }
                mixed = _normalize(mixed)

            newP[n] = mixed

        P = newP

    # write attrs
    for n in nodes:
        probs = P[n]
        G.nodes[n][out_prob_attr] = probs
        G.nodes[n][out_label_attr] = _argmax_key(probs)

    return P


def service_to_bc_mapping(
        G: nx.DiGraph,
        bc_list: List[str],
        service_attr: str = "service",       # int
        bc_prob_attr: str = "bc_probs",
        purity_threshold: float = 0.65,
        entropy_threshold_bits: float = 1.2,
) -> Dict[int, Dict[str, Any]]:
    by_service = defaultdict(list)
    for n in G.nodes():
        sid = G.nodes[n].get(service_attr)
        if sid is None:
            continue
        if not isinstance(sid, int):
            raise TypeError(f"{service_attr} must be int, got {type(sid)} for node {n}")
        by_service[sid].append(n)

    out: Dict[int, Dict[str, Any]] = {}
    for sid, nodes in by_service.items():
        agg = {bc: 0.0 for bc in bc_list}
        cnt = 0
        for n in nodes:
            p = G.nodes[n].get(bc_prob_attr)
            if not p:
                continue
            cnt += 1
            for bc in bc_list:
                agg[bc] += float(p.get(bc, 0.0))

        agg = _normalize(agg) if cnt > 0 else {bc: 1.0 / len(bc_list) for bc in bc_list}
        top = _argmax_key(agg)
        pur = float(agg.get(top, 0.0)) if top else 0.0
        H = _entropy_bits(agg)

        if top and (pur >= purity_threshold) and (H <= entropy_threshold_bits):
            label = top
        else:
            label = f"mixed:{top}" if top else "mixed:unknown"

        out[sid] = {
            "dist": agg,
            "bc_top": top,
            "purity": pur,
            "entropy": H,
            "label": label,
            "node_count": len(nodes),
            "nodes_with_probs": cnt,
        }

    return out


def write_service_bc_labels(
        G: nx.DiGraph,
        service_map: Dict[int, Dict[str, Any]],
        service_attr: str = "service",
        out_attr: str = "service_bc",
):
    for n in G.nodes():
        sid = G.nodes[n].get(service_attr)
        if sid is None:
            continue
        if sid in service_map:
            G.nodes[n][out_attr] = service_map[sid]["label"]


def read_graphml(path):
    G = nx.read_graphml(path)

    def try_json_load(value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    for _, attrs in G.nodes(data=True):
        for k, v in list(attrs.items()):
            attrs[k] = try_json_load(v)

    for _, _, attrs in G.edges(data=True):
        for k, v in list(attrs.items()):
            attrs[k] = try_json_load(v)

    return G


def ambiguous_nodes(G, probs_attr="bc_probs", tau_max=0.60, tau_gap=0.15):
    out = []
    for n in G.nodes():
        p = G.nodes[n].get(probs_attr)
        if not p:
            continue
        items = sorted(p.items(), key=lambda kv: kv[1], reverse=True)
        top1, v1 = items[0]
        top2, v2 = items[1] if len(items) > 1 else (None, 0.0)
        if v1 < tau_max or (v1 - v2) < tau_gap:
            out.append((n, top1, v1, top2, v2, p))
    return out



if __name__ == "__main__":
    import os
    import argparse
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[2])
    parser = argparse.ArgumentParser(description="Bounded context induction")
    parser.add_argument("--app", default="cargo",
                        help="Monolith system to analyse")
    args = parser.parse_args()
    project = args.app

    G = read_graphml(f"monoliths/{project}/vista_disenio_{project}.graphml")

    with open(f"monoliths/{project}/bc.json", "r") as f:
        bc = json.load(f)

    bc_list = bc["bc_list"]
    seeds = bc["seeds"]

    infer_canonical_bcs_soft(
        G,
        bc_list=bc_list,
        seeds=seeds,
        iters=60,
        undirected_propagation=True,
        teleport=0.12,
        seed_strength=0.95,
        seed_anchor=0.92,
        alpha_llm=0.25,
        use_freq_in_prop=False,
        prior=None,
        inertia=0.0,
        out_prob_attr="bc_probs",
        out_label_attr="bc",
    )

    svc_map = service_to_bc_mapping(
        G,
        bc_list=bc_list,
        service_attr="service",
        bc_prob_attr="bc_probs",
        purity_threshold=0.65,
        entropy_threshold_bits=1.2,
    )

    write_service_bc_labels(G, svc_map, service_attr="service", out_attr="service_bc")

    for node, data in G.nodes(data=True):
        print(f"{node}: {data["service_bc"]}")

    for n, d in G.nodes(data=True):
        print(n, d.get("bc_probs"))

    export_graph(G, f"monoliths/{project}/vista_disenio_{project}.graphml")  # overwrites existing graphml

