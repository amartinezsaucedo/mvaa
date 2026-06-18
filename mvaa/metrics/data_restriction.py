from __future__ import annotations

import json
import pickle

import networkx as nx
import pandas as pd
from collections import defaultdict, deque
from itertools import combinations
from typing import Dict, Set, Tuple, Any, List

from mvaa.utils.graph import read_graphml


def _tx_group_methods(root: str, call_adj: Dict[str, Set[str]], max_depth: int) -> Set[str]:
    visited = {root}
    q = deque([(root, 0)])
    while q:
        m, d = q.popleft()
        if d >= max_depth:
            continue
        for nxt in call_adj.get(m, ()):
            if nxt not in visited:
                visited.add(nxt)
                q.append((nxt, d + 1))
    return visited

def precompute_table_bits(access_by_class: Dict[str, Set[str]]) -> Tuple[Dict[str, int], Dict[str, int]]:
    all_tables = sorted({t for s in access_by_class.values() for t in s})
    table_bit = {t: (1 << i) for i, t in enumerate(all_tables)}

    class_access_bits: Dict[str, int] = {}
    for cls, tables in access_by_class.items():
        bits = 0
        for t in tables:
            b = table_bit.get(t)
            if b is not None:
                bits |= b
        class_access_bits[cls] = bits

    return table_bit, class_access_bits


def _jaccard_bits(a: int, b: int) -> float:
    if a == 0 and b == 0:
        return 0.0
    inter = (a & b).bit_count()
    uni = (a | b).bit_count()
    return inter / uni if uni else 0.0

def compute_do_metric_for_membership(
        membership: Dict[str, int],
        class_access_bits: Dict[str, int],
        class_tx_bits: Dict[str, int],
        *,
        only_pairs_with_overlap: bool = True,
) -> Dict[Tuple[int, int], float]:
    svc_access: Dict[int, int] = defaultdict(int)
    svc_tx: Dict[int, int] = defaultdict(int)

    table_to_svcs: Dict[int, Set[int]] = defaultdict(set)

    for cls, svc in membership.items():
        ab = class_access_bits.get(cls, 0)
        tb = class_tx_bits.get(cls, 0)
        if ab:
            svc_access[svc] |= ab
            x = ab
            while x:
                lsb = x & -x
                pos = (lsb.bit_length() - 1)
                table_to_svcs[pos].add(svc)
                x ^= lsb
        if tb:
            svc_tx[svc] |= tb

    services = sorted(set(svc_access.keys()) | set(svc_tx.keys()))
    if len(services) < 2:
        return {}

    candidate_pairs: Set[Tuple[int, int]] = set()
    for svcs in table_to_svcs.values():
        if len(svcs) < 2:
            continue
        svcs = sorted(svcs)
        for i in range(len(svcs)):
            for j in range(i + 1, len(svcs)):
                candidate_pairs.add((svcs[i], svcs[j]))

    if not only_pairs_with_overlap:
        for i in range(len(services)):
            for j in range(i + 1, len(services)):
                candidate_pairs.add((services[i], services[j]))

    penalties: Dict[Tuple[int, int], float] = {}
    for si, sj in candidate_pairs:
        overlap = _jaccard_bits(svc_access.get(si, 0), svc_access.get(sj, 0))
        if overlap == 0.0:
            continue
        tx = _jaccard_bits(svc_tx.get(si, 0), svc_tx.get(sj, 0))
        pen = overlap * tx
        if pen > 0:
            penalties[(si, sj)] = pen

    return penalties


def compute_do_metrics_for_many_decompositions(
        decompositions: Dict[str, Dict[str, Any]],
        class_access_bits: Dict[str, int],
        class_tx_bits: Dict[str, int],
        *,
        only_pairs_with_overlap: bool = True,
) -> Dict[str, Dict[Tuple[int, int], float]]:
    out: Dict[str, Dict[Tuple[int, int], float]] = {}
    for param, payload in decompositions.items():
        membership = payload.get("membership", {})
        out[param] = compute_do_metric_for_membership(
            membership=membership,
            class_access_bits=class_access_bits,
            class_tx_bits=class_tx_bits,
            only_pairs_with_overlap=only_pairs_with_overlap,
        )
    return out


def precompute_class_tx_bits(
        graphml_path: str,
        *,
        contains_edge_types: Set[str] = frozenset({"contiene"}),  # class -> method
        call_edge_types: Set[str] = frozenset({"invoca"}),        # method -> method
        tx_max_depth: int = 2,
) -> Tuple[Dict[str, int], int]:
    G = read_graphml(graphml_path)

    method_to_class: Dict[str, str] = {}
    for u, v, data in G.edges(data=True):
        if data.get("tipo") in contains_edge_types and str(u).startswith("class:") and str(v).startswith("method:"):
            method_to_class[str(v)] = str(u)

    call_adj: Dict[str, Set[str]] = defaultdict(set)
    for u, v, data in G.edges(data=True):
        if data.get("tipo") in call_edge_types and str(u).startswith("method:") and str(v).startswith("method:"):
            call_adj[str(u)].add(str(v))

    roots: List[str] = []
    for n, data in G.nodes(data=True):
        nid = str(n)
        if nid.startswith("method:") and data.get("transactional") is True:
            roots.append(nid)

    class_tx_bits: Dict[str, int] = defaultdict(int)

    for tx_id, root in enumerate(roots):
        ms = _tx_group_methods(root, call_adj, max_depth=tx_max_depth)
        classes = {method_to_class[m] for m in ms if m in method_to_class}
        bit = 1 << tx_id
        for cls in classes:
            class_tx_bits[cls] |= bit

    return dict(class_tx_bits), len(roots)



def jaccard(a: Set[Any], b: Set[Any]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return inter / uni if uni else 0.0

def build_method_to_class_from_contains(
        G: nx.DiGraph,
        contains_edge_types: Set[str] = frozenset({"contiene"}),
) -> Dict[str, str]:
    m2c: Dict[str, str] = {}
    for u, v, data in G.edges(data=True):
        et = data.get("tipo")
        if et in contains_edge_types and str(u).startswith("class:") and str(v).startswith("method:"):
            m2c[str(v)] = str(u)
    return m2c


def build_method_call_adj(
        G: nx.DiGraph,
        call_edge_types: Set[str] = frozenset({"invoca"}),
) -> Dict[str, Set[str]]:
    adj: Dict[str, Set[str]] = defaultdict(set)
    for u, v, data in G.edges(data=True):
        et = data.get("tipo")
        if et in call_edge_types and str(u).startswith("method:") and str(v).startswith("method:"):
            adj[str(u)].add(str(v))
    return dict(adj)


def transactional_roots(G: nx.DiGraph) -> Set[str]:
    roots: Set[str] = set()
    for n, data in G.nodes(data=True):
        nid = str(n)
        if nid.startswith("method:") and data.get("transactional") is True:
            roots.add(nid)
    return roots


def tx_group_methods(root: str, call_adj: Dict[str, Set[str]], max_depth: int = 2) -> Set[str]:
    visited: Set[str] = {root}
    q = deque([(root, 0)])
    while q:
        m, d = q.popleft()
        if d >= max_depth:
            continue
        for nxt in call_adj.get(m, ()):
            if nxt not in visited:
                visited.add(nxt)
                q.append((nxt, d + 1))
    return visited


def build_tx_groups_clusters(
        roots_tx: Set[str],
        call_adj: Dict[str, Set[str]],
        method_to_class: Dict[str, str],
        cluster_of_class: Dict[str, str],
        max_depth: int = 2,
        drop_empty: bool = True,
) -> List[Set[str]]:
    groups: List[Set[str]] = []
    for r in roots_tx:
        ms = tx_group_methods(r, call_adj, max_depth=max_depth)
        classes = {method_to_class[m] for m in ms if m in method_to_class}
        clusters = {cluster_of_class[c] for c in classes if c in cluster_of_class}
        if (not clusters) and drop_empty:
            continue
        groups.append(clusters)
    return groups


def build_txs_by_cluster(tx_groups: List[Set[str]]) -> Dict[str, Set[int]]:
    txs_by_cluster: Dict[str, Set[int]] = defaultdict(set)
    for tx_id, clusters in enumerate(tx_groups):
        for c in clusters:
            txs_by_cluster[c].add(tx_id)
    return dict(txs_by_cluster)


def build_access_by_cluster(
        cluster_of_class: Dict[str, str],
        access_by_class: Dict[str, Set[str]],
) -> Dict[str, Set[str]]:
    access_by_cluster: Dict[str, Set[str]] = defaultdict(set)
    for cls, c in cluster_of_class.items():
        access_by_cluster[c].update(access_by_class.get(cls, set()))
    return dict(access_by_cluster)


def compute_do_metric(
        graphml_path: str,
        cluster_of_class: Dict[str, str],
        access_by_class: Dict[str, Set[str]],
        *,
        contains_edge_types: Set[str] = frozenset({"contiene"}),
        call_edge_types: Set[str] = frozenset({"invoca"}),
        tx_max_depth: int = 2,
        only_pairs_with_overlap: bool = True,
) -> Tuple[
    Dict[Tuple[str, str], float],  # penalties
    Dict[Tuple[str, str], float],  # overlap
    Dict[Tuple[str, str], float],  # tx
    Dict[str, Set[str]],           # access_by_cluster
    Dict[str, Set[int]],           # txs_by_cluster
]:
    G = read_graphml(graphml_path)

    method_to_class = build_method_to_class_from_contains(G, contains_edge_types=contains_edge_types)
    call_adj = build_method_call_adj(G, call_edge_types=call_edge_types)
    roots_tx = transactional_roots(G)

    tx_groups = build_tx_groups_clusters(
        roots_tx=roots_tx,
        call_adj=call_adj,
        method_to_class=method_to_class,
        cluster_of_class=cluster_of_class,
        max_depth=tx_max_depth,
    )
    txs_by_cluster = build_txs_by_cluster(tx_groups)

    access_by_cluster = build_access_by_cluster(cluster_of_class, access_by_class)
    clusters = set(access_by_cluster.keys()) | set(txs_by_cluster.keys())

    penalties: Dict[Tuple[str, str], float] = {}
    overlap_map: Dict[Tuple[str, str], float] = {}
    tx_map: Dict[Tuple[str, str], float] = {}

    for ci, cj in combinations(sorted(clusters), 2):
        Ai = access_by_cluster.get(ci, set())
        Aj = access_by_cluster.get(cj, set())
        overlap = jaccard(Ai, Aj)

        if only_pairs_with_overlap and overlap == 0.0:
            continue

        Ti = txs_by_cluster.get(ci, set())
        Tj = txs_by_cluster.get(cj, set())
        tx = jaccard(Ti, Tj)

        pen = overlap * tx
        overlap_map[(ci, cj)] = overlap
        tx_map[(ci, cj)] = tx

        if pen > 0:
            penalties[(ci, cj)] = pen

    return penalties, overlap_map, tx_map, access_by_cluster, txs_by_cluster


def top_k(d: Dict[Tuple[str, str], float], k: int = 20):
    return sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:k]


if __name__ == "__main__":
    import os
    import argparse
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[2])
    parser = argparse.ArgumentParser(description="Data-restriction metric analysis")
    parser.add_argument("--app", default="jpetstore",
                        help="Monolith system to analyse")
    args = parser.parse_args()
    project = args.app
    project_path = f"monoliths/{project}"
    with open(f"{project_path}/decompositions_{project}.pkl", "rb") as f:
        decompositions = pickle.load(f)
    with open(f"{project_path}/mapper_to_tables.json", "r") as f:
        mapper_to_tables = json.load(f)
    class_tx_bits, num_txs = precompute_class_tx_bits(f"{project_path}/vista_implementacion_{project}.graphml", tx_max_depth=2)

    table_bit, class_access_bits = precompute_table_bits(mapper_to_tables)

    all_penalties = compute_do_metrics_for_many_decompositions(
        decompositions=decompositions,
        class_access_bits=class_access_bits,
        class_tx_bits=class_tx_bits,
        only_pairs_with_overlap=True,
    )
    print(all_penalties)

    pen_data_global = {
        dec_id: sum(pairs.values())
        for dec_id, pairs in all_penalties.items()
    }
    pd.DataFrame({
        "ID": list(pen_data_global.keys()),
        "pen_data": list(pen_data_global.values()),
    }).to_csv(f"results/RQ1/pen_data_{project}.csv", index=False)
