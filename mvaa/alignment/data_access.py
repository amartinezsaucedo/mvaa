import json
import math
import numpy as np
from pprint import pprint
from typing import Dict, Set, Hashable, List, Callable, Optional, Tuple

from mvaa.utils.graph import read_graphml

ImplId = Hashable
DesignId = Hashable
EntityId = Hashable
TableId = Hashable

def cosine(u: List[float], v: List[float]) -> float:
    dot = 0.0
    nu = 0.0
    nv = 0.0
    for a, b in zip(u, v):
        dot += a * b
        nu += a * a
        nv += b * b
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot / (math.sqrt(nu) * math.sqrt(nv))

def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

def make_sigmoid_f(alpha: float = 4.0, beta: float = 1.0) -> Callable[[float], float]:
    return lambda n: _sigmoid(alpha * (n - beta))


def build_Tables_from_entities_in_impl(
        *,
        G_design,
        G_impl,
        entity_to_tables: Dict[EntityId, Set[TableId]],
        design_emb_attr: str = "embedding",
        impl_emb_attr: str = "embedding",
        k_entities: int = 3,
        min_cos: float = 0.30,
        entity_node_filter=None,
) -> Dict[DesignId, Set[TableId]]:

    if entity_node_filter is None:
        entity_node_filter = lambda eid, edata: True

    concept_vecs: Dict[DesignId, List[float]] = {}
    for c, cdata in G_design.nodes(data=True):
        v = cdata.get(design_emb_attr)
        if isinstance(v, list) and v and isinstance(v[0], (int, float)):
            concept_vecs[c] = [float(x) for x in v]

    entity_vecs: Dict[EntityId, List[float]] = {}
    for e in entity_to_tables.keys():
        if not G_impl.has_node(e):
            continue
        edata = G_impl.nodes[e]
        if not entity_node_filter(e, edata):
            continue
        v = edata.get(impl_emb_attr)
        if isinstance(v, list) and v and isinstance(v[0], (int, float)):
            entity_vecs[e] = [float(x) for x in v]

    entity_items = list(entity_vecs.items())

    Tables: Dict[DesignId, Set[TableId]] = {}
    for c, vc in concept_vecs.items():
        scored: List[Tuple[EntityId, float]] = []
        for e, ve in entity_items:
            s = cosine(vc, ve)
            if s >= min_cos:
                scored.append((e, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        top_entities = [e for (e, _) in scored[:k_entities]]

        tabs = set()
        for e in top_entities:
            tabs |= set(entity_to_tables.get(e, set()))
        Tables[c] = tabs

    for c, _ in G_design.nodes(data=True):
        Tables.setdefault(c, set())

    return Tables


def sim_data(
        u: ImplId,
        c: DesignId,
        Access: Dict[ImplId, Set[TableId]],
        Tables: Dict[DesignId, Set[TableId]],
        *,
        alpha: float = 4.0,
        beta: float = 1.0,
        f: Optional[Callable[[float], float]] = None,
) -> float:
    tables_c = Tables.get(c, set())
    if not tables_c:
        return 0.0

    n = float(len(Access.get(u, set()) & tables_c))
    if n == 0:
        return 0.0
    if f is None:
        f = make_sigmoid_f(alpha=alpha, beta=beta)
    return float(f(n))


def compute_sim_data_all(
        Access: Dict[ImplId, Set[TableId]],
        Tables: Dict[DesignId, Set[TableId]],
        *,
        alpha: float = 4.0,
        beta: float = 1.0,
        keep_zeros: bool = False,
) -> List[Dict[str, object]]:
    f = make_sigmoid_f(alpha=alpha, beta=beta)
    out = []
    for u in Access.keys():
        for c in Tables.keys():
            s = sim_data(u, c, Access, Tables, f=f)
            if keep_zeros or s > 0.0:
                out.append({"impl": u, "concept": c, "sim_data": s})
    return out


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

def x_count(u, c, Access, Tables) -> float:
    return float(len(Access.get(u, set()) & Tables.get(c, set())))

def x_coverage(u, c, Access, Tables) -> float:
    T = Tables.get(c, set())
    if not T:
        return 0.0
    inter = Access.get(u, set()) & T
    return float(len(inter)) / float(len(T))

def x_jaccard(u, c, Access, Tables) -> float:
    A = Access.get(u, set())
    T = Tables.get(c, set())
    if not T:
        return 0.0
    inter = A & T
    if not inter:
        return 0.0
    union = A | T
    return float(len(inter)) / float(len(union))

def calibrate_alpha_beta(
        Access: Dict[ImplId, Set[TableId]],
        Tables: Dict[DesignId, Set[TableId]],
        *,
        x_func: Callable[[ImplId, DesignId, Dict, Dict], float] = x_count,
        beta_quantile: float = 0.95,
        target_hi: float = 0.9,
        delta: float = 1.0,
        sample_pairs: int = 200_000,
        seed: int = 0,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    U = list(Access.keys())
    C = [c for c, tabs in Tables.items() if tabs]

    if not U or not C:
        return 1.0, 0.0  # fallback

    xs: List[float] = []
    n = min(sample_pairs, len(U) * len(C))
    for _ in range(n):
        u = U[rng.integers(0, len(U))]
        c = C[rng.integers(0, len(C))]
        xs.append(x_func(u, c, Access, Tables))

    beta = float(np.quantile(xs, beta_quantile))

    logit = math.log(target_hi / (1.0 - target_hi))
    alpha = float(logit / max(delta, 1e-9))

    return alpha, beta


def compute_data_signal(
        *,
        G_design,
        G_impl,
        Access: Dict[ImplId, Set[TableId]],
        entity_to_tables: Dict[EntityId, Set[TableId]],
        k_entities: int = 3,
        min_cos: float = 0.30,
        alpha: float = 4.0,
        beta: float = 1.0,
) -> List[Dict[str, object]]:
    Tables = build_Tables_from_entities_in_impl(
        G_design=G_design,
        G_impl=G_impl,
        entity_to_tables=entity_to_tables,
        k_entities=k_entities,
        min_cos=min_cos,
        design_emb_attr="embedding",
        impl_emb_attr="embedding"
    )
    return compute_sim_data_all(Access, Tables, alpha=alpha, beta=beta)

if __name__ == "__main__":
    import os
    import argparse
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[2])
    parser = argparse.ArgumentParser(description="Data-access alignment signal computation")
    parser.add_argument("--app", default="jpetstore",
                        choices=["cargo", "jpetstore", "daytrader"],
                        help="Monolith system to analyse")
    args = parser.parse_args()
    project_name = args.app
    project = f"monoliths/{project_name}"
    Gi = read_graphml(f"{project}/vista_implementacion_{project_name}_c.graphml")
    Gd = read_graphml(f"{project}/vista_disenio_{project_name}.graphml")
    Gdt = read_graphml(f"{project}/vista_datos_{project_name}.graphml")
    with open(f"{project}/entity_to_tables.json", "r") as f:
        entity_to_tables = json.load(f)
        entity_to_tables = {key: set(values) for key, values in entity_to_tables.items()}
    with open(f"{project}/mapper_to_tables.json", "r") as f:
        mapper_to_tables = json.load(f)
        mapper_to_tables = {key: set(values) for key, values in mapper_to_tables.items()}


    results = compute_data_signal(G_design=Gd, entity_to_tables=entity_to_tables, G_impl=Gi, Access=mapper_to_tables, alpha=10.0, beta=1)
    pprint(results)
