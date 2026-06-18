import networkx as nx
import numpy as np

from mvaa.utils.graph import read_graphml


def normalize(v: np.ndarray):
    if v is None:
        return None
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    a = normalize(a)
    b = normalize(b)
    return float(np.dot(a, b))

def geometric_mean(values):
    prod = 1.0
    for v in values:
        prod *= v
    return prod ** (1.0 / len(values))


def get_data_nodes_by_kind(G_data, kind: str):
    return [
        n for n, d in G_data.nodes(data=True)
        if d.get("tipo") == kind and d.get("embedding") is not None
    ]


def align_by_embeddings(
        nodes_A,
        nodes_B,
        embeddings_A: dict,
        embeddings_B: dict,
        k: int = 1,
        min_sim: float = 0.0,
        mutual: bool = False,
):
    alignments = []
    best_B_for_A = {}
    best_A_for_B = {}

    # A → B
    for a in nodes_A:
        va = embeddings_A.get(a)
        if va is None:
            continue

        best = []
        for b in nodes_B:
            vb = embeddings_B.get(b)
            if vb is None:
                continue

            sim = cosine_sim(va, vb)
            if sim >= min_sim:
                best.append((b, sim))

        if not best:
            continue

        best.sort(key=lambda x: x[1], reverse=True)
        best = best[:k]

        best_B_for_A[a] = best
        for b, sim in best:
            if b not in best_A_for_B or sim > best_A_for_B[b][1]:
                best_A_for_B[b] = (a, sim)

    for a, matches in best_B_for_A.items():
        for b, sim in matches:
            if mutual:
                if best_A_for_B.get(b, (None,))[0] != a:
                    continue

            alignments.append({
                "a": a,
                "b": b,
                "similarity": sim,
                "source": "embedding"
            })

    return alignments


def build_tri_view_confidence(
        align_DI,
        align_DD,
        align_ID,
        min_conf: float = 0.4
):

    DI = {(x["a"], x["b"]): x["similarity"] for x in align_DI}
    DD = {(x["a"], x["b"]): x["similarity"] for x in align_DD}
    ID = {(x["a"], x["b"]): x["similarity"] for x in align_ID}

    results = []

    for (d, i), sim_di in DI.items():
        for (d2, t), sim_dd in DD.items():
            if d2 != d:
                continue

            if (i, t) not in ID:
                continue

            sim_id = ID[(i, t)]

            conf = geometric_mean([sim_di, sim_dd, sim_id])

            if conf >= min_conf:
                results.append({
                    "design": d,
                    "implementation": i,
                    "data": t,
                    "confidence": conf,
                    "sim_DI": sim_di,
                    "sim_DD": sim_dd,
                    "sim_ID": sim_id
                })

    return results

def align_columns_within_triad(
        design_attrs,
        table_columns,
        emb_design,
        emb_data,
        min_sim=0.4
):
    alignments = []

    for da in design_attrs:
        va = emb_design.get(da)
        if va is None:
            continue

        for col in table_columns:
            vc = emb_data.get(col)
            if vc is None:
                continue

            sim_dc = cosine_sim(va, vc)
            if sim_dc < min_sim:
                continue

            alignments.append({
                "design_attr": da,
                "column": col,
                "similarity": sim_dc,
                "source": "embedding"
            })

    return alignments


def compute_embeddings_similarity(G_design, G_impl, G_data):
    design_nodes = G_design.nodes()
    impl_nodes = G_impl.nodes()
    data_nodes = G_data.nodes()
    table_nodes = get_data_nodes_by_kind(G_data, "tabla")
    column_nodes = get_data_nodes_by_kind(G_data, "columna")

    design_emb = {n: G_design.nodes[n]["embedding"] for n in design_nodes}
    impl_emb   = {n: G_impl.nodes[n]["embedding"] for n in impl_nodes}
    data_emb = {n: G_data.nodes[n]["embedding"] for n in data_nodes}

    align_DI = align_by_embeddings(
        design_nodes,
        impl_nodes,
        design_emb,
        impl_emb,
        k=10,
        min_sim=0.1,
        mutual=False
    )
    align_DD = align_by_embeddings(
        design_nodes,
        table_nodes,
        design_emb,
        data_emb,
        k=1,
        min_sim=0.15,
        mutual=True
    )

    align_ID = align_by_embeddings(
        impl_nodes,
        table_nodes,
        impl_emb,
        data_emb,
        k=1,
        min_sim=0.2,
        mutual=True
    )

    return align_DI, align_DD, align_ID


if __name__ == "__main__":
    import os
    import argparse
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[2])
    parser = argparse.ArgumentParser(description="Embeddings-based alignment analysis")
    parser.add_argument("--app", default="cargo",
                        choices=["cargo", "jpetstore", "daytrader"],
                        help="Monolith system to analyse")
    args = parser.parse_args()
    project = args.app
    G_design = read_graphml(f"monoliths/{project}/vista_disenio_{project}.graphml")
    G_impl = read_graphml(f"monoliths/{project}//vista_implementacion_{project}_c.graphml")
    G_data = read_graphml(f"monoliths/{project}/vista_datos_{project}.graphml")
    design_nodes = G_design.nodes()
    impl_nodes = G_impl.nodes()
    data_nodes = G_data.nodes()
    table_nodes = get_data_nodes_by_kind(G_data, "tabla")
    column_nodes = get_data_nodes_by_kind(G_data, "columna")

    design_emb = {n: G_design.nodes[n]["embedding"] for n in design_nodes}
    impl_emb   = {n: G_impl.nodes[n]["embedding"] for n in impl_nodes}
    data_emb = {n: G_data.nodes[n]["embedding"] for n in data_nodes}

    align_DI = align_by_embeddings(
        design_nodes,
        impl_nodes,
        design_emb,
        impl_emb,
        k=1,
        min_sim=0.1,
        mutual=True
    )
    align_DD = align_by_embeddings(
        design_nodes,
        table_nodes,
        design_emb,
        data_emb,
        k=1,
        min_sim=0.15,
        mutual=True
    )

    align_ID = align_by_embeddings(
        impl_nodes,
        table_nodes,
        impl_emb,
        data_emb,
        k=1,
        min_sim=0.2,
        mutual=True
    )

    print("----Alignment DI ------")
    for a in align_DI:
        print(a)

    print("----Alignment DD ------")
    for a in align_DD:
        print(a)

    print("----Alignment ID ------")
    for a in align_ID:
        print(a)

    confidence = build_tri_view_confidence(align_DI, align_DD, align_ID)
    for c in confidence:
        print(c)

    align_columns_within_triad(design_nodes, column_nodes, design_emb, data_emb, min_sim=0.4)
