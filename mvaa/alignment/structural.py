import networkx as nx
import numpy as np
from scipy.stats import rankdata

from mvaa.utils.graph import read_graphml



def percentile_normalize(metrics):
    nodes = list(metrics.keys())
    metric_names = list(next(iter(metrics.values())).keys())

    values = {
        m: np.array([metrics[n][m] for n in nodes])
        for m in metric_names
    }

    percentiles = {}
    for m, vals in values.items():
        if np.allclose(vals, vals[0]):
            percentiles[m] = np.zeros(len(vals))
        else:
            ranks = rankdata(vals, method="average")
            percentiles[m] = (ranks - 1) / (len(vals) - 1)

    return {
        n: np.array([percentiles[m][i] for m in metric_names])
        for i, n in enumerate(nodes)
    }


def center_vectors(vectors):
    M = np.vstack(list(vectors.values()))
    mean = M.mean(axis=0)

    return {n: v - mean for n, v in vectors.items()}


def cosine_similarity_centered(v1, v2, eps=1e-6):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < eps or n2 < eps:
        return 0.0

    return float(np.dot(v1, v2) / (n1 * n2))


def similarity_matrix_centered(A, B):
    return {
        (a, b): cosine_similarity_centered(va, vb)
        for a, va in A.items()
        for b, vb in B.items()
    }


def compute_general_view_metrics(G):
    degree = dict(G.degree())
    betweenness = nx.betweenness_centrality(G, normalized=True)
    closeness = nx.closeness_centrality(G)

    return {
        n: {
            "degree": degree.get(n, 0.0),
            "betweenness": betweenness.get(n, 0.0),
            "closeness": closeness.get(n, 0.0),
        }
        for n in G.nodes()
    }


def compute_structural_profiles_centered(G):
    raw = compute_general_view_metrics(G)
    sigma = percentile_normalize(raw)
    return center_vectors(sigma)


def compute_data_view_metrics(G):
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1

    metrics = {}

    for n in G.nodes():
        tipo = G.nodes[n].get("tipo")

        is_table = tipo == "tabla"
        is_column = tipo == "columna"

        neighbor_tables = [
            nb for nb in G.neighbors(n)
            if G.nodes[nb].get("tipo") == "tabla"
        ]

        table_context_degree = (
            np.mean([degrees[t] for t in neighbor_tables])
            if neighbor_tables else 0.0
        )

        metrics[n] = {
            "entity": 1.0 if is_table else 0.0,
            "degree": degrees[n] / max_degree,
            "shared": len(neighbor_tables) if is_column else 0.0,
            "table_context": table_context_degree / max_degree,
        }

    return metrics


def compute_data_structural_profiles_centered(G):
    raw = compute_data_view_metrics(G)
    sigma = percentile_normalize(raw)
    return center_vectors(sigma)

def project_data_to_logical_space(sigma_data):
    projected = {}

    for n, v in sigma_data.items():
        entity, degree, shared, table_ctx = v

        projected[n] = np.array([
            degree + table_ctx,
            shared,
            entity
        ])

    return projected


def compute_structural_similarity(G_design, G_impl, G_data):
    sigma_design = compute_structural_profiles_centered(G_design)
    sigma_impl   = compute_structural_profiles_centered(G_impl)
    sigma_data   = compute_data_structural_profiles_centered(G_data)
    sigma_data_proj = project_data_to_logical_space(sigma_data)

    sim_data_design = similarity_matrix_centered(
        sigma_data_proj, sigma_design
    )


    sim_design_impl = similarity_matrix_centered(
        sigma_design, sigma_impl
    )

    return sim_data_design, sim_design_impl


if __name__ == "__main__":
    import os
    import argparse
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[2])
    parser = argparse.ArgumentParser(description="Structural alignment analysis")
    parser.add_argument("--app", default="jpetstore",
                        choices=["cargo", "jpetstore", "daytrader"],
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

    sigma_design = compute_structural_profiles_centered(G_design)
    sigma_impl   = compute_structural_profiles_centered(G_impl)
    sigma_data   = compute_data_structural_profiles_centered(G_data)
    sigma_data_proj = project_data_to_logical_space(sigma_data)

    sim_data_design = similarity_matrix_centered(
        sigma_data_proj, sigma_design
    )


    sim_design_impl = similarity_matrix_centered(
            sigma_design, sigma_impl
    )


    print(sim_design_impl)
