import networkx as nx

from mvaa.utils.graph import read_graphml, export_graph


def build_method_to_class_index(G: nx.MultiDiGraph) -> dict:
    method_to_class = {}

    for u, v, data in G.edges(data=True):
        if data.get("tipo") == "contiene":
            if (
                    G.nodes[u].get("tipo") == "clase"
                    and G.nodes[v].get("tipo") == "metodo"
            ):
                method_to_class[v] = u

    return method_to_class


def project_to_class_dependency_graph(G: nx.MultiDiGraph) -> nx.DiGraph:
    H = nx.DiGraph()

    for n, data in G.nodes(data=True):
        if data.get("tipo") == "clase":
            H.add_node(n, **data)


    method_to_class = build_method_to_class_index(G)


    aggregated = {}

    # invoca handled separately
    DEP_TYPES = {"usa", "hereda_de", "implementa", "depende_de"}

    for u, v, data in G.edges(data=True):
        tipo = data.get("tipo")

        src_class = None
        tgt_class = None


        if tipo == "invoca":
            if (
                    G.nodes[u].get("tipo") != "metodo"
                    or G.nodes[v].get("tipo") != "metodo"
            ):
                continue

            src_class = method_to_class.get(u)
            tgt_class = method_to_class.get(v)

            if not src_class or not tgt_class:
                continue


        elif tipo in DEP_TYPES:
            if (
                    G.nodes[u].get("tipo") == "clase"
                    and G.nodes[v].get("tipo") == "clase"
            ):
                src_class = u
                tgt_class = v
            else:
                continue
        else:
            continue


        if src_class == tgt_class:
            continue

        key = (src_class, tgt_class)

        if key not in aggregated:
            aggregated[key] = {
                "tipos": set(),
                "peso": 0,
                "origen": [],
            }

        entry = aggregated[key]
        entry["tipos"].add(tipo)
        entry["peso"] += data.get("peso", 1)
        entry["origen"].extend(data.get("origen", []))


    for (u, v), info in aggregated.items():
        H.add_edge(
            u,
            v,
            tipos=sorted(info["tipos"]),
            peso=info["peso"],
            origen=info["origen"],
        )

    return H


if __name__ == "__main__":
    import os
    import argparse
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[3])
    parser = argparse.ArgumentParser(description="Project implementation graph to class dependency graph")
    parser.add_argument("--app", default="cargo",
                        help="Monolith system to analyse")
    args = parser.parse_args()
    project = args.app
    G_impl = read_graphml(f"monoliths/{project}/vista_implementacion_{project}.graphml")

    G_class = project_to_class_dependency_graph(G_impl)

    export_graph(
        G_class,
        f"monoliths/{project}/vista_implementacion_{project}_c.graphml",
    )
