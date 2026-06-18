import json
from copy import deepcopy
import networkx as nx
import numpy as np


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


def export_graph(G: nx.Graph, out: str):
    G_export = deepcopy(G)
    def make_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(v) for v in obj]
        elif isinstance(obj, set):
            return [make_json_serializable(v) for v in obj]  # convert sets to lists
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):  # numpy scalar types
            return obj.item()
        else:
            return obj  # primitive types

    # Serialize node attributes
    for node_id in G_export.nodes:
        for key in list(G_export.nodes[node_id].keys()):
            G_export.nodes[node_id][key] = json.dumps(make_json_serializable(G_export.nodes[node_id][key]))

    # Serialize edge attributes
    for u, v, data in G_export.edges(data=True):
        for attr in list(data.keys()):
            data[attr] = json.dumps(make_json_serializable(data[attr]))


    nx.write_graphml(G_export, out)
