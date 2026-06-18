from collections import defaultdict

from mvaa.bc_induction.bc_induction import read_graphml
from mvaa.utils.decompositions import build_dict_from_directory

decompositions = build_dict_from_directory("monoliths/daytrader/mid_results/services")

TOP3    = ["04_03_18_02_02_826045", "04_03_17_49_08_364749", "04_03_17_52_00_575877"]
BOTTOM3 = ["04_03_17_54_20_240991", "04_03_18_03_52_016931", "04_03_17_34_49_331757"]

def print_decomposition_clean(dec_id, label, decompositions):
    mem = decompositions[dec_id]["membership"]
    clusters = defaultdict(list)
    for cls, cid in mem.items():
        simple = cls.replace("class:", "").split(".")[-1]
        clusters[cid].append(simple)
    print(f"\n{label}")
    print("-" * 40)
    for cid, classes in sorted(clusters.items()):
        print(f"  Service {cid+1}: {', '.join(sorted(classes))}")

print("=== DECOMPOSITIONS FOR EXPERT EVALUATION ===")
for i, dec_id in enumerate(TOP3 + BOTTOM3, 1):
    print_decomposition_clean(dec_id, f"Decomposition {i}", decompositions)


G = read_graphml("monoliths/daytrader/vista_implementacion_daytrader_c.graphml")

print("=== GRAPH ===")
for n, d in G.nodes(data=True):
    print(n, d.get("semantic_descriptor"))
