import json
import pickle
import statistics
import pandas as pd
from scipy.stats import spearmanr
from collections import defaultdict

from mvaa.metrics.data_restriction import precompute_class_tx_bits, precompute_table_bits, \
    compute_do_metrics_for_many_decompositions

if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[3])
    import argparse
    parser = argparse.ArgumentParser(description="Data-ownership (DO) metric analysis")
    parser.add_argument("--app", default="daytrader",
                        choices=["cargo", "jpetstore", "daytrader"],
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


    pen_data_global = {
        dec_id: sum(pairs.values())
        for dec_id, pairs in all_penalties.items()
    }

    pd.DataFrame({
        "ID": list(pen_data_global.keys()),
        "pen_data": list(pen_data_global.values()),
    }).to_csv(f"results/RQ1/pen_data_{project}.csv", index=False)

    vals = list(pen_data_global.values())
    print(f"Decompositions with Pen_data > 0: {sum(1 for v in vals if v > 0)} / {len(vals)}")
    print(f"Pen_data: min={min(vals):.4f} max={max(vals):.4f} std={statistics.stdev(vals):.4f}")

    df = pd.read_csv(f"results/RQ1/metrics_{project}.csv")
    df["pen_data"] = df["ID"].map(pen_data_global)

    print(f"\nCorrelation Pen_data vs Pen_bc:")
    rho, p = spearmanr(df["pen_data"].dropna(), df["penalty_swm"].dropna())
    print(f"  rho={rho:.3f}  p={p:.3f}")

    print(f"\nCorrelation Pen_data vs SERVICES:")
    rho, p = spearmanr(df["pen_data"].dropna(), df["SERVICES"].dropna())
    print(f"  rho={rho:.3f}  p={p:.3f}")

    best  = min(pen_data_global, key=pen_data_global.get)
    worst = max(pen_data_global, key=pen_data_global.get)
    print(f"\nBest  ({best}): {len(all_penalties[best])} penalized pairs, sum={pen_data_global[best]:.4f}")
    print(f"Worst ({worst}): {len(all_penalties[worst])} penalized pairs, sum={pen_data_global[worst]:.4f}")

    def cluster_summary(dec_id, decompositions, mapper_to_tables):
        mem = decompositions[dec_id]["membership"]

        clusters = defaultdict(list)
        for cls, cid in mem.items():
            simple = cls.replace("class:", "").split(".")[-1]
            clusters[cid].append(simple)

        cluster_tables = defaultdict(set)
        for cls, cid in mem.items():
            tables = mapper_to_tables.get(cls, [])
            cluster_tables[cid].update(t.replace("table:", "") for t in tables)

        return dict(clusters), dict(cluster_tables)

    for dec_id, label in [(best, "BEST"), (worst, "WORST")]:
        pairs    = all_penalties[dec_id]
        pen_sum  = pen_data_global[dec_id]
        clusters, cluster_tables = cluster_summary(dec_id, decompositions, mapper_to_tables)
        mem      = decompositions[dec_id]["membership"]
        n_clusters = len(set(mem.values()))

        print(f"\n{'='*60}")
        print(f"{label} ({dec_id})")
        print(f"  Pen_data={pen_sum:.4f}  n_clusters={n_clusters}  penalized_pairs={len(pairs)}")

        print(f"  Clusters with table access:")
        for cid, tables in sorted(cluster_tables.items()):
            if tables:
                classes = clusters.get(cid, [])
                print(f"    S{cid} {classes} → {tables}")

        print(f"  Penalized pairs:")
        for (si, sj), pen in sorted(pairs.items(), key=lambda x: x[1], reverse=True):
            ti = cluster_tables.get(si, set())
            tj = cluster_tables.get(sj, set())
            shared = ti & tj
            print(f"    S{si} ↔ S{sj}: pen={pen:.4f}  shared_tables={shared}")


        top3    = df.nlargest(3,  "NMI")[["ID", "K", "RESOLUTION", "NMI", "penalty_swm", "n_clusters"]]
        bottom3 = df.nsmallest(3, "NMI")[["ID", "K", "RESOLUTION", "NMI", "penalty_swm", "n_clusters"]]
        print("TOP 3:\n", top3.to_string())
        print("\nBOTTOM 3:\n", bottom3.to_string())

        def print_decomposition(dec_id, decompositions):
            mem = decompositions[dec_id]["membership"]
            clusters = defaultdict(list)
            for cls, cid in mem.items():
                simple = cls.replace("class:", "").split(".")[-1]
                clusters[cid].append(simple)
            print(f"\nDecomposition: {dec_id}")
            for cid, classes in sorted(clusters.items()):
                print(f"  S{cid}: {sorted(classes)}")

        print("\n=== TOP 3 ===")
        for dec_id in top3["ID"]:
            print_decomposition(dec_id, decompositions)

        print("\n=== BOTTOM 3 ===")
        for dec_id in bottom3["ID"]:
            print_decomposition(dec_id, decompositions)
