from __future__ import annotations

import json
import pickle
import copy
from math import log
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mvaa.utils.graph import read_graphml
from mvaa.bc_induction.bc_induction import infer_canonical_bcs_soft
from mvaa.alignment.alignment import MultiViewAlignments
from mvaa.utils.decompositions import build_dict_from_directory


def build_class_to_bc_from_graph(G, bc_list: List[str]) -> Dict[str, Tuple[str, str]]:
    result = {}
    for n, data in G.nodes(data=True):
        bc_probs = data.get("bc_probs", {})
        if not bc_probs:
            continue
        dominant = max(bc_probs, key=bc_probs.get)
        result[str(n)] = dominant
    return result


def build_impl_class_to_bc(
        P_u_to_c: Dict[str, List[Tuple[str, float]]],
        concept_to_bc: Dict[str, str],
) -> Dict[str, str]:
    result = {}
    for cls, alignments in P_u_to_c.items():
        bc_scores: Dict[str, float] = {}
        for item in alignments:
            if isinstance(item, (list, tuple)):
                concept, weight = item[0], item[1]
            else:
                concept, weight = item["concept"], item["weight"]
            bc = concept_to_bc.get(concept)
            if bc is None:
                continue
            bc_scores[bc] = bc_scores.get(bc, 0.0) + weight
        if bc_scores:
            result[cls] = max(bc_scores, key=bc_scores.get)
    return result


def nmi_decomposition(
        membership: Dict[str, int],
        class_to_bc: Dict[str, str],
) -> float:
    pairs = []
    for cls, cluster in membership.items():
        bc = class_to_bc.get(cls) or class_to_bc.get(cls.split(".")[-1])
        if bc is None:
            continue
        pairs.append((str(cluster), bc))

    if not pairs:
        return 0.0

    n = len(pairs)
    clusters = [p[0] for p in pairs]
    bcs      = [p[1] for p in pairs]

    p_c     = Counter(clusters)
    p_bc    = Counter(bcs)
    p_joint = Counter(pairs)

    mi = 0.0
    for (c, bc), cnt in p_joint.items():
        p_ij = cnt / n
        p_i  = p_c[c] / n
        p_j  = p_bc[bc] / n
        mi  += p_ij * log(p_ij / (p_i * p_j) + 1e-12)

    H_c  = -sum((v/n) * log(v/n + 1e-12) for v in p_c.values())
    H_bc = -sum((v/n) * log(v/n + 1e-12) for v in p_bc.values())

    return 2 * mi / (H_c + H_bc) if (H_c + H_bc) > 0 else 0.0


def make_seed_variants(seeds_full: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    by_bc: Dict[str, List[str]] = defaultdict(list)
    for node, bc in seeds_full.items():
        by_bc[bc].append(node)

    seeds_restricted = {}
    for bc, nodes in by_bc.items():
        for node in nodes[:2]:
            seeds_restricted[node] = bc

    return {
        "full":       seeds_full,
        "restricted": seeds_restricted,
        "extended":   dict(seeds_full),
    }


def run_sensitivity(
        project: str,
        seeds_path: str,
        graphml_design_path: str,
        graphml_impl_path: str,
        decompositions_dir: str,
        p_u_to_c_path: str,
        results_csv_path: str,
        *,
        propagation_params: Dict[str, Any] | None = None,
        extended_seeds_extra: Dict[str, str] | None = None,
):

    with open(seeds_path) as f:
        config = json.load(f)
    bc_list     = config["bc_list"]
    seeds_full  = config["seeds"]

    G_design = read_graphml(graphml_design_path)

    with open(p_u_to_c_path, "rb") as f:
        alignment = pickle.load(f)

    P_u_to_c = alignment.P_I_D

    decompositions = build_dict_from_directory(decompositions_dir)
    results_df     = pd.read_csv(results_csv_path)


    variants = make_seed_variants(seeds_full)
    if extended_seeds_extra:
        variants["extended"].update(extended_seeds_extra)


    prop_params = dict(
        iters=60,
        undirected_propagation=True,
        teleport=0.12,
        seed_strength=0.95,
        seed_anchor=0.92,
        alpha_llm=0.25,
    )
    if propagation_params:
        prop_params.update(propagation_params)


    nmi_by_variant: Dict[str, List[float]] = {}

    for variant_name, seeds in variants.items():
        print(f"\n{'='*50}")
        print(f"Variant: {variant_name} ({len(seeds)} seeds)")

        G = copy.deepcopy(G_design)

        infer_canonical_bcs_soft(
            G,
            bc_list=bc_list,
            seeds=seeds,
            **prop_params,
        )

        concept_to_bc = build_class_to_bc_from_graph(G, bc_list)
        print(f"  BC distribution: { {bc: sum(1 for v in concept_to_bc.values() if v == bc) for bc in bc_list} }")

        class_to_bc = build_impl_class_to_bc(P_u_to_c, concept_to_bc)
        print(f"  Classes with BC: {len(class_to_bc)}")

        nmis = []
        for dec_id, payload in decompositions.items():
            nmi = nmi_decomposition(payload["membership"], class_to_bc)
            nmis.append(nmi)

        nmi_by_variant[variant_name] = nmis
        print(f"  NMI: min={min(nmis):.4f}  max={max(nmis):.4f}  "
              f"std={np.std(nmis):.4f}  mean={np.mean(nmis):.4f}")

    print(f"\n{'='*50}")
    print(f"SENSITIVITY SUMMARY — {project.upper()}")
    print(f"{'Variant':<12} {'Min':>8} {'Max':>8} {'Std':>8} {'Mean':>8} {'Range':>8}")
    print("-" * 55)
    for variant_name, nmis in nmi_by_variant.items():
        print(f"{variant_name:<12} {min(nmis):>8.4f} {max(nmis):>8.4f} "
              f"{np.std(nmis):>8.4f} {np.mean(nmis):>8.4f} "
              f"{max(nmis)-min(nmis):>8.4f}")

    print(f"\nSpearman correlation between variant NMI rankings:")
    variant_names = list(nmi_by_variant.keys())
    for i in range(len(variant_names)):
        for j in range(i+1, len(variant_names)):
            a, b = variant_names[i], variant_names[j]
            rho, p = spearmanr(nmi_by_variant[a], nmi_by_variant[b])
            print(f"  {a} vs {b}: rho={rho:.3f}  p={p:.4f}")

    print(f"\nTop-10 overlap between variants:")
    dec_ids = list(decompositions.keys())
    top10_by_variant = {}
    for variant_name, nmis in nmi_by_variant.items():
        ranked = sorted(zip(nmis, dec_ids), reverse=True)
        top10_by_variant[variant_name] = set(d for _, d in ranked[:10])

    for i in range(len(variant_names)):
        for j in range(i+1, len(variant_names)):
            a, b = variant_names[i], variant_names[j]
            overlap = len(top10_by_variant[a] & top10_by_variant[b])
            print(f"  {a} vs {b}: {overlap}/10 decompositions in common")

    return nmi_by_variant


if __name__ == "__main__":
    import os
    import argparse
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[2])
    parser = argparse.ArgumentParser(description="Bounded context induction sensitivity analysis")
    parser.add_argument("--app", default="jpetstore",
                        choices=["cargo", "jpetstore", "daytrader"],
                        help="Monolith system to analyse")
    args = parser.parse_args()
    project = args.app

    G = read_graphml(f"monoliths/{project}/vista_disenio_{project}.graphml")
    print(sorted(G.nodes()))

    config = {
        "cargo": {
            "seeds_path":         f"monoliths/{project}/bc.json",
            "graphml_design":     f"monoliths/{project}/vista_disenio_{project}.graphml",
            "graphml_impl":       f"monoliths/{project}/vista_implementacion_{project}_c.graphml",
            "decompositions_dir": f"monoliths/{project}/mid_results/services",
            "p_u_to_c_path":      f"monoliths/{project}/alignment_results.pkl",
            "results_csv":        f"results/RQ1/metrics_{project}.csv",
            "extended_extra": {
                "delivery#0":  "booking",
                "delivery#1":  "tracking",
                "identifier#new_32f8fc64": "booking",
            },
        },
        "jpetstore": {
            "seeds_path":         f"monoliths/{project}/bc.json",
            "graphml_design":     f"monoliths/{project}/vista_disenio_{project}.graphml",
            "graphml_impl":       f"monoliths/{project}/vista_implementacion_{project}_c.graphml",
            "decompositions_dir": f"monoliths/{project}/mid_results/services",
            "p_u_to_c_path":      f"monoliths/{project}/alignment_results.pkl",
            "results_csv":        f"results/RQ1/metrics_{project}.csv",
            "extended_extra": {},
        },
        "daytrader": {
            "seeds_path":         f"monoliths/{project}/bc.json",
            "graphml_design":     f"monoliths/{project}/vista_disenio_{project}.graphml",
            "graphml_impl":       f"monoliths/{project}/vista_implementacion_{project}_c.graphml",
            "decompositions_dir": f"monoliths/{project}/mid_results/services",
            "p_u_to_c_path":      f"monoliths/{project}/alignment_results.pkl",
            "results_csv":        f"results/RQ1/metrics_{project}.csv",
            "extended_extra": {
                "current value#0":              "portfolio",
                "purchase price#0":             "portfolio",
                "quantity#0":                   "portfolio",
                "synchronous asynchronous order#0": "trade",
                "personal information#0":       "account",
                "initial balance#0":            "account",
                "account balance#0":            "account",
            },
        },
    }[project]

    nmi_by_variant = run_sensitivity(
        project=project,
        seeds_path=config["seeds_path"],
        graphml_design_path=config["graphml_design"],
        graphml_impl_path=config["graphml_impl"],
        decompositions_dir=config["decompositions_dir"],
        p_u_to_c_path=config["p_u_to_c_path"],
        results_csv_path=config["results_csv"],
        extended_seeds_extra=config.get("extended_extra", {}),
    )
