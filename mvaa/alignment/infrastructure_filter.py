from __future__ import annotations

import json
import pickle
from collections import defaultdict, Counter
from math import log
from typing import Dict, List, Set

import pandas as pd
from scipy.stats import spearmanr

from mvaa.utils.decompositions import build_dict_from_directory
from mvaa.utils.graph import read_graphml
from mvaa.alignment.alignment import MultiViewAlignments

INFRA_FILTER: Dict[str, Set[str]] = {
    "cargo": {
        "class:com.pathfinder.internal.GraphDAOStub",
        "class:se.citerus.dddsample.config.DDDSampleApplicationContext",
        "class:se.citerus.dddsample.domain.model.voyage.Voyage.Builder",
        "class:se.citerus.dddsample.infrastructure.messaging.jms.InfrastructureMessagingJmsConfig",
        "class:se.citerus.dddsample.infrastructure.sampledata.SampleDataGenerator",
        "class:se.citerus.dddsample.infrastructure.sampledata.SampleLocations",
        "class:se.citerus.dddsample.infrastructure.sampledata.SampleVoyages",
    },
    "jpetstore": {
        "class:org.mybatis.jpetstore.web.actions.AbstractActionBean",
    },
    "daytrader": {
        "class:com.ibm.websphere.samples.daytrader.util.Log",
        "class:com.ibm.websphere.samples.daytrader.util.TradeConfig",
    },
}

def nmi_decomposition(
        membership: Dict[str, int],
        class_to_bc: Dict[str, str],
        exclude: Set[str] | None = None,
) -> float:
    exclude = exclude or set()
    pairs = []
    for cls, cluster in membership.items():
        if cls in exclude:
            continue
        bc = class_to_bc.get(cls)
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

def entropy_normalized(dist: Dict[str, float], n_bcs: int) -> float:
    if n_bcs <= 1:
        return 0.0
    h = -sum(p * log(p + 1e-12) for p in dist.values() if p > 0)
    return h / log(n_bcs)


def pen_bc_decomposition(
        membership: Dict[str, int],
        class_bc_probs: Dict[str, Dict[str, float]],
        bc_list: List[str],
        exclude: Set[str] | None = None,
) -> float:
    exclude = exclude or set()
    services: Dict[int, List[str]] = defaultdict(list)
    for cls, svc in membership.items():
        if cls in exclude:
            continue
        if cls in class_bc_probs:
            services[svc].append(cls)

    total = sum(len(v) for v in services.values())
    if total == 0:
        return 0.0

    pen = 0.0
    for svc, classes in services.items():
        avg: Dict[str, float] = {bc: 0.0 for bc in bc_list}
        for cls in classes:
            probs = class_bc_probs[cls]
            for bc in bc_list:
                avg[bc] += probs.get(bc, 0.0)
        avg = {bc: v / len(classes) for bc, v in avg.items()}
        h = entropy_normalized(avg, len(bc_list))
        pen += len(classes) * h

    return pen / total


def run_infra_filter_analysis(project: str):
    print(f"\n{'='*60}")
    print(f"INFRA FILTER ANALYSIS — {project.upper()}")
    print(f"{'='*60}")

    infra_classes = INFRA_FILTER.get(project, set())
    print(f"Filtering {len(infra_classes)} infrastructure classes")

    decompositions = build_dict_from_directory(
        f"monoliths/{project}/mid_results/services"
    )

    with open(f"monoliths/{project}/bc.json") as f:
        config = json.load(f)
    bc_list = config["bc_list"]

    G_design = read_graphml(
        f"monoliths/{project}/vista_disenio_{project}.graphml"
    )

    with open(f"monoliths/{project}/alignment_results.pkl", "rb") as f:
        ar = pickle.load(f)
    P_I_D = ar.P_I_D  # {class_id: [(concept, weight), ...]}


    concept_bc: Dict[str, Dict[str, float]] = {}
    for n, data in G_design.nodes(data=True):
        probs = data.get("bc_probs")
        if probs:
            concept_bc[str(n)] = probs

    # P(BC|u) = sum_c P(BC|c) * P(c|u)
    class_bc_probs: Dict[str, Dict[str, float]] = {}
    for cls, alignments in P_I_D.items():
        bc_scores = {bc: 0.0 for bc in bc_list}
        for concept, weight in alignments:
            cbc = concept_bc.get(concept)
            if cbc is None:
                continue
            for bc in bc_list:
                bc_scores[bc] += weight * cbc.get(bc, 0.0)
        total = sum(bc_scores.values())
        if total > 0:
            class_bc_probs[cls] = {bc: v/total for bc, v in bc_scores.items()}

    print(f"Classes with BC probs: {len(class_bc_probs)}")

    class_to_bc = {
        cls: max(probs, key=probs.get)
        for cls, probs in class_bc_probs.items()
    }

    results = []
    for dec_id, payload in decompositions.items():
        mem = payload["membership"]

        nmi_full    = nmi_decomposition(mem, class_to_bc, exclude=set())
        nmi_filtered = nmi_decomposition(mem, class_to_bc, exclude=infra_classes)

        pen_full     = pen_bc_decomposition(mem, class_bc_probs, bc_list, exclude=set())
        pen_filtered = pen_bc_decomposition(mem, class_bc_probs, bc_list, exclude=infra_classes)

        results.append({
            "dec_id":       dec_id,
            "nmi_full":     nmi_full,
            "nmi_filtered": nmi_filtered,
            "pen_full":     pen_full,
            "pen_filtered": pen_filtered,
        })

    df = pd.DataFrame(results)

    print(f"\n{'Metric':<20} {'Full':>10} {'Filtered':>10} {'Delta mean':>12}")
    print("-" * 55)
    for metric in ["nmi", "pen"]:
        full_col     = f"{metric}_full"
        filtered_col = f"{metric}_filtered"
        delta = (df[filtered_col] - df[full_col]).mean()
        print(f"{metric.upper():<20} "
              f"mean={df[full_col].mean():.4f}  "
              f"mean={df[filtered_col].mean():.4f}  "
              f"Δ={delta:+.4f}")
        print(f"  std:  {'':<14} {df[full_col].std():.4f}  "
              f"{'':>5}{df[filtered_col].std():.4f}")
        print(f"  range:{'':<13} {df[full_col].max()-df[full_col].min():.4f}  "
              f"{'':>5}{df[filtered_col].max()-df[filtered_col].min():.4f}")

    rho_nmi, p_nmi = spearmanr(df["nmi_full"], df["nmi_filtered"])
    rho_pen, p_pen = spearmanr(df["pen_full"], df["pen_filtered"])
    print(f"\nRank correlation full vs filtered:")
    print(f"  NMI: rho={rho_nmi:.3f}  p={p_nmi:.4f}")
    print(f"  Pen_bc: rho={rho_pen:.3f}  p={p_pen:.4f}")

    top10_full     = set(df.nlargest(10, "nmi_full")["dec_id"])
    top10_filtered = set(df.nlargest(10, "nmi_filtered")["dec_id"])
    print(f"\nTop-10 NMI overlap: {len(top10_full & top10_filtered)}/10")

    return df


if __name__ == "__main__":
    import os
    import argparse
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[2])
    parser = argparse.ArgumentParser(description="Infrastructure filter analysis")
    parser.add_argument("--apps", nargs="+", default=["cargo", "jpetstore", "daytrader"],
                        metavar="APP", help="Systems to include (default: all three)")
    args = parser.parse_args()
    for project in args.apps:
        run_infra_filter_analysis(project)
