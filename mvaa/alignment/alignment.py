from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from typing import Dict, Tuple, List, Iterable, Optional, Set, Callable, Any
import math
import numpy as np

from mvaa.alignment.data_access import compute_data_signal
from mvaa.alignment.embeddings import compute_embeddings_similarity
from mvaa.alignment.structural import compute_structural_similarity
from mvaa.utils.graph import read_graphml

NodeId = str
Pair = Tuple[NodeId, NodeId]
ScoreMap = Dict[Pair, float]
AlignmentDist = Dict[NodeId, List[Tuple[NodeId, float]]]
AlignmentMap = Dict[NodeId, List[NodeId]]


def invert_scoremap(scores: ScoreMap) -> ScoreMap:
    return {(b, a): s for (a, b), s in scores.items()}


def scoremap_from_triplets(triplets: Iterable[Tuple[NodeId, NodeId, float]]) -> ScoreMap:
    out: ScoreMap = {}
    for a, b, s in triplets:
        out[(a, b)] = float(s)
    return out


def scoremap_from_records(records: Iterable[Dict[str, Any]], a_key="a", b_key="b", s_key="similarity") -> ScoreMap:
    return {(r[a_key], r[b_key]): float(r[s_key]) for r in records}

def scoremap_from_data_results(data_results: Iterable[Dict[str, Any]]) -> ScoreMap:
    return {(r["impl"], r["concept"]): float(r["sim_data"]) for r in data_results}


def normalize_scores(
        scores: ScoreMap,
        *,
        mode: str = "shift01",
        clip: Optional[Tuple[float, float]] = (0.0, 1.0),
        per_source: bool = False,
) -> ScoreMap:
    if not scores:
        return {}

    vals = np.array(list(scores.values()), dtype=float)

    if mode == "identity":
        norm_vals = vals
    elif mode == "shift01":
        norm_vals = 0.5 * (1.0 + vals)
    elif mode == "minmax":
        vmin, vmax = float(vals.min()), float(vals.max())
        if math.isclose(vmin, vmax):
            norm_vals = np.zeros_like(vals)
        else:
            norm_vals = (vals - vmin) / (vmax - vmin)
    elif mode == "zscore_sigmoid":
        mu = float(vals.mean())
        sd = float(vals.std(ddof=0))
        if math.isclose(sd, 0.0):
            z = np.zeros_like(vals)
        else:
            z = (vals - mu) / sd
        norm_vals = 1.0 / (1.0 + np.exp(-z))
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

    if clip is not None:
        lo, hi = clip
        norm_vals = np.clip(norm_vals, lo, hi)

    out: ScoreMap = {}
    for (pair, v) in zip(scores.keys(), norm_vals):
        out[pair] = float(v)
    return out


def combine_similarities(
        score_maps: Dict[str, ScoreMap],
        weights: Dict[str, float],
        *,
        default: float = 0.0,
) -> ScoreMap:
    pairs: Set[Pair] = set()
    for sm in score_maps.values():
        pairs.update(sm.keys())

    out: ScoreMap = {}
    for p in pairs:
        s = 0.0
        for k, sm in score_maps.items():
            w = float(weights.get(k, 0.0))
            if w == 0.0:
                continue
            s += w * float(sm.get(p, default))
        out[p] = s
    return out


def topk_candidates_from_scores(scores: ScoreMap, *, k: int) -> Dict[NodeId, List[NodeId]]:
    if k <= 0:
        raise ValueError("k must be > 0")

    grouped: Dict[NodeId, List[Tuple[NodeId, float]]] = {}
    for (a, b), s in scores.items():
        grouped.setdefault(a, []).append((b, float(s)))

    out: Dict[NodeId, List[NodeId]] = {}
    for a, lst in grouped.items():
        lst_sorted = sorted(lst, key=lambda x: x[1], reverse=True)
        out[a] = [b for (b, _) in lst_sorted[:k]]
    return out


def softmax_alignment(
        scores: ScoreMap,
        *,
        lambda_: float = 10.0,
        candidates: Optional[Dict[NodeId, List[NodeId]]] = None,
        eps: float = 1e-12,
) -> AlignmentDist:
    if lambda_ <= 0:
        raise ValueError("lambda_ must be > 0")

    by_a: Dict[NodeId, Dict[NodeId, float]] = {}
    for (a, b), s in scores.items():
        by_a.setdefault(a, {})[b] = float(s)

    P: AlignmentDist = {}

    for a, b_to_s in by_a.items():
        if candidates is not None:
            cand_bs = candidates.get(a, [])
            if not cand_bs:
                P[a] = []
                continue
            s_list = [b_to_s.get(b, float("-inf")) for b in cand_bs]
            bs = cand_bs
        else:
            bs = list(b_to_s.keys())
            s_list = [b_to_s[b] for b in bs]

        finite = [x for x in s_list if math.isfinite(x)]
        if not finite:
            P[a] = []
            continue

        m = max(finite)
        exps = []
        for s in s_list:
            if not math.isfinite(s):
                exps.append(0.0)
            else:
                exps.append(math.exp(lambda_ * (s - m)))

        Z = sum(exps)
        if Z <= eps:
            finite_bs = [(b, 1.0) for b, s in zip(bs, s_list) if math.isfinite(s)]
            if not finite_bs:
                P[a] = []
            else:
                u = 1.0 / len(finite_bs)
                P[a] = [(b, u) for (b, _) in finite_bs]
            continue

        P[a] = [(b, e / Z) for b, e in zip(bs, exps) if e > 0.0]

        P[a].sort(key=lambda x: x[1], reverse=True)

    return P

def select_alignment(
        P: AlignmentDist,
        *,
        tau: float = 0.2,
        min_keep: int = 0,
        max_keep: Optional[int] = None,
) -> AlignmentMap:
    if not (0.0 <= tau <= 1.0):
        raise ValueError("tau must be in [0,1]")
    if min_keep < 0:
        raise ValueError("min_keep must be >= 0")
    if max_keep is not None and max_keep <= 0:
        raise ValueError("max_keep must be > 0 or None")

    out: AlignmentMap = {}
    for a, lst in P.items():
        chosen = [b for (b, p) in lst if p >= tau]

        if min_keep > 0 and len(chosen) < min_keep:
            chosen = [b for (b, _) in lst[:min_keep]]

        if max_keep is not None:
            chosen = chosen[:max_keep]

        out[a] = chosen
    return out


def compose_distributions_via_bridge(
        P_A_B: AlignmentDist,
        P_B_C: AlignmentDist,
        *,
        mode: str = "sum",
        topk_bridge: Optional[int] = None,
        topk_out: Optional[int] = None,
        eps: float = 1e-15,
) -> AlignmentDist:
    if mode not in {"sum", "max", "product"}:
        raise ValueError("mode must be one of: sum, max, product")

    P_B_C_dict: Dict[NodeId, Dict[NodeId, float]] = {}
    for b, lst in P_B_C.items():
        P_B_C_dict[b] = {c: float(p) for (c, p) in lst}

    out: AlignmentDist = {}

    for a, lst_ab in P_A_B.items():
        if topk_bridge is not None:
            lst_ab = lst_ab[:topk_bridge]

        agg: Dict[NodeId, float] = {}

        if mode in {"sum", "product"}:
            for b, p_ba in lst_ab:
                p_ba = float(p_ba)
                if p_ba <= 0:
                    continue
                bc = P_B_C_dict.get(b)
                if not bc:
                    continue
                for c, p_cb in bc.items():
                    agg[c] = agg.get(c, 0.0) + p_ba * float(p_cb)

        elif mode == "max":
            for b, p_ba in lst_ab:
                p_ba = float(p_ba)
                if p_ba <= 0:
                    continue
                bc = P_B_C_dict.get(b)
                if not bc:
                    continue
                for c, p_cb in bc.items():
                    val = p_ba * float(p_cb)
                    prev = agg.get(c, 0.0)
                    if val > prev:
                        agg[c] = val

        # Normalize to distribution (soft normalization)
        Z = sum(agg.values())
        if Z <= eps:
            out[a] = []
            continue

        items = [(c, v / Z) for c, v in agg.items() if v > 0.0]
        items.sort(key=lambda x: x[1], reverse=True)

        if topk_out is not None:
            items = items[:topk_out]

        out[a] = items

    return out


def scoremap_from_anchors(anchors: Set[Pair], *, default: float = 0.0) -> ScoreMap:
    return {pair: 1.0 for pair in anchors}


def scoremap_access_IT(mapper_to_tables: Dict[NodeId, Set[NodeId]], *, weight: float = 1.0) -> ScoreMap:
    out: ScoreMap = {}
    for i, tables in mapper_to_tables.items():
        for t in tables:
            out[(i, t)] = float(weight)
    return out


def scoremap_tables_for_concept_DT(entity_to_tables: Dict[NodeId, Set[NodeId]], *, weight: float = 1.0) -> ScoreMap:
    out: ScoreMap = {}
    for d, tables in entity_to_tables.items():
        for t in tables:
            out[(d, t)] = float(weight)
    return out


@dataclass(frozen=True)
class AlignParams:
    lambda_: float = 10.0
    tau: float = 0.2
    topk: int = 50
    min_keep: int = 0
    max_keep: Optional[int] = None


def run_alignment_from_signals(
        *,
        signals: Dict[str, ScoreMap],
        weights: Dict[str, float],
        norm_modes: Dict[str, str],
        params: AlignParams,
        default_missing: float = 0.0,
) -> Tuple[ScoreMap, AlignmentDist, AlignmentMap]:
   # 1) normalize signals
    normed: Dict[str, ScoreMap] = {}
    for name, sm in signals.items():
        mode = norm_modes.get(name, "identity")
        normed[name] = normalize_scores(sm, mode=mode)

    # 2) combine
    sim = combine_similarities(normed, weights, default=default_missing)

    # 3) candidates
    cand = topk_candidates_from_scores(sim, k=params.topk)

    # 4) softmax
    P = softmax_alignment(sim, lambda_=params.lambda_, candidates=cand)

    # 5) selection
    Psi = select_alignment(P, tau=params.tau, min_keep=params.min_keep, max_keep=params.max_keep)

    return sim, P, Psi


@dataclass
class MultiViewAlignments:
    D_to_I: AlignmentMap
    I_to_D: AlignmentMap
    I_to_T: AlignmentMap
    T_to_I: AlignmentMap
    D_to_T: AlignmentMap
    T_to_D: AlignmentMap

    I_to_D_via_T: Optional[AlignmentMap] = None
    D_to_I_via_T: Optional[AlignmentMap] = None

    P_D_I: Optional[AlignmentDist] = None
    P_I_D: Optional[AlignmentDist] = None
    P_I_T: Optional[AlignmentDist] = None
    P_T_I: Optional[AlignmentDist] = None
    P_D_T: Optional[AlignmentDist] = None
    P_T_D: Optional[AlignmentDist] = None

    P_I_D_via_T: Optional[AlignmentDist] = None
    P_D_I_via_T: Optional[AlignmentDist] = None


def run_all_alignments(
        *,
        emb_DI: ScoreMap,
        emb_D_T: ScoreMap,
        emb_I_T: ScoreMap,
        struct_DI: Optional[ScoreMap] = None,
        struct_D_T: Optional[ScoreMap] = None,
        struct_I_T: Optional[ScoreMap] = None,


        sim_data_ID: Optional[ScoreMap] = None,
        access_I_T: Optional[ScoreMap] = None,

        tables_for_D_T: Optional[ScoreMap] = None,


        anchor_DI: Optional[ScoreMap] = None,
        anchor_I_T: Optional[ScoreMap] = None,
        anchor_D_T: Optional[ScoreMap] = None,

        weights_DI: Optional[Dict[str, float]] = None,
        weights_I_T: Optional[Dict[str, float]] = None,
        weights_D_T: Optional[Dict[str, float]] = None,

        norms_DI: Optional[Dict[str, str]] = None,
        norms_I_T: Optional[Dict[str, str]] = None,
        norms_D_T: Optional[Dict[str, str]] = None,

        params_DI: AlignParams = AlignParams(),
        params_I_T: AlignParams = AlignParams(),
        params_D_T: AlignParams = AlignParams(),


        compose_via_T: bool = True,
        compose_mode: str = "sum",
        compose_topk_bridge: int = 25,
        compose_topk_out: int = 50,
        compose_tau: Optional[float] = None,
) -> MultiViewAlignments:

    if weights_DI is None:
        weights_DI = {"emb": 0.55, "struct": 0.20, "data": 0.25, "anchor": 1.00}
    if weights_I_T is None:
        weights_I_T = {"access": 1.00, "emb": 0.20, "struct": 0.10, "anchor": 1.00}
    if weights_D_T is None:
        weights_D_T = {"tables": 0.70, "emb": 0.30, "struct": 0.10, "anchor": 1.00}


    if norms_DI is None:
        norms_DI = {"emb": "shift01", "struct": "shift01", "data": "identity", "anchor": "identity"}
    if norms_I_T is None:
        norms_I_T = {"access": "identity", "emb": "shift01", "struct": "shift01", "anchor": "identity"}
    if norms_D_T is None:
        norms_D_T = {"tables": "identity", "emb": "shift01", "struct": "shift01", "anchor": "identity"}


    signals_DI: Dict[str, ScoreMap] = {"emb": emb_DI}
    if struct_DI is not None:
        signals_DI["struct"] = struct_DI
    if sim_data_ID is not None:
        signals_DI["data"] = sim_data_ID
    if anchor_DI is not None:
        signals_DI["anchor"] = anchor_DI

    sim_DI, P_D_I, Psi_D_I = run_alignment_from_signals(
        signals=signals_DI,
        weights=weights_DI,
        norm_modes=norms_DI,
        params=params_DI,
        default_missing=0.0,
    )

    signals_ID: Dict[str, ScoreMap] = {k: invert_scoremap(v) for k, v in signals_DI.items()}
    sim_ID, P_I_D, Psi_I_D = run_alignment_from_signals(
        signals=signals_ID,
        weights=weights_DI,
        norm_modes=norms_DI,
        params=params_DI,
        default_missing=0.0,
    )

    signals_I_T: Dict[str, ScoreMap] = {"emb": emb_I_T}
    if access_I_T is not None:
        signals_I_T["access"] = access_I_T
    if struct_I_T is not None:
        signals_I_T["struct"] = struct_I_T
    if anchor_I_T is not None:
        signals_I_T["anchor"] = anchor_I_T

    sim_IT, P_I_T, Psi_I_T = run_alignment_from_signals(
        signals=signals_I_T,
        weights=weights_I_T,
        norm_modes=norms_I_T,
        params=params_I_T,
        default_missing=0.0,
    )

    signals_T_I: Dict[str, ScoreMap] = {k: invert_scoremap(v) for k, v in signals_I_T.items()}
    sim_TI, P_T_I, Psi_T_I = run_alignment_from_signals(
        signals=signals_T_I,
        weights=weights_I_T,
        norm_modes=norms_I_T,
        params=params_I_T,
        default_missing=0.0,
    )

    signals_D_T: Dict[str, ScoreMap] = {"emb": emb_D_T}
    if tables_for_D_T is not None:
        signals_D_T["tables"] = tables_for_D_T
    if struct_D_T is not None:
        signals_D_T["struct"] = struct_D_T
    if anchor_D_T is not None:
        signals_D_T["anchor"] = anchor_D_T

    sim_DT, P_D_T, Psi_D_T = run_alignment_from_signals(
        signals=signals_D_T,
        weights=weights_D_T,
        norm_modes=norms_D_T,
        params=params_D_T,
        default_missing=0.0,
    )

    signals_T_D: Dict[str, ScoreMap] = {k: invert_scoremap(v) for k, v in signals_D_T.items()}
    sim_TD, P_T_D, Psi_T_D = run_alignment_from_signals(
        signals=signals_T_D,
        weights=weights_D_T,
        norm_modes=norms_D_T,
        params=params_D_T,
        default_missing=0.0,
    )

    I_to_D_via_T: Optional[AlignmentMap] = None
    D_to_I_via_T: Optional[AlignmentMap] = None
    P_I_D_via_T: Optional[AlignmentDist] = None
    P_D_I_via_T: Optional[AlignmentDist] = None

    if compose_via_T:
        P_I_D_via_T = compose_distributions_via_bridge(
            P_A_B=P_I_T,
            P_B_C=P_T_D,
            mode=compose_mode,
            topk_bridge=compose_topk_bridge,
            topk_out=compose_topk_out,
        )
        P_D_I_via_T = compose_distributions_via_bridge(
            P_A_B=P_D_T,
            P_B_C=P_T_I,
            mode=compose_mode,
            topk_bridge=compose_topk_bridge,
            topk_out=compose_topk_out,
        )

        tau_comp = params_DI.tau if compose_tau is None else compose_tau
        I_to_D_via_T = select_alignment(P_I_D_via_T, tau=tau_comp, min_keep=params_DI.min_keep, max_keep=params_DI.max_keep)
        D_to_I_via_T = select_alignment(P_D_I_via_T, tau=tau_comp, min_keep=params_DI.min_keep, max_keep=params_DI.max_keep)

    return MultiViewAlignments(
        D_to_I=Psi_D_I,
        I_to_D=Psi_I_D,
        I_to_T=Psi_I_T,
        T_to_I=Psi_T_I,
        D_to_T=Psi_D_T,
        T_to_D=Psi_T_D,
        I_to_D_via_T=I_to_D_via_T,
        D_to_I_via_T=D_to_I_via_T,
        P_D_I=P_D_I,
        P_I_D=P_I_D,
        P_I_T=P_I_T,
        P_T_I=P_T_I,
        P_D_T=P_D_T,
        P_T_D=P_T_D,
        P_I_D_via_T=P_I_D_via_T,
        P_D_I_via_T=P_D_I_via_T,
    )


def glue_from_current_outputs(
        *,
        align_DI_records: Iterable[Dict[str, Any]],
        align_DD_records: Iterable[Dict[str, Any]],
        align_ID_records: Iterable[Dict[str, Any]],
        struct_DI_scores: Optional[ScoreMap] = None,
        struct_DD_scores: Optional[ScoreMap] = None,
        struct_ID_scores: Optional[ScoreMap] = None,
        sim_data_items: Optional[Iterable[Dict[str, Any]]] = None,  # expects {'concept':..., 'impl':..., 'sim_data':...}
        access_IT: Optional[ScoreMap] = None,
) -> Dict[str, ScoreMap]:
    emb_DI = scoremap_from_records(align_DI_records, a_key="a", b_key="b", s_key="similarity")
    emb_D_T = scoremap_from_records(align_DD_records, a_key="a", b_key="b", s_key="similarity")
    emb_I_T = scoremap_from_records(align_ID_records, a_key="a", b_key="b", s_key="similarity")

    out = {
        "emb_DI": emb_DI,
        "emb_D_T": emb_D_T,
        "emb_I_T": emb_I_T,
    }

    if struct_DI_scores is not None:
        out["struct_DI"] = struct_DI_scores
    if struct_DD_scores is not None:
        out["struct_D_T"] = struct_DD_scores
    if struct_ID_scores is not None:
        out["struct_I_T"] = struct_ID_scores

    if sim_data_items is not None:
        trip = [(d["concept"], d["impl"], float(d["sim_data"])) for d in sim_data_items]
        out["data_D_I"] = scoremap_from_triplets(trip)

    if access_IT is not None:
        out["access_I_T"] = access_IT

    return out


if __name__ == "__main__":
    import os
    import argparse
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[2])
    parser = argparse.ArgumentParser(description="Multi-view alignment analysis")
    parser.add_argument("--app", default="jpetstore",
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
    with open(f"monoliths/{project}/entity_to_tables.json", "r") as f:
        entity_to_tables = json.load(f)
        entity_to_tables = {key: set(values) for key, values in entity_to_tables.items()}
    with open(f"monoliths/{project}/mapper_to_tables.json", "r") as f:
        mapper_to_tables = json.load(f)
        mapper_to_tables = {key: set(values) for key, values in mapper_to_tables.items()}

    sim_struct_DD, sim_struct_DI = compute_structural_similarity(G_design, G_impl, G_data)
    align_DI, align_DD, align_ID = compute_embeddings_similarity(G_design, G_impl, G_data)
    sim_data = compute_data_signal(G_design=G_design, entity_to_tables=entity_to_tables, G_impl=G_impl,Access=mapper_to_tables)

    emb_DI = scoremap_from_records(align_DI)
    emb_D_T = scoremap_from_records(align_DD)
    emb_I_T = scoremap_from_records(align_ID)

    data_I_D = scoremap_from_data_results(sim_data )
    data_D_I = invert_scoremap(data_I_D)

    access_I_T = scoremap_access_IT(mapper_to_tables)

    params = AlignParams(lambda_=12.0, tau=0.25, topk=40, min_keep=0, max_keep=None)

    results = run_all_alignments(
        emb_DI=emb_DI,
        emb_D_T=emb_D_T,
        emb_I_T=emb_I_T,
        struct_DI=sim_struct_DI,
        struct_D_T=sim_struct_DD,
        sim_data_ID=data_D_I,
        access_I_T=access_I_T,
        params_DI=params,
        params_I_T=params,
        params_D_T=params,
        compose_via_T=True,
    )

    with open(f"monoliths/{project}/alignment_results.pkl", "wb") as f:
        pickle.dump(results, f)

    print(results)





