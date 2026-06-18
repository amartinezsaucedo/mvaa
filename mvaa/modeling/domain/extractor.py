import argparse
import json
import logging
import os
import re
import uuid
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional

import time

import nest_asyncio
import networkx as nx
import numpy as np
from community import community_louvain
from dotenv import load_dotenv
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.groq import Groq
from nltk.tokenize import sent_tokenize
from pydantic import BaseModel, field_validator
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from typing import List as TList
import spacy

nest_asyncio.apply()
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("design-view")


SPACY_MODEL      = "en_core_web_sm"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
GROQ_MODEL       = "llama-3.3-70b-versatile"
GROQ_TEMPERATURE = 0.0

WEIGHT_DEP_PATH      = 1.0
WEIGHT_SRL           = 2.0
WEIGHT_EMBEDDING     = 1.0
EMBED_SIM_THRESHOLD  = 0.85
MIN_CONCEPT_FREQ     = 1
LOUVAIN_RESOLUTION   = 0.6

ALPHA = 1.0
BETA  = 0.8
GAMMA = 2.0
DELTA = 1.0
MIN_WEIGHT_THRESHOLD = 1e-3

LLM_MAX_RETRIES   = 4
LLM_RETRY_WAIT_S  = 15  # base wait; doubles each retry

AGENT_LEMMAS = {"operator", "user", "actor", "staff", "clerk",
                "system", "application", "service", "i", "we", "you"}


nlp      = None
embedder = None
llm      = None


def load_models():
    global nlp, embedder, llm
    nlp = spacy.load(SPACY_MODEL)
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    llm = Groq(
        model=GROQ_MODEL,
        temperature=GROQ_TEMPERATURE,
        pydantic_program_mode="llm",
        api_key=os.getenv("GROQ_API_KEY"),
    )
    logger.info(f"Loaded spaCy={SPACY_MODEL}, embedder={EMBEDDING_MODEL}, llm={GROQ_MODEL}")



def load_inputs(monolith_dir: str) -> tuple[list[str], list[dict]]:
    req_path = os.path.join(monolith_dir, "requirements_processed.txt")
    srl_path = os.path.join(monolith_dir, "srl.json")
    with open(req_path) as f:
        reqs = sent_tokenize(f.read())
    with open(srl_path) as f:
        srl = json.load(f)
    return reqs, srl


def preprocess_requirements(reqs: list[str], srl: list[dict]) -> list[tuple]:
    return [(i + 1, s, srl[i]) for i, s in enumerate(reqs)]



def extract_srl_spans(description):
    spans = []
    for m in re.findall(r"\[(.*?)\]", description):
        if ":" not in m:
            continue
        role, text = m.split(":", 1)
        spans.append((role.strip(), text.strip()))
    return spans


def np_candidates(span_text):
    doc = nlp(span_text)
    candidates = []
    for chunk in doc.noun_chunks:
        candidates.append(chunk.text)
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN"} and token.text not in candidates:
            candidates.append(token.text)
    return candidates


def canonicalize(np_text):
    m = re.match(r"(.*?)\((.*?)\)", np_text)
    if m:
        return [canonicalize(m.group(1).strip()), canonicalize(m.group(2).strip())]
    doc = nlp(np_text)
    tokens = [t.lemma_.lower() for t in doc if not t.is_stop and t.is_alpha]
    return [" ".join(tokens)] if tokens else []


def flatten(lst):
    result = []
    for x in lst:
        if isinstance(x, list):
            result.extend(flatten(x))
        else:
            result.append(x)
    return result


def extract_concept_mentions_from_srl(requirements, srl_json):
    mentions, embeddings_list, seen = [], [], set()
    for sent_idx, sent_obj in enumerate(srl_json):
        if "verbs" not in sent_obj:
            continue
        sentence = requirements[sent_idx]
        for frame in sent_obj["verbs"]:
            for role, span_text in extract_srl_spans(frame.get("description", "")):
                if role == "V":
                    continue
                for np_text in np_candidates(span_text):
                    for lemma in flatten(canonicalize(np_text)):
                        lemma = lemma.strip()
                        if not lemma:
                            continue
                        key = (lemma, sentence, role, np_text.strip().lower())
                        if key in seen:
                            continue
                        context = f"{np_text} | {sentence}"
                        embeddings_list.append(embedder.encode(context))
                        mentions.append((lemma, sentence))
                        seen.add(key)
    if embeddings_list:
        return mentions, np.vstack(embeddings_list)
    return mentions, np.zeros((0, embedder.get_sentence_embedding_dimension()))



def disambiguate_senses(mentions, embeddings,
                        min_samples=2,
                        base_eps_percentiles=(0.1, 0.2, 0.3, 0.4),
                        max_eps=0.6,
                        use_pca=True,
                        pca_dim=50,
                        min_mentions_for_clustering=3,
                        min_silhouette=0.2,
                        min_k_for_split=8,
                        min_silhouette_small_k=0.35,
                        collapse_small_cluster_threshold=3,
                        collapse_small_cluster_ratio=0.5):
    sense_map, sense_meta = {}, {}
    lemma_to_idxs = defaultdict(list)
    for i, (lemma, _) in enumerate(mentions):
        lemma_to_idxs[lemma].append(i)

    for lemma, idxs in lemma_to_idxs.items():
        k = len(idxs)
        X = embeddings[idxs]

        if k < min_mentions_for_clustering or k < min_k_for_split:
            label = f"{lemma}#0"
            for gi in idxs:
                sense_map[gi] = label
            sense_meta[label] = {"members": idxs, "centroid": X.mean(axis=0),
                                 "avg_sim": 1.0, "silhouette": 1.0, "num_mentions": k}
            continue

        Xc = X
        if use_pca and X.shape[1] > 5:
            n_comp = min(pca_dim, X.shape[1], X.shape[0] - 1)
            if n_comp >= 2:
                Xc = PCA(n_components=n_comp).fit_transform(X)

        sim  = cosine_similarity(Xc)
        dist = np.clip(1 - sim, 0.0, 1.0)
        np.fill_diagonal(dist, 0.0)

        labels, best_sil = None, -1.0
        upper_tri = dist[np.triu_indices(k, k=1)]
        if upper_tri.size == 0:
            labels, best_sil = np.zeros(k, dtype=int), 1.0
        else:
            for p in base_eps_percentiles:
                eps = float(np.clip(np.percentile(upper_tri, p * 100), 0.02, max_eps))
                db  = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
                candidate = db.fit_predict(dist)
                if np.all(candidate == -1) or len(set(candidate)) <= 1:
                    continue
                try:
                    sil = silhouette_score(dist, candidate, metric="precomputed")
                except Exception:
                    sil = -1
                if sil > best_sil:
                    best_sil, labels = sil, candidate

        min_sil_local = min_silhouette_small_k if k < 15 else min_silhouette
        if labels is None or len(set(labels)) <= 1 or best_sil < min_sil_local:
            labels, best_sil = np.zeros(k, dtype=int), 1.0

        labels = np.asarray(labels).reshape(-1)
        noise_mask = labels == -1
        if np.any(noise_mask):
            maj = Counter(labels[~noise_mask]).most_common(1)[0][0]
            labels[noise_mask] = maj

        unique   = sorted(set(labels))
        mapping  = {old: new for new, old in enumerate(unique)}
        norm_lbl = np.array([mapping[int(l)] for l in labels], dtype=int)

        sizes = Counter(norm_lbl)
        if k <= 12 and len(sizes) >= 4:
            norm_lbl[:] = 0
            sizes = Counter(norm_lbl)
            best_sil = 1.0
        if len(sizes) > 1:
            small = sum(1 for s in sizes.values() if s < collapse_small_cluster_threshold)
            if small / len(sizes) > collapse_small_cluster_ratio:
                norm_lbl[:] = 0
                best_sil = 1.0

        for local_i, lab in enumerate(norm_lbl):
            sense_map[idxs[local_i]] = f"{lemma}#{int(lab)}"

        cluster_members = defaultdict(list)
        for local_i, lab in enumerate(norm_lbl):
            cluster_members[int(lab)].append(idxs[local_i])

        for lab, members in cluster_members.items():
            centroid = embeddings[members].mean(axis=0)
            avg_sim  = float(np.mean([
                cosine_similarity([embeddings[i]], [centroid])[0, 0] for i in members
            ]))
            sense_meta[f"{lemma}#{int(lab)}"] = {
                "members": members, "centroid": centroid,
                "avg_sim": avg_sim, "silhouette": float(best_sil),
                "num_mentions": len(members), "label": lemma,
            }

    for i, (lemma, _) in enumerate(mentions):
        if i not in sense_map:
            sense_map[i] = f"{lemma}#0"

    return sense_map, sense_meta


def sense_for(lemma, sent, mentions, sense_map):
    for i, (l, s) in enumerate(mentions):
        if l == lemma and s == sent:
            return sense_map.get(i)
    return None



class LLMRelation(BaseModel):
    subject:      str
    predicate:    str
    object:       str
    type:         Literal["action", "composition", "attribute", "dependency"]
    confidence:   Optional[float] = None
    rationale:    Optional[str]   = None
    evidence_span: Optional[str]  = None
    source:       Optional[str]   = "llm"
    tokens_used:  Optional[int]   = None
    requirement_id: Optional[str] = None

    @field_validator("subject", "object", "predicate", mode="before")
    @classmethod
    def coerce_to_str(cls, v):
        if v is None:
            return ""
        if isinstance(v, (list, dict)):
            return str(v)
        return v


class LLMRelationList(BaseModel):
    relations: TList[LLMRelation]


def _save_cache(cache: dict, path: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def llm_normalize_and_augment(req_id, req_text, srl_output, cache):
    cache_key = f"llm:{req_id}"
    if cache_key in cache:
        return LLMRelationList.model_validate(cache[cache_key])

    srl_json = json.dumps(srl_output, ensure_ascii=False)
    prompt = PromptTemplate(f"""
You are an assistant that extracts and normalizes DOMAIN relations from requirement text.
RULES:
- Ignore actors such as {AGENT_LEMMAS} — do not create nodes for them.
- Use only domain entities and relations that are supported by the text or the provided SRL frames.
- Output a JSON object matching the schema: {{ "relations": [ {{subject, predicate, object, type, confidence, rationale, evidence_span}} ] }}
- predicate should be normalized (verb lemma or short verb phrase), not long text.
- type must be one of: action, composition, attribute, dependency.
- evidence_span should be a substring of the requirement demonstrating the relation.
- If you propose a relation that is not explicit, set confidence <= 0.6 and explain in rationale.
- If relation is explicit in SRL frames, set <confidence> >= 0.8.
- subject and object MUST be a singular noun phrase naming a concrete domain entity or concept.
- NEVER use adjectives, adverbs, constraint phrases, or verb phrases as subject or object.
- NEVER use a comma-separated list of values as a single subject or object — omit the relation entirely.
- State values, format constraints, and enumerations of attribute values are NOT domain entities — skip them.

Requirement (id={req_id}):
{req_text}

SRL (PropBank-style frames) for this requirement (JSON):
{srl_json}

Return ONLY JSON.
""")
    for attempt in range(LLM_MAX_RETRIES):
        try:
            out = llm.structured_predict(LLMRelationList, prompt, llm_kwargs={"tool_choice": "none"})
            cache[cache_key] = out.model_dump()
            cache.pop(f"__failed__:{req_id}", None)
            return out
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = "429" in err_str or "rate_limit" in err_str or "rate limit" in err_str
            if is_rate_limit and attempt < LLM_MAX_RETRIES - 1:
                wait = LLM_RETRY_WAIT_S * (2 ** attempt)
                logger.warning(f"Rate limit on req {req_id}, retrying in {wait}s (attempt {attempt + 1}/{LLM_MAX_RETRIES})")
                time.sleep(wait)
                continue
            logger.warning(f"LLM extraction failed for req {req_id}: {e}")
            cache[f"__failed__:{req_id}"] = str(e)
            return LLMRelationList(relations=[])

    cache[f"__failed__:{req_id}"] = "rate_limit_exhausted"
    return LLMRelationList(relations=[])



def build_sense_index(sense_meta, embeddings):
    node_list, rows, per_node_emb = [], [], {}
    emb_dim = embeddings.shape[1] if embeddings.size > 0 else embedder.get_sentence_embedding_dimension()
    num_embeddings = embeddings.shape[0]

    for sense_id, meta in sense_meta.items():
        label = meta.get("label")
        vec   = embedder.encode(label) if label else embedder.encode(sense_id.split("#", 1)[0].replace("_", " "))
        if vec is None:
            vecs = [embeddings[m] for m in meta.get("members", []) if 0 <= m < num_embeddings]
            vec  = np.mean(np.vstack(vecs), axis=0) if vecs else None
        if vec is None:
            continue
        vec = np.asarray(vec).reshape(-1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        node_list.append(sense_id)
        rows.append(vec)
        per_node_emb[sense_id] = vec

    if not rows:
        return [], np.zeros((0, emb_dim)), per_node_emb
    return node_list, np.vstack(rows), per_node_emb


def base_key(text):
    doc = nlp(text.lower().strip())
    nouns = [t for t in doc if t.pos_ in ("NOUN", "PROPN")]
    if nouns:
        return nouns[-1].lemma_
    for t in doc:
        if t.is_alpha:
            return t.lemma_
    return text.lower().strip()


def map_label_to_sense(label, node_list, matrix, threshold=0.75):
    if not node_list or matrix.size == 0:
        return None, -1.0, None
    v = np.asarray(embedder.encode(label)).reshape(-1)
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    sims = matrix.dot(v)
    idx  = int(np.argmax(sims))
    sim  = float(sims[idx])
    best = node_list[idx]
    return (best, sim, best) if sim >= threshold else (None, sim, best)


def align_llm_relations_to_senses(llm_relations, sense_meta, embeddings,
                                  similarity_threshold=0.75,
                                  create_new_when_missing=True):
    node_list, matrix, per_node_emb = build_sense_index(sense_meta, embeddings)
    mapped = []

    def _is_noun_headed(label: str) -> bool:
        doc = nlp(label.strip())
        head = next((t for t in doc if t.dep_ == "ROOT"), None)
        if head is None:
            head = doc[-1] if doc else None
        return head is not None and head.pos_ in {"NOUN", "PROPN"}

    def ensure_sense(label):
        nonlocal node_list, matrix
        sense, sim, best = map_label_to_sense(label, node_list, matrix, similarity_threshold)
        if sense is None and sim >= 0.62:
            sense = best
        if sense is None and create_new_when_missing:
            if not _is_noun_headed(label):
                return None, 0.0
            new_id = f"{label.lower().replace(' ', '_')}#new_{uuid.uuid4().hex[:8]}"
            vec = np.asarray(embedder.encode(label)).reshape(-1)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            sense_meta[new_id] = {"members": [], "label": label}
            per_node_emb[new_id] = vec
            node_list.append(new_id)
            matrix = np.expand_dims(vec, 0) if matrix.size == 0 else np.vstack([matrix, vec])
            sense, sim = new_id, 1.0
        return sense, sim

    for rel in llm_relations:
        if not rel.subject.strip() or not rel.object.strip():
            continue
        subj_sense, subj_sim = ensure_sense(rel.subject)
        obj_sense,  obj_sim  = ensure_sense(rel.object)
        mapped.append({
            "req_id": rel.requirement_id, "orig_subject": rel.subject,
            "orig_object": rel.object, "predicate": rel.predicate.lower(),
            "subject_sense": subj_sense, "object_sense": obj_sense,
            "subject_similarity": subj_sim, "object_similarity": obj_sim,
            "mapped": subj_sense is not None and obj_sense is not None,
            "confidence": rel.confidence,
        })

    return mapped, sense_meta, per_node_emb



def is_agent_lemma(lemma):
    return lemma.lower() in AGENT_LEMMAS


def base_lemma_from_sense(sense_id):
    return sense_id.split("#", 1)[0].lower().strip()


def concepts_from_span(span_txt):
    raw_nps = np_candidates(span_txt)
    canon   = flatten([canonicalize(np) for np in raw_nps])
    return canon


def extract_triples_srl_aligned(srl_out, sent, known_concepts):
    doc = nlp(sent)
    triples = []

    def span_text(tokens):
        return " ".join(doc[i].text for i in tokens) if tokens else None

    for frame in srl_out.get("verbs", []):
        tags = frame["tags"]
        verb = frame["verb"]
        arg_spans = {"ARG0": [], "ARG1": [], "ARG2": []}
        cur_role, cur_span = None, []

        def flush():
            nonlocal cur_role, cur_span
            if cur_role and cur_span and cur_role in arg_spans:
                arg_spans[cur_role].append(cur_span)
            cur_role, cur_span = None, []

        for i, tag in enumerate(tags):
            if tag == "O":
                flush()
            elif tag.startswith("B-"):
                flush()
                role = tag[2:]
                if role != "V":
                    cur_role, cur_span = role, [i]
            elif tag.startswith("I-") and cur_role == tag[2:]:
                cur_span.append(i)
        flush()

        subj_span = arg_spans["ARG0"][0] if arg_spans["ARG0"] else None
        obj_span  = arg_spans["ARG1"][0] if arg_spans["ARG1"] else (arg_spans["ARG2"][0] if arg_spans["ARG2"] else None)
        if subj_span is None or obj_span is None:
            continue

        subj_concepts = [c for c in concepts_from_span(span_text(subj_span)) if c in known_concepts]
        obj_concepts  = [c for c in concepts_from_span(span_text(obj_span))  if c in known_concepts]
        if not subj_concepts or not obj_concepts:
            continue

        verb_lemma = nlp(verb)[0].lemma_.lower()
        for s in subj_concepts:
            for o in obj_concepts:
                if s != o:
                    triples.append((s, verb_lemma, o))

    return triples


def build_design_graph(reqs_complete, mentions, sense_map, sense_meta, embeddings,
                       enable_llm=True, llm_cache=None, cache_path=None) -> tuple:
    G = nx.DiGraph()
    node_reqs       = defaultdict(set)
    relation_data   = defaultdict(list)
    relation_types  = defaultdict(set)
    known_concepts  = {lemma for lemma, _ in mentions}
    per_node_emb    = {}

    # SRL phase
    for req_id, sent, srl in reqs_complete:
        for s, v, o in extract_triples_srl_aligned(srl, sent, known_concepts):
            s_lemma = nlp(s)[0].lemma_.lower()
            o_lemma = nlp(o)[0].lemma_.lower()
            s_sense = sense_for(s_lemma, sent, mentions, sense_map)
            o_sense = sense_for(o_lemma, sent, mentions, sense_map)
            if not s_sense or not o_sense or s_sense == o_sense:
                continue
            if is_agent_lemma(s_lemma) or is_agent_lemma(o_lemma):
                continue
            G.add_edge(s_sense, o_sense, confidence=1.0)
            G.nodes[s_sense]["confidence"] = 1.0
            G.nodes[o_sense]["confidence"] = 1.0
            relation_data[(s_sense, o_sense)].append(v.lower().strip())
            relation_types[(s_sense, o_sense)].add("srl-direct")
            node_reqs[s_sense].add(req_id)
            node_reqs[o_sense].add(req_id)

    # LLM phase
    failed_req_ids = []
    if enable_llm and llm_cache is not None:
        llm_relations = []
        n_empty_skipped = 0
        per_req_counts: list[tuple] = []
        for req_id, text, srl_output in reqs_complete:
            result = llm_normalize_and_augment(req_id, text, srl_output, llm_cache)
            if cache_path:
                _save_cache(llm_cache, cache_path)
            if f"__failed__:{req_id}" in llm_cache:
                failed_req_ids.append(req_id)
            raw = len(result.relations)
            useful = sum(1 for r in result.relations if r.subject.strip() and r.object.strip())
            per_req_counts.append((req_id, raw, useful))
            for r in result.relations:
                r.requirement_id = req_id
                llm_relations.append(r)

        total_raw    = sum(r for _, r, _ in per_req_counts)
        total_useful = sum(u for _, _, u in per_req_counts)
        n_empty_skipped = total_raw - total_useful
        zero_reqs = [req_id for req_id, _, u in per_req_counts if u == 0 and req_id not in failed_req_ids]
        logger.info(
            f"LLM relations: {total_raw} raw, {total_useful} useful, "
            f"{n_empty_skipped} skipped (null subject/object)"
        )
        if zero_reqs:
            logger.warning(f"Reqs with 0 useful relations (not failed): {zero_reqs}")

        aligned, sense_meta, per_node_emb = align_llm_relations_to_senses(
            llm_relations, sense_meta, embeddings
        )
        n_added = 0
        for rel in aligned:
            if not rel["mapped"]:
                continue
            u, v_node = rel["subject_sense"], rel["object_sense"]
            if u == v_node:
                continue
            if is_agent_lemma(base_lemma_from_sense(u)) or is_agent_lemma(base_lemma_from_sense(v_node)):
                continue
            G.add_edge(u, v_node, confidence=rel["confidence"])
            G.nodes[u]["confidence"] = rel["confidence"] if "new" in u else 1.0
            G.nodes[v_node]["confidence"] = rel["confidence"] if "new" in v_node else 1.0
            relation_data[(u, v_node)].append(rel["predicate"])
            relation_types[(u, v_node)].add("llm")
            node_reqs[u].add(rel["req_id"])
            node_reqs[v_node].add(rel["req_id"])
            n_added += 1
        logger.info(f"LLM edges added to graph: {n_added} / {len(aligned)} aligned")

    # Node metadata
    for node in G.nodes():
        meta    = sense_meta.get(node, {})
        members = meta.get("members", [])
        vecs    = [embeddings[i] for i in members if 0 <= i < embeddings.shape[0]]
        if vecs:
            mean = np.mean(np.vstack(vecs), axis=0)
            norm = np.linalg.norm(mean)
            G.nodes[node]["embedding"] = (mean / norm if norm > 0 else mean)
            G.nodes[node]["mention_embeddings"] = vecs
        elif node in per_node_emb:
            G.nodes[node]["embedding"] = per_node_emb[node]
            G.nodes[node]["mention_embeddings"] = []
        else:
            G.nodes[node]["embedding"] = None
            G.nodes[node]["mention_embeddings"] = []
        G.nodes[node]["requirements"] = list(node_reqs[node])
        G.nodes[node]["mention_texts"] = list({
            mentions[m][1] for m in members if 0 <= m < len(mentions)
        })
        G.nodes[node]["label"]        = meta.get("label")

    # Remove any node whose base lemma is an agent (can slip through as edge objects)
    agent_nodes = [n for n in list(G.nodes()) if is_agent_lemma(base_lemma_from_sense(n))]
    if agent_nodes:
        logger.info(f"Removing {len(agent_nodes)} agent node(s): {agent_nodes}")
        G.remove_nodes_from(agent_nodes)

    # Edge metadata
    verb_to_pairs = defaultdict(set)
    for (u, v_node), verbs in relation_data.items():
        for vb in verbs:
            verb_to_pairs[vb].add((u, v_node))
    verb_genericity = {vb: len(pairs) for vb, pairs in verb_to_pairs.items()}

    for (u, v_node), verbs in relation_data.items():
        freq         = len(verbs)
        verb_counts  = Counter(verbs)
        dominant     = max(verb_counts, key=verb_counts.get)
        dom_ratio    = max(verb_counts.values()) / freq
        gen          = verb_genericity.get(dominant, 1)
        generic_pen  = 1.0 / (np.log1p(gen) ** 2)
        weight       = np.log1p(freq) * dom_ratio * generic_pen
        G[u][v_node]["verbs"]          = dict(verb_counts)
        G[u][v_node]["freq"]           = freq
        G[u][v_node]["verb_consistency"] = dom_ratio
        G[u][v_node]["weight"]         = float(weight)
        G[u][v_node]["relation_types"] = list(relation_types[(u, v_node)])

    return G, sense_meta, failed_req_ids



def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, set)):
        return [make_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def export_graph(G: nx.DiGraph, out_path: str):
    G_export = deepcopy(G)
    for node_id in G_export.nodes:
        for key in list(G_export.nodes[node_id].keys()):
            G_export.nodes[node_id][key] = json.dumps(make_serializable(G_export.nodes[node_id][key]))
    for u, v, data in G_export.edges(data=True):
        for attr in list(data.keys()):
            data[attr] = json.dumps(make_serializable(data[attr]))
    nx.write_graphml(G_export, out_path)
    logger.info(f"Graph written to {out_path}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", required=True,
                        help="Monolith name (e.g. cargo, petclinic, ftgo)")
    parser.add_argument("--monolith-dir",
                        help="Path to monolith dir (default: monoliths/{app})")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM augmentation phase (faster, less complete)")
    parser.add_argument("--cache-file",
                        help="Path to persistent LLM cache JSON (default: {monolith_dir}/llm_cache.json)")
    args = parser.parse_args()

    app          = args.app
    monolith_dir = args.monolith_dir or f"monoliths/{app}"
    out_graphml  = f"monoliths/{app}/vista_disenio_{app}.graphml"
    cache_file   = args.cache_file or os.path.join(monolith_dir, "llm_cache.json")

    print(f"App          : {app}")
    print(f"Monolith dir : {monolith_dir}")
    print(f"Output       : {out_graphml}")

    load_models()

    print("\n[1/4] Loading inputs...")
    reqs, srl = load_inputs(monolith_dir)
    reqs_complete = preprocess_requirements(reqs, srl)
    print(f"  → {len(reqs)} sentences")

    print("\n[2/4] Extracting concept mentions...")
    mentions, embeddings = extract_concept_mentions_from_srl(reqs, srl)
    sense_map, sense_meta = disambiguate_senses(mentions, embeddings)
    print(f"  → {len(mentions)} mentions, {len(set(sense_map.values()))} senses")

    if not args.no_llm:
        if os.path.exists(cache_file):
            with open(cache_file, encoding="utf-8") as f:
                llm_cache = json.load(f)
            cached_hits = sum(1 for k in llm_cache if k.startswith("llm:"))
            print(f"  → Loaded LLM cache from {cache_file} ({cached_hits} cached entries)")
        else:
            llm_cache = {}
    else:
        llm_cache = None

    print(f"\n[3/4] Building design graph (LLM={'enabled' if llm_cache is not None else 'disabled'})...")
    G, sense_meta, failed_reqs = build_design_graph(
        reqs_complete, mentions, sense_map, sense_meta, embeddings,
        enable_llm=(llm_cache is not None),
        llm_cache=llm_cache,
        cache_path=cache_file if llm_cache is not None else None,
    )
    print(f"  → {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("\n[4/4] Applying Louvain partition and exporting...")
    partition = community_louvain.best_partition(
        G.to_undirected(), weight="weight", resolution=LOUVAIN_RESOLUTION
    )
    for n, cid in partition.items():
        G.nodes[n]["service"] = cid

    export_graph(G, out_graphml)
    print(f"\n✓  vista_disenio_{app}.graphml written")

    if failed_reqs:
        n = len(failed_reqs)
        total = len(reqs)
        print(f"\n⚠  WARNING: LLM extraction failed for {n}/{total} requirements "
              f"(ids: {failed_reqs}).")
        print(f"   Concepts appearing only in those requirements may be missing from the graph.")
        print(f"   Re-run when token budget is available to get a complete design view.")


if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[3])
    main()
