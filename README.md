# Multi-View Evaluation of Microservice Decompositions using Bounded Context and Data Ownership Metrics

Artifact for the paper **Multi-View Evaluation of Microservice Decompositions using Bounded Context and Data Ownership Metrics** submitted to **ECSA 2026** (Research Track).

## Overview

This artifact provides the implementation of **MVAA** (Multi-View Architectural Analysis), an approach for evaluating candidate microservice decompositions by jointly reasoning over three complementary architectural views extracted from a monolith application:

| View | Source | Purpose |
|------|--------|---------|
| **Domain view** | Natural-language requirements | Encodes domain concepts and their relationships |
| **Implementation view** | Java source code (static AST analysis) | Encodes classes and their dependencies |
| **Data view** | SQL schema + ORM mappings | Encodes tables and data-access patterns |

The approach (1) builds each view from sources, (2) induces Bounded Context (BC) labels on domain-view nodes via semi-supervised propagation, (3) aligns the three views using weighted similarity signals (embeddings, structural similarity, data-access), and (4) evaluates decomposition quality through two architectural metrics: **BCA** (Bounded Context Alignment — quantifies how much services mix classes from different BCs) and **DO** (Data Ownership — captures transactional data ownership violations across services). 

Three open-source monolith systems are used as case studies: **[Cargo](https://github.com/citerus/dddsample-core)**, **[JPetStore](https://github.com/mybatis/jpetstore-6)**, and **[DayTrader](https://github.com/WASdev/sample.daytrader7)**.

## Structure

```
mvaa/                            # Python source package
├── modeling/                    # Step 1: Views construction
│   ├── data/                    # Data view construction (SQL → graphml)
│   ├── domain/                  # Domain view construction (Requirements + Coref + SRL → graphml)
│   └── implementation/          # Implementation view construction (Java AST → graphml)
├── bc_induction/                # Step 2: Semi-supervised BC label propagation on the domain view
├── alignment/                   # Step 3: Multi-view alignment (embedding + structural + data signals)
├── metrics/                     # Step 4: BC-alignment and data-ownership metrics evaluation
├── tests/                       # Evaluation and figure-generation scripts
│   ├── RQ1/                     # Per-decomposition metrics and dataset characterization
│   │   ├── bc_penalty_analysis.py   # BCA metric evaluation -> results/RQ1, results/RQ2, results/graphics
│   │   ├── data_penalty_analysis.py # DO metric evaluation -> results/RQ1
│   │   ├── decomposition_space.py   # Decomposition space summary: classes, BCs, decompositions, duplicates
│   │   └── discriminative_power.py  # Discriminative power of NMI, NMI_filtered, and penalty_swm across decompositions
│   ├── RQ2/                     # Structural-metric correlations
│   │   └── partial_spearman.py      # Partial Spearman correlation, controlling for #services
│   ├── RQ3/                     # Expert evaluation
│   │   ├── questionnaire.py         # Prints candidate decompositions used for the expert survey
│   │   └── questionnaire_results.py # Expert evaluation questionnaire analysis
│   ├── graphics/                # Figure scripts (cross_view_alignment.png, etc.)
│   └── reproduce_all.py         # Runs the full suite above, regenerating results/RQ1-3 and graphics
└── utils/                       # Shared utilities (graph I/O, embeddings, Groq client)

monoliths/                       # Three example monolith systems
├── cargo/                       # Cargo Tracker (DDD reference application)
├── jpetstore/                   # MyBatis JPetStore
└── daytrader/                   # Apache DayTrader
    # Each system contains:
    #   *.java                   — source classes (implementation view input)
    #   data.sql                 — SQL schema (data view input)
    #   requirements.txt         — raw requirements 
    #   requirements_processed.txt — processed requirements (after applying coreference resolution and SRL, domain view input)
    #   srl.json                 — pre-parsed SRL dependency relations
    #   bc.json                  — expert-annotated BCs and seed labels
    #   decompositions_*.pkl     — candidate decompositions from topic-model strategy (mid)
    #   vista_*.graphml          — pre-computed graph views (all three views)
    #   alignment_results.pkl    — pre-computed multi-view alignments
    #   mid_results/             — raw candidate decompositions and structural metrics per decomposition (CSV) from Brito et al. strategy

results/                         # Pre-generated evaluation results, grouped by research question
├── RQ1/                          # Per-decomposition metrics (input to RQ2/RQ3 analyses)
│   ├── metrics_{system}.csv      # Structural metrics + NMI + BCA metric per decomposition
│   ├── metrics_combined.csv      # metrics_*.csv concatenated across all three systems
│   ├── pen_data_{system}.csv     # DO metric (Pen_data) per decomposition
│   ├── decomposition_space.csv   # Classes, BCs, decompositions, duplicates per system
│   └── discriminative_power.csv  # NMI, NMI_filtered, and penalty_swm min/max/std/range per system
├── RQ2/                           # BCA/DO metrics vs. structural-metric correlations
│   ├── spearman_nmi_{system}.csv
│   ├── spearman_penalty_swm_{system}.csv
│   ├── spearman_penalty_mean_{system}.csv
│   ├── partial_spearman_{system}.csv  # Partial Spearman ρ (NMI vs. structural metrics, controlling for #services)
│   └── partial_spearman_pattern.csv   # Cross-system comparison of partial Spearman ρ per metric
├── RQ3/                           # Expert evaluation
│   ├── Expert Evaluation_ Microservice Decomposition Quality (respuestas).xlsx
│   ├── spearman_vs_nmi.csv        # Per-evaluator Spearman ρ (expert ranking vs. NMI ranking)
│   ├── consensus_ranking.csv      # Mean/std rank per decomposition + consensus position
│   ├── inter_rater_agreement.csv  # Kendall's W, chi2, and consensus-vs-NMI correlation
│   └── data_ownership_preference.csv  # ALPHA/BETA/no-difference preference counts
└── graphics/                      # Figures
    ├── cross_view_alignment.png       # Multi-view alignment figure (paper Fig. 2)
    ├── spearman_comparative_*.png     # Comparative Spearman figures
    └── distributions_comparative.png

appendix/                        # Jupyter notebooks
├── results.ipynb                # Full interactive evaluation analysis
└── decomposition_appendix.ipynb # Decomposition walkthrough
```

## Requirements

| Requirement | Notes |
|-------------|-------|
| Python ≥ 3.13 | Required by the package |
| [uv](https://docs.astral.sh/uv/) | Fast locked-dependency installer |
| Disk ≥ 9 GB | Docker image is ~8.7 GB; local `uv sync` installs ~7.7 GB |
| GPU (optional) | CPU-only inference works; the embedding model (`all-MiniLM-L6-v2`) is small and runs fast on CPU |
| Groq API key | **Only needed for full pipeline re-execution** — not required to reproduce paper results |

All intermediate outputs (graphml views, alignment `.pkl` files, evaluation CSVs) are pre-computed and committed to the repository. Reproducing the paper's quantitative results and figures requires no API key.

## Setup

### Option A — Docker

Provides a fully reproducible environment. First build takes ~5 minutes and ~9 GB of disk space.

```bash
# Build the image
docker build -t mvaa .

# Quick test: reproduce all results
docker run --rm --user "$(id -u):$(id -g)" mvaa

# Regenerate the cross-view alignment figure
docker run --rm --user "$(id -u):$(id -g)" -v "$(pwd)/results:/app/results" mvaa \
    python mvaa/tests/graphics/graphics.py

# Start an interactive Jupyter session
docker run --rm --user "$(id -u):$(id -g)" -p 8888:8888 -v "$(pwd)/results:/app/results" mvaa \
    jupyter notebook --ip=0.0.0.0 --allow-root appendix/results.ipynb
```

### Option B — Local setup with uv

```bash
# 1. Install uv (https://docs.astral.sh/uv/getting-started/installation/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install all locked dependencies
uv sync

# 3. Optional: configure Groq API key for full pipeline re-execution
cp .env.example .env
# Edit .env and set: GROQ_API_KEY=<your-key>
```

## Reproducing paper results (no API key required)

The steps below use only pre-computed intermediate artifacts already included in the repository. Each script regenerates a specific subset of `results/`:

| Script | Regenerates | RQ |
|--------|-------------|----|
| `mvaa/tests/RQ1/bc_penalty_analysis.py` | `results/RQ1/metrics_{system}.csv`, `results/RQ1/metrics_combined.csv`, `results/RQ2/spearman_*.csv`, `results/graphics/spearman_comparative_*.png`, `results/graphics/distributions_comparative.png` | RQ1, RQ2 |
| `mvaa/tests/RQ1/data_penalty_analysis.py --app <system>` | `results/RQ1/pen_data_{system}.csv` | RQ1 |
| `mvaa/tests/RQ1/decomposition_space.py` | `results/RQ1/decomposition_space.csv` | RQ1 |
| `mvaa/tests/RQ1/discriminative_power.py` | `results/RQ1/discriminative_power.csv` | RQ1 |
| `mvaa/tests/RQ2/partial_spearman.py` | `results/RQ2/partial_spearman_{system}.csv`, `results/RQ2/partial_spearman_pattern.csv` | RQ2 |
| `mvaa/tests/RQ3/questionnaire_results.py` | `results/RQ3/spearman_vs_nmi.csv`, `results/RQ3/consensus_ranking.csv`, `results/RQ3/inter_rater_agreement.csv`, `results/RQ3/data_ownership_preference.csv` | RQ3 |
| `mvaa/tests/graphics/graphics.py` | `results/graphics/cross_view_alignment.png` | — |
| `mvaa/tests/reproduce_all.py` | all of the above, for all three systems | RQ1, RQ2, RQ3 |

### Basic test — partial Spearman correlation (RQ2)

Reproduces the partial Spearman ρ table (controlling for number of services):

```bash
# Local
uv run python mvaa/tests/RQ2/partial_spearman.py

# Docker
docker run --rm --user "$(id -u):$(id -g)" -v "$(pwd)/results:/app/results" mvaa python mvaa/tests/RQ2/partial_spearman.py
```

Diff the regenerated `results/RQ2/partial_spearman_{system}.csv` and `results/RQ2/partial_spearman_pattern.csv` against the committed versions. The table lists the partial correlation of each structural metric (CHM, CHD, IFN, IRN, OPN, SMQ, SCOH, SCOP, CMQ, CCOH, CCOP) with NMI across the three systems, with significance markers. Values should match exactly (the computation is deterministic and uses only pre-computed data).

### Regenerate everything (one command)

```bash
# Local
uv run python mvaa/tests/reproduce_all.py

# Docker
docker run --rm --user "$(id -u):$(id -g)" -v "$(pwd)/results:/app/results" mvaa python mvaa/tests/reproduce_all.py
```

### Full reproduction sequence (individual scripts)

```bash
# BC-alignment analysis -> results/RQ1, results/RQ2, results/graphics
uv run python mvaa/tests/RQ1/bc_penalty_analysis.py
docker run --rm --user "$(id -u):$(id -g)" -v "$(pwd)/results:/app/results" mvaa python mvaa/tests/RQ1/bc_penalty_analysis.py

# Data-ownership analysis -> results/RQ1/pen_data_<app>.csv (one per system)
for app in cargo jpetstore daytrader; do
  uv run python mvaa/tests/RQ1/data_penalty_analysis.py --app "$app"
  docker run --rm --user "$(id -u):$(id -g)" -v "$(pwd)/results:/app/results" mvaa python mvaa/tests/RQ1/data_penalty_analysis.py --app "$app"
done

# Decomposition space summary (RQ1)
uv run python mvaa/tests/RQ1/decomposition_space.py
docker run --rm --user "$(id -u):$(id -g)" -v "$(pwd)/results:/app/results" mvaa python mvaa/tests/RQ1/decomposition_space.py

# Discriminative power of NMI/penalty_swm (RQ1)
uv run python mvaa/tests/RQ1/discriminative_power.py
docker run --rm --user "$(id -u):$(id -g)" -v "$(pwd)/results:/app/results" mvaa python mvaa/tests/RQ1/discriminative_power.py

# Partial Spearman correlation (RQ2)
uv run python mvaa/tests/RQ2/partial_spearman.py
docker run --rm --user "$(id -u):$(id -g)" -v "$(pwd)/results:/app/results" mvaa python mvaa/tests/RQ2/partial_spearman.py

# Expert questionnaire analysis (RQ3)
uv run python mvaa/tests/RQ3/questionnaire_results.py
docker run --rm --user "$(id -u):$(id -g)" -v "$(pwd)/results:/app/results" mvaa python mvaa/tests/RQ3/questionnaire_results.py

# Cross-view alignment figure -> results/graphics/cross_view_alignment.png
uv run python mvaa/tests/graphics/graphics.py
docker run --rm --user "$(id -u):$(id -g)" -v "$(pwd)/results:/app/results" mvaa python mvaa/tests/graphics/graphics.py
```

### Interactive analysis via Jupyter

```bash
# Local
uv run jupyter notebook appendix/results.ipynb

# Docker
docker run --rm --user "$(id -u):$(id -g)" -p 8888:8888 -v "$(pwd)/results:/app/results" mvaa \
    jupyter notebook --ip=0.0.0.0 --allow-root appendix/results.ipynb
```

## Full pipeline re-execution

To re-run the pipeline from raw sources (Java classes, SQL, requirements) a Groq API key is required for the LLM-assisted node description generation integrated into implementation, domain, and data views.

All steps below run via `docker run` against the `mvaa` image (substitute `<your-key>` with your Groq API key). Every script takes an `--app <name>` flag selecting `cargo`, `jpetstore`, or `daytrader`; the loops re-run each step for all three case studies. Drop the loop and pass a single app name to target one system.

```bash
# 1. Build the image (if not already built) and have your Groq API key ready;
#    substitute it for <your-key> in the commands below.
docker build -t mvaa .

# 2. Construct the data view (LLM-assisted table/column descriptions)
for app in cargo jpetstore daytrader; do
  docker run --rm --user "$(id -u):$(id -g)" -e GROQ_API_KEY=<your-key> -v "$(pwd)/monoliths:/app/monoliths" mvaa \
      python mvaa/modeling/data/extractor.py --app "$app"
done

# 3. Generate entity/table and class/table access mappings (entity_to_tables.json,
#    mapper_to_tables.json). table_access.py picks an extraction strategy per app:
#    JPA repositories for cargo, MyBatis mapper XML for jpetstore, EJB/JDBC
#    (propagated through the implementation graph) for daytrader.
for app in cargo jpetstore daytrader; do
  docker run --rm --user "$(id -u):$(id -g)" -v "$(pwd)/monoliths:/app/monoliths" mvaa \
      python mvaa/modeling/data/table_access.py --app "$app"
done

# 4a. Construct the implementation view (LLM-assisted class descriptions)
for app in cargo jpetstore daytrader; do
  docker run --rm --user "$(id -u):$(id -g)" -e GROQ_API_KEY=<your-key> -v "$(pwd)/monoliths:/app/monoliths" mvaa \
      python mvaa/modeling/implementation/extractor.py --app "$app"
done

# 4b. (Optional) Project the implementation graph into a class-level dependency graph (needed for most monolith decomposition strategies)
for app in cargo jpetstore daytrader; do
  docker run --rm --user "$(id -u):$(id -g)" -v "$(pwd)/monoliths:/app/monoliths" mvaa \
      python mvaa/modeling/implementation/projection.py --app "$app"
done

# 5. Construct the domain view (LLM-assisted relation augmentation)
#    (reads requirements_processed.txt + srl.json, writes vista_disenio_*.graphml)
for app in cargo jpetstore daytrader; do
  docker run --rm --user "$(id -u):$(id -g)" -e GROQ_API_KEY=<your-key> -v "$(pwd)/monoliths:/app/monoliths" mvaa \
      python mvaa/modeling/domain/extractor.py --app "$app"
done

# 6. Induce BC labels on the domain view
#    (reads vista_disenio_*.graphml, writes updated graphml)
for app in cargo jpetstore daytrader; do
  docker run --rm --user "$(id -u):$(id -g)" -v "$(pwd)/monoliths:/app/monoliths" mvaa \
      python mvaa/bc_induction/bc_induction.py --app "$app"
done

# 7. Compute multi-view alignments
#    (reads all three vista_*.graphml, writes alignment_results.pkl)
for app in cargo jpetstore daytrader; do
  docker run --rm --user "$(id -u):$(id -g)" -v "$(pwd)/monoliths:/app/monoliths" mvaa \
      python mvaa/alignment/alignment.py --app "$app"
done

# 8. Evaluate metrics (data_restriction.py also writes results/RQ1/pen_data_<app>.csv)
for app in cargo jpetstore daytrader; do
  docker run --rm --user "$(id -u):$(id -g)" -v "$(pwd)/monoliths:/app/monoliths" mvaa \
      python mvaa/metrics/bounded_contexts.py --app "$app"
  docker run --rm --user "$(id -u):$(id -g)" -v "$(pwd)/monoliths:/app/monoliths" -v "$(pwd)/results:/app/results" mvaa \
      python mvaa/metrics/data_restriction.py --app "$app"
done
```

## Expert evaluation data

The `results/` directory includes the anonymized expert evaluation spreadsheet (`Expert Evaluation_ Microservice Decomposition Quality (respuestas).xlsx`). Participants ranked six candidate decompositions (A–F) by perceived quality. Rankings are cross-referenced with NMI scores in `mvaa/tests/RQ3/questionnaire_results.py`.

## Module API reference

Each module is independently importable. Below are the primary entry points with their inputs and outputs.

### `mvaa.modeling.data` — Data view construction

```python
from mvaa.modeling.data.extractor import build_data_graph, enrich_tables_with_descriptors

G = build_data_graph("monoliths/cargo/data.sql")
# Input:  path to a SQL file containing CREATE TABLE statements
# Output: nx.MultiDiGraph with nodes typed "tabla"/"columna" and FK edges
```

### `mvaa.modeling.domain` — Domain view construction

```python
from mvaa.modeling.domain.extractor import (
    load_inputs, preprocess_requirements,
    extract_concept_mentions_from_srl, disambiguate_senses,
    build_design_graph,
)

reqs, srl = load_inputs("monoliths/cargo")
reqs_complete = preprocess_requirements(reqs, srl)
mentions, embeddings = extract_concept_mentions_from_srl(reqs, srl)
sense_map, sense_meta = disambiguate_senses(mentions, embeddings)

G, sense_meta, failed_reqs = build_design_graph(
    reqs_complete, mentions, sense_map, sense_meta, embeddings,
    enable_llm=False,
)
# Input:  monoliths/<app>/requirements_processed.txt (processed requirements, one
#         sentence per line) and monoliths/<app>/srl.json (pre-parsed SRL relations)
# Output: nx.DiGraph of domain concept nodes (label, embedding, confidence,
#         requirements, mention_texts) and weighted relation edges (dependency-path,
#         SRL, and embedding-similarity signals); failed_reqs lists requirement ids
#         where LLM augmentation failed (empty when enable_llm=False)
# Note:   set enable_llm=True (with llm_cache=dict and cache_path=...) and a
#         GROQ_API_KEY to enable LLM-assisted relation augmentation. Then apply
#         community_louvain.best_partition(G.to_undirected(), weight="weight") to
#         set each node's "service" attribute and export_graph(G,
#         "monoliths/<app>/vista_disenio_<app>.graphml") to reproduce the CLI output.
```

### `mvaa.modeling.implementation` — Implementation view construction

```python
from mvaa.modeling.implementation.extractor import build_implementation_graph

G = build_implementation_graph("monoliths/cargo/")
# Input:  path to a directory containing *.java source files
# Output: nx.MultiDiGraph with class/method nodes and dependency edges
#         (invoca, hereda, implementa, accede, contiene)
```

### `mvaa.bc_induction` — Bounded Context label propagation

```python
from mvaa.bc_induction.bc_induction import infer_canonical_bcs_soft

probs = infer_canonical_bcs_soft(
    G,           # nx.DiGraph: domain view (vista_disenio_*.graphml)
    bc_list,     # List[str]: bounded context names (from bc.json)
    seeds,       # Dict[str, str]: node -> BC label for seed nodes
)
# Output: Dict[str, Dict[str, float]] — node -> {bc: probability}
#         Also writes bc_probs and bc attributes onto each graph node.
```

### `mvaa.alignment` — Multi-view alignment

```python
from mvaa.alignment.alignment import run_all_alignments, AlignParams

alignments = run_all_alignments(
    emb_DI=...,     # ScoreMap: embedding similarity between domain and implementation nodes
    emb_D_T=...,    # ScoreMap: embedding similarity between domain and table nodes
    emb_I_T=...,    # ScoreMap: embedding similarity between implementation and table nodes
    # Optional signals:
    struct_DI=...,  # ScoreMap: structural similarity (domain ↔ implementation)
    sim_data_ID=...,# ScoreMap: data-access similarity (implementation → domain)
    access_I_T=..., # ScoreMap: table-access similarity (implementation → tables)
    tables_for_D_T=..., # ScoreMap: entity-to-table mapping (domain → tables)
    params_DI=AlignParams(topk=40, lambda_=5.0, tau=0.25),
)
# Output: MultiViewAlignments dataclass with six pairwise alignment maps
#         (D_to_I, I_to_D, I_to_T, T_to_I, D_to_T, T_to_D) and their
#         probability distributions.
```

`AlignParams` fields: `topk` (candidate pool size), `lambda_` (softmax temperature), `tau` (minimum probability threshold), `min_keep`, `max_keep`.

### `mvaa.metrics.bounded_contexts` — BCA metric

```python
from mvaa.metrics.bounded_contexts import (
    bca_metric_for_decomposition,
    nmi_decomposition,
)

result = bca_metric_for_decomposition(
    membership={"ClassName": "service_id", ...},  # impl node -> cluster id
    G_design=G_design,  # domain view graph with bc / bc_probs node attributes
    P_u_to_c=alignments.P_I_D,  # impl node -> [(design_node, prob), ...]
    top_k=1,
    bc_list=["BC1", "BC2", ...],
    alpha=0.1,
)
# Output dict keys: penalty, penalty_rel, score_rel, H_global, H_Ci, P_Cd_given_Ci

nmi = nmi_decomposition(membership, class_to_bc, impl_nodes)
# Output: float in [0, 1] — Normalized Mutual Information of the decomposition
#         with respect to ground-truth BC labels.
```

### `mvaa.metrics.data_restriction` — DO metric

```python
from mvaa.metrics.data_restriction import compute_do_metric

penalties, overlap, tx, access_by_cluster, txs_by_cluster =
    compute_do_metric(
        graphml_path="monoliths/cargo/vista_implementacion_cargo.graphml",
        cluster_of_class={"Cargo": "0", "Booking": "0", "Location": "1", ...},
        access_by_class={"Cargo": {"CARGO", "LEGS"}, ...},
    )
# penalties: Dict[(cluster_i, cluster_j), float] — cross-cluster data overlap × tx overlap
# overlap:   Dict[(cluster_i, cluster_j), float] — Jaccard of table access sets
# tx:        Dict[(cluster_i, cluster_j), float] — Jaccard of transaction groups
```

---

## Applying the pipeline to a new monolith

To run the full pipeline on a different monolith:

**1. Prepare the inputs** (place them in `monoliths/<your_system>/`):
- `*.java` — all Java source files
- `data.sql` — SQL schema with `CREATE TABLE` statements
- `requirements.txt` — one requirement sentence per line
- `bc.json` — bounded context definitions and seed node labels (see `monoliths/cargo/bc.json` for format)

**2. Preprocess the requirements** to obtain `requirements_processed.txt` and `srl.json`, both required by the domain view in step 3. This is a two-stage pipeline external to this repository:

- **Coreference resolution** — `requirements.txt` is passed to [pocs_coref](https://doi.org/10.5281/zenodo.20750116), which resolves references across requirement sentences and writes the result to `requirements_processed.txt`.
- **Semantic role labeling (SRL)** — `requirements_processed.txt` is then passed to [pocs_srl](https://zenodo.org/records/20750388) (AllenNLP BERT-based SRL model, run via Docker), which writes the parsed sentence relations to `srl.json`.

See `monoliths/cargo/` (or `jpetstore`, `daytrader`) for example outputs of both stages.

**3. Build the views** (Groq API key needed — each script also runs LLM-assisted
descriptor/embedding enrichment used by alignment):

```bash
# Data view (writes vista_datos_<your_system>.graphml)
uv run python mvaa/modeling/data/extractor.py --app <your_system>

# Implementation view (writes vista_implementacion_<your_system>.graphml)
uv run python mvaa/modeling/implementation/extractor.py --app <your_system>

# Domain view (requires requirements_processed.txt and srl.json from step 2;
# writes vista_disenio_<your_system>.graphml)
uv run python mvaa/modeling/domain/extractor.py --app <your_system>
```

> The `--app` flag for these three scripts accepts any system name since they read/write `monoliths/<app>/...` directly.


**4. Induce BC labels, project the implementation graph, align views, and evaluate** — these scripts also accept any `--app <name>` value, so point them directly at `<your_system>`:

```bash
# Project the implementation graph to a class-level dependency graph
# (writes vista_implementacion_<your_system>_c.graphml, used by alignment and metrics)
uv run python mvaa/modeling/implementation/projection.py --app <your_system>

# Induce BC labels on the domain view (reads bc.json from step 1)
uv run python mvaa/bc_induction/bc_induction.py --app <your_system>

# Compute multi-view alignments (writes alignment_results.pkl)
uv run python mvaa/alignment/alignment.py --app <your_system>

# Evaluate metrics (requires decompositions_<your_system>.pkl — candidate
# decompositions to score; see monoliths/cargo/decompositions_cargo.pkl for the
# expected format)
uv run python mvaa/metrics/bounded_contexts.py --app <your_system>

# Also writes results/RQ1/pen_data_<your_system>.csv (Pen_data per decomposition)
uv run python mvaa/metrics/data_restriction.py --app <your_system>
```

**5. Use alignment results programmatically**:

```python
import pickle
from mvaa.metrics.bounded_contexts import nmi_decomposition, bca_metric_for_decomposition

with open("monoliths/<your_system>/alignment_results.pkl", "rb") as f:
    alignments = pickle.load(f)

# Evaluate any membership dict (class -> service_id) against BC ground truth
nmi = nmi_decomposition(membership, class_to_bc, impl_nodes)
result = bca_metric_for_decomposition(
    membership=membership, G_design=G_design,
    P_u_to_c=alignments.P_I_D, top_k=1, bc_list=bc_list, alpha=0.1,
)
print(f"NMI={nmi:.3f}  BCA={result['penalty_rel']:.3f}")
```

## License

[MIT](LICENSE) — © 2026 Ana Martínez Saucedo, J. Andrés Díaz-Pace, Guillermo Rodríguez.

> This repository is archived on Zenodo at [https://doi.org/10.5281/zenodo.20277403](https://doi.org/10.5281/zenodo.20277403).
