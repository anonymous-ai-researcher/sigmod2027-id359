# Phantom Neighbors: Information Leakage and Its Prevention in Access-Controlled Vector Databases

This repository contains the implementation and experimental code for the paper:

> **Phantom Neighbors: Information Leakage and Its Prevention in Access-Controlled Vector Databases**
> Anonymous Authors, SIGMOD 2027 Submission


## Overview

Access-controlled vector databases ensure that each user retrieves only authorized vectors, but similarity search results systematically leak information about *restricted* vectors through three geometric channels:

1. **Distance Distribution Skew (Channel 1):** The k-th nearest neighbor distance shifts when restricted vectors are removed, producing a statistically detectable distributional change.
2. **Neighborhood Topology Distortion (Channel 2):** Returned neighbors exhibit directional asymmetry, leaving angular gaps that point toward restricted vectors.
3. **Cross-Query Triangulation (Channel 3):** An adaptive adversary combining evidence from multiple queries can localize restricted vectors with precision improving as O(1/√m).

We call these geometric signatures **phantom neighbors** and develop a defense framework combining:
- **Private top-k selection** via the Gumbel mechanism with zero-sensitivity scores (§4.1)
- **Geometry-aware noise calibration** that adapts per-query privacy budgets to local leakage risk ρ(q) (§4.2)
- **Synthetic decoy augmentation** (isotropic and manifold-aware) to close the topology distortion channel (§4.3)

The composed mechanism satisfies (ε, δ)-differential privacy via Rényi DP composition (Mironov, Propositions 1, 3, 9).

## Datasets

The paper evaluates on four datasets spanning clinical, legal, enterprise, and vision domains:

| Dataset | Vectors | Roles | α range | Domain | Dimensionality |
|---------|---------|-------|---------|--------|----------------|
| MIMIC-IV | 331K | 4 | 0.12–0.35 | Clinical | 1024 (BGE-M3) |
| LegalBench-RAG | 48K | 3 | 0.20–0.30 | Legal | 1024 (BGE-M3) |
| Wiki-Enterprise | 500K | 5 | 0.10–0.20 | Enterprise | 1024 (BGE-M3) |
| SIFT1M | 1M | 5 | 0.15 | Vision | 128 (native SIFT) |

## Repository Structure

```
phantom-neighbors/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── configs/
│   ├── default.yaml          # Default hyperparameters
│   ├── mimic.yaml             # MIMIC-IV specific config
│   ├── legalbench.yaml        # LegalBench-RAG specific config
│   ├── wiki.yaml              # Wiki-Enterprise specific config
│   └── sift1m.yaml            # SIFT1M specific config
├── src/
│   ├── __init__.py
│   ├── defense/
│   │   ├── __init__.py
│   │   ├── private_topk.py    # Private top-k via Gumbel noise (§4.1)
│   │   ├── geometry_aware.py  # Geometry-aware budget allocation (§4.2)
│   │   ├── decoy.py           # Decoy generation: isotropic + manifold-aware (§4.3)
│   │   ├── composed.py        # Composed mechanism + RDP accounting (§4.4)
│   │   └── risk_map.py        # Risk map construction under DP (§4.2)
│   ├── attacks/
│   │   ├── __init__.py
│   │   ├── channel1.py        # Distance skew attack (§3.1)
│   │   ├── channel2.py        # Topology distortion attack (§3.2)
│   │   └── channel3.py        # Cross-query triangulation attack (§3.3)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py         # AUC, Recall@k, NDCG@k, TPR@FPR
│   │   ├── protocol.py        # Evaluation protocol (Appx. D.5)
│   │   └── significance.py    # Statistical significance tests (paired t-test)
│   └── utils/
│       ├── __init__.py
│       ├── index.py           # HNSW / IVF-PQ index wrapper
│       ├── access_control.py  # RBAC / ABAC access control policy (§2.1)
│       ├── embeddings.py      # Embedding model wrapper (BGE-M3, MiniLM, etc.)
│       └── poisson.py         # Local Poisson density estimation
├── scripts/
│   ├── preprocess_mimic.py    # MIMIC-IV preprocessing
│   ├── preprocess_legal.py    # LegalBench-RAG preprocessing
│   ├── preprocess_wiki.py     # Wiki-Enterprise preprocessing
│   ├── preprocess_sift1m.py   # SIFT1M preprocessing
│   ├── run_leakage.py         # RQ1: Leakage without defense
│   ├── run_defense.py         # RQ2: Defense effectiveness
│   ├── run_tradeoff.py        # RQ3: Privacy-utility tradeoff
│   ├── run_ablation.py        # RQ4: Ablation and sensitivity
│   ├── run_overhead.py        # RQ5: System overhead
│   └── run_all.sh             # Full reproduction pipeline
├── notebooks/
│   └── visualization.ipynb    # Figures and tables from the paper
└── data/
    └── .gitkeep               # Placeholder (see Data Preparation below)
```

## Installation

### Requirements

- Python 3.10+
- CUDA 11.8+ (for embedding computation)

### Setup

```bash
git clone https://github.com/anonymous/phantom-neighbors.git
cd phantom-neighbors
pip install -e .
```

### Dependencies

Key dependencies (see `requirements.txt` for full list):
- `faiss-gpu==1.7.4` (IVF-PQ index)
- `hnswlib==0.7.0` (HNSW index)
- `torch==2.1.0` (embedding computation)
- `transformers==4.35.0` (embedding models)
- `numpy`, `scipy`, `scikit-learn`

## Data Preparation

### MIMIC-IV

1. Obtain access to MIMIC-IV v3.1 from [PhysioNet](https://physionet.org/content/mimiciv/3.1/).
2. Download discharge summaries.
3. Run preprocessing:

```bash
python scripts/preprocess_mimic.py \
    --input /path/to/mimic-iv/discharge.csv \
    --output data/mimic/ \
    --model BAAI/bge-m3
```

Access control assigns 4 roles based on care unit (ICU, general ward, ED, non-psychiatric) with restriction ratios α = 0.12–0.35.

### LegalBench-RAG

```bash
python scripts/preprocess_legal.py \
    --input /path/to/legalbench/corpus.json \
    --output data/legalbench/ \
    --model BAAI/bge-m3
```

Access control applies ABA Rule 1.6 confidentiality tiers (3 roles, α = 0.20–0.30). This is the only dataset with ground-truth QA pairs (6.8K) for end-to-end RAG evaluation.

### Wiki-Enterprise

```bash
python scripts/preprocess_wiki.py \
    --input /path/to/wikipedia/dump-2024-04/ \
    --output data/wiki/ \
    --n_articles 500000 \
    --model BAAI/bge-m3
```

Synthetic department-level access policies (5 roles, α = 0.10–0.20). Extended to 2M–20M vectors for scalability experiments.

### SIFT1M

```bash
python scripts/preprocess_sift1m.py \
    --input /path/to/sift1m/ \
    --output data/sift1m/
```

Download SIFT1M from [INRIA](http://corpus-texmex.irisa.fr/). Uses native 128-dimensional SIFT descriptors (no embedding model). Synthetic access control with 5 roles and α = 0.15. This dataset places Channel 2 in its strongest regime due to improved angular discrimination at low dimensionality.

## Reproducing Results

### Full Reproduction

```bash
# Run all experiments (estimated time: ~8 hours on A100 + EPYC)
bash scripts/run_all.sh
```

### Individual Experiments

```bash
# RQ1: Leakage severity without defense
python scripts/run_leakage.py --config configs/default.yaml --dataset mimic

# RQ2: Defense effectiveness
python scripts/run_defense.py --config configs/default.yaml --dataset mimic

# RQ3: Privacy-utility tradeoff
python scripts/run_tradeoff.py --config configs/default.yaml --dataset mimic \
    --epsilon 0.1 0.5 1.0 3.0 5.0 10.0

# RQ4: Ablation study
python scripts/run_ablation.py --config configs/default.yaml --dataset mimic

# RQ5: System overhead
python scripts/run_overhead.py --config configs/default.yaml --dataset mimic

# Run on SIFT1M (uses native descriptors, no embedding model needed)
python scripts/run_defense.py --config configs/sift1m.yaml --dataset sift1m
```

## Hyperparameters

All default hyperparameters match the paper (Appendix D, Table 10):

| Category | Parameter | Default | Sweep Range |
|----------|-----------|---------|-------------|
| **Defense** | ε₀ (privacy budget) | 1.0 | {0.1, 0.5, 1.0, 3.0, 5.0, 10.0} |
| | k (neighbors) | 10 | {5, 10, 20, 50} |
| | k' (over-retrieval) | 2k = 20 | — |
| | c (decoys/vector) | 1 | {0, 1, 2, 3} |
| | δ (DP delta) | 10⁻⁶ | — |
| | δ_dec (decoy DP delta) | 10⁻⁶ | — |
| | ρ_max | 95th percentile of ρ(q) | — |
| | ρ_min | ρ_max / 10 | — |
| | σ_dec (decoy noise) | Median dist. to 10-th authorized NN | — |
| | ℓ (PCA neighbors) | 50 | — |
| **Risk Map** | Grid cells | Adaptive (≥100 vectors/cell) | — |
| | ε_ρ (map release budget) | 0.1 | — |
| **HNSW** | M (edges/node) | 32 | — |
| | ef_construction | 200 | — |
| | ef_search | 200 | — |
| **IVF-PQ** | n_list | 256 | — |
| | n_probe | 16 | — |
| | PQ sub-quantizers | 64 | — |
| **Evaluation** | Queries per dataset per role | 1,000 | — |
| | Random seeds | 5 | — |

## Key API

### Defense Framework

```python
from src.defense import ComposedDefense

defense = ComposedDefense(
    epsilon_0=1.0,
    delta=1e-6,
    k=10,
    c=1,              # decoys per restricted vector
    manifold_aware=True,
    d_int=48,          # intrinsic dimensionality (auto-estimated if None)
)

# Build augmented index with decoys and risk map
defense.build(database, access_policy)

# Private query (returns k genuine vectors after decoy filtering)
results = defense.query(q, user_id, k=10)

# Privacy accounting
print(defense.get_privacy_spent())  # cumulative (ε_total, δ_total)
```

### Attack Evaluation

```python
from src.attacks import DistanceSkewAttack, TopologyAttack, TriangulationAttack

# Channel 1: distance distribution skew
ch1 = DistanceSkewAttack(k=10)
auc_ch1 = ch1.evaluate(queries, results, ground_truth)

# Channel 2: topology distortion (Rayleigh test)
ch2 = TopologyAttack(k=10, d_int=48)
auc_ch2 = ch2.evaluate(queries, results, ground_truth)

# Channel 3: cross-query triangulation (m adaptive queries + MLE)
ch3 = TriangulationAttack(k=10, m=20)
auc_ch3 = ch3.evaluate(queries, results, ground_truth)
```

## Hardware

Experiments were run on:
- **CPU:** 64-core AMD EPYC, 256 GB RAM
- **GPU:** NVIDIA A100 40 GB (embedding computation only; all defense and evaluation runs on CPU)
- **Software:** Python 3.10, FAISS 1.7.4, hnswlib 0.7.0, PyTorch 2.1

## Embedding Models

| Model | Dim | Used For | HuggingFace ID |
|-------|-----|----------|----------------|
| BGE-M3 (default) | 1024 | MIMIC-IV, LegalBench, Wiki-Ent. | `BAAI/bge-m3` |
| MiniLM-L6 | 384 | Sensitivity to d (§6.4) | `sentence-transformers/all-MiniLM-L6-v2` |
| BioClinicalBERT | 768 | Sensitivity to d (§6.4) | `emilyalsentzer/Bio_ClinicalBERT` |
| text-embedding-3-large | 3072 | Sensitivity to d (§6.4) | OpenAI API |
| SIFT descriptors | 128 | SIFT1M | Native (no model) |

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

This repository will be de-anonymized upon acceptance.
