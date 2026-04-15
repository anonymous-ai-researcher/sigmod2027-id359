# Phantom Neighbors: Information Leakage and Its Prevention in Access-Controlled Vector Databases

This repository contains the implementation and experimental code for the paper:

> **Phantom Neighbors: Information Leakage and Its Prevention in Access-Controlled Vector Databases**
> Anonymous Authors, SIGMOD 2027 Submission

## Overview

Access-controlled vector databases ensure that each user retrieves only authorized vectors, but similarity search results systematically leak information about *restricted* vectors through three geometric channels:

1. **Distance Distribution Skew (Channel 1):** The k-th nearest neighbor distance shifts when restricted vectors are removed.
2. **Neighborhood Topology Distortion (Channel 2):** Returned neighbors exhibit directional asymmetry pointing toward restricted vectors.
3. **Cross-Query Triangulation (Channel 3):** Multiple queries can localize restricted vectors via geometric evidence.

We call these geometric signatures **phantom neighbors** and develop a defense framework combining:
- **Private top-k selection** via the joint exponential mechanism (Gumbel noise)
- **Geometry-aware noise calibration** that adapts privacy budgets to local leakage risk
- **Synthetic decoy augmentation** (isotropic and manifold-aware) to close topology distortion

The composed mechanism satisfies (ε, δ)-differential privacy via Rényi composition.

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
│   └── wiki.yaml              # Wiki-Enterprise specific config
├── src/
│   ├── __init__.py
│   ├── defense/
│   │   ├── __init__.py
│   │   ├── private_topk.py    # Private top-k via Gumbel noise (§4.1)
│   │   ├── geometry_aware.py  # Geometry-aware budget allocation (§4.2)
│   │   ├── decoy.py           # Decoy generation: isotropic + manifold-aware (§4.3)
│   │   ├── composed.py        # Composed mechanism + RDP accounting (§4.4)
│   │   └── risk_map.py        # Risk map construction (§4.2)
│   ├── attacks/
│   │   ├── __init__.py
│   │   ├── channel1.py        # Distance skew attack (§3.1)
│   │   ├── channel2.py        # Topology distortion attack (§3.2)
│   │   └── channel3.py        # Cross-query triangulation attack (§3.3)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py         # AUC, Recall@k, NDCG@k
│   │   ├── protocol.py        # Evaluation protocol (App D.5)
│   │   └── significance.py    # Statistical significance tests
│   └── utils/
│       ├── __init__.py
│       ├── index.py           # HNSW / IVF-PQ index wrapper
│       ├── access_control.py  # Access control policy (§2.1)
│       ├── embeddings.py      # Embedding model wrapper
│       └── poisson.py         # Local Poisson density estimation
├── scripts/
│   ├── preprocess_mimic.py    # MIMIC-IV preprocessing
│   ├── preprocess_legal.py    # LegalBench-RAG preprocessing
│   ├── preprocess_wiki.py     # Wiki-Enterprise preprocessing
│   ├── run_leakage.py         # RQ1: Leakage without defense
│   ├── run_defense.py         # RQ2: Defense effectiveness
│   ├── run_tradeoff.py        # RQ3: Privacy-utility tradeoff
│   ├── run_ablation.py        # RQ4: Ablation and sensitivity
│   ├── run_overhead.py        # RQ5: System overhead
│   └── run_all.sh             # Full reproduction pipeline
├── notebooks/
│   └── visualization.ipynb    # Figures and tables from the paper
└── data/
    └── .gitkeep               # Placeholder (see Data section below)
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

1. Obtain access to MIMIC-IV v2.2 from [PhysioNet](https://physionet.org/content/mimiciv/2.2/).
2. Download discharge summaries.
3. Run preprocessing:

```bash
python scripts/preprocess_mimic.py \
    --input /path/to/mimic-iv/discharge.csv \
    --output data/mimic/ \
    --model BAAI/bge-m3
```

### LegalBench-RAG

```bash
python scripts/preprocess_legal.py \
    --input /path/to/legalbench/corpus.json \
    --output data/legalbench/ \
    --model BAAI/bge-m3
```

### Wiki-Enterprise

```bash
python scripts/preprocess_wiki.py \
    --input /path/to/wikipedia/dump-2024-04/ \
    --output data/wiki/ \
    --n_articles 500000 \
    --model BAAI/bge-m3
```

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
```

## Hyperparameters

All default hyperparameters match the paper (Table in Appendix D):

| Parameter | Default | Sweep Range |
|-----------|---------|-------------|
| ε₀ (privacy budget) | 1.0 | {0.1, 0.5, 1.0, 3.0, 5.0, 10.0} |
| k (neighbors) | 10 | {5, 10, 20, 50} |
| k' (over-retrieval) | 2k = 20 | — |
| c (decoys/vector) | 1 | {0, 1, 2, 3} |
| δ (DP delta) | 10⁻⁶ | — |
| ρ_min/ρ_max | 0.10 | — |
| HNSW M | 32 | — |
| HNSW ef | 200 | — |

## Key API

### Defense Framework

```python
from src.defense import ComposedDefense

defense = ComposedDefense(
    epsilon_0=1.0,
    delta=1e-6,
    k=10,
    c=1,  # decoys per restricted vector
    manifold_aware=True,
    d_int=48,  # intrinsic dimensionality
)

# Build augmented index with decoys and risk map
defense.build(database, access_policy)

# Private query
results = defense.query(q, user_id, k=10)
```

### Attack Evaluation

```python
from src.attacks import DistanceSkewAttack, TopologyAttack, TriangulationAttack

# Channel 1
ch1 = DistanceSkewAttack(k=10)
auc_ch1 = ch1.evaluate(queries, results, ground_truth)

# Channel 2
ch2 = TopologyAttack(k=10, d_int=48)
auc_ch2 = ch2.evaluate(queries, results, ground_truth)

# Channel 3
ch3 = TriangulationAttack(k=10, m=20)
auc_ch3 = ch3.evaluate(queries, results, ground_truth)
```

## Hardware

Experiments were run on:
- **CPU:** 64-core AMD EPYC, 256 GB RAM
- **GPU:** NVIDIA A100 40 GB (embedding computation only)
- **Software:** Python 3.10, FAISS 1.7.4, hnswlib 0.7.0, PyTorch 2.1

## Embedding Models

| Model | Dimension | HuggingFace ID |
|-------|-----------|----------------|
| BGE-M3 (default) | 1024 | `BAAI/bge-m3` |
| MiniLM-L6 | 384 | `sentence-transformers/all-MiniLM-L6-v2` |
| BioClinicalBERT | 768 | `emilyalsentzer/Bio_ClinicalBERT` |
| text-embedding-3-large | 3072 | OpenAI API |

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

This repository will be de-anonymized upon acceptance.
