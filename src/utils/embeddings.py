"""
Embedding Model Wrapper

Provides unified interface for computing embeddings with different models.
See Appendix D.2 for model identifiers and configuration.
"""

import numpy as np
from typing import Optional

MODEL_REGISTRY = {
    "bge-m3": {"hf_id": "BAAI/bge-m3", "dim": 1024},
    "minilm": {"hf_id": "sentence-transformers/all-MiniLM-L6-v2", "dim": 384},
    "bioclinbert": {"hf_id": "emilyalsentzer/Bio_ClinicalBERT", "dim": 768},
    "text-emb-3": {"provider": "openai", "model": "text-embedding-3-large", "dim": 3072},
}


class EmbeddingModel:
    """Wrapper for embedding computation."""

    def __init__(self, model_name: str = "bge-m3", device: str = "cuda"):
        self.model_name = model_name
        self.config = MODEL_REGISTRY[model_name]
        self.device = device
        self.model = None
        self.tokenizer = None

    def load(self):
        """Load the embedding model."""
        if self.config.get("provider") == "openai":
            return  # Use API, no local model
        from transformers import AutoModel, AutoTokenizer
        import torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["hf_id"])
        self.model = AutoModel.from_pretrained(self.config["hf_id"]).to(self.device)
        self.model.eval()

    def encode(
        self, texts: list[str], batch_size: int = 128, normalize: bool = True
    ) -> np.ndarray:
        """Encode texts to embeddings."""
        import torch
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token

            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)
