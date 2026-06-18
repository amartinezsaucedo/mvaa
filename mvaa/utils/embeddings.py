import json
from copy import deepcopy

import networkx as nx
import re
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(_EMBEDDING_MODEL_NAME)
    return _model



def embed_texts(
        texts: Union[str, List[str]],
        normalize: bool = True
) -> np.ndarray:
    single = False
    if isinstance(texts, str):
        texts = [texts]
        single = True

    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

    return embeddings[0] if single else embeddings

def normalize_identifier(name: str) -> str:
    if not name:
        return ""

    name = name.replace("_", " ")
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    name = re.sub(r"\s+", " ", name)

    return name.lower().strip()
