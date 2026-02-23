import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import torch
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", device: str = "None"):
        if device == "None":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)

    # Documents - list[str]
    # Embeddings - list[list[float]]
    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(input, normalize_embeddings=True, convert_to_numpy = True).tolist()