from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import chromadb

# ✅ Correct imports for your directory structure
from rag_agent.utils.Embedding import SentenceTransformerEmbeddingFunction
from rag_agent.utils.ContentUtils import ContentUtils
from rag_agent.tools.web_addition import WebAddition
from rag_agent.tools.pdf_addition import PDFAddition


class _DryRunCollection:
    """
    A tiny shim to prevent writing to disk during --dry-run while still exercising ingestion logic.
    Mimics the subset of Chroma Collection used by rag_agent tools:
      - get(where=...)
      - add(documents=..., metadatas=..., ids=...)
    """
    def __init__(self, real_collection, logger=None):
        self.real = real_collection
        self.logger = logger

    def get(self, *args, **kwargs):
        return self.real.get(*args, **kwargs)

    def add(self, *args, **kwargs):
        if self.logger:
            docs = kwargs.get("documents") or []
            self.logger.info(f"[DRY-RUN] Would add {len(docs)} docs")
        return None


def create_rag_agent_collection_and_utils(
    *,
    persist_dir: Path,
    collection_name: str,
    embed_model: str,
    device: str,
    dry_run: bool,
    logger=None,
) -> Tuple[Any, ContentUtils, WebAddition, PDFAddition]:
    """
    Creates the same Chroma collection + ContentUtils + tools that rag_agent uses,
    so chunking + dedupe + document formatting match exactly.
    """
    embedding_fn = SentenceTransformerEmbeddingFunction(embed_model, device)
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_fn)

    # ContentUtils uses transformers tokenizer with embed_model
    content_utils = ContentUtils(embed_model=embed_model)

    # Optionally wrap collection to prevent writes in dry run
    tool_collection = _DryRunCollection(collection, logger=logger) if dry_run else collection

    web_adder = WebAddition(collection=tool_collection, content_utils=content_utils)
    pdf_adder = PDFAddition(collection=tool_collection, content_utils=content_utils)

    return collection, content_utils, web_adder, pdf_adder