from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb

from preload.utils.hashing import sha1_hex
from preload.transforms.normalize import split_into_chunks


@dataclass
class UpsertStats:
    chunks_created: int = 0
    chunks_upserted: int = 0
    chunks_failed: int = 0


class ChromaUpserter:
    """
    Standalone upserter that writes text chunks into a Chroma persistent collection.

    This does NOT depend on your RAG agent internals, so it works even if you can't import them.
    """

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str,
        embedding_model_label: str,
        dry_run: bool = False,
        logger=None,
    ):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.embedding_model_label = embedding_model_label
        self.dry_run = dry_run
        self.logger = logger

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def chunk_and_upsert(self, text: str, metadata: Dict[str, Any], stable_id: Optional[str]) -> Dict[str, int]:
        chunks = split_into_chunks(text)
        stats = UpsertStats(chunks_created=len(chunks))

        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict[str, Any]] = []

        for idx, chunk in enumerate(chunks):
            chunk_hash = sha1_hex(chunk)
            base = stable_id or metadata.get("url") or metadata.get("path") or metadata.get("source_name") or "unknown"
            chunk_id = sha1_hex(f"{base}::chunk{idx}::${chunk_hash}")

            m = dict(metadata)
            m["chunk_index"] = idx
            m["content_hash"] = chunk_hash
            m["embedding_model_label"] = self.embedding_model_label

            ids.append(chunk_id)
            docs.append(chunk)
            metas.append(m)

        if self.dry_run:
            stats.chunks_upserted = len(ids)
            return {
                "chunks_created": stats.chunks_created,
                "chunks_upserted": stats.chunks_upserted,
                "chunks_failed": stats.chunks_failed,
            }

        # Chroma upsert is idempotent for ids (overwrites existing with same id).
        # If your Chroma version lacks upsert, swap to add and handle conflicts.
        try:
            self.collection.upsert(documents=docs, metadatas=metas, ids=ids)
            stats.chunks_upserted = len(ids)
        except Exception:
            stats.chunks_failed = len(ids)
            if self.logger:
                self.logger.exception("Chroma upsert failed.")
        return {
            "chunks_created": stats.chunks_created,
            "chunks_upserted": stats.chunks_upserted,
            "chunks_failed": stats.chunks_failed,
        }

    def close(self):
        # PersistentClient writes as it goes; nothing mandatory here.
        return