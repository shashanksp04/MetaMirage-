# Preload Pipeline Design Documentation

## Overview

The **Preload Pipeline** is a manifest-driven ingestion system that pre-populates the Chroma vector database used by the `rag_agent`.

Its primary purpose is to:

* Seed the vector database with authoritative reference sources
* Prevent cold-start retrieval failures
* Ensure first queries have meaningful semantic context
* Maintain versioned backups before every ingestion run
* Reuse as much of the existing `rag_agent` ingestion logic as possible

This pipeline is designed to operate independently of the runtime RAG agent, while producing a fully compatible persistent Chroma database directory.

---

# Architectural Principles

The pipeline is built around five core principles:

1. Safety-first ingestion (automatic versioned backups)
2. Reuse rag_agent chunking and deduplication logic
3. Manifest-driven configuration
4. Source-type modular adapters
5. Idempotent ingestion behavior

---

# Directory Structure

```
parent/
  rag_agent/
    tools/
      pdf_addition.py
      web_addition.py
    utils/
      Embedding.py
      ContentUtils.py

  preload_pipeline/
    bootstrap.py
    manifest.yaml
    PRELOAD_PIPELINE_DESIGN.md
    preload/
      ...
```

The `preload_pipeline` and `rag_agent` directories exist as siblings under the same parent directory.

---

# High-Level Pipeline Stages

## Stage 0 — Lock + Backup (Safety Layer)

This stage runs before any ingestion occurs.

Steps:

1. Acquire a file lock to prevent concurrent preload runs
2. Copy the entire Chroma persistence directory to:

   ```
   <persist_parent>/backups/<timestamp>_before_preload/
   ```
3. Optionally prune older backups

Why this matters:

* Guarantees rollback capability
* Protects against partial ingestion failures
* Allows experimentation without risk

This stage makes the preload pipeline behave like a transactional system.

---

## Stage 1 — Manifest Loading

The pipeline reads `manifest.yaml`.

The manifest defines:

* Source name
* Source type
* Paths or URLs
* Optional metadata fields
* Unique ID field (for CSV sources)

Example:

```yaml
sources:
  - name: usda_plants_csv
    type: csv
    path: data/seeds/usda_plants.csv
    entity_type: plant
    source_org: USDA
    tags: [plants, usa]
```

The manifest ensures ingestion is configuration-driven rather than hardcoded.

---

## Stage 2 — rag_agent Integration Layer

The pipeline intentionally reuses major parts of `rag_agent`.

### Reused Components

From `rag_agent.utils.Embedding`:

* `SentenceTransformerEmbeddingFunction`

From `rag_agent.utils.ContentUtils`:

* `chunk_by_tokens(...)`
* `compute_content_hash(...)`
* `content_hash_exists(...)`
* tokenizer configuration
* chunk_config

From `rag_agent.tools.web_addition`:

* `WebAddition.add_web_content(...)`

From `rag_agent.tools.pdf_addition`:

* `PDFAddition.add_pdf_content(...)`

This guarantees:

* Identical chunk boundaries
* Identical deduplication behavior
* Identical document formatting
* Identical embedding function
* Identical metadata structure (for web/pdf)

The preload pipeline creates the same Chroma client + collection as the rag agent:

```python
SentenceTransformerEmbeddingFunction(...)
chromadb.PersistentClient(...)
get_or_create_collection(...)
```

This ensures the produced persistence directory is fully compatible with the RAG agent.

---

## Stage 3 — Source Adapters

Each source type has its own adapter.

### 1) Web Sources

Adapter: `WebPageListAdapter`

Behavior:

* Calls `WebAddition.add_web_content(url)`
* Extraction handled via trafilatura (inside rag_agent)
* Chunking handled via `ContentUtils.chunk_by_tokens`
* Deduplication handled via content hash logic
* Documents written via `collection.add(...)`

This is full reuse of rag_agent ingestion logic.

---

### 2) PDF Sources

Adapter: `PDFDirAdapter`

Behavior:

* Iterates over PDFs in directory
* Calls `PDFAddition.add_pdf_content(...)`
* Extraction via `pypdf`
* Chunking via rag_agent token chunker
* Deduplication via content hash
* Writes via `collection.add(...)`

Again, full reuse.

---

### 3) CSV Sources

Adapter: `CSVAdapter`

Since rag_agent does not have CSV ingestion logic, preload implements:

Row → structured narrative text → rag_agent chunking → rag_agent dedupe → collection.add()

Key details:

* Uses `ContentUtils.chunk_by_tokens`
* Uses `compute_content_hash`
* Uses `content_hash_exists`
* Stores metadata including:

  * source_name
  * record_id
  * entity_type
  * tags
  * content_hash

This ensures CSV data behaves identically to web/pdf chunks.

---

## Stage 4 — Run Report

After ingestion completes, a JSON report is written:

```
preload_run_report_<timestamp>.json
```

Includes:

* Sources processed
* Items added
* Items skipped (duplicates)
* Failures
* Backup path
* Timestamps

This makes ingestion auditable.

---

# How We Reuse rag_agent Code

The preload pipeline is not a parallel ingestion system.

Instead, it:

* Imports rag_agent as a module
* Uses the same embedding class
* Uses the same ContentUtils
* Uses the same chunking config
* Uses the same deduplication method
* Uses the same document formatting conventions

This ensures:

If a document was ingested via preload, it is indistinguishable from one ingested dynamically during runtime by the rag agent.

There is zero divergence in behavior.

---

# Metadata Behavior

For:

* Web sources → metadata comes from rag_agent tool
* PDF sources → metadata comes from rag_agent tool
* CSV sources → metadata is explicitly attached

If desired, you can modify rag_agent tools to accept `extra_metadata` for richer provenance.

---

# Persistence Directory Strategy

The pipeline writes directly to the Chroma persistence directory used by rag_agent.

You have two options:

Option A:

* Preload writes to a temp directory
* Copy full directory into rag_agent

Option B (recommended):

* Preload writes directly to rag_agent’s persistence directory

In both cases:
Always copy the entire directory, never just SQLite files.

---

# Running the Pipeline

## Step 1 — Install Requirements

From `preload_pipeline/`:

```
pip install -r requirements.txt
```

---

## Step 2 — Prepare Manifest

Create:

```
preload_pipeline/manifest.yaml
```

Based on the example.

---

## Step 3 — Run

From `preload_pipeline/`:

```
python bootstrap.py \
  --manifest manifest.yaml \
  --persist-dir ../rag_agent/chroma_database/chroma_db \
  --collection meta-mirage_collection \
  --rag-agent-dir ../rag_agent
```

Arguments explained:

* `--manifest` → Path to manifest.yaml
* `--persist-dir` → Chroma persistence directory
* `--collection` → Chroma collection name (must match rag_agent)
* `--rag-agent-dir` → Path to rag_agent directory
* `--embed-model` → Must match rag_agent embedding model
* `--device` → Must match rag_agent device setting
* `--dry-run` → Runs pipeline without writing to DB

---

# Recommended Operational Workflow

1. Stop rag_agent if running
2. Run preload pipeline
3. Review run report
4. Restart rag_agent

---

# Safety Guarantees

Every run:

* Creates versioned backup
* Prevents concurrent runs
* Logs ingestion results
* Maintains deduplication
* Preserves chunk consistency

---

# Future Improvements

Possible enhancements:

* Add extra_metadata support to rag_agent tools
* Add incremental update mode
* Add source-level refresh policies
* Add validation queries after ingestion
* Add checksum validation of persistence directory

---

# Summary

The preload pipeline is:

A versioned, manifest-driven, safety-first ingestion system that fully reuses rag_agent chunking, deduplication, and embedding logic to ensure database consistency and high-quality retrieval from the very first query.

It avoids architectural drift and ensures the vector database built offline behaves identically to runtime ingestion.

---
