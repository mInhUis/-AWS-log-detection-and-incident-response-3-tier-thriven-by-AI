"""ChromaDB-backed RAG retriever for MITRE ATT&CK technique lookup (Day 4).

**Why ChromaDB?**
We need a vector store that satisfies three constraints:
(1) zero-server: no Docker daemon, no network port, safe for the laptop demo,
(2) persistent: survives Colab disconnects (CLAUDE.md §13) by writing to disk,
(3) idempotent build: re-running ``build_rag_index.py`` should not re-embed
    documents that are already indexed.

ChromaDB's ``PersistentClient`` uses SQLite + Parquet on disk.  The collection
survives restarts and can be mounted from Google Drive on Colab.

**Alternatives rejected:**
  - Qdrant: excellent production choice but requires a running server or Docker
    container — adds an operational dependency to the laptop demo.
  - FAISS: CPU-fast but no persistence layer and no metadata filtering; would
    require manual serialisation.
  - Weaviate: cloud-first, not local-first.

**Index build time:** ~30 s for 706 documents (700 ATT&CK + 6 flaws) on the
Dell Ryzen 7 5825U in CPU-only mode.  This is a one-time cost.

**Query latency:** < 50 ms per query.  ChromaDB does brute-force ANN at
this corpus size, which is fine — we are not building a production SIEM.

**Pipeline position (CLAUDE.md §3):**
    build_rag_index.py  →  RAGRetriever.build_index(docs)   [once, Day 4]
    Tier 3 inference    →  RAGRetriever.query(text, top_k)  [per flagged session]
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Final

from .corpus import AttackDocument
from .embedder import LocalEmbedder

# ---------------------------------------------------------------------------
#  Paths + constants (env-var overridable per CLAUDE.md §5.5)
# ---------------------------------------------------------------------------

# parents[3]: src/llm/rag/retriever.py → rag → llm → src → project_root
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]

CHROMA_DB_PATH: Final[Path] = Path(
    os.environ.get(
        "CHROMA_DB_PATH",
        str(_PROJECT_ROOT / "data" / "rag" / "chroma_db"),
    )
)

_COLLECTION_NAME: Final[str] = "attack_corpus"

DEFAULT_TOP_K: Final[int] = int(os.environ.get("RAG_TOP_K", "4"))

# ChromaDB upsert batch size — avoids SQLite row-count limits on large corpora
_UPSERT_BATCH: Final[int] = 500

# ---------------------------------------------------------------------------
#  RAGRetriever
# ---------------------------------------------------------------------------


class RAGRetriever:
    """ChromaDB-backed retriever over the ATT&CK + flaws.cloud corpus.

    **Two primary operations:**
    1. ``build_index(docs, overwrite=False)`` — embed all documents and persist.
       Idempotent: skips documents already in the collection unless overwrite.
    2. ``query(text, top_k)`` — embed the query and return the top-k most
       similar document texts ordered by cosine similarity.

    Both operations share the same ``LocalEmbedder`` instance so the same
    model weights are used for document embedding (passage mode) and query
    embedding (instruction mode), as required for correct cosine similarity.

    **Cosine similarity via L2-normalised embeddings:**
    BGE-small-en-v1.5 returns L2-normalised vectors.  ChromaDB's ``cosine``
    HNSW space computes ``1 - cosine_similarity``, meaning smaller distance =
    more similar.  We set ``hnsw:space=cosine`` when creating the collection
    so ChromaDB uses cosine distance rather than the default L2 distance.

    Usage::

        retriever = RAGRetriever()
        retriever.build_index(docs)                  # one-time build
        results = retriever.query("CreateUser AttachUserPolicy", top_k=4)
        # results: list[str], ordered by similarity
    """

    def __init__(
        self,
        db_path: Path = CHROMA_DB_PATH,
        embedder: LocalEmbedder | None = None,
    ) -> None:
        self._db_path: Path = db_path
        # Shared embedder so doc and query embeddings use identical weights
        self._embedder: LocalEmbedder = embedder or LocalEmbedder()
        self._client: Any = None       # chromadb.PersistentClient; lazy-loaded
        self._collection: Any = None   # chromadb.Collection; lazy-loaded

    # ------------------------------------------------------------------ #
    #  Internal: lazy loading                                              #
    # ------------------------------------------------------------------ #

    def _ensure_client(self) -> None:
        """Initialise the ChromaDB persistent client (idempotent)."""
        if self._client is not None:
            return

        try:
            import chromadb  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "chromadb is required for the RAG retriever.\n"
                "Install: pip install chromadb~=0.5"
            ) from exc

        # Create the directory if it doesn't exist yet
        self._db_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._db_path))

    def _ensure_collection(self) -> None:
        """Get or create the ChromaDB collection (idempotent)."""
        self._ensure_client()
        if self._collection is None:
            # hnsw:space=cosine means ChromaDB computes 1-cosine_similarity as
            # the distance metric, so lower distance = more similar
            self._collection = self._client.get_or_create_collection(
                name=_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

    # ------------------------------------------------------------------ #
    #  Index building                                                      #
    # ------------------------------------------------------------------ #

    def build_index(
        self,
        docs: list[AttackDocument],
        overwrite: bool = False,
    ) -> None:
        """Embed all documents and persist to ChromaDB.

        Already-indexed documents (by doc_id) are skipped unless
        ``overwrite=True``.  This makes the build idempotent — safe to re-run
        after a partial failure, a Colab disconnect, or an incremental corpus
        update.

        The upsert is batched in chunks of ``_UPSERT_BATCH`` (default 500) to
        avoid SQLite's row-count limit per transaction.

        Args:
            docs:      Corpus from ``corpus.py`` — ATT&CK techniques + flaws
                       writeups combined into a single list.
            overwrite: If True, delete the existing collection and rebuild from
                       scratch.  Use this after updating the STIX bundle.
        """
        self._ensure_collection()

        if overwrite:
            # Delete and recreate so the HNSW index is rebuilt cleanly
            self._client.delete_collection(_COLLECTION_NAME)
            self._collection = self._client.get_or_create_collection(
                name=_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

        # Find which doc IDs are already in the collection
        existing_result = self._collection.get(include=[])  # IDs only, no data
        existing_ids: set[str] = set(
            existing_result.get("ids", []) if existing_result else []
        )

        # Filter out already-indexed documents (idempotent build)
        new_docs: list[AttackDocument] = [
            d for d in docs if d.doc_id not in existing_ids
        ]

        if not new_docs:
            print(
                f"[RAG] Index already up to date "
                f"({len(existing_ids)} documents in '{_COLLECTION_NAME}')."
            )
            return

        print(f"[RAG] Embedding {len(new_docs)} new documents ...")

        texts: list[str] = [d.text for d in new_docs]
        ids: list[str] = [d.doc_id for d in new_docs]
        metadatas: list[dict[str, str]] = [d.metadata for d in new_docs]

        # embeddings: list[list[float]], each length 384 — from LocalEmbedder.embed()
        embeddings: list[list[float]] = self._embedder.embed(texts)

        # Upsert in batches to avoid SQLite row limits
        for start in range(0, len(new_docs), _UPSERT_BATCH):
            end: int = min(start + _UPSERT_BATCH, len(new_docs))
            self._collection.upsert(
                ids=ids[start:end],
                documents=texts[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
            )
            print(
                f"[RAG]   Upserted batch [{start}:{end}] "
                f"({end - start} documents)"
            )

        total: int = len(existing_ids) + len(new_docs)
        print(f"[RAG] Index built. Total documents in collection: {total}")

    # ------------------------------------------------------------------ #
    #  Querying                                                            #
    # ------------------------------------------------------------------ #

    def query(
        self,
        query_text: str,
        top_k: int = DEFAULT_TOP_K,
        source_filter: str | None = None,
    ) -> list[str]:
        """Retrieve the top-k most relevant documents for a query.

        The query is embedded with the BGE query-instruction prefix (see
        ``embedder.py``).  ChromaDB returns the nearest neighbours by cosine
        distance (lower = more similar) from the pre-built index.

        This is called at Tier 3 inference time, once per flagged session.  At
        our corpus size (~700 docs) the brute-force ANN takes < 50 ms.

        Args:
            query_text:    Natural-language description of the anomaly, e.g.
                           ``"CreateUser AttachUserPolicy from unexpected IP —
                           possible privilege escalation"``.
            top_k:         Number of documents to return.  Default is 4 (matches
                           ``tier3.rag_top_k`` in configs/default.yaml).
            source_filter: If ``'mitre_attack'`` or ``'flaws_cloud'``, restrict
                           retrieval to that source.  ``None`` searches all docs.

        Returns:
            ``list[str]`` of retrieved document texts, ordered by similarity
            (most similar first).  Returns an empty list if the index is empty.

        Raises:
            RuntimeError: If the index has not been built yet.
        """
        self._ensure_collection()

        n_docs: int = self._collection.count()
        if n_docs == 0:
            raise RuntimeError(
                "RAG index is empty.  Build it first:\n"
                "  python scripts/build_rag_index.py --mode mock   # quick test\n"
                "  python scripts/build_rag_index.py --mode full   # full corpus"
            )

        # query_embedding: list[float], length 384, L2-normalised (query mode)
        query_embedding: list[float] = self._embedder.embed_query(query_text)

        # Build the where clause for optional source filtering
        where_clause: dict[str, str] | None = (
            {"source": source_filter} if source_filter else None
        )

        # n_results must not exceed the collection size — ChromaDB raises if it does
        n_results: int = min(top_k, n_docs)

        query_kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents"],   # only need the text, not embeddings or distances
        }
        if where_clause:
            query_kwargs["where"] = where_clause

        results = self._collection.query(**query_kwargs)

        # results["documents"]: list[list[str]], outer list is per query (always 1 here)
        retrieved: list[str] = (
            results["documents"][0] if results.get("documents") else []
        )
        return retrieved

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

    def count(self) -> int:
        """Return the number of documents currently in the index."""
        self._ensure_collection()
        return self._collection.count()

    def is_built(self) -> bool:
        """Return True if the index exists and contains at least one document."""
        try:
            self._ensure_collection()
            return self._collection.count() > 0
        except Exception:
            return False
