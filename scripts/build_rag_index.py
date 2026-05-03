"""Build the ChromaDB RAG index from the MITRE ATT&CK STIX bundle (Day 4).

This script is the Day 4 entry point.  Run it once to build the persistent
ChromaDB index that Tier 3 uses at inference time.  The index survives
restarts and does not need to be rebuilt unless the corpus changes.

**Modes:**
    mock  — Build a 6-document index from the built-in flaws.cloud writeups
            only.  No download, no heavy embedding.  Completes in < 5 s.
            Use this to verify the pipeline end-to-end before the full build.

    full  — Download the MITRE ATT&CK STIX bundle (~60 MB, once) and build
            the complete index: ~700 ATT&CK techniques + 6 flaws writeups.
            Embedding takes ~30 s on CPU (Ryzen 7 5825U).  The model
            (BAAI/bge-small-en-v1.5, ~130 MB) is downloaded on first run.

**Usage:**
    # From the project root:
    python scripts/build_rag_index.py --mode mock

    python scripts/build_rag_index.py --mode full

    # Force rebuild even if index already exists:
    python scripts/build_rag_index.py --mode full --overwrite

    # Custom paths via env vars:
    ATTACK_STIX_PATH=data/rag/enterprise-attack.json \\
    CHROMA_DB_PATH=data/rag/chroma_db \\
    python scripts/build_rag_index.py --mode full

**On Colab:**
    Mount Google Drive first so the index persists across sessions:
        from google.colab import drive
        drive.mount('/content/drive')
    Then set CHROMA_DB_PATH=/content/drive/MyDrive/thesis/data/rag/chroma_db
    and ATTACK_STIX_PATH=/content/drive/MyDrive/thesis/data/rag/enterprise-attack.json.

**STIX bundle source:**
    https://github.com/mitre/cti/blob/master/enterprise-attack/enterprise-attack.json
    The bundle is ~60 MB compressed.  It is gitignored (CLAUDE.md §8) and
    must be downloaded on first run.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path
from typing import Final

# Make project root importable when running as a standalone script
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.llm.rag.corpus import (
    AttackDocument,
    load_attack_stix,
    load_flaws_writeups,
)
from src.llm.rag.retriever import RAGRetriever, CHROMA_DB_PATH

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

_STIX_DOWNLOAD_URL: Final[str] = (
    "https://raw.githubusercontent.com/mitre/cti/master/"
    "enterprise-attack/enterprise-attack.json"
)

_DEFAULT_STIX_PATH: Final[Path] = (
    _PROJECT_ROOT / "data" / "rag" / "enterprise-attack.json"
)

# ---------------------------------------------------------------------------
#  STIX download
# ---------------------------------------------------------------------------


def download_stix(output_path: Path, force: bool = False) -> None:
    """Download the MITRE ATT&CK Enterprise STIX bundle from GitHub.

    The bundle is ~60 MB.  It is saved to ``output_path`` and reused on
    subsequent runs (unless ``force=True``).

    Args:
        output_path: Where to save the JSON file.
        force:       Re-download even if the file already exists.

    Raises:
        urllib.error.URLError: If the download fails (no network, GitHub down).
    """
    if output_path.exists() and not force:
        print(f"[STIX] Bundle already present at {output_path} (skip download).")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[STIX] Downloading MITRE ATT&CK bundle from GitHub (~60 MB) ...")
    print(f"[STIX] Source: {_STIX_DOWNLOAD_URL}")

    # urllib.request is stdlib — no extra dependency needed
    def _progress_hook(block_count: int, block_size: int, total_size: int) -> None:
        downloaded: int = block_count * block_size
        if total_size > 0:
            pct: float = min(downloaded / total_size * 100, 100.0)
            mb: float = downloaded / 1_048_576
            print(f"\r[STIX]   {pct:.1f}%  ({mb:.1f} MB)", end="", flush=True)

    urllib.request.urlretrieve(
        _STIX_DOWNLOAD_URL,
        str(output_path),
        reporthook=_progress_hook,
    )
    print()  # newline after progress bar
    size_mb: float = output_path.stat().st_size / 1_048_576
    print(f"[STIX] Saved {size_mb:.1f} MB to {output_path}")


# ---------------------------------------------------------------------------
#  Mock corpus builder
# ---------------------------------------------------------------------------


def _build_mock_corpus() -> list[AttackDocument]:
    """Build a minimal 6-document corpus from built-in flaws writeups only.

    Used in ``--mode mock`` to verify the full pipeline (embedder + ChromaDB)
    without downloading the 60 MB STIX bundle or waiting for ~700 embeddings.

    Returns:
        6 ``AttackDocument`` objects — one per flaws.cloud level.
    """
    # load_flaws_writeups falls back to the built-in descriptions when the
    # external JSONL file is absent — exactly right for mock mode
    docs: list[AttackDocument] = load_flaws_writeups()
    print(f"[Mock] Corpus: {len(docs)} documents (flaws.cloud writeups only).")
    return docs


# ---------------------------------------------------------------------------
#  Full corpus builder
# ---------------------------------------------------------------------------


def _build_full_corpus(
    stix_path: Path,
    download: bool = True,
) -> list[AttackDocument]:
    """Build the complete corpus: ATT&CK STIX + flaws.cloud writeups.

    Args:
        stix_path: Local path to the STIX bundle JSON.
        download:  If True and ``stix_path`` does not exist, download it.

    Returns:
        Combined ``list[AttackDocument]`` (~706 documents).
    """
    if not stix_path.exists():
        if download:
            download_stix(stix_path)
        else:
            raise FileNotFoundError(
                f"STIX bundle not found at {stix_path} and --no-download was set.\n"
                "Run without --no-download or copy the file manually."
            )

    print("[Full] Loading MITRE ATT&CK STIX bundle ...")
    attack_docs: list[AttackDocument] = load_attack_stix(stix_path)
    print(f"[Full] ATT&CK techniques loaded: {len(attack_docs)}")

    print("[Full] Loading flaws.cloud writeups ...")
    flaws_docs: list[AttackDocument] = load_flaws_writeups()
    print(f"[Full] flaws.cloud documents: {len(flaws_docs)}")

    combined: list[AttackDocument] = attack_docs + flaws_docs
    print(f"[Full] Total corpus: {len(combined)} documents")
    return combined


# ---------------------------------------------------------------------------
#  Index build orchestrator
# ---------------------------------------------------------------------------


def build_index(
    mode: str,
    stix_path: Path = _DEFAULT_STIX_PATH,
    chroma_path: Path = CHROMA_DB_PATH,
    overwrite: bool = False,
    no_download: bool = False,
) -> None:
    """Build (or update) the ChromaDB index from the selected corpus.

    Args:
        mode:        ``'mock'`` (6 flaws docs) or ``'full'`` (~706 docs).
        stix_path:   Path to the STIX JSON file.
        chroma_path: Directory for the ChromaDB persistent store.
        overwrite:   Delete and rebuild the index from scratch.
        no_download: Skip automatic STIX download (raises if file absent).
    """
    print(f"\n{'='*60}")
    print(f" Building RAG index  |  mode={mode}  |  overwrite={overwrite}")
    print(f" ChromaDB path: {chroma_path}")
    print(f"{'='*60}\n")

    if mode == "mock":
        docs: list[AttackDocument] = _build_mock_corpus()
    elif mode == "full":
        docs = _build_full_corpus(stix_path, download=not no_download)
    else:
        raise ValueError(f"Unknown mode '{mode}'.  Use 'mock' or 'full'.")

    retriever = RAGRetriever(db_path=chroma_path)
    retriever.build_index(docs, overwrite=overwrite)

    # ---- Sanity-check retrieval ----
    print("\n[Verify] Running sanity-check query ...")
    test_query: str = "privilege escalation via new IAM account creation"
    results: list[str] = retriever.query(test_query, top_k=2)
    if results:
        print(f"[Verify] Query: '{test_query}'")
        for i, doc in enumerate(results, start=1):
            # Print only the first line of each result for brevity
            first_line: str = doc.split("\n")[0]
            print(f"[Verify]   Result {i}: {first_line}")
        print(f"[Verify] Retrieval working correctly ({len(results)} results).")
    else:
        print("[Verify] WARNING: query returned no results — index may be empty.")

    print(f"\n[Done] RAG index built. Total documents: {retriever.count()}")
    print(
        "[Done] Use RAGRetriever to query at inference time:\n"
        "  from src.llm.rag.retriever import RAGRetriever\n"
        "  r = RAGRetriever()\n"
        "  docs = r.query('CreateUser AttachUserPolicy', top_k=4)"
    )


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for Day 4 RAG index build."""
    parser = argparse.ArgumentParser(
        description="Build the ChromaDB RAG index for Tier 3 LLM retrieval."
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "full"],
        default="mock",
        help=(
            "mock: 6-document index from built-in flaws writeups (no download). "
            "full: ~706-document index from ATT&CK STIX + flaws writeups."
        ),
    )
    parser.add_argument(
        "--stix-path",
        type=Path,
        default=_DEFAULT_STIX_PATH,
        help="Path to enterprise-attack.json STIX bundle.",
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=CHROMA_DB_PATH,
        help="Directory for ChromaDB persistent store.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing index and rebuild from scratch.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download the STIX bundle if absent (fail instead).",
    )
    args = parser.parse_args()

    build_index(
        mode=args.mode,
        stix_path=args.stix_path,
        chroma_path=args.chroma_path,
        overwrite=args.overwrite,
        no_download=args.no_download,
    )


if __name__ == "__main__":
    main()
