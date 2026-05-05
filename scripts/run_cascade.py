"""End-to-end cascade CLI: load all three tiers and run the full pipeline.

WHY THIS SCRIPT EXISTS:
    The cascade integration (src/pipeline/cascade.py) accepts pre-loaded model
    objects.  This script handles the I/O layer: reading raw CloudTrail events
    from a JSONL file, loading each model from disk with the correct format,
    wiring them into a CascadePipeline, and saving the incident reports.

    It also exposes the ablation flags (--skip-tier1) and the mock mode
    (--mode mock for Tier 3 without Ollama) that the thesis evaluation chapter
    requires.

USAGE:
    # Quick smoke test (no Ollama needed, uses mock Tier 3):
    python scripts/run_cascade.py --input data/processed/val_events.jsonl \\
        --output reports/ --mode mock

    # Full run with real Llama inference:
    python scripts/run_cascade.py --input data/raw/flaws_merged.jsonl \\
        --output reports/ --mode full --top-k 9 --prompt-version v2

    # Ablation: Tier 2 alone (no Isolation Forest pre-filter):
    python scripts/run_cascade.py --input data/processed/val_events.jsonl \\
        --output reports/ --mode mock --skip-tier1

ENVIRONMENT VARIABLES (all path arguments have env-var equivalents):
    CASCADE_INPUT            Path to input JSONL (--input default)
    CASCADE_OUTPUT           Path to output directory (--output default)
    ISOLATION_FOREST_PATH    Path to isolation_forest.pkl
    TIER1_BASELINE_PATH      Path to tier1_baseline.json
    DEEPLOG_MODEL_PATH       Path to deeplog_best.pt checkpoint
    DRAIN_STATE_PATH         Path to drain3_state.json
    CHROMA_DB_PATH           Path to ChromaDB directory
    RESPONDER_MODEL          Ollama model tag (e.g. llama3.1:8b-instruct-q4_K_M)
    RESPONDER_MOCK           'true' to force mock Tier 3

EXIT CODES:
    0 — success (even if no anomalies found; check reports/ for output)
    1 — fatal error (model file missing, invalid input, etc.)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
#  Project root on sys.path — must happen before any src.* imports
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import torch

from src.detector.deeplog import DeepLogModel
from src.detector.drain_parser import DrainParser
from src.llm.llama_inference import LlamaResponder, ResponderConfig
from src.llm.rag.retriever import RAGRetriever
from src.pipeline.cascade import CascadePipeline, CascadeResult
from src.tier1_ml.baseline import load_baseline

# ---------------------------------------------------------------------------
#  Default paths (all env-var overridable per CLAUDE.md §5.5)
# ---------------------------------------------------------------------------

_IF_PATH: Final[Path] = Path(
    os.environ.get(
        "ISOLATION_FOREST_PATH",
        str(_PROJECT_ROOT / "models" / "isolation_forest.pkl"),
    )
)
_BASELINE_PATH: Final[Path] = Path(
    os.environ.get(
        "TIER1_BASELINE_PATH",
        str(_PROJECT_ROOT / "models" / "tier1_baseline.json"),
    )
)
_DEEPLOG_PATH: Final[Path] = Path(
    os.environ.get(
        "DEEPLOG_MODEL_PATH",
        str(_PROJECT_ROOT / "models1" / "deeplog" / "deeplog_best.pt"),
    )
)
_DRAIN_STATE_PATH: Final[Path] = Path(
    os.environ.get(
        "DRAIN_STATE_PATH",
        str(_PROJECT_ROOT / "data" / "processeded1" / "drain3_state.json"),
    )
)
_CASCADE_INPUT: Final[Path] = Path(
    os.environ.get(
        "CASCADE_INPUT",
        str(_PROJECT_ROOT / "data" / "processed" / "val_events.jsonl"),
    )
)
_CASCADE_OUTPUT: Final[Path] = Path(
    os.environ.get(
        "CASCADE_OUTPUT",
        str(_PROJECT_ROOT / "reports"),
    )
)


# ---------------------------------------------------------------------------
#  Model loading helpers
# ---------------------------------------------------------------------------


def _load_deeplog(path: Path, device: torch.device) -> tuple[DeepLogModel, int]:
    """Load DeepLog checkpoint and return (model, window_size).

    The checkpoint format (from train_deeplog.py) stores all architecture
    hyperparameters alongside the weights, so we reconstruct the model from
    the checkpoint rather than from a separate config file — no mismatch risk.

    Args:
        path:   Path to deeplog.pt or deeplog_best.pt.
        device: Torch device to map model parameters to.

    Returns:
        (DeepLogModel, window_size) — model is in eval mode.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        KeyError:          If required architecture keys are missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"DeepLog checkpoint not found: {path}\n"
            "Train DeepLog first: python src/preprocessing/run_drain.py --mode full"
        )

    ckpt: dict = torch.load(path, map_location=device, weights_only=True)

    # Reconstruct architecture from stored hyperparameters.
    # train_deeplog.py always stores these keys in every checkpoint.
    model = DeepLogModel(
        num_keys=ckpt["num_keys"],
        embedding_dim=ckpt["embedding_dim"],
        hidden_size=ckpt["hidden_size"],
        num_layers=ckpt["num_layers"],
        dropout=ckpt["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    window_size: int = int(ckpt["window_size"])
    print(
        f"[cascade] DeepLog loaded: vocab={ckpt['num_keys']}  "
        f"window={window_size}  device={device}  "
        f"checkpoint_epoch={ckpt.get('epoch', '?')}"
    )
    return model, window_size


def _load_drain(path: Path) -> DrainParser:
    """Load a fitted DrainParser from its serialised state.

    Args:
        path: Path to the drain3_state.json file produced by DrainParser.save_state().

    Returns:
        DrainParser in read-only transform mode.

    Raises:
        FileNotFoundError: If the state file is missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Drain3 state file not found: {path}\n"
            "Run the Drain parser on training data first to generate the state file."
        )
    parser = DrainParser()
    parser.load_state(path)
    print(f"[cascade] Drain3 loaded: vocab={parser.num_keys} templates  path={path}")
    return parser


def _load_if_bundle(path: Path) -> dict:
    """Load the Isolation Forest joblib bundle.

    Args:
        path: Path to isolation_forest.pkl from train.py.

    Returns:
        Bundle dict with keys: model, feature_columns, contamination, etc.

    Raises:
        FileNotFoundError: If the pkl file is missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Isolation Forest bundle not found: {path}\n"
            "Train Tier 1: python -m src.tier1_ml.train"
        )
    bundle: dict = joblib.load(path)
    print(
        f"[cascade] IsolationForest loaded: "
        f"n_estimators={bundle.get('n_estimators', '?')}  "
        f"contamination={bundle.get('contamination', '?')}"
    )
    return bundle


# ---------------------------------------------------------------------------
#  CLI argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run the 3-tier CloudTrail anomaly detection cascade.  "
            "Use --mode mock to test the plumbing without a running Ollama daemon."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Input / output ---------------------------------------------------
    p.add_argument(
        "--input", type=Path, default=_CASCADE_INPUT,
        help="JSONL file of raw CloudTrail events (one JSON dict per line).",
    )
    p.add_argument(
        "--output", type=Path, default=_CASCADE_OUTPUT,
        help="Directory to write incident report JSON files.",
    )

    # ---- Mode -------------------------------------------------------------
    p.add_argument(
        "--mode", choices=["mock", "full"], default="mock",
        help=(
            "'mock': Tier 3 uses a template report (no Ollama needed) — "
            "tests the Tier 1 + Tier 2 plumbing.  "
            "'full': real Ollama inference (requires ollama serve + model pull)."
        ),
    )

    # ---- DeepLog parameters -----------------------------------------------
    p.add_argument(
        "--top-k", type=int, default=9,
        help=(
            "DeepLog top-k anomaly threshold.  Tune on validation only "
            "(CLAUDE.md §7).  Higher k → fewer false positives, lower recall."
        ),
    )
    p.add_argument(
        "--prompt-version", choices=["v1", "v2"], default="v2",
        help=(
            "Llama prompt template version.  "
            "v1: zero-shot (ablation baseline).  "
            "v2: RAG-augmented (primary evaluation config)."
        ),
    )

    # ---- Ablation ---------------------------------------------------------
    p.add_argument(
        "--skip-tier1", action="store_true",
        help=(
            "Bypass Tier 1 Isolation Forest and send ALL events to Tier 2.  "
            "Use this for the ablation study: 'cascade vs Tier 2 alone'."
        ),
    )

    # ---- Model paths (override defaults via env vars or these flags) ------
    p.add_argument(
        "--if-model", type=Path, default=_IF_PATH,
        help="Path to isolation_forest.pkl.",
    )
    p.add_argument(
        "--baseline", type=Path, default=_BASELINE_PATH,
        help="Path to tier1_baseline.json.",
    )
    p.add_argument(
        "--deeplog", type=Path, default=_DEEPLOG_PATH,
        help="Path to deeplog_best.pt checkpoint.",
    )
    p.add_argument(
        "--drain-state", type=Path, default=_DRAIN_STATE_PATH,
        help="Path to drain3_state.json.",
    )

    # ---- RAG (optional) ---------------------------------------------------
    p.add_argument(
        "--skip-rag", action="store_true",
        help="Disable RAG retrieval (uses v1 prompt even if --prompt-version v2).",
    )

    # ---- Misc -------------------------------------------------------------
    p.add_argument(
        "--max-events", type=int, default=None,
        help="Maximum number of events to read from --input (None = all).",
    )

    return p


# ---------------------------------------------------------------------------
#  Output helpers
# ---------------------------------------------------------------------------


def _print_metrics_table(metrics: dict) -> None:
    """Print a formatted cost/latency table for the thesis evaluation chapter."""
    print("\n" + "═" * 62)
    print("  CASCADE METRICS  (fill values into thesis cost table §3)")
    print("═" * 62)

    # ---- Volume counts ---------------------------------------------------
    rows = [
        ("Tier 1 input events",       metrics.get("tier1_input_events",       "N/A")),
        ("Tier 1 windows scored",     metrics.get("tier1_input_windows",      "N/A")),
        ("Tier 1 windows flagged",    metrics.get("tier1_flagged_windows",     "N/A")),
        ("Tier 1 principals flagged", metrics.get("tier1_flagged_principals",  "N/A")),
        ("Tier 2 input events",       metrics.get("tier2_input_events",        "N/A")),
        ("Tier 2 events flagged",     metrics.get("tier2_flagged_events",      "N/A")),
        ("Tier 3 sessions analysed",  metrics.get("tier3_input_sessions",      "N/A")),
    ]
    for label, val in rows:
        print(f"  {label:<30s} {val!s:>10s}")

    # ---- Reduction ratios (useful for the thesis cost argument) ----------
    t1_in  = metrics.get("tier1_input_events", 0)
    t2_in  = metrics.get("tier2_input_events", 0)
    t2_out = metrics.get("tier2_flagged_events", 0)
    if t1_in and t1_in > 0:
        print(f"\n  Tier 1 pass-through rate: {t2_in / t1_in:.2%}")
    if t2_in and t2_in > 0:
        print(f"  Tier 2 anomaly rate:      {t2_out / t2_in:.2%}")
    if t1_in and t1_in > 0 and t2_out is not None:
        print(f"  End-to-end anomaly rate:  {t2_out / t1_in:.4%}")

    # ---- Latency ---------------------------------------------------------
    print(f"\n  Tier 1 latency:  {metrics.get('tier1_latency_s', '?'):>8.3f} s")
    print(f"  Tier 2 latency:  {metrics.get('tier2_latency_s', '?'):>8.3f} s")
    print(f"  Tier 3 latency:  {metrics.get('tier3_latency_s', '?'):>8.3f} s")
    print(f"  Total latency:   {metrics.get('total_latency_s', '?'):>8.3f} s")

    if metrics.get("skip_tier1"):
        print("\n  [NOTE] Tier 1 was skipped (ablation mode --skip-tier1).")
    print("═" * 62 + "\n")


def _save_reports(reports: list[dict], output_dir: Path) -> None:
    """Write each incident report to a separate JSON file in output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, report in enumerate(reports):
        # Sanitise session_id for use as a filename.
        safe_id: str = str(report.get("session_id", f"session_{i}"))
        safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in safe_id)
        out_path: Path = output_dir / f"report_{i:04d}_{safe_id[:60]}.json"

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)

        print(
            f"  [saved] {out_path.name}  "
            f"(anomaly_ratio={report.get('anomaly_ratio', 0):.1%}  "
            f"flagged_events={len(report.get('flagged_events', []))})"
        )


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # ---- Device -----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cascade] Device: {device}")

    # ---- Load input events ------------------------------------------------
    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    events: list[dict] = []
    with open(args.input, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(
                    f"WARNING: skipping malformed JSON at line {lineno}: {exc}",
                    file=sys.stderr,
                )
            if args.max_events is not None and len(events) >= args.max_events:
                break

    print(f"[cascade] Loaded {len(events):,} events from {args.input}")
    if not events:
        print("[cascade] No events to process — exiting.")
        sys.exit(0)

    # ---- Load Tier 1 models -----------------------------------------------
    try:
        if_bundle: dict = _load_if_bundle(args.if_model)
        hour_baseline = load_baseline(args.baseline)
        print(
            f"[cascade] HourBaseline loaded: "
            f"{hour_baseline.train_principal_count} principals"
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # ---- Load Tier 2 models -----------------------------------------------
    try:
        deeplog_model, window_size = _load_deeplog(args.deeplog, device)
        drain_parser: DrainParser = _load_drain(args.drain_state)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # ---- Load Tier 3 (Llama + RAG) ----------------------------------------
    mock_mode: bool = (args.mode == "mock")

    responder_cfg = ResponderConfig(
        mock_mode=mock_mode,
        backend="ollama",
        prompt_version=args.prompt_version,
    )
    responder = LlamaResponder(config=responder_cfg)
    print(
        f"[cascade] LlamaResponder: mock={mock_mode}  "
        f"prompt_version={args.prompt_version}"
    )

    retriever: RAGRetriever | None = None
    if not args.skip_rag:
        retriever = RAGRetriever()
        if retriever.is_built():
            print(f"[cascade] RAGRetriever: {retriever.count()} documents in index")
        else:
            print(
                "[cascade] WARNING: RAG index is empty or not built. "
                "Run: python scripts/build_rag_index.py --mode full. "
                "Falling back to v1 prompt (no RAG context)."
            )
            retriever = None

    # ---- Instantiate and run the cascade ----------------------------------
    try:
        pipeline = CascadePipeline(
            if_bundle=if_bundle,
            hour_baseline=hour_baseline,
            deeplog_model=deeplog_model,
            window_size=window_size,
            drain_parser=drain_parser,
            responder=responder,
            retriever=retriever,
            top_k=args.top_k,
            skip_tier1=args.skip_tier1,
        )
    except ValueError as exc:
        print(f"ERROR: Pipeline initialisation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(
        f"\n[cascade] Running cascade on {len(events):,} events "
        f"(top_k={args.top_k}  skip_tier1={args.skip_tier1}) ...\n"
    )

    result: CascadeResult = pipeline.process(events)

    # ---- Print metrics table ----------------------------------------------
    _print_metrics_table(result.metrics)

    # ---- Save reports -----------------------------------------------------
    if result.reports:
        print(f"[cascade] {len(result.reports)} incident report(s) generated:")
        _save_reports(result.reports, args.output)
    else:
        print("[cascade] No anomalous sessions found — no reports generated.")
        print(
            "  This is expected for clean / synthetic-normal-only input data."
        )

    # ---- Summary ----------------------------------------------------------
    print(
        f"\n[cascade] Done.  {len(result.reports)} report(s) saved to {args.output}"
    )


if __name__ == "__main__":
    main()
