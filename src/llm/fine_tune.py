"""QLoRA fine-tuning of Llama-3.1-8B-Instruct on CloudTrail incident reports.

**Path A — QLoRA fine-tuning with Unsloth (Days 5-6, Colab GPU only)**

Path A is attempted on Day 5 and evaluated by end of Day 5.  If training is
unstable or outputs are poor, fall back to **Path B** immediately:
    Path B = prompt engineering + RAG only (skip this script entirely).
Path B still uses ``llama_inference.py`` with ``backend='ollama'`` and
``prompt_version='v2'``.  The thesis must document which path was taken.

**Why QLoRA?**
Hu et al. (2021) introduced Low-Rank Adaptation (LoRA): instead of updating
all model weights, LoRA learns rank-decomposed weight *deltas* (rank r=8 adds
~1M trainable parameters vs ~8B full).  Dettmers et al. (2023) extended this
to QLoRA: a 4-bit quantised frozen base + 16-bit LoRA adapters.  At Llama-3-8B:
  - 4-bit quantised base: ~5 GB VRAM
  - 16-bit adapters (r=8, two attention matrices): ~0.05 GB
  - Gradient buffers and optimizer state: ~3 GB
  Total: ~8 GB — fits comfortably in a Colab T4 GPU (15 GB).

**Why r=8, alpha=16?**
Hu et al. (2021) Table 4 shows that r=8 matches full fine-tuning quality on
instruction-following tasks when the dataset is small (< 1 000 examples).
alpha=16 means the LoRA update scale = alpha / r = 2, which is the standard
stable initialisation (Unsloth default).

**Why target_modules = ["q_proj", "v_proj"]?**
Adapting only the query and value attention projection matrices is the minimal
set that achieves instruction-following improvements.  Expanding to all linear
layers (gate_proj, up_proj, down_proj) marginally improves quality but doubles
adapter size and training time — not worth it for a 100-example dataset.

**Training data format:**
Each fine-tuning pair is a JSONL line with a ``text`` field containing the
full Llama-3 chat conversation (system + user + assistant completion), already
formatted with the special tokens.  SFTTrainer from the ``trl`` library reads
this format natively.

    {
        "text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\n
                 {system}<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\n
                 {user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n
                 {completion}<|eot_id|>"
    }

CRITICAL — data discipline (CLAUDE.md §9):
  - Fine-tune only on VALIDATION-set scenarios (levels 3+4).
  - NEVER use test-set scenarios (levels 2+5) for training data.
  - Synthetic data augmentation is permitted but must be quality-controlled.
  - The fine-tuning set should be small (50–200 pairs) to avoid overfitting
    on a 6-scenario dataset.

**CRITICAL: Colab-only script.**
This script must NOT be run on the Dell laptop (no CUDA GPU).  Mock mode
exits immediately without importing Unsloth or allocating any GPU memory.

**Environment variables:**
    FINETUNE_DATA_PATH   — Path to finetune_pairs.jsonl
    ADAPTER_OUTPUT_DIR   — Where to save adapter weights
    FINETUNE_BASE_MODEL  — HuggingFace model ID for the 4-bit base
    CUDA_MEMORY_MAX      — VRAM fraction cap (default: 0.85 — more than
                           llama_inference.py because this script has the
                           whole GPU to itself on Colab)
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

# ---------------------------------------------------------------------------
#  Paths (env-var overridable per CLAUDE.md §5.5)
# ---------------------------------------------------------------------------

# parents[2]: src/llm/fine_tune.py → llm → src → project_root
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

FINETUNE_DATA_PATH: Final[Path] = Path(
    os.environ.get(
        "FINETUNE_DATA_PATH",
        str(_PROJECT_ROOT / "data" / "rag" / "finetune_pairs.jsonl"),
    )
)

ADAPTER_OUTPUT_DIR: Final[Path] = Path(
    os.environ.get(
        "ADAPTER_OUTPUT_DIR",
        str(_PROJECT_ROOT / "models" / "llama" / "adapter"),
    )
)

_BASE_MODEL: Final[str] = os.environ.get(
    "FINETUNE_BASE_MODEL",
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
)

_CUDA_MEMORY_MAX: Final[float] = float(
    os.environ.get("CUDA_MEMORY_MAX", "0.85")
)

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------


@dataclass
class FineTuneConfig:
    """QLoRA fine-tuning hyperparameters.

    Attributes:
        lora_r:           LoRA rank.  8 is the sweet spot for small datasets —
                          see Hu et al. (2021) Table 4.
        lora_alpha:       Scaling factor.  alpha/r = 2 is standard (Unsloth
                          default).
        lora_dropout:     Light regularisation.  0.05 for a ~100-example set.
        target_modules:   Attention matrices to adapt.  q + v is minimal set.
        max_seq_length:   Truncate prompt+completion.  2 048 is safe for the
                          system + user + 5-section report.
        num_epochs:       Epochs over the fine-tuning set.  3 converges on
                          ~100 examples; use 5 for < 50.
        batch_size:       Per-device batch.  1 is safest on T4 (15 GB).
        grad_accum_steps: Effective batch = batch_size * steps.  4 → effective
                          batch of 4, which is standard for instruction tuning.
        learning_rate:    2e-4 is the Unsloth-recommended QLoRA default.
        warmup_ratio:     10% warmup to avoid early divergence.
        output_dir:       Where adapter weights are saved after training.
        mock_mode:        Exit immediately without loading any model.
                          Default True — Colab only.
    """

    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    max_seq_length: int = 2048
    num_epochs: int = 3
    batch_size: int = 1
    grad_accum_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    output_dir: Path = field(default_factory=lambda: ADAPTER_OUTPUT_DIR)
    mock_mode: bool = True  # default mock per CLAUDE.md §5.6


# ---------------------------------------------------------------------------
#  Fine-tuning data helpers
# ---------------------------------------------------------------------------


def _build_mock_pairs() -> list[dict[str, str]]:
    """Generate 3 synthetic fine-tuning pairs for mock mode / pipeline testing.

    These are NOT real incidents — they are minimal examples that verify the
    training loop can process the data format without errors.  Do not confuse
    these with real evaluation data.
    """
    from .prompts.v1 import SYSTEM_PROMPT

    def _make_pair(
        events: list[str], evt_names: list[str], completion: str
    ) -> dict[str, str]:
        user_block = (
            "Session ID: MOCK-001\n"
            f"Total events in session: {len(events)}  |  "
            f"Flagged as anomalous: {len(events)} (100.0%)\n\n"
            "Flagged events (DeepLog top-k anomaly detection):\n"
        )
        for i, (log, name) in enumerate(zip(events, evt_names)):
            user_block += (
                f"  {i+1}. [2024-01-01T00:00:00Z] {name}\n"
                f"     Drain template : {log}\n"
                f"     Source IP      : 1.2.3.4\n"
                f"     Identity (ARN) : arn:aws:iam::123456789012:user/test\n"
            )

        # Full Llama-3 conversation with the special tokens
        text = (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            f"{SYSTEM_PROMPT}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_block}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{completion}<|eot_id|>"
        )
        return {"text": text}

    pairs = [
        _make_pair(
            events=["CreateUser <*>", "AttachUserPolicy <*>"],
            evt_names=["CreateUser", "AttachUserPolicy"],
            completion=(
                "## 1. Incident Summary\n\n"
                "Two consecutive IAM API calls — CreateUser followed by "
                "AttachUserPolicy — were flagged as anomalous by DeepLog.\n\n"
                "## 2. Anomaly Analysis\n\n"
                "  - CreateUser: creating an IAM user outside normal provisioning "
                "workflows is a privilege escalation indicator.\n"
                "  - AttachUserPolicy: immediately attaching a policy to the new "
                "user suggests an automated persistence mechanism.\n\n"
                "## 3. Severity Assessment\n\n"
                "**High** — sequential account creation + policy attachment "
                "is a classic persistence pattern.\n\n"
                "## 4. Attack Pattern Classification\n\n"
                "MITRE ATT&CK: T1136 (Create Account), T1098 (Account "
                "Manipulation) — Privilege Escalation / Persistence.\n\n"
                "## 5. Recommended Mitigation\n\n"
                "1. Disable and investigate the newly created IAM user.\n"
                "2. Audit all policies attached within the last 24 hours.\n"
                "3. Rotate credentials of the identity that performed these calls.\n\n"
                "**[Requires analyst review before any automated action.]**"
            ),
        ),
        _make_pair(
            events=["GetObject <*>", "PutObject <*>"],
            evt_names=["GetObject", "PutObject"],
            completion=(
                "## 1. Incident Summary\n\n"
                "Anomalous S3 GetObject followed by PutObject from an "
                "unexpected source suggests data staging for exfiltration.\n\n"
                "## 2. Anomaly Analysis\n\n"
                "  - GetObject: reading from a bucket that is not in this "
                "identity's normal access pattern.\n"
                "  - PutObject: writing to a second bucket — possible staging.\n\n"
                "## 3. Severity Assessment\n\n"
                "**Medium** — data movement pattern warrants investigation but "
                "may be legitimate cross-bucket migration.\n\n"
                "## 4. Attack Pattern Classification\n\n"
                "MITRE ATT&CK: T1530 (Data from Cloud Storage Object) — "
                "Collection.\n\n"
                "## 5. Recommended Mitigation\n\n"
                "1. Review S3 bucket policies and access logs.\n"
                "2. Enable S3 access logging if not active.\n"
                "3. Quarantine the destination bucket pending investigation.\n\n"
                "**[Requires analyst review before any automated action.]**"
            ),
        ),
        _make_pair(
            events=["AssumeRole <*>", "GetCallerIdentity <*>"],
            evt_names=["AssumeRole", "GetCallerIdentity"],
            completion=(
                "## 1. Incident Summary\n\n"
                "AssumeRole followed by GetCallerIdentity from a new principal "
                "indicates credential verification after role assumption.\n\n"
                "## 2. Anomaly Analysis\n\n"
                "  - AssumeRole: the source principal is not in the normal role "
                "assumption baseline.\n"
                "  - GetCallerIdentity: common first call by attackers to verify "
                "stolen credentials are still valid.\n\n"
                "## 3. Severity Assessment\n\n"
                "**High** — stolen credential verification pattern.\n\n"
                "## 4. Attack Pattern Classification\n\n"
                "MITRE ATT&CK: T1078.004 (Valid Cloud Accounts), "
                "T1550.001 (Application Access Token) — Lateral Movement.\n\n"
                "## 5. Recommended Mitigation\n\n"
                "1. Revoke the assumed role session tokens immediately.\n"
                "2. Investigate the source of the AssumeRole call.\n"
                "3. Enable CloudTrail log-file validation.\n\n"
                "**[Requires analyst review before any automated action.]**"
            ),
        ),
    ]
    return pairs


def load_finetune_data(data_path: Path = FINETUNE_DATA_PATH) -> list[dict[str, str]]:
    """Load fine-tuning pairs from JSONL.

    Each line must be a JSON object with a ``text`` field containing the full
    Llama-3 conversation in special-token format (system + user + assistant
    completion with ``<|eot_id|>`` at the end of the assistant turn).

    **Data discipline (CLAUDE.md §9):**
    The pairs in this file MUST come only from the VALIDATION set scenarios
    (flaws.cloud levels 3 and 4).  Do NOT include test-set scenarios (levels
    2 and 5) — that would contaminate the test evaluation.

    Args:
        data_path: Path to the JSONL file.

    Returns:
        List of ``{"text": str}`` dicts, one per training example.

    Raises:
        FileNotFoundError: If the JSONL file does not exist and the caller
            is not in mock mode (error message explains how to create the file).
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"Fine-tuning data not found at {data_path}.\n"
            "To create it:\n"
            "  1. Run the cascade on flaws.cloud validation scenarios (levels 3+4).\n"
            "  2. Manually review and annotate the incident reports.\n"
            "  3. Save as JSONL: one {\"text\": \"<full_llama3_conversation>\"} per line.\n"
            "  4. Store at data/rag/finetune_pairs.jsonl\n"
            "Or use mock mode: --mode mock"
        )

    pairs: list[dict[str, str]] = []
    with data_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_no} of {data_path}: {exc}"
                ) from exc
            if "text" not in obj:
                raise ValueError(
                    f"Missing 'text' key on line {line_no} of {data_path}.  "
                    "Each line must be {\"text\": \"...\"}."
                )
            pairs.append({"text": obj["text"]})

    print(f"[FineTune] Loaded {len(pairs)} training pairs from {data_path}")
    return pairs


# ---------------------------------------------------------------------------
#  QLoRA training
# ---------------------------------------------------------------------------


def train(config: FineTuneConfig, data_path: Path = FINETUNE_DATA_PATH) -> None:
    """Fine-tune Llama-3.1-8B-Instruct with QLoRA on incident report pairs.

    **Colab-only.**  Do not run on the Dell laptop.  Mock mode exits without
    importing Unsloth or allocating any GPU memory.

    Steps:
    1. Load the 4-bit quantised base model with Unsloth.
    2. Attach LoRA adapters (r=8, alpha=16, q+v attention matrices).
    3. Load training data (or mock pairs if mock_mode=True).
    4. Run SFTTrainer for ``config.num_epochs`` epochs.
    5. Save adapter weights to ``config.output_dir``.
    6. Print GGUF conversion instructions (for Ollama deployment).

    Args:
        config:    Fine-tuning hyperparameters.
        data_path: Path to the JSONL fine-tuning data.

    Raises:
        RuntimeError: If CUDA is not available and mock_mode is False.
        FileNotFoundError: If data_path does not exist and mock_mode is False.
    """
    if config.mock_mode:
        print("[FineTune] Mock mode — exiting without loading model.")
        print("[FineTune] Set mock_mode=False and run on Colab GPU to actually train.")
        return

    # ---- CUDA guard ----
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for fine-tuning.") from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required for QLoRA fine-tuning.  "
            "Run this script on Google Colab Pro (not the Dell laptop).  "
            "Use mock_mode=True for local testing."
        )

    torch.cuda.set_per_process_memory_fraction(_CUDA_MEMORY_MAX)
    print(
        f"[FineTune] CUDA device: {torch.cuda.get_device_name(0)}, "
        f"VRAM cap: {_CUDA_MEMORY_MAX:.0%}"
    )

    # ---- Unsloth dependency guard ----
    try:
        from unsloth import FastLanguageModel  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "Unsloth is required for QLoRA fine-tuning.\n"
            'On Colab: pip install "unsloth[colab-new] @ '
            'git+https://github.com/unslothai/unsloth.git"'
        ) from exc

    try:
        from trl import SFTTrainer  # type: ignore[import-untyped]
        from transformers import TrainingArguments  # type: ignore[import-untyped]
        from datasets import Dataset  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "trl, transformers, and datasets are required for fine-tuning.\n"
            "On Colab: pip install trl transformers datasets"
        ) from exc

    # ---- Load 4-bit base model ----
    print(f"[FineTune] Loading base model: {_BASE_MODEL} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=_BASE_MODEL,
        max_seq_length=config.max_seq_length,
        load_in_4bit=True,
        # dtype=None lets Unsloth auto-detect: float16 on T4, bfloat16 on A100
        dtype=None,
    )

    # ---- Attach LoRA adapters ----
    # r=8 is the rank; alpha=16 sets the scaling factor alpha/r=2 (standard init)
    # target_modules: query + value projections are the minimum for instruction tuning
    # bias='none': do not adapt bias terms (doubles parameters for marginal gain)
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",  # reduces VRAM by ~30%
        random_state=42,
    )
    print(
        f"[FineTune] LoRA adapter attached: "
        f"r={config.lora_r}, alpha={config.lora_alpha}, "
        f"modules={config.target_modules}"
    )

    # ---- Load training data ----
    if config.mock_mode:
        # Already handled above, but defensive guard
        raw_pairs = _build_mock_pairs()
    else:
        raw_pairs = load_finetune_data(data_path)

    if len(raw_pairs) < 3:
        raise ValueError(
            f"Too few training pairs ({len(raw_pairs)}).  "
            "Need at least 3.  Check the JSONL file format."
        )

    # Wrap in HuggingFace Dataset for SFTTrainer
    dataset: Any = Dataset.from_list(raw_pairs)

    # ---- SFTTrainer configuration ----
    # packing=False: don't pack multiple short examples into one sequence —
    # our examples are already close to max_seq_length so packing would truncate
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.grad_accum_steps,
            warmup_ratio=config.warmup_ratio,
            num_train_epochs=config.num_epochs,
            learning_rate=config.learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),  # bfloat16 on A100, fp16 on T4
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",      # 8-bit Adam — saves ~0.5 GB VRAM
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            output_dir=str(config.output_dir / "checkpoints"),
            save_strategy="epoch",
            report_to="none",        # no W&B, wandb, or MLflow on Colab
        ),
    )

    print(
        f"[FineTune] Starting training: {len(raw_pairs)} pairs × "
        f"{config.num_epochs} epochs "
        f"(effective batch={config.batch_size * config.grad_accum_steps}) ..."
    )
    trainer.train()

    # ---- Save adapter weights ----
    config.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))
    print(f"[FineTune] Adapter saved to {config.output_dir}")

    # ---- GGUF conversion instructions ----
    # After fine-tuning, merge the adapter into the base and convert to GGUF
    # so Ollama can serve the model.  Unsloth can do this directly:
    print(
        "\n[FineTune] To convert to GGUF for Ollama deployment, run on Colab:\n"
        "  model.save_pretrained_gguf(\n"
        "      'models/llama/gguf',\n"
        "      tokenizer,\n"
        "      quantization_method='q4_k_m'\n"
        "  )\n"
        "Then create a Modelfile and register with Ollama:\n"
        "  ollama create my-cloudtrail-llama -f Modelfile"
    )


# ---------------------------------------------------------------------------
#  CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """QLoRA fine-tuning CLI (Colab only)."""
    parser = argparse.ArgumentParser(
        description=(
            "QLoRA fine-tune Llama-3.1-8B-Instruct on CloudTrail incident "
            "reports (Path A, Colab GPU only)."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "full"],
        default="mock",
        help="'mock': exit immediately; 'full': run QLoRA training on Colab.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=FINETUNE_DATA_PATH,
        help="Path to finetune_pairs.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ADAPTER_OUTPUT_DIR,
        help="Directory to save LoRA adapter weights.",
    )
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    config = FineTuneConfig(
        lora_r=args.lora_r,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output,
        mock_mode=(args.mode == "mock"),
    )

    train(config, data_path=args.data)


if __name__ == "__main__":
    main()
