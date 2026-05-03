"""Train the DeepLog LSTM on normal-only log-key sequences.

**Why train on normal data only (semi-supervised)?**
In real cloud environments, labelled attack data is scarce and constantly
evolving.  By training exclusively on *normal* sequences (Root user
baseline from preprocess.py), the LSTM learns the legitimate transition
grammar.  Any deviation at inference — regardless of attack type — is
flagged automatically.  This makes DeepLog robust to *novel* attacks that
were never seen during training.

**Why track validation loss and save the best epoch?**
Training loss alone is a poor stopping criterion: the LSTM can memorise the
training corpus after enough epochs, fitting to noise rather than the true
transition grammar.  Tracking loss on a held-out *validation* set lets us
pick the checkpoint that generalises best.  This is mandatory for honest
evaluation (CLAUDE.md §7): if we tune top_k on the validation set, the
model weights must also be selected on the validation set.

Pipeline position (CLAUDE.md):
    Drain log_keys (train_log_keys.json + val_log_keys.json)
        -> LogKeyDataset sliding windows
        -> DeepLog LSTM training   (this script)
        -> saved best checkpoint   (models/deeplog_best.pt)
        -> saved final checkpoint  (models/deeplog.pt)

Usage:
    python src/models/train_deeplog.py [--epochs N] [--mode mock|full]
                                       [--seed N]

Environment Variables:
    TRAIN_KEYS_PATH  - Path to the training log-key JSON file
                       (default: data/processed/train_log_keys.json)
    VAL_KEYS_PATH    - Path to the validation log-key JSON file
                       (default: data/processed/val_log_keys.json)
    MODEL_DIR        - Directory for saved checkpoints (default: models/)
    DEEPLOG_MODE     - "mock" (default) or "full"
    CUDA_MEMORY_MAX  - Max VRAM fraction (0.0-1.0).  Default: 1.0
                       (no cap on Colab dedicated GPU).  Set to 0.5 if
                       running on a consumer GPU with a display driver.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Final

# Ensure project root is on sys.path so src.* imports resolve when the
# script is invoked directly.
_PROJECT_ROOT_EARLY = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT_EARLY not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT_EARLY)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.detector.dataset import DEFAULT_WINDOW_SIZE, LogKeyDataset
from src.detector.deeplog import DeepLogModel

# ---------------------------------------------------------------------------
# Configurable paths (env-var friendly per CLAUDE.md ss5 #5)
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent

TRAIN_KEYS_PATH: Final[Path] = Path(
    os.environ.get(
        "TRAIN_KEYS_PATH",
        str(_PROJECT_ROOT / "data" / "processeded1" / "train_log_keys.json"),
    )
)
VAL_KEYS_PATH: Final[Path] = Path(
    os.environ.get(
        "VAL_KEYS_PATH",
        str(_PROJECT_ROOT / "data" / "processeded1" / "val_log_keys.json"),
    )
)
MODEL_DIR: Final[Path] = Path(
    os.environ.get("MODEL_DIR", str(_PROJECT_ROOT / "models1" / "deeplog"))
)

# ---------------------------------------------------------------------------
# VRAM cap -- default 1.0 (no cap) for Colab dedicated GPU.
# Override via CUDA_MEMORY_MAX=0.5 if running on a consumer GPU with a
# display driver sharing VRAM.  (CLAUDE.md ss13: never train on the laptop.)
# ---------------------------------------------------------------------------
_CUDA_MEMORY_MAX: Final[float] = float(
    os.environ.get("CUDA_MEMORY_MAX", "1.0")
)

# ---------------------------------------------------------------------------
# Default hyper-parameters (module-level Finals per CLAUDE.md ss5 #4).
# DEFAULT_WINDOW_SIZE is imported from dataset.py -- single source of truth.
# ---------------------------------------------------------------------------
DEFAULT_EMBEDDING_DIM: Final[int] = 32
DEFAULT_HIDDEN_SIZE: Final[int] = 64
DEFAULT_NUM_LAYERS: Final[int] = 2
DEFAULT_DROPOUT: Final[float] = 0.1
DEFAULT_LR: Final[float] = 1e-3
DEFAULT_BATCH_SIZE: Final[int] = 1024
DEFAULT_EPOCHS: Final[int] = 50
DEFAULT_SEED: Final[int] = 42


# ======================================================================== #
#  Reproducibility                                                          #
# ======================================================================== #

def set_seed(seed: int) -> None:
    """Fix all relevant random seeds for reproducible training.

    Sets Python random, PyTorch CPU/CUDA seeds, and enables CuDNN
    deterministic mode.  Record the seed in the checkpoint and in the
    thesis evaluation chapter so results can be reproduced exactly.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    # torch.manual_seed covers both CPU and all CUDA devices.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Deterministic CuDNN: slightly slower but fully reproducible.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ======================================================================== #
#  Mock data generator                                                      #
# ======================================================================== #

def generate_mock_sequences(
    num_keys: int = 5,
    num_sessions: int = 10,
    session_length: int = 30,
) -> list[list[int]]:
    """Create tiny synthetic sessions for safe logic verification.

    Generates deterministic repeating patterns so the LSTM can learn
    them quickly, confirming the training loop converges.

    Args:
        num_keys:       Vocabulary size (number of distinct log keys).
        num_sessions:   Number of sessions to generate.
        session_length: Events per session.

    Returns:
        List of sessions, each a list[int] of log-key IDs.
    """
    sessions: list[list[int]] = []
    for s in range(num_sessions):
        # Deterministic repeating pattern: 0,1,2,...,num_keys-1,0,1,...
        session: list[int] = [
            (s + i) % num_keys for i in range(session_length)
        ]
        sessions.append(session)
    return sessions


# ======================================================================== #
#  Data loading helpers                                                     #
# ======================================================================== #

def load_key_file(path: Path) -> tuple[list[list[int]], int]:
    """Load a log-key JSON file produced by the Drain parser.

    Expected format:
        {"sequences": [[0, 1, 2, ...], ...], "num_keys": 42}

    Args:
        path: Path to the JSON file.

    Returns:
        (sequences, num_keys) tuple.

    Raises:
        SystemExit: If the file does not exist.
    """
    if not path.exists():
        print(
            f"ERROR: {path} not found. Run the Drain parser first.",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as fh:
        payload: dict = json.load(fh)
    return payload["sequences"], payload["num_keys"]


def print_data_summary(
    label: str,
    sequences: list[list[int]],
    num_keys: int,
    skipped: int = 0,
) -> None:
    """Print corpus statistics for the thesis methodology section.

    Args:
        label:     "train" or "val".
        sequences: The loaded session list.
        num_keys:  Drain vocabulary size.
        skipped:   Sessions dropped for being shorter than window+1.
    """
    lengths: list[int] = [len(s) for s in sequences]
    total_events: int = sum(lengths)
    min_len: int = min(lengths) if lengths else 0
    max_len: int = max(lengths) if lengths else 0
    avg_len: float = total_events / len(lengths) if lengths else 0.0
    print(
        f"[{label}] {len(sequences)} sessions | "
        f"{total_events:,} events | "
        f"vocab={num_keys} | "
        f"len min={min_len} avg={avg_len:.1f} max={max_len} | "
        f"skipped={skipped}"
    )


# ======================================================================== #
#  Validation helper                                                        #
# ======================================================================== #

def validate(
    model: DeepLogModel,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Compute average cross-entropy loss on the validation DataLoader.

    Runs in torch.no_grad() / model.eval() mode -- no gradients computed,
    dropout and batch-norm behave in inference mode.  Model is restored to
    training mode after the call.

    Why validate with loss instead of top-k F1?
    Loss is fast (one forward pass per batch, no set construction),
    continuous, and sensitive to overfitting before the discrete metric
    degrades.  Top-k F1 on validation is computed separately in run_eval.py
    for hyperparameter selection.

    Args:
        model:      DeepLogModel -- temporarily set to eval mode.
        dataloader: Validation DataLoader yielding (X, y) batches.
        device:     Torch device.

    Returns:
        Average cross-entropy loss over the validation set.

    Tensor shapes per batch:
        X      : (batch_size, window_size) int64
        y      : (batch_size,)             int64
        logits : (batch_size, num_keys)    float32
        loss   : ()                        float32 scalar
    """
    model.eval()
    criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    running_loss: float = 0.0
    num_batches: int = 0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            # X_batch: (batch_size, window_size) int64
            # y_batch: (batch_size,)             int64
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # logits: (batch_size, num_keys) float32
            logits, _ = model(X_batch)

            # loss: () scalar float32
            loss: torch.Tensor = criterion(logits, y_batch)
            running_loss += loss.item()
            num_batches += 1

    model.train()
    return running_loss / max(num_batches, 1)


# ======================================================================== #
#  Training loop                                                            #
# ======================================================================== #

def train(
    model: DeepLogModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    epochs: int,
    lr: float,
    device: torch.device,
    start_epoch: int = 0,
    optimiser_state: dict | None = None,
    save_every: int = 10,
    on_checkpoint: Callable[[nn.Module, torch.optim.Optimizer, int, float], None] | None = None,
    on_best: Callable[[nn.Module, int, float], None] | None = None,
) -> tuple[list[float], list[float], torch.optim.Optimizer]:
    """Train the DeepLog model with cross-entropy loss.

    Why cross-entropy?
    The task is multi-class classification -- predicting which of K possible
    log keys comes next.  Cross-entropy directly optimises the predicted
    probability of the correct key, which is exactly what the top-k anomaly
    check relies on at inference time.

    Why Adam?
    Log sequences have very uneven key frequencies (Zipfian distribution).
    Adam's per-parameter adaptive learning rates converge substantially
    faster than SGD on such sparse inputs.

    Args:
        model:           DeepLogModel instance (moved to device).
        train_loader:    Training DataLoader yielding (X, y) batches.
        val_loader:      Optional validation DataLoader.  If provided, val
                         loss is tracked and on_best fires on improvement.
                         If None, training loss is used as best criterion.
        epochs:          Total epochs to reach (absolute, not additional).
        lr:              Learning rate for Adam.
        device:          Torch device (cpu / cuda).
        start_epoch:     Epoch index to resume from (0-based).
        optimiser_state: Saved optimiser.state_dict() for resumption.
        save_every:      Save a periodic checkpoint every N epochs.
        on_checkpoint:   Callback every save_every epochs (not on final).
                         Signature: (model, opt, epoch, train_loss) -> None
        on_best:         Callback when a new best monitor loss is found.
                         Signature: (model, epoch, monitor_loss) -> None

    Returns:
        (train_losses, val_losses, optimiser) where train_losses and
        val_losses are per-epoch averages for epochs actually run.
        val_losses is empty if no val_loader was provided.

    Tensor shapes per batch:
        X      : (batch_size, window_size) int64
        y      : (batch_size,)             int64
        logits : (batch_size, num_keys)    float32
        loss   : ()                        float32 scalar
    """
    model.to(device)
    model.train()

    criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    # Typed as Optimizer base class so a future AdamW swap doesn't break
    # the annotation.
    optimiser: torch.optim.Optimizer = torch.optim.Adam(
        model.parameters(), lr=lr
    )
    if optimiser_state is not None:
        optimiser.load_state_dict(optimiser_state)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_monitor_loss: float = float("inf")

    for epoch in range(start_epoch, epochs):
        # ---- Training pass ----
        model.train()
        running_loss: float = 0.0
        num_batches: int = 0

        for X_batch, y_batch in train_loader:
            # X_batch: (batch_size, window_size) int64
            # y_batch: (batch_size,)             int64
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimiser.zero_grad()

            # logits: (batch_size, num_keys) float32
            logits, _ = model(X_batch)

            # loss: () scalar float32
            loss: torch.Tensor = criterion(logits, y_batch)
            loss.backward()
            optimiser.step()

            running_loss += loss.item()
            num_batches += 1

        avg_train_loss: float = running_loss / max(num_batches, 1)
        train_losses.append(avg_train_loss)

        # ---- Validation pass ----
        if val_loader is not None:
            avg_val_loss: float = validate(model, val_loader, device)
            val_losses.append(avg_val_loss)
            monitor_loss: float = avg_val_loss
        else:
            avg_val_loss = float("nan")
            monitor_loss = avg_train_loss  # fall back to train loss

        # ---- Logging every 10 epochs and on the first epoch ----
        if (epoch + 1) % 10 == 0 or epoch == start_epoch:
            val_str: str = (
                f"  val: {avg_val_loss:.4f}" if val_loader is not None else ""
            )
            print(
                f"  Epoch {epoch + 1:>4d}/{epochs}"
                f"  train: {avg_train_loss:.4f}{val_str}"
            )

        # ---- Best-epoch checkpoint ----
        if monitor_loss < best_monitor_loss:
            best_monitor_loss = monitor_loss
            if on_best is not None:
                on_best(model, epoch + 1, monitor_loss)

        # ---- Periodic mid-training checkpoint ----
        # Skip the final epoch -- the caller handles the terminal save.
        is_save_epoch: bool = (epoch + 1) % save_every == 0
        is_final_epoch: bool = (epoch + 1) == epochs
        if on_checkpoint is not None and is_save_epoch and not is_final_epoch:
            on_checkpoint(model, optimiser, epoch + 1, avg_train_loss)

    return train_losses, val_losses, optimiser


# ======================================================================== #
#  Main entry point                                                         #
# ======================================================================== #

def main() -> None:
    """Parse arguments and run the full training pipeline."""
    parser = argparse.ArgumentParser(description="Train DeepLog LSTM")
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--mode", type=str, default=os.environ.get("DEEPLOG_MODE", "mock"),
        choices=["mock", "full"],
        help="'mock' for synthetic data, 'full' for real data.",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )
    args = parser.parse_args()

    # ---- Reproducibility ----
    set_seed(args.seed)
    print(f"[seed] {args.seed}")

    # ---- Device selection ----
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(_CUDA_MEMORY_MAX)
        device = torch.device("cuda")
        print(
            f"[device] CUDA ({torch.cuda.get_device_name(0)})"
            f" -- VRAM cap: {_CUDA_MEMORY_MAX * 100:.0f}%"
        )
    else:
        device = torch.device("cpu")
        print("[device] CPU")

    # ---- Load or generate sequences ----
    val_sequences: list[list[int]] | None = None

    if args.mode == "mock":
        print("[data] Generating mock sequences (safe for interactive use) ...")
        num_keys: int = 5
        train_sequences: list[list[int]] = generate_mock_sequences(
            num_keys=num_keys, num_sessions=10
        )
        # Separate mock val sessions to exercise the validation code path.
        val_sequences = generate_mock_sequences(
            num_keys=num_keys, num_sessions=4
        )
    else:
        print(f"[data] Loading training keys from {TRAIN_KEYS_PATH} ...")
        train_sequences, num_keys = load_key_file(TRAIN_KEYS_PATH)

        if VAL_KEYS_PATH.exists():
            print(f"[data] Loading validation keys from {VAL_KEYS_PATH} ...")
            val_seqs_loaded, val_num_keys = load_key_file(VAL_KEYS_PATH)
            if val_num_keys != num_keys:
                # Vocabulary mismatch means the two files were produced by
                # different Drain states -- train/val must share the same
                # fit_transform call (CLAUDE.md drain section).
                print(
                    f"WARNING: val num_keys={val_num_keys} != "
                    f"train num_keys={num_keys}. "
                    f"Ensure both files were produced by the same Drain "
                    f"fit_transform call. Skipping validation.",
                    file=sys.stderr,
                )
            else:
                val_sequences = val_seqs_loaded
        else:
            print(
                f"[data] {VAL_KEYS_PATH} not found -- training without "
                f"validation. Best-epoch selection will use training loss.",
                file=sys.stderr,
            )

    # ---- Datasets ----
    train_dataset = LogKeyDataset(train_sequences, window_size=DEFAULT_WINDOW_SIZE)
    print_data_summary(
        "train", train_sequences, num_keys, train_dataset.skipped_sessions
    )

    val_dataset: LogKeyDataset | None = None
    if val_sequences is not None:
        val_dataset = LogKeyDataset(val_sequences, window_size=DEFAULT_WINDOW_SIZE)
        print_data_summary(
            "val", val_sequences, num_keys, val_dataset.skipped_sessions
        )

    # ---- DataLoaders ----
    # num_workers=2 on CUDA reduces the Python-loop bottleneck between
    # batches; pin_memory=True enables faster host->GPU transfers.
    # Both are safe on Linux (Colab); use num_workers=0 on Windows.
    _workers: int = 2 if device.type == "cuda" else 0
    _pin: bool = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=True,
        num_workers=_workers,
        pin_memory=_pin,
        drop_last=False,
    )
    val_loader: DataLoader | None = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=DEFAULT_BATCH_SIZE,
            shuffle=False,   # val order is irrelevant; keep deterministic
            num_workers=_workers,
            pin_memory=_pin,
            drop_last=False,
        )

    # ---- Model ----
    model = DeepLogModel(
        num_keys=num_keys,
        embedding_dim=DEFAULT_EMBEDDING_DIM,
        hidden_size=DEFAULT_HIDDEN_SIZE,
        num_layers=DEFAULT_NUM_LAYERS,
        dropout=DEFAULT_DROPOUT,
    )

    # ---- Checkpoint paths ----
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path: Path = MODEL_DIR / "deeplog.pt"          # latest / final
    best_ckpt_path: Path = MODEL_DIR / "deeplog_best.pt"  # best val loss

    # ---- Resume from checkpoint ----
    start_epoch: int = 0
    optimiser_state: dict | None = None

    if ckpt_path.exists():
        print(f"[resume] Loading checkpoint from {ckpt_path} ...")
        ckpt: dict = torch.load(
            ckpt_path, map_location=device, weights_only=True
        )

        # ---- Full architecture mismatch guard (CLAUDE.md ss5 #8) ----
        # Checking only num_keys is insufficient -- a change to any
        # architecture hyper-parameter (embedding_dim, hidden_size,
        # num_layers, window_size) also produces an opaque shape-mismatch
        # error deep inside load_state_dict.  Report all mismatches at once.
        ckpt_arch: dict[str, object] = {
            "num_keys":      ckpt.get("num_keys"),
            "embedding_dim": ckpt.get("embedding_dim"),
            "hidden_size":   ckpt.get("hidden_size"),
            "num_layers":    ckpt.get("num_layers"),
            "window_size":   ckpt.get("window_size"),
        }
        current_arch: dict[str, object] = {
            "num_keys":      num_keys,
            "embedding_dim": DEFAULT_EMBEDDING_DIM,
            "hidden_size":   DEFAULT_HIDDEN_SIZE,
            "num_layers":    DEFAULT_NUM_LAYERS,
            "window_size":   DEFAULT_WINDOW_SIZE,
        }
        mismatches: dict[str, tuple[object, object]] = {
            k: (ckpt_arch[k], current_arch[k])
            for k in current_arch
            if ckpt_arch[k] is not None and ckpt_arch[k] != current_arch[k]
        }
        if mismatches:
            detail: str = "\n  ".join(
                f"{k}: checkpoint={old}  current={new}"
                for k, (old, new) in mismatches.items()
            )
            raise ValueError(
                f"Checkpoint architecture mismatch:\n  {detail}\n"
                f"Delete the stale checkpoint and retrain:\n"
                f"    rm {ckpt_path}\n"
                f"    python src/models/train_deeplog.py"
            )

        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        optimiser_state = ckpt.get("optimizer_state_dict")
        prev_loss: float | None = ckpt.get("final_loss")
        # Two separate prints to avoid f-string eager-evaluation of prev_loss
        # when it might be None (would raise TypeError on format call).
        if prev_loss is not None:
            print(f"         Resuming from epoch {start_epoch} (last loss: {prev_loss:.4f})")
        else:
            print(f"         Resuming from epoch {start_epoch}")

    total_params: int = sum(p.numel() for p in model.parameters())
    print(f"[model] DeepLogModel -- {total_params:,} parameters")
    print(
        f"        embedding_dim={DEFAULT_EMBEDDING_DIM}  "
        f"hidden_size={DEFAULT_HIDDEN_SIZE}  "
        f"num_layers={DEFAULT_NUM_LAYERS}  "
        f"window_size={DEFAULT_WINDOW_SIZE}\n"
    )

    if start_epoch >= args.epochs:
        print(
            f"Already trained to epoch {start_epoch} "
            f"(requested {args.epochs}). Nothing to do."
        )
        return

    # ---- Checkpoint payload builder ----
    def _build_payload(
        mdl: nn.Module,
        opt: torch.optim.Optimizer,
        epoch: int,
        avg_loss: float,
    ) -> dict:
        """Assemble the dict saved to every checkpoint file."""
        return {
            "model_state_dict":     mdl.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "epoch":                epoch,
            "num_keys":             num_keys,
            "window_size":          DEFAULT_WINDOW_SIZE,
            "embedding_dim":        DEFAULT_EMBEDDING_DIM,
            "hidden_size":          DEFAULT_HIDDEN_SIZE,
            "num_layers":           DEFAULT_NUM_LAYERS,
            "dropout":              DEFAULT_DROPOUT,
            "seed":                 args.seed,
            "final_loss":           avg_loss,
        }

    # ---- Periodic mid-training checkpoint callback ----
    def _save_checkpoint(
        mdl: nn.Module,
        opt: torch.optim.Optimizer,
        epoch: int,
        avg_loss: float,
    ) -> None:
        torch.save(_build_payload(mdl, opt, epoch, avg_loss), ckpt_path)
        print(f"  [checkpoint] Saved at epoch {epoch}  (train: {avg_loss:.4f})")

    # ---- Best-epoch checkpoint callback ----
    def _save_best(
        mdl: nn.Module,
        epoch: int,
        monitor_loss: float,
    ) -> None:
        # Save model weights only -- no optimiser state needed for inference.
        torch.save(
            {
                "model_state_dict": mdl.state_dict(),
                "epoch":            epoch,
                "num_keys":         num_keys,
                "window_size":      DEFAULT_WINDOW_SIZE,
                "embedding_dim":    DEFAULT_EMBEDDING_DIM,
                "hidden_size":      DEFAULT_HIDDEN_SIZE,
                "num_layers":       DEFAULT_NUM_LAYERS,
                "dropout":          DEFAULT_DROPOUT,
                "seed":             args.seed,
                "best_val_loss":    monitor_loss,
            },
            best_ckpt_path,
        )
        print(
            f"  [best]       New best at epoch {epoch}"
            f"  (monitor: {monitor_loss:.4f}) -> {best_ckpt_path.name}"
        )

    # ---- Train ----
    print(f"Training epochs {start_epoch} -> {args.epochs} ...")
    train_losses, val_losses, optimiser = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=DEFAULT_LR,
        device=device,
        start_epoch=start_epoch,
        optimiser_state=optimiser_state,
        on_checkpoint=_save_checkpoint,
        on_best=_save_best,
    )

    # ---- Final checkpoint (latest weights + optimiser state) ----
    final_loss: float | None = train_losses[-1] if train_losses else None
    torch.save(
        _build_payload(
            model, optimiser, args.epochs, final_loss or float("nan")
        ),
        ckpt_path,
    )
    print(f"\n[done] Final checkpoint -> {ckpt_path}")
    if final_loss is not None:
        print(f"       Final train loss : {final_loss:.4f}")
    if val_losses:
        print(f"       Final val loss   : {val_losses[-1]:.4f}")
        print(f"       Best model       -> {best_ckpt_path}")
        print(
            "       Use deeplog_best.pt for evaluation and top_k tuning "
            "(generalises better than the final epoch)."
        )


if __name__ == "__main__":
    main()