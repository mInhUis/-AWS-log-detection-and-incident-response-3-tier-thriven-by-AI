"""DeepLog: Deep Learning for Anomaly Detection in System Logs.

Implementation based on:
    Du, M., Li, F., Zheng, G., & Srikumar, V. (2017).
    "DeepLog: Anomaly Detection and Diagnosis from System Logs through
    Deep Learning."  ACM CCS.

**Why an LSTM for log-key anomaly detection?**
Cloud log sequences exhibit strong *temporal ordering* — certain API calls
routinely follow others (e.g., AssumeRole → GetCallerIdentity → ListBuckets).
An LSTM is ideal here because:
    1. It models *sequential dependencies* of arbitrary length via its gated
       cell state, capturing the normal workflow grammar of CloudTrail events.
    2. It naturally handles the *variable-length* sessions that arise from
       different user activities.
    3. By training exclusively on *normal* sequences (Root user baseline),
       the model learns the expected transition distribution.  At inference,
       any key that falls outside the top-k most-likely next keys signals a
       deviation from learned behaviour — an anomaly — without needing
       labelled attack data for training (semi-supervised approach).

Architecture:
    Embedding(num_keys, embedding_dim)
        → LSTM(embedding_dim, hidden_size, num_layers)
        → Linear(hidden_size, num_keys)

    The final Linear layer produces logits over all possible log keys
    (multi-class classification), and the predicted distribution is used
    for both training (cross-entropy loss) and inference (top-k check).

**Why no __slots__ on nn.Module?**
``nn.Module.__setattr__`` is overridden to register parameters, submodules,
and buffers into internal dicts (``_parameters``, ``_modules``,
``_buffers``).  Declaring ``__slots__`` on a subclass can interfere with
this registration mechanism on certain PyTorch versions, breaking
``state_dict()`` round-trips silently.  ``__slots__`` is safe on plain
Python dataclasses (``LogCluster``, ``_Node``, ``LogKeyDataset``) but
should be avoided on ``nn.Module`` subclasses.

Data Flow (see CLAUDE.md architecture):
    Drain log_keys → LogKeyDataset sliding windows → DeepLogModel → logits
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DeepLogModel(nn.Module):
    """LSTM-based next-log-key predictor for anomaly detection.

    Tensor-shape documentation is provided inline for every operation
    in ``forward()``.

    Args:
        num_keys:      Total number of unique Drain log-key IDs (vocabulary
                       size).  Determines embedding table rows and output
                       logit width.
        embedding_dim: Dimensionality of the learned key embeddings.
                       Default ``32``.  For small Drain vocabularies
                       (50–300 templates typical on CloudTrail), 32 matches
                       the scale used by Du et al. (2017) on HDFS (28 keys)
                       and avoids over-parameterising the embedding table.
        hidden_size:   LSTM hidden-state width.  Default ``64``.
        num_layers:    Stacked LSTM depth.  Default ``2``.
        dropout:       Dropout probability between LSTM layers (only
                       active when ``num_layers > 1``).  Default ``0.1``.
    """

    def __init__(
        self,
        num_keys: int,
        embedding_dim: int = 32,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Embedding: integer log key → dense vector.
        # Input:  (batch_size, seq_len) int64
        # Output: (batch_size, seq_len, embedding_dim) float32
        self._embedding: nn.Embedding = nn.Embedding(
            num_embeddings=num_keys,
            embedding_dim=embedding_dim,
        )

        # LSTM: captures temporal dependencies across the key sequence.
        # Input:  (batch_size, seq_len, embedding_dim)
        # Output: (batch_size, seq_len, hidden_size), (h_n, c_n)
        self._lstm: nn.LSTM = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Fully-connected head: last hidden state → logits over all keys.
        # Input:  (batch_size, hidden_size)
        # Output: (batch_size, num_keys)
        self._fc: nn.Linear = nn.Linear(hidden_size, num_keys)

        self._hidden_size: int = hidden_size
        self._num_layers: int = num_layers

    # ------------------------------------------------------------------ #
    #  Forward pass                                                        #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass: predict logits for the next log key.

        Args:
            x:      Input log-key indices.
                    Shape: ``(batch_size, seq_len)`` — int64.
            hidden: Optional LSTM initial state ``(h_0, c_0)``.
                    h_0 shape: ``(num_layers, batch_size, hidden_size)``
                    c_0 shape: ``(num_layers, batch_size, hidden_size)``
                    If ``None``, zeros are used (PyTorch default).

        Returns:
            logits: Raw (unnormalised) scores over all log keys.
                    Shape: ``(batch_size, num_keys)`` — float32.
            hidden: Updated LSTM state ``(h_n, c_n)``, same shapes as
                    the input hidden state.
        """
        # x: (batch_size, seq_len) int64
        embedded: torch.Tensor = self._embedding(x)
        # embedded: (batch_size, seq_len, embedding_dim) float32

        lstm_out, hidden = self._lstm(embedded, hidden)
        # lstm_out: (batch_size, seq_len, hidden_size) float32
        # hidden:   ( (num_layers, batch_size, hidden_size),
        #             (num_layers, batch_size, hidden_size) )

        # Use only the last time-step's output for next-key prediction.
        last_step: torch.Tensor = lstm_out[:, -1, :]
        # last_step: (batch_size, hidden_size) float32

        logits: torch.Tensor = self._fc(last_step)
        # logits: (batch_size, num_keys) float32

        return logits, hidden

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def num_keys(self) -> int:
        """Vocabulary size — number of Drain log-key IDs the model knows.

        Used by ``detect.py`` for the stale-model guard without reaching
        into private attributes (``model._embedding.num_embeddings``).
        """
        return self._embedding.num_embeddings

    @property
    def hidden_size(self) -> int:
        """LSTM hidden state dimensionality."""
        return self._hidden_size

    @property
    def num_layers(self) -> int:
        """Number of stacked LSTM layers."""
        return self._num_layers