"""Drain3-backed log parser for CloudTrail log-key sequence extraction.

Why Drain3 instead of a hand-rolled implementation?
----------------------------------------------------
The hand-rolled DrainParser (kept in drain_parser_reference.py for
pedagogical reference) is faithful to He et al. (2017) but has one critical
limitation for a thesis pipeline: it lacks a *separate transform-only mode*.
Every call to fit_transform() can mint new cluster IDs and mutate existing
templates, meaning calling it on test data leaks test-set structure back into
the parser state (CLAUDE.md ss5 -- stale-artifact guards, data-leakage).

Drain3 (IBM open-source, the de-facto reference implementation) solves this
with a built-in TemplateMiner.save_state() / load_state() mechanism.  The
workflow is:

    1. fit_transform(train_messages)  -- builds the template tree
    2. save_state(path)               -- snapshot the parser to disk
    3. transform(val_messages)        -- read-only: matches existing clusters,
       transform(test_messages)          returns OOV_KEY for any unmatched msg
    4. The OOV_KEY (= num_keys, one beyond the vocabulary) is handled in
       detect.py's stale-model guard and in the thesis methodology discussion.

Same algorithm, better persistence.  Citation: "We use the IBM Drain3
reference implementation [cite] of the Drain algorithm (He et al., 2017)."

Data Flow (CLAUDE.md architecture):
    Raw CloudTrail log strings (list[str])
        -> DrainParser.fit_transform()  [train only]
        -> (log_keys: list[int], templates: dict[int, str])
        -> saved as train_log_keys.json + drain3_state.json

    At inference:
        -> DrainParser.transform()      [val / test, read-only]
        -> list[int]  (OOV_KEY for unrecognised messages)

Input Shape:
    log_messages : list[str]  -- N raw log event strings.

Output Shape:
    fit_transform -> (list[int] of length N, dict[int, str] of K clusters)
    transform     -> list[int] of length N  (may contain OOV_KEY)

Vocabulary note:
    num_keys = len(templates)   -- the integer written to the JSON file.
    OOV_KEY  = num_keys         -- one beyond the last valid key; never seen
                                   by the trained model, triggers the stale-
                                   model guard in detect.py if encountered.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Final

# drain3 is the IBM open-source Drain implementation.
# Install: pip install drain3
try:
    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig
except ImportError as exc:
    raise ImportError(
        "drain3 is required: pip install drain3"
    ) from exc

# ======================================================================== #
#  Variable-masking regex patterns                                          #
# ======================================================================== #
# Drain3 supports masking via its config file, but we keep a Python-side
# pre-processing step for CloudTrail-specific patterns (AWS ARNs, access
# key prefixes) that are not in Drain3's default mask list.
# Order matters: most specific patterns first to avoid partial-match
# interference (e.g. IPv4 before bare integers).

_VARIABLE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),               # IPv4
    re.compile(
        r"[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}"
    ),                                                           # UUID
    # Full AWS ARN grammar: arn:partition:service:region:account:resource
    # Handles IAM (empty region), EC2, S3 bucket ARNs, etc.
    re.compile(
        r"arn:aws[a-z\-]*:[a-z0-9\-]*:[a-z0-9\-]*:[0-9]*:"
        r"[a-zA-Z0-9/\-_:.]+"
    ),
    re.compile(r"AKIA[0-9A-Z]{16}"),                            # Access key
    re.compile(r"AIDA[0-9A-Z]{16}"),                            # IAM user ID
    re.compile(r"AROA[0-9A-Z]{16}"),                            # Role ID
    re.compile(r"ASIA[0-9A-Z]{16}"),                            # Temp cred
    # Long hex/decimal IDs -- includes AWS account IDs (12 decimal digits).
    # Must come before the bare-integer pattern to win the first-match.
    re.compile(r"\b[0-9a-fA-F]{12,}\b"),
    re.compile(r"\b0x[0-9a-fA-F]+\b"),                         # Hex literal
    re.compile(r"\b\d+\.\d+\.\d+\b"),                          # Version x.y.z
    re.compile(r"\b\d+\b"),                                     # Remaining ints
)

_WILDCARD: Final[str] = "<*>"


# ======================================================================== #
#  DrainParser                                                              #
# ======================================================================== #

class DrainParser:
    """Thin wrapper around Drain3's TemplateMiner with the same public API
    as the hand-rolled DrainParser, plus a read-only transform() method.

    The wrapper:
        - Pre-processes messages with the same CloudTrail-tuned regexes.
        - Exposes fit_transform() (mutating) and transform() (read-only).
        - Provides save_state() / load_state() backed by Drain3 persistence.
        - Returns (list[int], dict[int, str]) from fit_transform(), matching
          the shape the rest of the pipeline expects.

    OOV (out-of-vocabulary) handling:
        transform() assigns OOV_KEY (= num_keys) to any message that does
        not match an existing cluster above sim_threshold.  The calling
        script should log the OOV count.  detect.py's stale-model guard
        will catch OOV keys before they reach the LSTM embedding table.

    Args:
        sim_threshold:  Drain3 similarity threshold [0.0, 1.0].  Default 0.4.
        depth:          Drain3 tree depth.  Default 4.
        max_children:   Max children per internal node.  Default 100.
    """

    def __init__(
        self,
        sim_threshold: float = 0.4,
        depth: int = 4,
        max_children: int = 100,
    ) -> None:
        self._sim_threshold: float = sim_threshold
        self._depth: int = depth
        self._max_children: int = max_children

        # _miner is created (or re-created) in fit_transform / load_state.
        # None signals the parser has not yet been fitted.
        self._miner: TemplateMiner | None = None

        # Mapping from Drain3's internal cluster_id (1-based str) to our
        # 0-based integer log-key IDs.
        self._drain_id_to_key: dict[str, int] = {}
        self._next_key: int = 0

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    @property
    def num_keys(self) -> int:
        """Number of distinct templates seen during fit_transform."""
        return self._next_key

    @property
    def oov_key(self) -> int:
        """Sentinel key assigned to messages that match no known template.

        Equal to num_keys (one beyond the last valid key).  The calling
        code should treat this as an unknown event, not as log key 0.
        """
        return self._next_key

    def fit_transform(
        self,
        log_messages: list[str],
    ) -> tuple[list[int], dict[int, str]]:
        """Parse training messages, build template tree, return log-key IDs.

        This is a MUTATING call -- it updates the internal Drain3 state.
        Call it ONLY on training data.  For validation and test data use
        transform() to avoid leaking test-set structure into the parser.

        Args:
            log_messages: Raw log strings, one per event (length N).

        Returns:
            (log_keys, templates) where:
            - log_keys:  list[int] of length N, one key per message.
            - templates: dict[int, str] mapping key ID -> template text.
        """
        # Build a fresh TemplateMiner for this fit.
        cfg = self._build_config()
        self._miner = TemplateMiner(config=cfg)
        self._drain_id_to_key = {}
        self._next_key = 0

        log_keys: list[int] = []

        for message in log_messages:
            preprocessed: str = _preprocess(message)
            result = self._miner.add_log_message(preprocessed)
            key: int = self._resolve_key(result["cluster_id"])
            log_keys.append(key)

        templates: dict[int, str] = self._build_templates()
        return log_keys, templates

    def transform(
        self,
        log_messages: list[str],
    ) -> list[int]:
        """Read-only inference: assign existing cluster IDs, never create new.

        Messages that match no existing cluster above sim_threshold are
        assigned oov_key (= num_keys).  The template tree is NEVER mutated.

        Call this on validation and test data ONLY.  Calling fit_transform()
        on test data is a data-leakage error.

        Args:
            log_messages: Raw log strings to classify (length N).

        Returns:
            list[int] of length N.  oov_key appears for unrecognised messages.

        Raises:
            RuntimeError: If called before fit_transform() or load_state().
        """
        if self._miner is None:
            raise RuntimeError(
                "DrainParser has not been fitted.  Call fit_transform() "
                "or load_state() before transform()."
            )

        log_keys: list[int] = []

        for message in log_messages:
            preprocessed: str = _preprocess(message)
            # match_only=True: Drain3 will not create new clusters.
            cluster = self._miner.match(preprocessed, full_search_strategy="never")
            if cluster is None:
                # No match above threshold -- out-of-vocabulary event
                log_keys.append(self.oov_key)
            else:
                key: int = self._resolve_key(cluster.cluster_id)
                log_keys.append(key)

        return log_keys

    def save_state(self, path: Path | str) -> None:
        """Persist the Drain3 parser state and the key-ID mapping to disk."""
        if self._miner is None:
            raise RuntimeError(
                "DrainParser has not been fitted.  Nothing to save."
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # --- NEW: Drain3 uses jsonpickle for serialization ---
        import jsonpickle
        state_str: str = jsonpickle.encode(self._miner, keys=True)
        path.write_text(state_str, encoding="utf-8")

        # Our 0-based key mapping (Drain3 IDs are 1-based strings internally)
        keymap_path: Path = path.with_suffix(".keymap.json")
        keymap_payload: dict = {
            "drain_id_to_key": self._drain_id_to_key,
            "next_key": self._next_key,
        }
        keymap_path.write_text(
            json.dumps(keymap_payload, indent=2), encoding="utf-8"
        )
        print(f"[drain] State saved -> {path}  ({self.num_keys} templates)")

    def load_state(self, path: Path | str) -> None:
        """Restore a previously saved Drain3 state from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Drain3 state file not found: {path}\n"
                f"Run the Drain parser on the training corpus first:\n"
                f"    python src/preprocessing/run_drain.py --mode full"
            )

        # --- NEW: Drain3 uses jsonpickle for deserialization ---
        import jsonpickle
        state_str: str = path.read_text(encoding="utf-8")
        self._miner = jsonpickle.decode(state_str, keys=True)

        keymap_path: Path = path.with_suffix(".keymap.json")
        if not keymap_path.exists():
            raise FileNotFoundError(
                f"Key-map file not found: {keymap_path}\n"
                f"Delete the state file and rerun fit_transform()."
            )
        keymap_payload: dict = json.loads(
            keymap_path.read_text(encoding="utf-8")
        )
        self._drain_id_to_key = keymap_payload["drain_id_to_key"]
        self._next_key = keymap_payload["next_key"]
        print(f"[drain] State loaded <- {path}  ({self.num_keys} templates)")

    def get_templates(self) -> dict[int, str]:
        """Return the current key-ID -> template-string mapping.

        Returns:
            dict[int, str] suitable for the reverse-mapping step before
            the Llama prompt (Tier 3 of the cascade).
        """
        return self._build_templates()

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _build_config(self) -> TemplateMinerConfig:
        """Construct a TemplateMinerConfig from our hyper-parameters."""
        cfg = TemplateMinerConfig()
        cfg.drain_sim_th = self._sim_threshold
        cfg.drain_depth = self._depth
        cfg.drain_max_children = self._max_children
        # Disable Drain3's own persistence -- we handle that in save_state().
        cfg.snapshot_interval_minutes = 0
        cfg.snapshot_interval_lines = 0
        return cfg

    def _resolve_key(self, drain_cluster_id: object) -> int:
        """Map a Drain3 cluster_id (any hashable) to our 0-based int key.

        Drain3 cluster IDs are integers internally but may be returned as
        ints or strings depending on version.  We normalise to str for the
        dict key to be safe across versions.

        Args:
            drain_cluster_id: The cluster_id field from Drain3's result dict.

        Returns:
            0-based integer log-key ID.
        """
        cid_str: str = str(drain_cluster_id)
        if cid_str not in self._drain_id_to_key:
            self._drain_id_to_key[cid_str] = self._next_key
            self._next_key += 1
        return self._drain_id_to_key[cid_str]

    def _build_templates(self) -> dict[int, str]:
        """Build the int-key -> template-string dict from the current miner."""
        if self._miner is None:
            return {}
        templates: dict[int, str] = {}
        for cluster in self._miner.drain.clusters:
            cid_str: str = str(cluster.cluster_id)
            if cid_str in self._drain_id_to_key:
                key: int = self._drain_id_to_key[cid_str]
                templates[key] = " ".join(cluster.log_template_tokens)
        return templates


# ======================================================================== #
#  Module-level pre-processing (shared by fit_transform and transform)     #
# ======================================================================== #

def _preprocess(message: str) -> str:
    """Mask known variable patterns with the <*> wildcard.

    Applied before Drain3 tokenisation so that variable fields (IPs, UUIDs,
    ARNs, numeric IDs) do not fragment templates unnecessarily.

    Args:
        message: A single raw log string.

    Returns:
        The message with all matched variable tokens replaced by '<*>'.
    """
    result: str = message
    for pattern in _VARIABLE_PATTERNS:
        result = pattern.sub(_WILDCARD, result)
    return result