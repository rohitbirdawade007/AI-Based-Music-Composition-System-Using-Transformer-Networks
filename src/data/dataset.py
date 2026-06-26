"""Music Dataset Module.

Provides :class:`MusicDataset` (a :class:`torch.utils.data.Dataset` subclass)
and :class:`DataModule` for managing train / validation / test splits.

The module is intentionally *import-safe*: if PyTorch is not installed a
helpful error is raised only when the classes are *instantiated*, not at
module import time.  This makes the module importable in environments where
only NumPy is available (e.g., for inspection or unit-testing the data
shapes).

Typical usage::

    from src.data.dataset import MusicDataset, DataModule, music_collate_fn
    import torch
    from torch.utils.data import DataLoader

    dataset = MusicDataset(sequences, mode="seq2seq")
    loader  = DataLoader(dataset, batch_size=32, collate_fn=music_collate_fn)

    dm = DataModule(sequences, batch_size=64, val_split=0.1, test_split=0.1)
    dm.setup()
    train_loader = dm.train_dataloader()
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional PyTorch import
# ---------------------------------------------------------------------------
try:
    import torch
    from torch.utils.data import DataLoader, Dataset, Subset, random_split

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    logger.warning(
        "PyTorch is not installed.  Install with `pip install torch` to use "
        "MusicDataset and DataModule."
    )


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for MusicDataset / DataModule.  "
            "Install it with:  pip install torch"
        )


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

NoteArray = np.ndarray  # shape (seq_len, 5)
ModeType = Literal["language_model", "seq2seq", "classification"]


# ---------------------------------------------------------------------------
# MusicDataset
# ---------------------------------------------------------------------------


class MusicDataset:
    """PyTorch Dataset for variable-length music note sequences.

    Three operating modes are supported:

    * ``"language_model"`` (default) – returns ``(src, tgt)`` where *tgt* is
      *src* shifted left by one step (standard auto-regressive LM objective).
    * ``"seq2seq"`` – returns ``(src, tgt)`` from separate source and target
      lists (useful for music arrangement / harmonisation tasks).
    * ``"classification"`` – returns ``(src, label)`` for genre / style
      classification tasks.

    Args:
        sequences:    List of ``(seq_len, 5)`` float32 NumPy arrays **or**
                      list of 1-D integer token-ID arrays.
        targets:      Optional list of targets.  Interpretation depends on
                      *mode*:

                      * ``"seq2seq"`` – list of target arrays (same length as
                        *sequences*).
                      * ``"classification"`` – list of integer labels.
                      * ``"language_model"`` – ignored (targets are derived
                        from *sequences*).
        mode:         Dataset mode.  One of ``"language_model"``,
                      ``"seq2seq"``, ``"classification"``.
        pad_value:    Scalar used to pad shorter sequences in the collate
                      function.  Defaults to ``0``.
        dtype:        PyTorch dtype for the main sequence tensor.  Defaults
                      to ``torch.float32``.

    Examples::

        seqs    = preprocessor.create_sequences(notes)
        dataset = MusicDataset(seqs, mode="language_model")
        print(len(dataset))          # number of windows
        src, tgt = dataset[0]        # first window
    """

    def __init__(
        self,
        sequences: List[Union[NoteArray, List[int]]],
        targets: Optional[List[Any]] = None,
        mode: ModeType = "language_model",
        pad_value: float = 0.0,
        dtype: Optional[Any] = None,
    ) -> None:
        _require_torch()
        if not sequences:
            raise ValueError("sequences must be a non-empty list")
        if mode not in {"language_model", "seq2seq", "classification"}:
            raise ValueError(
                f"Unknown mode {mode!r}. Choose from 'language_model', "
                "'seq2seq', 'classification'."
            )
        if mode == "seq2seq" and targets is None:
            raise ValueError("targets must be provided when mode='seq2seq'")
        if mode == "seq2seq" and len(targets) != len(sequences):  # type: ignore[arg-type]
            raise ValueError(
                f"sequences ({len(sequences)}) and targets ({len(targets)}) "  # type: ignore[arg-type]
                "must have equal length"
            )
        if mode == "classification" and targets is None:
            raise ValueError("targets must be provided when mode='classification'")

        self.sequences = sequences
        self.targets = targets
        self.mode = mode
        self.pad_value = pad_value
        self._dtype = dtype if dtype is not None else torch.float32

        logger.info(
            "MusicDataset created: %d samples, mode=%s", len(sequences), mode
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.sequences)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Any, Any]:
        """Return the (src, tgt) pair at index *idx*.

        Args:
            idx: Sample index (0-based).

        Returns:
            A ``(src_tensor, tgt_tensor)`` tuple.  Both tensors have dtype
            :attr:`_dtype` (or ``torch.long`` for classification labels).
        """
        seq = np.asarray(self.sequences[idx], dtype=np.float32)
        src = torch.tensor(seq, dtype=self._dtype)

        if self.mode == "language_model":
            # Target = source shifted left by 1 (predict next token)
            tgt = torch.roll(src, -1, dims=0)
            tgt[-1] = self.pad_value  # last position has no ground-truth
            return src, tgt

        if self.mode == "seq2seq":
            tgt_array = np.asarray(self.targets[idx], dtype=np.float32)  # type: ignore[index]
            tgt = torch.tensor(tgt_array, dtype=self._dtype)
            return src, tgt

        # classification
        label = int(self.targets[idx])  # type: ignore[index]
        return src, torch.tensor(label, dtype=torch.long)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def subset(self, indices: Sequence[int]) -> "MusicDataset":
        """Return a new :class:`MusicDataset` containing only *indices*.

        Args:
            indices: Integer indices to select.

        Returns:
            A new :class:`MusicDataset` with the selected samples.
        """
        subs = [self.sequences[i] for i in indices]
        tgts = None
        if self.targets is not None:
            tgts = [self.targets[i] for i in indices]
        return MusicDataset(
            subs,
            targets=tgts,
            mode=self.mode,
            pad_value=self.pad_value,
            dtype=self._dtype,
        )

    def __repr__(self) -> str:  # noqa: D401
        return (
            f"MusicDataset(samples={len(self)}, mode={self.mode!r}, "
            f"seq_shape={np.asarray(self.sequences[0]).shape})"
        )


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------


def music_collate_fn(
    batch: List[Tuple[Any, Any]],
    pad_value: float = 0.0,
) -> Tuple[Any, Any]:
    """Pad and stack a list of (src, tgt) pairs into batched tensors.

    Sequences in the same batch that differ in length along the **time
    dimension** (dim-0) are right-padded with *pad_value*.

    Args:
        batch:     List of ``(src, tgt)`` tuples as returned by
                   :meth:`MusicDataset.__getitem__`.
        pad_value: Scalar fill value used for padding.  Defaults to ``0``.

    Returns:
        ``(src_batch, tgt_batch)`` where each has shape
        ``(batch_size, max_seq_len, ...)`` for sequence tasks or
        ``(batch_size,)`` for classification labels.
    """
    _require_torch()
    srcs, tgts = zip(*batch)

    def _pad(tensors: Tuple[Any, ...]) -> Any:
        if tensors[0].dim() == 0:
            # scalar labels
            return torch.stack(list(tensors))
        max_len = max(t.shape[0] for t in tensors)
        padded = []
        for t in tensors:
            pad_len = max_len - t.shape[0]
            if pad_len > 0:
                pad_shape = (pad_len,) + t.shape[1:]
                padding = torch.full(pad_shape, pad_value, dtype=t.dtype)
                t = torch.cat([t, padding], dim=0)
            padded.append(t)
        return torch.stack(padded)

    return _pad(srcs), _pad(tgts)


def make_collate_fn(pad_value: float = 0.0) -> Callable:
    """Factory that returns a collate function with a fixed *pad_value*.

    Args:
        pad_value: Scalar fill value.

    Returns:
        A callable suitable for use as ``DataLoader(collate_fn=...)``.
    """

    def _fn(batch: List[Tuple[Any, Any]]) -> Tuple[Any, Any]:
        return music_collate_fn(batch, pad_value=pad_value)

    return _fn


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class DataModule:
    """Manages train / validation / test splits and :class:`DataLoader` creation.

    Inspired by the PyTorch-Lightning ``LightningDataModule`` interface but
    without requiring that library as a dependency.

    Args:
        sequences:   Full list of sequence arrays.
        targets:     Optional target list (forwarded to :class:`MusicDataset`).
        mode:        Dataset mode (``"language_model"``, ``"seq2seq"``, or
                     ``"classification"``).
        batch_size:  Mini-batch size for all loaders.  Defaults to ``32``.
        val_split:   Fraction of data reserved for validation.  Must be in
                     ``(0, 1)``.  Defaults to ``0.1``.
        test_split:  Fraction of data reserved for testing.  Must be in
                     ``(0, 1)``.  Defaults to ``0.1``.
        num_workers: Number of worker processes for :class:`DataLoader`.
                     Defaults to ``0`` (main process).
        pin_memory:  Whether to pin memory in :class:`DataLoader`.  Useful
                     when training on GPU.  Defaults to ``False``.
        seed:        Random seed for the train/val/test split.  Defaults to
                     ``42``.
        pad_value:   Padding value forwarded to the collate function.

    Examples::

        dm = DataModule(seqs, batch_size=64, val_split=0.1, test_split=0.1)
        dm.setup()
        for src, tgt in dm.train_dataloader():
            ...   # training step
    """

    def __init__(
        self,
        sequences: List[NoteArray],
        targets: Optional[List[Any]] = None,
        mode: ModeType = "language_model",
        batch_size: int = 32,
        val_split: float = 0.1,
        test_split: float = 0.1,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int = 42,
        pad_value: float = 0.0,
    ) -> None:
        _require_torch()
        if not (0.0 < val_split < 1.0):
            raise ValueError(f"val_split must be in (0, 1), got {val_split}")
        if not (0.0 < test_split < 1.0):
            raise ValueError(f"test_split must be in (0, 1), got {test_split}")
        if val_split + test_split >= 1.0:
            raise ValueError("val_split + test_split must be < 1.0")

        self.sequences = sequences
        self.targets = targets
        self.mode = mode
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.pad_value = pad_value

        self._train_dataset: Optional[MusicDataset] = None
        self._val_dataset: Optional[MusicDataset] = None
        self._test_dataset: Optional[MusicDataset] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        """Compute train / val / test splits and build sub-datasets.

        Args:
            stage: Optional string (``"fit"``, ``"test"``, ``"predict"``)
                   following the Lightning convention.  Currently unused but
                   kept for API compatibility.
        """
        n = len(self.sequences)
        n_test = math.ceil(n * self.test_split)
        n_val = math.ceil(n * self.val_split)
        n_train = n - n_val - n_test

        if n_train <= 0:
            raise ValueError(
                f"Not enough data ({n} samples) for the requested splits "
                f"(val={self.val_split}, test={self.test_split})."
            )

        rng = np.random.default_rng(self.seed)
        indices = rng.permutation(n).tolist()

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        full_dataset = MusicDataset(
            self.sequences,
            targets=self.targets,
            mode=self.mode,
            pad_value=self.pad_value,
        )

        self._train_dataset = full_dataset.subset(train_idx)
        self._val_dataset = full_dataset.subset(val_idx)
        self._test_dataset = full_dataset.subset(test_idx)

        logger.info(
            "DataModule split: train=%d  val=%d  test=%d",
            n_train,
            n_val,
            n_test,
        )

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def _check_setup(self) -> None:
        if self._train_dataset is None:
            raise RuntimeError("Call DataModule.setup() before requesting DataLoaders")

    def train_dataloader(self) -> "DataLoader":
        """Return a shuffled :class:`DataLoader` for the training split.

        Returns:
            :class:`torch.utils.data.DataLoader` with shuffling enabled.
        """
        self._check_setup()
        return DataLoader(
            self._train_dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=make_collate_fn(self.pad_value),
            drop_last=False,
        )

    def val_dataloader(self) -> "DataLoader":
        """Return a non-shuffled :class:`DataLoader` for the validation split.

        Returns:
            :class:`torch.utils.data.DataLoader` without shuffling.
        """
        self._check_setup()
        return DataLoader(
            self._val_dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=make_collate_fn(self.pad_value),
        )

    def test_dataloader(self) -> "DataLoader":
        """Return a non-shuffled :class:`DataLoader` for the test split.

        Returns:
            :class:`torch.utils.data.DataLoader` without shuffling.
        """
        self._check_setup()
        return DataLoader(
            self._test_dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=make_collate_fn(self.pad_value),
        )

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        """Return sample counts for each split.

        Returns:
            Dict with ``"train"``, ``"val"``, ``"test"`` keys.
        """
        self._check_setup()
        return {
            "train": len(self._train_dataset),  # type: ignore[arg-type]
            "val": len(self._val_dataset),  # type: ignore[arg-type]
            "test": len(self._test_dataset),  # type: ignore[arg-type]
        }

    def __repr__(self) -> str:  # noqa: D401
        return (
            f"DataModule(n={len(self.sequences)}, batch_size={self.batch_size}, "
            f"val_split={self.val_split}, test_split={self.test_split})"
        )
