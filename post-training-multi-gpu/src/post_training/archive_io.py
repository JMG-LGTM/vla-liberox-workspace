"""Read replay archives with one integrity-checked retry for transient CRC errors."""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Iterable
import zipfile

import numpy as np


def load_npz_arrays(path: str | Path, *, keys: Iterable[str] | None = None,
                    expected_sha256: str | None = None) -> dict[str, np.ndarray]:
    """Eagerly read the same arrays; only a CRC failure permits one checked retry.

    The normal path is unchanged. Recovery closes the failed archive, advises the
    OS to evict its clean file pages, then hashes and decodes one newly opened
    read-only file handle. It never substitutes another sample or changes data.
    """
    path = Path(path)
    selected = None if keys is None else tuple(keys)
    try:
        with np.load(path, allow_pickle=False) as source:
            return {key: source[key] for key in (source.files if selected is None else selected)}
    except zipfile.BadZipFile as exc:
        if not str(exc).startswith("Bad CRC-32"):
            raise
        if not expected_sha256:
            raise ValueError(f"CRC recovery requires a frozen SHA-256 for {path}") from exc

    logging.getLogger(__name__).warning(
        "CRC read failure in %s; retrying the same archive once after SHA-256 verification", path,
    )
    advise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if advise is not None and dontneed is not None:
        with path.open("rb") as evict:
            try:
                advise(evict.fileno(), 0, 0, dontneed)
            except OSError:
                # This hint is optional; content verification remains mandatory.
                logging.getLogger(__name__).warning("File-cache eviction unavailable for %s", path)

    try:
        with path.open("rb") as stream:
            digest = hashlib.sha256()
            while block := stream.read(1024 * 1024):
                digest.update(block)
            actual_sha256 = digest.hexdigest()
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"CRC recovery SHA-256 mismatch for {path}: "
                    f"expected {expected_sha256}, got {actual_sha256}"
                )
            stream.seek(0)
            with np.load(stream, allow_pickle=False) as source:
                return {key: source[key] for key in (source.files if selected is None else selected)}
    except zipfile.BadZipFile as exc:
        raise zipfile.BadZipFile(f"CRC recovery retry failed for {path}: {exc}") from exc
