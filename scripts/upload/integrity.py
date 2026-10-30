"""Checksums that read files from disk rather than from the page cache.

Copies on this machine have come out corrupted (zeroed or foreign 4 KiB pages) while the page cache still served the
correct bytes, so a plain re-read right after a copy is not a reliable check.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

CHUNK = 16 * 1024 * 1024


def sync_tree(root: Path) -> None:
    """Flush every file under ``root`` so that write-back errors surface here instead of being lost."""
    for path in sorted(root.rglob("*")):
        if path.is_file():
            fd = os.open(path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)


def evict_page_cache(fd: int) -> None:
    # posix_fadvise is Linux-only; elsewhere the file is read through the cache.
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)


def disk_sha256(path: Path) -> str:
    """SHA-256 of ``path`` after evicting its pages from the page cache where the OS supports it."""
    digest = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY)
    try:
        evict_page_cache(fd)
        while chunk := os.read(fd, CHUNK):
            digest.update(chunk)
        evict_page_cache(fd)
    finally:
        os.close(fd)
    return digest.hexdigest()
