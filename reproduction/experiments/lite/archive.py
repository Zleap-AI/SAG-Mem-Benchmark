"""Restore a verified, lossless archive without regenerating embeddings."""
import fcntl
import gzip
import json
import os
import shutil
from pathlib import Path

from ..shared.runtime import file_digest


def restore_index(directory):
    directory = Path(directory)
    source = directory / "sag2.sqlite"
    meta = json.loads((directory / "index_archive.json").read_text())
    with (directory / ".archive.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if source.exists():
            if source.stat().st_size != meta["original_bytes"] or file_digest(source) != meta["original_sha256"]:
                raise ValueError("Restored index does not match its source hash")
            return source
        compressed = directory / meta["archive_name"]
        if file_digest(compressed) != meta["archive_sha256"]:
            raise ValueError("Compressed index checksum mismatch")
        temporary = directory / ".sag2.sqlite.restore"
        with gzip.open(compressed, "rb") as incoming, temporary.open("wb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        if temporary.stat().st_size != meta["original_bytes"] or file_digest(temporary) != meta["original_sha256"]:
            raise ValueError("Decompressed index checksum mismatch")
        temporary.replace(source)
    return source
