import gzip
import hashlib
import json

import pytest

from reproduction.experiments.lite.archive import restore_index


def test_archived_index_restores_exact_bytes_and_checks_hash(tmp_path):
    original = b"complete index\0" * 1000
    compressed = gzip.compress(original)
    (tmp_path / "sag2.sqlite.gz").write_bytes(compressed)
    (tmp_path / "index_archive.json").write_text(json.dumps({
        "archive_name": "sag2.sqlite.gz", "archive_sha256": hashlib.sha256(compressed).hexdigest(),
        "original_sha256": hashlib.sha256(original).hexdigest(), "original_bytes": len(original),
    }))
    assert restore_index(tmp_path).read_bytes() == original
    (tmp_path / "sag2.sqlite").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="source hash"):
        restore_index(tmp_path)
