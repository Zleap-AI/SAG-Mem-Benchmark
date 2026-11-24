"""Read LF-delimited JSON without splitting Unicode separators inside strings."""
import json


def read_jsonl(path, *, allow_incomplete_tail=False):
    records = []
    with path.open("rb") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if allow_incomplete_tail and not line.endswith(b"\n"):
                    break
                raise
            records.append(record)
    return records
