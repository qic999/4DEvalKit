"""Small, strict IO helpers. JSONL is detected by content, not its suffix."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def read_json(path):
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      default=json_default, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False,
                      default=json_default)
            stream.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def source_signature(path):
    """Record local source file identity; per-sample fingerprints verify contents."""
    p = Path(path).expanduser()
    if not p.exists():
        return {"hub_id": str(path)}
    if p.is_file():
        return {"path": str(p.resolve()), "size": p.stat().st_size,
                "mtime_ns": p.stat().st_mtime_ns}
    files = sorted(x for x in p.rglob("*") if x.is_file()
                   and x.suffix in {".json", ".jsonl", ".parquet", ".tsv", ".csv"}
                   and ".cache" not in x.parts)
    return {"path": str(p.resolve()), "files": [
        [str(x.relative_to(p)), x.stat().st_size, x.stat().st_mtime_ns] for x in files]}
