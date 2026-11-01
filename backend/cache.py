"""Versioned, content-addressed JSON cache. No credentials or source filenames.

Atomic replacement keeps interrupted writes from corrupting existing entries;
flock serializes section updates across workers. Analysis variants are keyed by
provider, while the expensive SAM masks are shared by both providers.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

CACHE_DIR = Path(__file__).resolve().parent / "cache"
SCHEMA_VERSION = 1


def video_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DiskCache:
    def __init__(self, directory=None):
        self.directory = Path(directory) if directory is not None else CACHE_DIR

    def _path(self, digest):
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid video SHA256")
        return self.directory / f"{digest}.json"

    def load(self, digest):
        path = self._path(digest)
        try:
            entry = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            return {}
        if not isinstance(entry, dict) or entry.get("schema_version") != SCHEMA_VERSION:
            return {}
        return entry

    @contextmanager
    def locked(self, digest):
        self.directory.mkdir(parents=True, exist_ok=True)
        with self._path(digest).with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def update(self, digest, *, reset_analyses=False, **sections):
        with self.locked(digest):
            entry = self.load(digest)
            # Merge individual provider results without discarding the other.
            if "analyses" in sections and not reset_analyses:
                sections["analyses"] = {**entry.get("analyses", {}), **sections["analyses"]}
            entry.update(sections, schema_version=SCHEMA_VERSION)
            payload = json.dumps(entry, allow_nan=False, separators=(",", ":"))
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", dir=self.directory, delete=False) as out:
                    temporary = out.name
                    out.write(payload)
                    out.flush()
                    os.fsync(out.fileno())
                os.replace(temporary, self._path(digest))
            finally:
                if temporary and os.path.exists(temporary):
                    os.unlink(temporary)
