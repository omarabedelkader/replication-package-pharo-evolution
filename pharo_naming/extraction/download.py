"""Download release archives into a local cache and read files out of them."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import tarfile
import time
import urllib.error
import urllib.request
import zipfile

from ..releases import Release


ATTEMPTS = 4
TIMEOUT_SECONDS = 120
USER_AGENT = "pharo-naming-replication/1.0"


class DownloadError(RuntimeError):
    """Raised when an archive cannot be obtained or fails verification."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(release: Release, cache_dir: Path) -> Path:
    """Return the cached archive of ``release``, downloading it when missing."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / release.archive

    if not target.exists():
        _download(release.url, target)

    if release.sha256 and sha256_of(target) != release.sha256:
        target.unlink()
        raise DownloadError(
            f"{release.name}: checksum mismatch for {release.url}; the cached "
            "file was removed, run again to download it afresh"
        )
    return target


def _download(url: str, target: Path) -> None:
    partial = target.with_name(target.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None

    for attempt in range(1, ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                with partial.open("wb") as stream:
                    shutil.copyfileobj(response, stream)
            partial.replace(target)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            last_error = error
            partial.unlink(missing_ok=True)
            if attempt < ATTEMPTS:
                time.sleep(2 ** attempt)

    raise DownloadError(f"cannot download {url}: {last_error}")


def read_image(archive: Path) -> bytes:
    """Return the bytes of the single ``.image`` file stored in a zip archive."""
    with zipfile.ZipFile(archive) as bundle:
        images = [name for name in bundle.namelist() if name.endswith(".image")]
        if len(images) != 1:
            raise DownloadError(
                f"{archive.name}: expected one .image file, found {len(images)}"
            )
        return bundle.read(images[0])


def read_tarball(archive: Path, subdirectory: str = "") -> dict[str, bytes]:
    """Read the regular files of a GitHub tarball into memory.

    Paths are returned relative to ``subdirectory`` of the repository root.
    Nothing is written to disk, so archive paths never touch the file system.
    """
    prefix = f"{subdirectory.strip('/')}/" if subdirectory else ""
    files: dict[str, bytes] = {}

    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            if not member.isfile():
                continue
            _, _, path = member.name.partition("/")  # drop "<repo>-<commit>/"
            if not path.startswith(prefix):
                continue
            stream = bundle.extractfile(member)
            if stream is not None:
                files[path[len(prefix):]] = stream.read()

    return files
