"""The Pharo releases under study and where their code comes from.

Each release is pinned to an immutable artifact: a checksummed archive from
``files.pharo.org`` or a Git commit.  For every release line we take its last
published state, which is the only state available for all of them.

Three kinds of artifact are needed because Pharo changed how it is published:

``image``
    Pharo 1.0.  Only a binary image exists, so the image is the sole
    description of what the release contains.  The image is decoded directly.
``filetree``
    Pharo 2 to 6.  The official FileTree mirror ``pharo-project/pharo-core``.
``tonel``
    Pharo 7 and later.  The development repository ``pharo-project/pharo``.
"""

from __future__ import annotations

from dataclasses import dataclass


IMAGE_URL = "https://files.pharo.org/image/{path}.zip"
PHARO_CORE_URL = "https://codeload.github.com/pharo-project/pharo-core/tar.gz/{commit}"
PHARO_URL = "https://codeload.github.com/pharo-project/pharo/tar.gz/{commit}"

# pharo-core was produced by an exporter that was itself loaded in the image
# being exported.  These packages are that tool, not part of the release.
EXPORTER_PACKAGES = frozenset({
    "ConfigurationOfFileSystemST",
    "FileSystem-Relative",
    "FileSystem-Smalltalk",
    "FileSystem-Tests-Relative",
})


@dataclass(frozen=True)
class Release:
    label: str  # "1.0", "2", "14"
    kind: str  # "image", "filetree" or "tonel"
    url: str
    archive: str  # file name used in the download cache
    revision: str  # build name or Git commit
    sha256: str | None = None  # known for files.pharo.org archives
    excluded_packages: frozenset[str] = frozenset()
    note: str = ""

    @property
    def name(self) -> str:
        return f"Pharo {self.label}"


def _image(label: str, path: str, sha256: str, note: str) -> Release:
    return Release(
        label=label,
        kind="image",
        url=IMAGE_URL.format(path=path),
        archive=f"pharo-{label}-image.zip",
        revision=path.split("/")[1],
        sha256=sha256,
        note=note,
    )


def _filetree(label: str, commit: str) -> Release:
    return Release(
        label=label,
        kind="filetree",
        url=PHARO_CORE_URL.format(commit=commit),
        archive=f"pharo-{label}-{commit[:12]}.tar.gz",
        revision=commit,
        excluded_packages=EXPORTER_PACKAGES,
        note=f"pharo-core, branch {label}.0",
    )


def _tonel(label: str, commit: str, branch: str) -> Release:
    return Release(
        label=label,
        kind="tonel",
        url=PHARO_URL.format(commit=commit),
        archive=f"pharo-{label}-{commit[:12]}.tar.gz",
        revision=commit,
        note=f"pharo, branch {branch}",
    )


RELEASES: tuple[Release, ...] = (
    _image("1.0", "10/10517", "578dc23ea12b2c5c4b32521e0c8538edfcaca24e057a9e7f3cd7fe8d864833ff", "Pharo 1.0 (build 10517), full image"),
    _filetree("2", "28cc87dd94fa3e8fef6278635a3f9abc0904869e"),
    _filetree("3", "18c33b2573dbb03c9826acfa5d04edcc63540ab7"),
    _filetree("4", "e5c8e4d89ca795062fdab1f50f6e7cd6ac692099"),
    _filetree("5", "8d6c095bda8026b36cb51d1fe4fe3fa0e9e933d7"),
    _filetree("6", "f35877b624a127a40c6abc4aad553ff063fab522"),
    _tonel("7", "fd8c156d959653f52be6fd7f8c71edf8cfb8fe90", "Pharo7.0"),
    _tonel("8", "6b47e520dba66ca19ffd98f0ae0b031d677f7099", "Pharo8.0"),
    _tonel("9", "e1bb0fb7b6c4aaadd5106d19b58633c94dff1b69", "Pharo9.0"),
    _tonel("10", "28169ae414c78e07602b31545f43061b4fa7a88a", "Pharo10"),
    _tonel("11", "963dc3a7de9ee8e5543f6e62d25a2a283f7dcd48", "Pharo11"),
    _tonel("12", "7d5f14cb47c000853271e85a27706569614966e7", "Pharo12"),
    _tonel("13", "d7c6f761d5445e8b04a0aa5137e0d546ad9bcd89", "Pharo13"),
    _tonel("14", "3221b560a7cbcb4361c8be367d3efdc8e2303459", "Pharo14"),
)

RELEASE_LABELS: tuple[str, ...] = tuple(release.label for release in RELEASES)
_ORDER = {label: index for index, label in enumerate(RELEASE_LABELS, start=1)}


def release_index(label: str) -> int:
    """Position of a release in the historical sequence, starting at 1."""
    try:
        return _ORDER[str(label)]
    except KeyError:
        raise ValueError(f"unknown Pharo release {label!r}") from None


def find_release(label: str) -> Release:
    return RELEASES[release_index(label) - 1]
