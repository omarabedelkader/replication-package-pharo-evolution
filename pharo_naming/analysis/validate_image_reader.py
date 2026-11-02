"""Check the image reader against an export of the same release.

    python -m pharo_naming.analysis.validate_image_reader

Pharo 1.0 to 1.4 exist only as binary images, read by ``extraction/image.py``.
To check that reader, it is applied to the Pharo 2 image whose update number
(20628) is the last one of the Pharo 2 FileTree export used for the corpus:
every method of the export must be found in the image.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..extraction.download import fetch, read_image, read_tarball
from ..extraction.filetree import read_filetree_declarations
from ..extraction.image import read_image_declarations
from ..paths import CACHE_DIR
from ..releases import IMAGE_URL, Release, find_release

PHARO_2_IMAGE = Release(
    label="2",
    kind="image",
    url=IMAGE_URL.format(path="20/20628"),
    archive="pharo-2-image.zip",
    revision="20628",
    sha256="aa69e4b995b25ae4c547179cd2f9152f74719e30ac8a7e4487923c2683663be1",
    note="Pharo 2.0 (build 20628), used only to check the image reader",
)


def method_keys(declarations) -> set[tuple[str, str | None, str | None]]:
    return {
        (declaration.class_name, declaration.side, declaration.selector)
        for declaration in declarations if declaration.kind == "method"
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    args = parser.parse_args()

    export_release = find_release("2")
    export = method_keys(
        declaration
        for declaration in read_filetree_declarations(read_tarball(fetch(export_release, args.cache_dir)))
        if declaration.package not in export_release.excluded_packages
    )
    image = method_keys(read_image_declarations(read_image(fetch(PHARO_2_IMAGE, args.cache_dir))))
    found = len(export & image)
    print(
        f"Pharo 2: {found:,} of the {len(export):,} methods of the export are found by the image "
        f"reader in image {PHARO_2_IMAGE.revision} ({len(image):,} methods in the image)"
    )
    if found != len(export):
        sys.exit("The image reader missed methods of the export.")


if __name__ == "__main__":
    main()
