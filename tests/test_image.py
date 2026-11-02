import struct

import pytest

from pharo_naming.extraction.image import ImageFormatError, ObjectMemory, PackageResolver


def test_category_is_assigned_to_the_most_specific_package():
    resolver = PackageResolver({"Kernel", "Collections", "Collections-Strings"})

    assert resolver.package_of_category("Kernel-Classes") == "Kernel"
    assert resolver.package_of_category("Collections-Strings-Base") == "Collections-Strings"
    assert resolver.package_of_category("Collections-Abstract") == "Collections"
    assert resolver.package_of_category("collections-abstract") == "Collections"


def test_category_without_package_stays_as_it_is():
    assert PackageResolver({"Kernel"}).package_of_category("Morphic-Core") == "Morphic-Core"


def test_extension_protocol_names_the_extending_package():
    resolver = PackageResolver({"Kernel", "Fuel"})

    assert resolver.package_of_extension("*Fuel") == "Fuel"
    assert resolver.package_of_extension("*fuel-core") == "Fuel"
    assert resolver.package_of_extension("*Unknown-Tools") == "Unknown-Tools"
    assert resolver.package_of_extension("accessing") is None


def test_unsupported_image_format_is_rejected():
    spur_header = struct.pack("<5I", 68021, 64, 0, 0, 0) + bytes(44)
    with pytest.raises(ImageFormatError):
        ObjectMemory(spur_header)
