"""Read classes, methods and packages directly from a pre-Spur Pharo image.

Pharo 1.0 to 1.4 were never published as source trees.  Every 1.x release
ships the same ``PharoV10.sources`` file plus a release-specific image and
``.changes`` log, so the only artifact that states exactly which classes and
methods a release contains is the image itself.  This module decodes the
32-bit Squeak V3 object memory (image formats 6504 and 6505) far enough to
enumerate the globals of the system dictionary, their method dictionaries and
their categories.  Nothing is executed and no virtual machine is required.

The reader is self-describing: slot positions are resolved from the instance
variable names stored in the image rather than from hard-coded offsets, so it
adapts to the layout changes that happened between releases.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Iterator

from .model import Declaration


SUPPORTED_FORMATS = {6504, 6505}

# Indices into the VM's special objects array.
SPECIAL_NIL = 0
SPECIAL_COMPACT_CLASSES = 28

# Slots of a class the VM itself relies on, identical in every V3 image.
SLOT_SUPERCLASS = 0
SLOT_FORMAT = 2


class ImageFormatError(ValueError):
    """Raised when a file is not a readable V3 Pharo image."""


@dataclass(frozen=True)
class Behavior:
    """One side (instance or class) of a class or trait found in the image."""

    name: str
    side: str
    category: str
    selectors: tuple[tuple[str, str], ...]  # (selector, protocol)


class ObjectMemory:
    """Random access to the objects of a little-endian 32-bit V3 image."""

    def __init__(self, data: bytes) -> None:
        if len(data) < 64:
            raise ImageFormatError("file is too small to be an image")

        image_format, header_size, data_size, base, special = struct.unpack_from(
            "<5I", data, 0
        )
        if image_format not in SUPPORTED_FORMATS:
            raise ImageFormatError(
                f"unsupported image format {image_format}; expected one of "
                f"{sorted(SUPPORTED_FORMATS)}"
            )

        self._data = data
        self._delta = header_size - base  # file offset = oop + delta
        self._end = header_size + data_size
        self._words = struct.Struct("<I")
        self._class_of: dict[int, int] = {}
        self._header: dict[int, int] = {}
        self._size: dict[int, int] = {}

        self._scan(header_size, base)
        self.special_objects = self.pointers(special)
        self.nil = self.special_objects[SPECIAL_NIL]
        compact = self.pointers(self.special_objects[SPECIAL_COMPACT_CLASSES])
        for oop, header in self._header.items():
            compact_index = (header >> 12) & 0x1F
            if compact_index:
                self._class_of[oop] = compact[compact_index - 1]

    def _scan(self, start: int, base: int) -> None:
        data = self._data
        unpack = self._words.unpack_from
        position = start

        while position < self._end:
            (word,) = unpack(data, position)
            header_type = word & 3

            if header_type == 2:  # free chunk
                position += word & ~3
                continue

            if header_type == 0:  # size word, class word, base header
                size = word & ~3
                (class_word,) = unpack(data, position + 4)
                position += 8
                (header,) = unpack(data, position)
            elif header_type == 1:  # class word, base header
                class_word = word
                position += 4
                (header,) = unpack(data, position)
                size = header & 0xFC
            else:  # base header only, compact class
                class_word = 0
                header = word
                size = header & 0xFC

            oop = position - self._delta
            self._header[oop] = header
            self._size[oop] = size
            self._class_of[oop] = class_word & ~3
            position += size

    def __iter__(self) -> Iterator[int]:
        return iter(self._header)

    def __contains__(self, oop: int) -> bool:
        return oop in self._header

    def class_of(self, oop: int) -> int:
        return self._class_of[oop]

    def object_format(self, oop: int) -> int:
        return (self._header[oop] >> 8) & 0xF

    def is_object(self, oop: int) -> bool:
        return not oop & 1 and oop in self._header

    def pointers(self, oop: int) -> list[int]:
        count = (self._size[oop] - 4) // 4
        return list(
            struct.unpack_from(f"<{count}I", self._data, oop + self._delta + 4)
        )

    def is_bytes(self, oop: int) -> bool:
        return self.is_object(oop) and 8 <= self.object_format(oop) <= 11

    def text(self, oop: int) -> str:
        length = self._size[oop] - 4 - (self.object_format(oop) & 3)
        start = oop + self._delta + 4
        return self._data[start:start + length].decode("latin-1")

    def is_array_of_strings(self, oop: int) -> bool:
        if not self.is_object(oop) or self.object_format(oop) != 2:
            return False
        elements = self.pointers(oop)
        return bool(elements) and all(self.is_bytes(item) for item in elements)

    @staticmethod
    def small_integer(oop: int) -> int | None:
        return oop >> 1 if oop & 1 else None


class ImageReader:
    """Enumerates the classes and traits reachable from the system dictionary."""

    def __init__(self, memory: ObjectMemory) -> None:
        self.memory = memory
        self.nil = memory.nil
        self._layout_cache: dict[int, list[str]] = {}

        self.metaclass = self._find_metaclass()
        self._inst_var_slot = self._find_inst_var_names_slot()
        self._this_class_slot = self._slot(self.metaclass, "thisClass")

    # -- bootstrap ---------------------------------------------------------

    def _find_metaclass(self) -> int:
        """Find Metaclass: the class whose class is one of its own instances.

        ``Metaclass class`` satisfies the same cycle, but being a metaclass it
        carries fewer slots than ``Metaclass``, which is a full class.
        """
        memory = self.memory
        candidates = []
        for class_oop in set(memory._class_of.values()):
            if not memory.is_object(class_oop):
                continue
            meta = memory.class_of(class_oop)
            if memory.is_object(meta) and memory.class_of(meta) == class_oop:
                candidates.append(class_oop)
        if not candidates:
            raise ImageFormatError("cannot locate Metaclass in the image")
        return max(candidates, key=lambda oop: len(memory.pointers(oop)))

    def _find_inst_var_names_slot(self) -> int:
        """Locate ``instanceVariables`` by looking at how Metaclass describes itself."""
        memory = self.memory
        slots = memory.pointers(self.metaclass)
        for index in range(SLOT_FORMAT + 1, len(slots)):
            if not memory.is_array_of_strings(slots[index]):
                continue
            names = {memory.text(item) for item in memory.pointers(slots[index])}
            if "thisClass" in names:
                return index
        raise ImageFormatError("cannot locate instance variable names")

    def _layout(self, class_oop: int) -> list[str]:
        """All instance variable names of ``class_oop``, superclass first."""
        cached = self._layout_cache.get(class_oop)
        if cached is not None:
            return cached

        memory = self.memory
        slots = memory.pointers(class_oop)
        superclass = slots[SLOT_SUPERCLASS]
        names = list(self._layout(superclass)) if superclass != self.nil else []
        own = slots[self._inst_var_slot]
        if memory.is_array_of_strings(own):
            names.extend(memory.text(item) for item in memory.pointers(own))

        self._layout_cache[class_oop] = names
        return names

    def _slot(self, class_oop: int, name: str) -> int:
        try:
            return self._layout(class_oop).index(name)
        except ValueError:
            raise ImageFormatError(
                f"{self._class_name(class_oop)} has no instance variable {name!r}"
            ) from None

    def _field(self, oop: int, name: str, default: int | None = None) -> int:
        layout = self._layout(self.memory.class_of(oop))
        if name not in layout:
            if default is None:
                raise ImageFormatError(f"object has no instance variable {name!r}")
            return default
        return self.memory.pointers(oop)[layout.index(name)]

    def _class_name(self, class_oop: int) -> str:
        memory = self.memory
        if memory.class_of(class_oop) == self.metaclass:
            base = memory.pointers(class_oop)[self._this_class_slot]
            return f"{self._class_name(base)} class"
        name = self._field(class_oop, "name", self.nil)
        return memory.text(name) if memory.is_bytes(name) else "?"

    # -- collections -------------------------------------------------------

    def _hashed_elements(self, oop: int) -> list[int]:
        """Non-nil elements of a Set/Dictionary-like object with an ``array`` slot."""
        array = self._field(oop, "array")
        return [item for item in self.memory.pointers(array) if item != self.nil]

    def _strings(self, array_oop: int) -> list[str]:
        memory = self.memory
        if not memory.is_object(array_oop) or array_oop == self.nil:
            return []
        return [
            memory.text(item) if memory.is_bytes(item) else ""
            for item in memory.pointers(array_oop)
        ]

    def _categorized_elements(self, organizer: int) -> dict[str, str]:
        """Decode a Categorizer: map each element to the category it is filed under."""
        memory = self.memory
        if organizer == self.nil or not memory.is_object(organizer):
            return {}
        layout = self._layout(memory.class_of(organizer))
        if "categoryArray" not in layout:
            return {}

        categories = self._strings(self._field(organizer, "categoryArray"))
        elements = self._strings(self._field(organizer, "elementArray"))
        stops_oop = self._field(organizer, "categoryStops")
        stops = [
            memory.small_integer(item) or 0
            for item in memory.pointers(stops_oop)
        ]

        result: dict[str, str] = {}
        start = 0
        for category, stop in zip(categories, stops):
            for element in elements[start:stop]:
                result[element] = category
            start = stop
        return result

    # -- system structure --------------------------------------------------

    def _system_dictionary(self) -> int:
        memory = self.memory
        candidates = [
            oop for oop in memory
            if memory.is_object(memory.class_of(oop))
            and memory.class_of(memory.class_of(oop)) != self.metaclass
            and memory.class_of(memory.class_of(memory.class_of(oop))) == self.metaclass
            and self._class_name(memory.class_of(oop)) == "SystemDictionary"
        ]
        if not candidates:
            raise ImageFormatError("cannot locate the system dictionary")
        return max(candidates, key=lambda oop: len(self._hashed_elements(oop)))

    def _class_categories(self, globals_by_name: dict[str, int]) -> dict[str, str]:
        organization = globals_by_name.get("SystemOrganization")
        if organization is None:
            return {}
        return self._categorized_elements(organization)

    def _local_selectors(self, behavior: int) -> list[str]:
        """Selectors defined by ``behavior`` itself, excluding trait-provided ones."""
        memory = self.memory
        dictionary = self._field(behavior, "methodDict", self.nil)
        if dictionary == self.nil:
            return []

        fixed = len(self._layout(memory.class_of(dictionary)))
        selectors = [
            memory.text(key)
            for key in memory.pointers(dictionary)[fixed:]
            if key != self.nil and memory.is_bytes(key)
        ]

        local = self._field(behavior, "localSelectors", self.nil)
        if local == self.nil:
            return selectors
        allowed = {
            memory.text(item)
            for item in self._hashed_elements(local)
            if memory.is_bytes(item)
        }
        return [selector for selector in selectors if selector in allowed]

    def _behavior(self, oop: int, name: str, side: str, category: str) -> Behavior:
        protocols = self._categorized_elements(
            self._field(oop, "organization", self.nil)
        )
        selectors = tuple(
            (selector, protocols.get(selector, ""))
            for selector in sorted(self._local_selectors(oop))
        )
        return Behavior(name=name, side=side, category=category, selectors=selectors)

    def behaviors(self) -> Iterator[Behavior]:
        """Yield the instance and class side of every global class and trait."""
        memory = self.memory
        globals_by_name: dict[str, int] = {}

        for association in self._hashed_elements(self._system_dictionary()):
            key, value = memory.pointers(association)[:2]
            if memory.is_bytes(key):
                globals_by_name[memory.text(key)] = value

        categories = self._class_categories(globals_by_name)

        for name, oop in sorted(globals_by_name.items()):
            if not memory.is_object(oop) or oop == self.nil:
                continue
            kind = memory.class_of(oop)
            if not memory.is_object(kind):
                continue

            is_class = memory.class_of(kind) == self.metaclass
            is_trait = not is_class and self._class_name(kind) == "Trait"
            if not is_class and not is_trait:
                continue

            own_name = self._field(oop, "name", self.nil)
            if not memory.is_bytes(own_name) or memory.text(own_name) != name:
                continue  # an alias such as a global pointing at another class

            category = categories.get(name, "")
            if not category:
                stored = self._field(oop, "category", self.nil)
                category = memory.text(stored) if memory.is_bytes(stored) else ""

            class_side = kind if is_class else self._field(oop, "classTrait", self.nil)
            yield self._behavior(oop, name, "instance", category)
            if class_side != self.nil and memory.is_object(class_side):
                yield self._behavior(class_side, name, "class", category)

    def package_names(self) -> set[str]:
        """Names of the Monticello packages loaded in the image."""
        memory = self.memory
        names: set[str] = set()
        for oop in memory:
            kind = memory.class_of(oop)
            if not memory.is_object(kind) or memory.class_of(kind) == self.metaclass:
                continue
            if memory.class_of(memory.class_of(kind)) != self.metaclass:
                continue
            if self._class_name(kind) != "MCPackage":
                continue
            name = self._field(oop, "name", self.nil)
            if memory.is_bytes(name):
                names.add(memory.text(name))
        return names


class PackageResolver:
    """Assign categories and extension protocols to Monticello packages.

    Before Pharo 2 a package was a naming convention, not an object owning its
    contents: package ``P`` contains the classes whose category is ``P`` or
    starts with ``P-``, and the methods filed under a ``*P`` protocol.  This
    reproduces that rule (``PackageInfo``), preferring the most specific match.
    """

    def __init__(self, package_names: set[str]) -> None:
        # Lookups ignore case, and some images hold two packages whose names
        # differ only in case (``BogusExt`` and ``Bogusext``).  Sorting makes the
        # choice independent of set order: the last name in sorted order wins.
        self._packages = {name.lower(): name for name in sorted(package_names) if name}

    def _lookup(self, label: str) -> str | None:
        key = label.lower()
        while key:
            if key in self._packages:
                return self._packages[key]
            if "-" not in key:
                return None
            key = key.rsplit("-", 1)[0]
        return None

    def package_of_category(self, category: str) -> str:
        return self._lookup(category) or category

    def package_of_extension(self, protocol: str) -> str | None:
        if not protocol.startswith("*"):
            return None
        label = protocol[1:].strip()
        return self._lookup(label) or label or None


def read_image_declarations(image: bytes) -> list[Declaration]:
    """Extract every class and method declaration from the bytes of a V3 image."""
    reader = ImageReader(ObjectMemory(image))
    resolver = PackageResolver(reader.package_names())
    declarations: list[Declaration] = []

    for behavior in reader.behaviors():
        home = resolver.package_of_category(behavior.category)
        if behavior.side == "instance":
            declarations.append(
                Declaration(kind="class", package=home, class_name=behavior.name)
            )
        for selector, protocol in behavior.selectors:
            package = resolver.package_of_extension(protocol) or home
            declarations.append(
                Declaration(
                    kind="method",
                    package=package,
                    class_name=behavior.name,
                    side=behavior.side,
                    selector=selector,
                )
            )

    return declarations
