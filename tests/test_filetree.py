from pharo_naming.extraction.filetree import read_filetree_declarations
from pharo_naming.extraction.model import Declaration


def test_metadata_layout_of_pharo_2():
    files = {
        "Kernel.package/Boolean.class/properties.json": b'{ "name" : "Boolean" }',
        "Kernel.package/Boolean.class/instance/and..st": b"controlling\nand: aBlock\n\t^ self",
        "Kernel.package/Boolean.class/instance/^and.st": b"logical operations\n& aBoolean\n\t^ self",
        "Kernel.package/Boolean.class/class/new.st": b"instance creation\nnew\n\t^ self error",
        "Kernel.package/TComparable.trait/properties.json": b"{}",
        "AST-Core.package/Boolean.extension/instance/isLiteral.st": b"*AST-Core\nisLiteral\n\t^ true",
    }

    assert set(read_filetree_declarations(files)) == {
        Declaration("class", "Kernel", "Boolean"),
        Declaration("class", "Kernel", "TComparable"),
        Declaration("method", "Kernel", "Boolean", "instance", "and:"),
        Declaration("method", "Kernel", "Boolean", "instance", "&"),
        Declaration("method", "Kernel", "Boolean", "class", "new"),
        Declaration("method", "AST-Core", "Boolean", "instance", "isLiteral"),
    }


def test_metadata_less_layout_of_pharo_3_to_6():
    files = {
        "Kernel.package/Boolean.class/definition.st": b"Object subclass: #Boolean",
        "Kernel.package/Boolean.class/README.md": b"I am a boolean.",
        "Kernel.package/Boolean.class/instance/controlling/and_.st": b"and: aBlock\n\t^ self",
        "Kernel.package/Boolean.class/instance/-- all --/and_.st": b"and: aBlock\n\t^ self",
        "Kernel.package/Boolean.class/class/instance creation/new.st": b"new\n\t^ self error",
        "AST-Core.package/extension/Boolean/instance/isLiteral.st": b"isLiteral\n\t^ true",
    }

    assert sorted(read_filetree_declarations(files), key=repr) == sorted(
        [
            Declaration("class", "Kernel", "Boolean"),
            Declaration("method", "Kernel", "Boolean", "instance", "and:"),
            Declaration("method", "Kernel", "Boolean", "class", "new"),
            Declaration("method", "AST-Core", "Boolean", "instance", "isLiteral"),
        ],
        key=repr,
    )


def test_wide_string_sources_are_decoded():
    files = {
        "A.package/B.class/definition.st": b"",
        "A.package/B.class/instance/x/bswap_.st": "bswap: aRegister\n\t^ 1".encode("utf-32-be"),
    }

    assert Declaration("method", "A", "B", "instance", "bswap:") in read_filetree_declarations(files)
