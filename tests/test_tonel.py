import pytest

from pharo_naming.extraction.model import Declaration
from pharo_naming.extraction.tonel import TonelError, parse_tonel, read_tonel_declarations


CLASS_FILE = """\
"
I am a comment that quotes code:

Sample >> notAMethod [ ^ 1 ]
"
Class {
	#name : #Sample,
	#superclass : #Object,
	#instVars : [
		'value'
	],
	#category : #'Sample-Package'
}

{ #category : #accessing }
Sample >> value [
	^ value
]

{ #category : #accessing }
Sample >> at: index
	put: anObject [
	^ anObject
]

{ #category : #comparing }
Sample >> >= other [
	^ true
]

{ #category : #examples }
Sample >> quoted [
	^ 'Other >> fake [ ^ $] ]' , "a ] comment" (String with: $[)
]

{ #category : #'instance creation' }
Sample class >> from: start to: stop [
	^ [ :x | [ x ] value ] value: self new
]
"""


def selectors(text):
    return [(method.side, method.selector) for method in parse_tonel(text).methods]


def test_methods_are_parsed_and_bodies_skipped():
    assert selectors(CLASS_FILE) == [
        ("instance", "value"),
        ("instance", "at:put:"),
        ("instance", ">="),
        ("instance", "quoted"),
        ("class", "from:to:"),
    ]


def test_class_name_is_read_in_every_tonel_dialect():
    for name in ("#Sample", "'Sample'", "#'Sample'"):
        parsed = parse_tonel(f"Class {{\n\t#name : {name},\n\t#superclass : #Object\n}}\n")
        assert parsed.kind == "Class"
        assert parsed.name == "Sample"


def test_trait_class_side_is_recognized():
    text = (
        "Trait {\n\t#name : #TSample\n}\n\n"
        "{ #category : #testing }\nTSample classTrait >> isSample [\n\t^ true\n]\n"
    )
    assert selectors(text) == [("class", "isSample")]


def test_unbalanced_literal_array_does_not_swallow_the_next_method():
    text = (
        "Class {\n\t#name : #Sample\n}\n\n"
        "{ #category : #a }\nSample >> odd [\n\t^ #( 1 ] 2 )\n]\n\n"
        "{ #category : #a }\nSample >> next [\n\t^ 1\n]\n"
    )
    assert selectors(text) == [("instance", "odd"), ("instance", "next")]


def test_garbage_is_rejected():
    with pytest.raises(TonelError):
        parse_tonel("this is not Tonel")


def test_tree_assigns_packages_and_extensions():
    files = {
        "Sample-Package/Sample.class.st": CLASS_FILE.encode(),
        "Sample-Package/package.st": b"Package { #name : #'Sample-Package' }\n",
        "Other-Package/Sample.extension.st": (
            b"Extension { #name : #Sample }\n\n"
            b"{ #category : #'*Other-Package' }\nSample >> extended [\n\t^ 1\n]\n"
        ),
    }

    declarations = read_tonel_declarations(files)

    assert Declaration("class", "Sample-Package", "Sample") in declarations
    assert Declaration("method", "Other-Package", "Sample", "instance", "extended") in declarations
    assert sum(item.kind == "class" for item in declarations) == 1
    assert sum(item.kind == "method" for item in declarations) == 6
