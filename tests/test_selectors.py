import pytest

from pharo_naming.extraction.selectors import selector_from_source


@pytest.mark.parametrize(
    ("source", "selector"),
    [
        ("size\n\t^ 0", "size"),
        ("isEmpty", "isEmpty"),
        ("at: index\n\t^ nil", "at:"),
        ("at: index put: value\n\t^ value", "at:put:"),
        ("at: index\n\tput: value\n\t^ value", "at:put:"),
        ('at: index "the position" put: value', "at:put:"),
        ("from: start to: stop by: step", "from:to:by:"),
        ("+ aNumber", "+"),
        (">= other", ">="),
        ("==> aBlock", "==>"),
        (", aCollection", ","),
        ("-> anObject", "->"),
        ("| aBoolean", "|"),
        ("x\n\tx := 3", "x"),
        ("value: anObject\n\tvalue := anObject", "value:"),
        ("_underscore: a", "_underscore:"),
    ],
)
def test_selector_is_recovered(source, selector):
    assert selector_from_source(source) == selector


@pytest.mark.parametrize("source", ["", "   ", "123", "at: 3"])
def test_invalid_pattern_is_rejected(source):
    assert selector_from_source(source) is None
