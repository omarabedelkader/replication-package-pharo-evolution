import math

from pharo_naming.analysis.family_contribution_analysis import contributions
from pharo_naming.corpus import IdentifierRecord


def klass(version, package, name):
    return IdentifierRecord(version, package, name, "class")


def test_a_family_that_leaves_carries_the_whole_change():
    stable = ["Apple", "Apricot", "Banana", "Berry", "Cherry"]
    corpus = {
        "1.0": [klass("1.0", "Fruit-Core", name) for name in stable]
        + [klass("1.0", "OB-Core", name) for name in ("OBA", "OBB", "OBC", "OBD")],
        "1.1": [klass("1.1", "Fruit-Core", name) for name in stable],
    }

    result = contributions(corpus, "class")
    rows = {row["family"]: row for row in result["families"]}

    assert rows["OB"]["present"] == ["1.0"]
    assert rows["OB"]["first_size"] == 4 and rows["OB"]["last_size"] == 0
    assert math.isclose(rows["OB"]["change"], result["change"])
    assert math.isclose(result["groups"]["left"]["change_without"], 0.0, abs_tol=1e-12)
