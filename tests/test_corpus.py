import pytest

from pharo_naming.analysis.validate_corpus import validate_records
from pharo_naming.corpus import (
    IdentifierRecord,
    available_versions,
    corpus_path,
    read_corpus,
    read_records,
    write_records,
)
from pharo_naming.releases import RELEASE_LABELS, release_index


RECORDS = [
    IdentifierRecord("2", "Kernel", "Object", "class"),
    IdentifierRecord("2", "Kernel", "Object", "method", "instance", ">="),
    IdentifierRecord("2", "Kernel", "Object", "method", "class", "year:month:day:"),
]


def test_records_survive_a_round_trip(tmp_path):
    path = corpus_path(tmp_path, "2")
    write_records(path, RECORDS)
    assert list(read_records(path)) == RECORDS


def test_corpus_files_are_reproducible(tmp_path):
    write_records(tmp_path / "a.jsonl.gz", RECORDS)
    write_records(tmp_path / "b.jsonl.gz", RECORDS)
    assert (tmp_path / "a.jsonl.gz").read_bytes() == (tmp_path / "b.jsonl.gz").read_bytes()


def test_corpus_is_read_in_historical_order(tmp_path):
    for version in ("10", "3", "2", "1.0"):
        write_records(
            corpus_path(tmp_path, version),
            [IdentifierRecord(version, "Kernel", "Object", "class")],
        )
    assert available_versions(tmp_path) == ["1.0", "2", "3", "10"]
    assert list(read_corpus(tmp_path)) == ["1.0", "2", "3", "10"]


def test_pharo_1_precedes_pharo_2():
    assert RELEASE_LABELS[:3] == ("1.0", "2", "3")
    assert release_index("1.0") < release_index("2") < release_index("14")
    with pytest.raises(ValueError):
        release_index("1.4")
    assert release_index("1.0") == 1
    with pytest.raises(ValueError):
        release_index("0.9")


def test_valid_release_is_summarized():
    row = validate_records("2", RECORDS)
    assert row["class_declarations"] == 1
    assert row["method_declarations"] == 2
    assert row["methods_in_undeclared_classes"] == 0


def test_duplicate_declaration_is_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        validate_records("2", RECORDS + RECORDS[-1:])


def test_record_of_another_release_is_rejected():
    with pytest.raises(ValueError, match="belongs to version"):
        validate_records("3", RECORDS)
