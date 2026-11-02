import itertools
import math
from collections import Counter

import numpy as np

from pharo_naming.analysis.conceptual_cohesion_analysis import class_documents, partial_spearman
from pharo_naming.analysis.turnover_analysis import transition, turnover
from pharo_naming.concepts import (
    between_group_similarity,
    mean_pairwise_similarity,
    prefix_words,
    random_group_similarities,
    split_identifier,
    terms,
    tfidf_matrix,
)
from pharo_naming.corpus import IdentifierRecord


def test_identifiers_are_split_into_words():
    assert split_identifier("at:put:") == ["at", "put"]
    assert split_identifier("OBBrowserModel") == ["ob", "browser", "model"]
    assert split_identifier("printOn:") == ["print", "on"]
    assert split_identifier("+") == []


def test_terms_drop_stop_words_and_merge_plurals():
    assert terms("addAll:withOccurrences:") == ["add", "occurrence"]
    assert terms("FileReferences") == ["file", "reference"]
    assert terms("classes") == terms("class") == ["class"]
    assert terms("dictionaries") == ["dictionary"]


def test_prefix_words_are_the_shared_first_words():
    assert prefix_words(["AthensCanvas", "AthensPath", "AthensSurface", "Color"]) == {"athen"}
    assert prefix_words(["Canvas", "Path", "Surface"]) == set()


def test_mean_pairwise_similarity_matches_the_pairs():
    documents = [Counter(words) for words in (
        ["file", "read"], ["file", "write"], ["file", "path"], ["socket"], ["socket", "read"],
    )]
    vectors = tfidf_matrix(documents)
    dense = vectors.toarray()
    pairs = [float(dense[i] @ dense[j]) for i, j in itertools.combinations(range(len(dense)), 2)]
    assert math.isclose(mean_pairwise_similarity(vectors), np.mean(pairs))
    assert math.isclose(mean_pairwise_similarity(vectors[[0, 1, 2]]), np.mean(pairs[:2] + pairs[4:5]))


def test_a_word_used_by_every_class_weighs_nothing():
    vectors = tfidf_matrix([Counter(["test", "a"]), Counter(["test", "b"])])
    assert math.isclose(mean_pairwise_similarity(vectors), 0.0, abs_tol=1e-12)


def test_random_groups_average_to_the_release_mean():
    documents = [Counter([f"w{i % 4}", f"v{i % 3}"]) for i in range(30)]
    vectors = tfidf_matrix(documents)
    random = random_group_similarities(vectors, 5, 400, np.random.default_rng(0))
    assert abs(random.mean() - mean_pairwise_similarity(vectors)) < 0.03


def test_class_documents_keep_only_the_methods_of_the_family():
    records = [
        IdentifierRecord("1", "Files-Core", "FileReader", "class"),
        IdentifierRecord("1", "Files-Core", "FileReader", "method", "instance", "readStream"),
        IdentifierRecord("1", "Other-Ext", "FileReader", "method", "instance", "drawOn:"),
    ]
    family, document = class_documents(records)["FileReader"]
    assert family == "Files"
    assert document == Counter({"file": 1, "reader": 1, "read": 1, "stream": 1})


def test_partial_spearman_removes_a_common_cause():
    generator = np.random.default_rng(1)
    size = generator.uniform(0, 1, 400)
    x = size + generator.normal(scale=0.3, size=400)
    y = size + generator.normal(scale=0.3, size=400)
    assert np.corrcoef(x, y)[0, 1] > 0.3
    assert abs(partial_spearman(x, y, size)) < 0.15


def test_turnover_counts_kept_added_removed_and_relocated():
    before = {"a": frozenset({"F"}), "b": frozenset({"F"}), "c": frozenset({"F"})}
    after = {"a": frozenset({"F"}), "b": frozenset({"G"}), "d": frozenset({"G"})}
    assert transition(before, after) == {"kept": 2, "added": 1, "removed": 1, "relocated": 1}


def test_turnover_over_the_history():
    def klass(version, package, name):
        return IdentifierRecord(version, package, name, "class")

    corpus = {
        "1": [klass("1", "A-x", "Kept"), klass("1", "A-x", "Gone")],
        "2": [klass("2", "A-x", "Kept"), klass("2", "B-x", "Passing")],
        "3": [klass("3", "B-x", "Kept"), klass("3", "B-x", "New")],
    }
    result = turnover(corpus, "class")
    assert result["overall"] == {"kept": 1, "added": 1, "removed": 1, "relocated": 1}
    assert result["transient"] == 1
    assert result["ever"] == 4


def test_between_group_similarity_matches_the_cross_pairs():
    documents = [Counter(words) for words in (
        ["file", "read"], ["file", "write"], ["socket"], ["socket", "read"], ["path"],
    )]
    dense = tfidf_matrix(documents).toarray()
    vectors = tfidf_matrix(documents)
    pairs = [float(dense[i] @ dense[j]) for i in (0, 1) for j in (2, 3, 4)]
    assert math.isclose(between_group_similarity(vectors[[0, 1]], vectors[[2, 3, 4]]), np.mean(pairs))
