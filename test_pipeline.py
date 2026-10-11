import pandas as pd
import pytest

from pipeline import build_model, clean_text, find_similar


def tiny_dataset():
    topics = {"sport": "football match goal team coach cup",
              "money": "market shares profit bank economy company",
              "film": "movie actor award music star cinema"}
    rows = []
    for name, words in topics.items():
        for i in range(10):
            rows.append({"title": f"{name} story {i}", "text": f"{words} {words.split()[i % 6]}"})
    return pd.DataFrame(rows)


def test_clean_text():
    assert clean_text("The <b>UK</b> said: Prices ROSE in 2004!") == "uk prices rose"
    assert clean_text(None) == "" and clean_text("   ") == ""


def test_model_assigns_every_article():
    model = build_model(tiny_dataset(), k=3)
    assert len(model["df"]) == 30
    assert model["df"]["cluster"].nunique() == 3
    assert -1 <= model["silhouette"] <= 1


def test_invalid_k():
    with pytest.raises(ValueError):
        build_model(tiny_dataset(), k=1)


def test_similar_excludes_itself_and_returns_five():
    model = build_model(tiny_dataset(), k=3)
    result = find_similar(model, index=0)
    assert len(result) == 5 and 0 not in result.index
    assert list(result["similarity"]) == sorted(result["similarity"], reverse=True)


def test_empty_search_is_rejected():
    model = build_model(tiny_dataset(), k=3)
    with pytest.raises(ValueError):
        find_similar(model, text="   ")