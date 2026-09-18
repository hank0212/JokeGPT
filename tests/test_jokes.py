"""Dataset loaders, exercised against synthetic files with the real schemas."""

import json

import pandas as pd
import pytest

from jokegpt.data.jokes import (
    build_corpus,
    load_dadjokes,
    load_joke_dataset,
    load_short_jokes,
    write_corpus,
)


@pytest.fixture
def joke_dataset_dir(tmp_path):
    root = tmp_path / "joke-dataset"
    root.mkdir()
    (root / "reddit_jokes.json").write_text(
        json.dumps([{"title": "A title.", "body": "A body. Yes."}])
    )
    (root / "stupidstuff.json").write_text(json.dumps([{"body": "Stupid. Joke."}]))
    (root / "wocka.json").write_text(json.dumps([{"body": "Wocka. Joke."}]))
    return root


@pytest.fixture
def dadjokes_dir(tmp_path):
    root = tmp_path / "dadjokes"
    root.mkdir()
    for name in ("train.csv", "test.csv"):
        pd.DataFrame(
            {"question": ["Why the road?"], "response": ["To cross. Indeed."]}
        ).to_csv(root / name, index=False)
    return root


@pytest.fixture
def short_jokes_dir(tmp_path):
    root = tmp_path / "short-jokes-dataset"
    root.mkdir()
    pd.DataFrame({"Joke": ["Short. Joke here."]}).to_csv(
        root / "shortjokes.csv", index=False
    )
    return root


def test_load_joke_dataset_concatenates_all_three_files(joke_dataset_dir):
    series = load_joke_dataset(joke_dataset_dir)
    assert len(series) == 3
    assert "A title." in series.iloc[0] and "A body." in series.iloc[0]


def test_load_dadjokes_joins_question_and_response(dadjokes_dir):
    series = load_dadjokes(dadjokes_dir)
    assert len(series) == 2
    assert "Why the road?" in series.iloc[0] and "To cross." in series.iloc[0]


def test_load_short_jokes_reads_joke_column(short_jokes_dir):
    assert load_short_jokes(short_jokes_dir).iloc[0] == "Short. Joke here."


def test_missing_dataset_file_names_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="reddit_jokes.json"):
        load_joke_dataset(tmp_path)


def test_build_corpus_cleans_and_deduplicates():
    source = pd.Series(["Hello. World here.", "Hello. World here.", "Other. Text now."])
    lines = build_corpus([source])
    assert len(lines) == 2
    assert all(line == line.lower() for line in lines)


def test_build_corpus_drops_invalid_lines():
    lines = build_corpus([pd.Series(["!!!", "a real joke here"])])
    assert lines == ["a real joke here"]


def test_build_corpus_rejects_empty_source_list():
    with pytest.raises(ValueError, match="no joke sources"):
        build_corpus([])


def test_write_corpus_writes_one_joke_per_line(tmp_path):
    path = write_corpus(["first joke", "second joke"], tmp_path / "out" / "jokes.txt")
    assert path.read_text(encoding="utf-8") == "first joke\nsecond joke\n"


def test_nan_fields_do_not_produce_literal_nan(joke_dataset_dir):
    """Reddit rows with a null body must not leak the string 'nan' into the corpus."""
    (joke_dataset_dir / "reddit_jokes.json").write_text(
        json.dumps([{"title": "Only a title.", "body": None}])
    )
    series = load_joke_dataset(joke_dataset_dir)
    assert "nan" not in series.iloc[0].lower()


def test_build_corpus_can_disable_sentence_trimming():
    source = pd.Series(["Setup ends here. Punchline lands."])
    trimmed = build_corpus([source])
    whole = build_corpus([source], trim_sentences=False)
    assert trimmed == ["punchline lands"]
    assert whole == ["setup ends here. punchline lands."]
