from jokegpt.data.cleaning import clean_text, is_valid_line


def test_lowercases_and_keeps_allowed_punctuation():
    assert clean_text("Why? Yes!") == "why? yes!"


def test_strips_disallowed_characters():
    assert "@" not in clean_text("a@b")
    assert "#" not in clean_text("tag #joke")


def test_removes_urls_and_emails():
    cleaned = clean_text("see http://example.com or me@example.com now")
    assert "example" not in cleaned
    assert "see" in cleaned and "now" in cleaned


def test_removes_words_containing_digits():
    assert clean_text("abc123 def") == "def"


def test_removes_very_long_words():
    assert "x" * 25 not in clean_text("a " + "x" * 25 + " b")


def test_collapses_repeated_whitespace():
    assert clean_text("a    b\n\nc") == "a b c"


def test_accepts_non_string_input():
    assert clean_text(42) == ""


def test_trims_to_sentence_bounds():
    # Content before the first period and after the last is dropped.
    assert clean_text("tail end. keep this. start") == "keep this"


def test_leaves_text_without_periods_intact():
    assert clean_text("no periods here") == "no periods here"


def test_is_valid_line_rejects_empty_short_and_letterless():
    assert not is_valid_line("")
    assert not is_valid_line("a")
    assert not is_valid_line("!!!")
    assert is_valid_line("ok")


def test_trim_sentences_destroys_setup_by_default():
    """Documents the original notebook behaviour: the setup is lost."""
    joke = "I told my wife she was drawing her eyebrows too high. She looked surprised."
    assert clean_text(joke) == "she looked surprised"


def test_trim_sentences_disabled_keeps_whole_joke():
    joke = "I told my wife she was drawing her eyebrows too high. She looked surprised."
    cleaned = clean_text(joke, trim_sentences=False)
    assert cleaned.startswith("i told my wife")
    assert cleaned.endswith("she looked surprised.")


def test_question_setups_survive_trimming():
    """Jokes whose setup ends in '?' are unaffected, which is why dad jokes work."""
    joke = "What do you call a fake noodle? An impasta."
    assert clean_text(joke) == "what do you call a fake noodle? an impasta."
