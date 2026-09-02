"""Believability proxies must separate adult register from middle-school voice."""

from __future__ import annotations

from eval import believability as bel

ADULT = [
    "To determine the point at which the two plans are equivalent, we can "
    "construct an equation for each option and solve for the number of texts.",
    "The monthly fee represents a constant term, whereas the per-message charge "
    "constitutes the coefficient of the independent variable.",
]

KID = [
    "wait so is it like 20 plus the texts?",
    "idk i think plan b is cheaper maybe",
    "i got 100 i think but im not sure",
]


def test_reading_grade_separates_the_two_registers():
    assert (
        bel.linguistic_proxies(ADULT)["flesch_kincaid_grade"]
        > bel.linguistic_proxies(KID)["flesch_kincaid_grade"]
    )


def test_kid_voice_hedges_more():
    assert (
        bel.linguistic_proxies(KID)["hedge_rate"]
        > bel.linguistic_proxies(ADULT)["hedge_rate"]
    )


def test_kid_voice_uses_shorter_sentences():
    assert (
        bel.linguistic_proxies(KID)["mean_sentence_words"]
        < bel.linguistic_proxies(ADULT)["mean_sentence_words"]
    )


def test_unterminated_turns_count_as_fragments():
    assert bel.linguistic_proxies(["so like 20 plus 10"])["fragment_rate"] == 1.0
    assert bel.linguistic_proxies(["So it is twenty."])["fragment_rate"] == 0.0


def test_hedges_match_whole_words_only():
    """'number' contains 'um'; it must not register as a hedge."""
    assert bel.linguistic_proxies(["The number of texts is fixed."])["hedge_rate"] == 0.0
    assert bel.linguistic_proxies(["um i think so"])["hedge_rate"] == 1.0


def test_long_turns_are_flagged():
    long_reply = " ".join(["word"] * (bel.LONG_TURN_WORDS + 5)) + "."
    assert bel.linguistic_proxies([long_reply])["long_turn_rate"] == 1.0
    assert bel.linguistic_proxies(["short one."])["long_turn_rate"] == 0.0


def test_empty_and_blank_input_is_safe():
    for payload in ([], [""], ["   ", ""]):
        proxies = bel.linguistic_proxies(payload)
        assert proxies["n_replies"] == 0
        assert proxies["flesch_kincaid_grade"] == 0.0


def test_judge_parses_a_json_verdict(monkeypatch):
    monkeypatch.setattr(
        bel, "complete", lambda *a, **k: '{"score": 4, "tell": null}'
    )
    assert bel.believability({"reply": "idk maybe plan b"})["score"] == 4


def test_judge_failure_degrades_to_zero(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("api down")

    monkeypatch.setattr(bel, "complete", boom)
    verdict = bel.believability({"reply": "idk maybe plan b"})
    assert verdict["score"] == 0
    assert "api down" in verdict["tell"]


def test_empty_reply_skips_the_llm(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("should not call the judge on an empty reply")

    monkeypatch.setattr(bel, "complete", boom)
    assert bel.believability({"reply": "   "})["score"] == 0
