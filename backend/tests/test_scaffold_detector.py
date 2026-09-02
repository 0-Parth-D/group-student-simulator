"""Scaffold detection must stay scoped to the task and neutral about topic."""

from __future__ import annotations

from app.scaffold_detector import _explanation_quality, detect_scaffold

PHONE_TASK = {
    "required_constructs": [
        "unit_rate",
        "additive_vs_multiplicative",
        "rate_comparison",
        "linear_relationship",
        "symbolic_modeling",
    ],
    "description": (
        "Plan A charges a $20 monthly fee plus $0.10 per text. Plan B charges "
        "$0.30 per text and no monthly fee. When do the plans cost the same?"
    ),
}

FRACTION_TASK = {
    "required_constructs": ["part_whole", "fraction_equivalence", "fraction_operations"],
    "description": "Maria ate 3/8 of a pizza and her brother ate 1/4. How much is left?",
}


def _zero_embed(texts):
    """Stand-in embedder that never clears the similarity threshold."""
    return [[0.0, 0.0] for _ in texts]


def test_constructs_outside_the_task_never_fire():
    # "compare" is a part_whole keyword, but part_whole is not in this task.
    events = detect_scaffold(
        "can you compare the two plans for me",
        PHONE_TASK,
        "math_scaffold",
        embed_fn=_zero_embed,
    )
    fired = {e.construct_id for e in events}
    assert "part_whole" not in fired
    assert fired <= set(PHONE_TASK["required_constructs"])


def test_keyword_hit_still_maps_to_its_construct():
    events = detect_scaffold(
        "which plan is the better deal",
        PHONE_TASK,
        "math_scaffold",
        embed_fn=_zero_embed,
    )
    assert "rate_comparison" in {e.construct_id for e in events}


def test_quality_is_independent_of_topic():
    scaffold = "what happens to the total because each one costs more"
    on_fractions = _explanation_quality(scaffold, FRACTION_TASK["description"])
    on_rates = _explanation_quality(scaffold, PHONE_TASK["description"])
    assert on_fractions == on_rates


def test_quality_rewards_reasoning_and_probing():
    bare = _explanation_quality("do that")
    rich = _explanation_quality(
        "what changes in the total, because each one adds to the amount you already have"
    )
    assert rich > bare


def test_semantic_fallback_covers_constructs_keywords_miss():
    """A construct with no keyword hit still earns credit when embeddings align."""

    def aligned_embed(texts):
        # Only symbolic_modeling's vocabulary and the message share a direction.
        return [
            [1.0, 0.0] if "equation" in t.lower() else [0.0, 1.0]
            for t in texts
        ]

    events = detect_scaffold(
        "try writing an equation for each one",
        PHONE_TASK,
        "math_scaffold",
        embed_fn=aligned_embed,
    )
    assert "symbolic_modeling" in {e.construct_id for e in events}


def test_semantic_events_stay_within_required_constructs():
    def always_aligned(texts):
        return [[1.0, 0.0] for _ in texts]

    events = detect_scaffold(
        "think about how this quantity grows",
        FRACTION_TASK,
        "math_scaffold",
        embed_fn=always_aligned,
    )
    assert {e.construct_id for e in events} <= set(FRACTION_TASK["required_constructs"])


def test_non_math_turns_produce_no_events():
    assert detect_scaffold("good morning everyone", PHONE_TASK, "social") == []


def test_detection_degrades_without_an_embedder():
    """No embed_fn and no sentence-transformers installed must not raise."""

    def exploding_embed(texts):
        raise RuntimeError("no embedder available")

    events = detect_scaffold(
        "which plan is the better deal",
        PHONE_TASK,
        "math_scaffold",
        embed_fn=exploding_embed,
    )
    assert "rate_comparison" in {e.construct_id for e in events}
