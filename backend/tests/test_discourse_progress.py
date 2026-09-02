"""Deterministic discourse scorer for PST replay."""

from eval.discourse_progress import (
    jordan_passive_hits,
    peer_reference_rate,
    reasoning_warrant_metrics,
    repetition_streak,
    score_export,
    unprompted_crossover,
)


def _transcript():
    return [
        {"speaker_type": "teacher", "content": "Explain your reasoning."},
        {
            "speaker_type": "student",
            "speaker_id": "maya",
            "content": "On my table Plan B is cheaper at 50.",
        },
        {
            "speaker_type": "student",
            "speaker_id": "jordan",
            "content": "Jordan, but 0.10 is less per text so Plan A.",
        },
        {"speaker_type": "teacher", "content": "When is each plan better?"},
        {
            "speaker_type": "student",
            "speaker_id": "jordan",
            "content": "So x = 100 texts where they match.",
        },
    ]


def test_peer_reference_rate_counts_reactions():
    t = _transcript()
    rate = peer_reference_rate(t, {"maya": "Maya", "jordan": "Jordan"})
    assert rate > 0


def test_unprompted_crossover_counts_before_cue():
    t = [
        {"speaker_type": "teacher", "content": "Explain your reasoning."},
        {
            "speaker_type": "student",
            "speaker_id": "jordan",
            "content": "So x = 100 texts where they match.",
        },
    ]
    assert unprompted_crossover(t) == 1


def test_repetition_streak_detects_duplicates():
    t = [
        {
            "speaker_type": "student",
            "speaker_id": "maya",
            "content": "Plan B is cheaper at 50 on my table for sure.",
        },
        {
            "speaker_type": "student",
            "speaker_id": "maya",
            "content": "Plan B is cheaper at 50 on my table for sure.",
        },
        {
            "speaker_type": "student",
            "speaker_id": "maya",
            "content": "Plan B is cheaper at 50 on my table for sure.",
        },
    ]
    assert repetition_streak(t) >= 3


def test_jordan_passive_hits():
    t = [
        {
            "speaker_type": "student",
            "speaker_id": "jordan",
            "content": "I'm not sure what I'm supposed to do.",
        }
    ]
    assert jordan_passive_hits(t) == 1


def test_score_export_shape():
    export = {
        "transcript": _transcript(),
        "display_names": {"maya": "Maya", "jordan": "Jordan"},
        "breakthrough_bullets": ["Phase: fight"],
        "fight_phase": "fight",
    }
    metrics = score_export(export)
    assert "peer_reference_rate" in metrics
    assert "repetition_streak" in metrics
    assert "jordan_passive_hits" in metrics


def test_score_export_jordan_fee_uptake():
    export = {
        "transcript": [],
        "display_names": {"jordan": "Jordan"},
        "fight_phase": "nuanced",
        "slot_flags": {
            "jordan": {"acknowledged_fee": True, "crossover_intuition": False},
        },
    }
    metrics = score_export(export)
    assert metrics["jordan_fee_uptake"] is True
    assert metrics["phase_at_end"] == "nuanced"


def test_score_export_includes_warrant_metrics():
    export = {
        "transcript": [
            {"speaker_type": "teacher", "content": "Jordan what's your reasoning?"},
            {
                "speaker_type": "student",
                "speaker_id": "jordan",
                "content": "0.10 per text is less than 0.30 so Plan A.",
            },
        ],
        "display_names": {"jordan": "Jordan"},
        "fight_phase": "fight",
        "task_id": "phone_plans_linear_01",
        "slot_flags": {
            "jordan": {
                "behavior_mode": "WRONG",
                "student_stack_level": 2,
                "primary_construct": "rate_comparison",
            },
        },
    }
    metrics = score_export(export)
    assert metrics["reasoning_press_turns"] == 1
    assert metrics["warrant_on_press_rate"] == 1.0
    assert metrics["bare_claim_on_press"] == 0


def test_reasoning_warrant_metrics_bare_claim():
    transcript = [
        {"speaker_type": "teacher", "content": "explain your reasoning"},
        {
            "speaker_type": "student",
            "speaker_id": "maya",
            "content": "I still think my answer is right.",
        },
    ]
    m = reasoning_warrant_metrics(transcript)
    assert m["reasoning_press_turns"] == 1
    assert m["bare_claim_on_press"] == 1
