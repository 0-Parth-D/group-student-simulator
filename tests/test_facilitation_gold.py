"""Track C.0 — facilitation gold vs regex parse_speak_constraints."""

from eval.fixtures.group_facilitation_gold import (
    load_facilitation_gold,
    score_regex_baseline,
)


def test_gold_file_loads_and_has_personality_brief():
    gold = load_facilitation_gold()
    assert gold["roster"]["profile_ids"] == ["jordan", "sam", "alex"]
    assert set(gold["personality_brief"]) == {"jordan", "sam", "alex"}
    assert len(gold["cases"]) >= 15


def test_regex_baseline_scores_against_gold():
    summary = score_regex_baseline()
    # Core demo / classic phrasing should mostly pass; paraphrases may fail.
    assert summary["n"] >= 40
    assert summary["by_field"]["mode"] >= 0.7
    # Critical classic watch/listen IDs must keep excludes (not paraphrase gaps).
    classic_ids = {
        "discuss_watch_jordan_classic",
        "discuss_watch_jordan_semicolon",
        "discuss_jordan_listen",
        "demo_04_watch",
        "discuss_exclude_sam_watch",
        "discuss_exclude_alex_listen",
        "paraphrase_observe_quietly",
        "edge_be_quiet_alex",
    }
    classic_hard = [r for r in summary["rows"] if r["id"] in classic_ids]
    assert classic_hard, "expected classic hard exclude cases"
    for row in classic_hard:
        assert row["fields"]["must_not_speak"], (
            f"classic hard case lost exclude: {row['id']} "
            f"pred={row['predicted']} exp={row['expected']}"
        )


def test_demo_script_watch_case_excludes_jordan():
    summary = score_regex_baseline()
    row = next(r for r in summary["rows"] if r["id"] == "demo_04_watch")
    assert row["fields"]["must_not_speak"]
    assert row["predicted"]["must_not_speak"] == ["jordan"]


def test_topic_only_keeps_jordan_out_of_must_speak():
    summary = score_regex_baseline()
    row = next(
        r for r in summary["rows"] if r["id"] == "topic_jordan_method_sam_explains"
    )
    assert "jordan" not in row["predicted"]["must_speak"]
    assert row["fields"]["must_speak"]
