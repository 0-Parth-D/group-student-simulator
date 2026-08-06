"""The group script must work for any bank task, not just the one it was written for."""

import pytest

from app.knowledge.constructs import load_task_metadata
from eval.group_script import (
    NEUTRAL_SCAFFOLDS,
    build_group_script,
    pick_drill,
)

ALL_TASKS = {t["task_id"]: t for t in load_task_metadata().get("tasks", [])}
NAMES = ["Jordan", "Sam", "Alex"]

# Words that would betray a script written for one specific problem.
SCENARIO_WORDS = (
    "pizza",
    "phone",
    "plan a",
    "plan b",
    "marble",
    "juice",
    "text message",
    "fraction",
    "denominator",
)


@pytest.mark.parametrize("task_id", sorted(ALL_TASKS))
def test_script_builds_for_every_bank_task(task_id):
    script = build_group_script(NAMES, ALL_TASKS[task_id])
    assert len(script) >= 10
    for step in script:
        assert step["text"].strip()
        assert step["label"]


@pytest.mark.parametrize("task_id", sorted(ALL_TASKS))
def test_scaffolds_never_name_a_scenario(task_id):
    """Only the drill may carry task-specific words, since it quotes a real task."""
    script = build_group_script(NAMES, ALL_TASKS[task_id])
    for step in script:
        if step["label"] == "math_eval":
            continue
        lowered = step["text"].lower()
        for word in SCENARIO_WORDS:
            assert word not in lowered, f"{task_id}: {step['text']!r} mentions {word!r}"


def test_script_covers_the_evaluated_turn_types():
    script = build_group_script(NAMES, ALL_TASKS["phone_plans_linear_01"])
    labels = {s["label"] for s in script}
    assert {"math_scaffold", "vague_ack", "peer_prompt", "peer_discussion"} <= labels
    assert "off_task" in labels
    assert "closing" in labels


def test_two_consecutive_stall_probes_are_present():
    """The consecutive-stall path only fires on a second vague turn in a row."""
    script = build_group_script(NAMES, ALL_TASKS["phone_plans_linear_01"])
    labels = [s["label"] for s in script]
    pairs = list(zip(labels, labels[1:]))
    assert ("vague_ack", "vague_ack") in pairs


def test_off_task_probe_is_optional():
    script = build_group_script(
        NAMES, ALL_TASKS["phone_plans_linear_01"], include_off_task=False
    )
    assert all(s["label"] != "off_task" for s in script)


@pytest.mark.parametrize("task_id", sorted(ALL_TASKS))
def test_drill_shares_a_construct_and_is_never_the_task_itself(task_id):
    meta = ALL_TASKS[task_id]
    drill = pick_drill(meta)
    if not drill:
        return
    assert drill != (meta.get("description") or "").strip()
    source = next(
        t for t in ALL_TASKS.values() if (t.get("description") or "").strip() == drill
    )
    assert set(source["required_constructs"]) & set(meta["required_constructs"])


def test_names_are_substituted_into_every_slot():
    script = build_group_script(NAMES, ALL_TASKS["phone_plans_linear_01"])
    text = " ".join(s["text"] for s in script)
    for name in NAMES:
        assert name in text
    assert "{" not in text


def test_fewer_than_three_students_still_produces_a_valid_script():
    script = build_group_script(["Riley"], ALL_TASKS["frac_compare_01"])
    text = " ".join(s["text"] for s in script)
    assert "{" not in text
    assert "Riley" in text


def test_missing_metadata_degrades_to_a_drill_free_script():
    script = build_group_script(NAMES, {})
    assert all(s["label"] != "math_eval" for s in script)
    assert len(script) >= len(NEUTRAL_SCAFFOLDS)
