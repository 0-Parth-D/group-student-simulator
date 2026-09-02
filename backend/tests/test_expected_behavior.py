"""Expected behavior must exist for every construct and name no scenario."""

from __future__ import annotations

import pytest

from app.construct_text import construct_vocabulary, mastery_descriptor
from app.expected_behavior import (
    ROLE_CLAUSES,
    expected_behavior,
    expected_behavior_for_record,
    role_for_turn,
)
from app.kg_config import get_construct_def, list_construct_ids

ALL_CONSTRUCTS = list_construct_ids()


def test_knowledge_graph_is_not_empty():
    assert ALL_CONSTRUCTS


@pytest.mark.parametrize("construct_id", ALL_CONSTRUCTS)
def test_every_construct_supplies_mastery_wording(construct_id):
    """Each construct needs lpf_stack or dlm_linkage_levels, or descriptors go generic."""
    cdef = get_construct_def(construct_id) or {}
    assert cdef.get("lpf_stack") or cdef.get("dlm_linkage_levels"), (
        f"{construct_id} has neither lpf_stack nor dlm_linkage_levels"
    )


@pytest.mark.parametrize("construct_id", ALL_CONSTRUCTS)
def test_descriptors_are_present_and_distinct_across_levels(construct_id):
    descriptors = [mastery_descriptor(construct_id, lvl) for lvl in range(4)]
    assert all(d.strip() for d in descriptors)
    # Levels 1-3 are the ones that drive routing; they must not collapse together.
    assert len(set(descriptors[1:])) == 3


@pytest.mark.parametrize("construct_id", ALL_CONSTRUCTS)
def test_vocabulary_is_substantial_enough_to_embed(construct_id):
    assert len(construct_vocabulary(construct_id).split()) >= 8


def test_out_of_range_levels_are_clamped():
    assert mastery_descriptor("part_whole", -5) == mastery_descriptor("part_whole", 0)
    assert mastery_descriptor("part_whole", 99) == mastery_descriptor("part_whole", 3)


def test_unknown_construct_degrades_without_raising():
    text = mastery_descriptor("not_a_real_construct", 2)
    assert text.strip()
    assert expected_behavior("not_a_real_construct", 2).strip()


def test_expected_behavior_carries_state_error_and_role():
    text = expected_behavior(
        "rate_comparison",
        1,
        misconception="pr_lower_rate_always_wins",
        turn_role="opening_attempt",
        behavior_mode="WRONG",
    )
    assert "level 1/3" in text
    assert mastery_descriptor("rate_comparison", 1) in text
    assert "cheaper rate always wins" in text
    assert ROLE_CLAUSES["opening_attempt"] in text


def test_expected_behavior_names_no_task():
    """The whole point: expectations must generalize to unseen scenarios."""
    banned = ("pizza", "plan a", "plan b", "maria", "phone", "0.1x", "4/5")
    for construct_id in ALL_CONSTRUCTS:
        for role in ROLE_CLAUSES:
            text = expected_behavior(construct_id, 2, turn_role=role).lower()
            assert not any(word in text for word in banned), (
                f"{construct_id}/{role} leaked a scenario detail"
            )


def test_unknown_role_falls_back_rather_than_raising():
    assert expected_behavior("unit_rate", 2, turn_role="nonsense").strip()


def test_record_wrapper_reads_the_eval_log_shape():
    record = {
        "construct_id": "linear_relationship",
        "mastery": 2,
        "misconception_id": "lr_line_starts_at_origin",
        "behavior_mode": "PARTIAL_ATTEMPT_THEN_STUCK",
    }
    text = expected_behavior_for_record(record, "post_scaffold")
    assert mastery_descriptor("linear_relationship", 2) in text
    assert ROLE_CLAUSES["post_scaffold"] in text


def test_record_wrapper_falls_back_to_stack_level():
    record = {"primary_construct": "unit_rate", "student_stack_level": 3}
    assert mastery_descriptor("unit_rate", 3) in expected_behavior_for_record(record)


def test_role_derivation_from_dialogue_context():
    assert role_for_turn("math_scaffold") == "post_scaffold"
    assert role_for_turn("vague_acknowledgment") == "vague_ack"
    assert role_for_turn("vague_directive") == "vague_ack"
    assert role_for_turn("math_eval") == "drill"
    assert role_for_turn("math_scaffold", is_peer_turn=True) == "peer_response"
    assert role_for_turn("", turn_role="peer_critique") == "peer_critique"
    assert role_for_turn("math_eval", is_first_turn=True) == "opening_attempt"
    assert role_for_turn("social") in ROLE_CLAUSES


def test_scripted_roles_all_resolve():
    """Every role named in the teacher script must have a clause."""
    from eval.fixtures.battery_fixtures import get_turn_roles

    for turn, role in get_turn_roles().items():
        assert role in ROLE_CLAUSES, f"turn {turn} uses unknown role {role}"


# --- Gating: the yardstick must match the instructions the student actually got ---

ALL_MODES = [
    "WRONG",
    "CONFUSED_HELPSEEKING",
    "PARTIAL_ATTEMPT_THEN_STUCK",
    "NORMAL_ERROR_PROFILE",
]


def _demands_the_error(text: str) -> bool:
    return "enact this specific error" in text.lower()


@pytest.mark.parametrize("mode", ALL_MODES)
@pytest.mark.parametrize("role", sorted(ROLE_CLAUSES))
def test_stalled_turns_never_demand_a_misconception(mode, role):
    """`app.prompts` suppresses the playbook while stalling, so demanding it is unfair."""
    text = expected_behavior(
        "unit_rate",
        2,
        misconception="ur_inverts_ratio",
        turn_role=role,
        behavior_mode=mode,
        stall_active=True,
    )
    assert not _demands_the_error(text)


@pytest.mark.parametrize("mode", ALL_MODES)
def test_vague_roles_never_demand_a_misconception(mode):
    """A bare 'okay' turn is scored the same whether or not the stall flag was set."""
    for role in ("vague_ack", "closing_ack"):
        text = expected_behavior(
            "unit_rate", 2, misconception="ur_inverts_ratio",
            turn_role=role, behavior_mode=mode,
        )
        assert not _demands_the_error(text)


def test_normal_error_profile_is_never_asked_to_enact_a_misconception():
    """The student prompt tells this mode not to copy the major error patterns."""
    for role in ROLE_CLAUSES:
        text = expected_behavior(
            "unit_rate", 3, misconception="ur_inverts_ratio",
            turn_role=role, behavior_mode="NORMAL_ERROR_PROFILE",
        )
        assert not _demands_the_error(text)


@pytest.mark.parametrize("mode", ["WRONG", "PARTIAL_ATTEMPT_THEN_STUCK"])
def test_productive_turns_still_demand_the_misconception(mode):
    """Gating must not silently disarm the yardstick on the turns that matter."""
    text = expected_behavior(
        "unit_rate", 1, misconception="ur_inverts_ratio",
        turn_role="opening_attempt", behavior_mode=mode,
    )
    assert _demands_the_error(text)


def test_post_scaffold_states_the_error_without_mandating_it():
    """A hint that corrects the error must not make correct uptake a failure."""
    text = expected_behavior(
        "unit_rate", 2, misconception="ur_inverts_ratio",
        turn_role="post_scaffold", behavior_mode="PARTIAL_ATTEMPT_THEN_STUCK",
    )
    assert not _demands_the_error(text)
    # The error is still named, so the judge can recognize it when it does appear.
    assert "standing error on this construct" in text.lower()


def test_misconception_is_never_both_mandated_and_optional():
    for role in ROLE_CLAUSES:
        text = expected_behavior(
            "unit_rate", 2, misconception="ur_inverts_ratio",
            turn_role=role, behavior_mode="WRONG",
        ).lower()
        assert not (
            _demands_the_error(text) and "standing error on this construct" in text
        )


@pytest.mark.parametrize("mode", ALL_MODES)
@pytest.mark.parametrize("role", sorted(ROLE_CLAUSES))
def test_expectation_never_both_requires_and_forbids_a_step(mode, role):
    """The regression that tanked cognitive scores: contradictory instructions."""
    for stall in (True, False):
        text = expected_behavior(
            "unit_rate", 2, misconception="ur_inverts_ratio",
            turn_role=role, behavior_mode=mode, stall_active=stall,
        ).lower()
        forbids = "do not compute another step" in text
        requires = "move one step" in text or "take one small step" in text
        assert not (forbids and requires), f"{mode}/{role}/stall={stall} is self-contradictory"


def test_off_task_turns_carry_no_math_expectation():
    text = expected_behavior(
        "unit_rate", 2, misconception="ur_inverts_ratio",
        turn_role="off_task", behavior_mode="WRONG",
    )
    assert not _demands_the_error(text)
    assert "knowledge state" not in text.lower()
    assert ROLE_CLAUSES["off_task"] in text


def test_social_labels_route_to_the_off_task_role():
    assert role_for_turn("social") == "off_task"
    assert role_for_turn("off_topic") == "off_task"


def test_record_wrapper_honours_the_stall_flag():
    record = {
        "construct_id": "unit_rate",
        "mastery": 2,
        "misconception_id": "ur_inverts_ratio",
        "behavior_mode": "PARTIAL_ATTEMPT_THEN_STUCK",
        "stall_active": True,
    }
    stalled = expected_behavior_for_record(record, "opening_attempt")
    assert not _demands_the_error(stalled)
    assert "standing error on this construct" not in stalled.lower()

    record["stall_active"] = False
    assert _demands_the_error(expected_behavior_for_record(record, "opening_attempt"))
