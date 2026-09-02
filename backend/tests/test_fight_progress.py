"""Fight session phase progression, listen restraint, and critic helpers."""

from app.fight_progress import (
    apply_listen_restraint,
    breakthrough_memory_block,
    claim_lock_block,
    detect_crossover_in_text,
    detect_crossover_intuition,
    detect_peer_critique_turn,
    detect_table_critique_peer,
    detect_teacher_fee_press,
    on_student_reply,
    on_teacher_message,
    parse_listen_directive,
    phase_after_breakthroughs,
    suppress_fight_misconception,
)
from app.group_sessions import GroupSession, StudentSlot
from app.models import Student
from app.response_refinement import (
    check_adult_register,
    check_claim_consistency,
    check_semantic_repetition,
)


def _fight_group() -> GroupSession:
    ids = ["maya", "jordan"]
    slots = {
        pid: StudentSlot(
            pid,
            Student(
                student_id=pid.capitalize(),
                Openness="High",
                Conscientiousness="High" if pid == "maya" else "Low",
                Extraversion="Low" if pid == "maya" else "High",
                Agreeableness="High" if pid == "maya" else "Low",
                Neuroticism="High" if pid == "maya" else "Low",
            ),
            implied_plan="B" if pid == "maya" else "A",
            misc_pinned=True,
        )
        for pid in ids
    }
    return GroupSession(
        session_id="fight-progress",
        task_text="Phone plans",
        task_metadata={"task_id": "phone_plans_linear_01"},
        students=slots,
        profile_ids=ids,
        display_names={"maya": "Maya", "jordan": "Jordan"},
    )


def test_parse_listen_directive_holds_back_non_focus():
    g = _fight_group()
    restraint, focus = parse_listen_directive(
        "Jordan, let Maya speak.", g.profile_ids, g.display_names
    )
    assert focus == "maya"
    assert restraint.get("jordan", 0) >= 2


def test_apply_listen_restraint_drops_jordan():
    g = _fight_group()
    g.listen_restraint = {"jordan": 2}
    g.focus_speaker = "maya"
    out = apply_listen_restraint(g, ["jordan", "maya"], "What do you think?")
    assert out == ["maya"]


def test_crossover_advances_phase_and_memory():
    g = _fight_group()
    on_student_reply(
        g,
        "jordan",
        "Okay so 20 + 0.10x = 0.30x, divide and x = 100 texts.",
        teacher_msg="Correct!",
    )
    assert g.students["jordan"].solved_crossover is True
    assert g.students["jordan"].stated_equations is True
    assert g.fight_phase in ("nuanced", "resolved")
    mem = breakthrough_memory_block(g, "jordan")
    assert "SESSION MEMORY" in mem
    assert "100" in mem or "crossover" in mem.lower()


def test_nuanced_claim_lock_mentions_crossover():
    block = claim_lock_block(phase="nuanced", plan="A", profile_id="jordan")
    assert "100" in block
    assert "NUANCED" in block


def test_suppress_misconception_after_crossover():
    g = _fight_group()
    g.fight_phase = "nuanced"
    assert suppress_fight_misconception(g, g.students["jordan"]) is True


def test_nuanced_claim_critic_allows_range_not_always():
    ok = check_claim_consistency(
        "Yeah under 100 Plan B is cheaper but over 100 Plan A wins.",
        "A",
        fight_phase="nuanced",
    )
    assert ok is None
    fail = check_claim_consistency(
        "Plan A is always better because 0.10 is less.",
        "A",
        fight_phase="nuanced",
    )
    assert fail is not None
    assert "rate_always_regression" in fail["issues"]


def test_semantic_repetition_critic():
    prior = [
        "I made a table up to 50 texts and Plan B is cheaper at 50.",
        "I made a table up to 50 texts and Plan B is cheaper at 50.",
    ]
    fail = check_semantic_repetition(
        "I made a table up to 50 texts and Plan B is cheaper at 50.",
        prior,
    )
    assert fail is not None
    assert "semantic_repetition" in fail["issues"]


def test_tutor_register_peer_explanation():
    fail = check_adult_register("You divide both sides by 0.20 to get x.")
    assert fail is not None
    assert "tutor_tone" in fail["issues"]


def test_on_teacher_message_sets_restraint():
    g = _fight_group()
    on_teacher_message(g, "Jordan, let Maya speak.")
    assert g.listen_restraint.get("jordan", 0) >= 2
    assert g.focus_speaker == "maya"


def test_detect_crossover_in_text():
    assert detect_crossover_in_text("So x = 100 texts where they match.")
    assert not detect_crossover_in_text("Plan B at 50 is cheaper.")


def test_concessive_jordan_fee_ack_not_claim_flip():
    ok = check_claim_consistency(
        "Yeah 0.1 is smaller, but Plan A still has the better rate so I think Plan A.",
        "A",
        fight_phase="fight",
    )
    assert ok is None


def test_plan_switch_still_fails_claim_flip():
    fail = check_claim_consistency(
        "You're right — Plan B is cheaper in my table so Plan B is better.",
        "A",
        fight_phase="fight",
    )
    assert fail is not None
    assert "claim_flip" in fail["issues"]


def test_equations_alone_advance_to_nuanced():
    g = _fight_group()
    on_student_reply(
        g,
        "jordan",
        "Plan A is 20 plus 0.1x and Plan B is 0.3x.",
    )
    assert g.students["jordan"].stated_equations is True
    assert g.fight_phase == "nuanced"


def test_teacher_crossover_cue_unlocks_group():
    g = _fight_group()
    on_teacher_message(g, "Can you set them equal and find the crossover?")
    assert g.crossover_unlocked is True


def test_fee_acknowledgment_sets_slot_flag():
    g = _fight_group()
    on_student_reply(
        g,
        "maya",
        "I think the $20 fee makes Plan A start higher at first.",
    )
    assert g.students["maya"].acknowledged_fee is True


def test_crossover_intuition_without_algebra():
    assert detect_crossover_intuition(
        "I think Plan A eventually catches up after enough texts."
    )
    assert not detect_crossover_intuition("So x = 100 texts where they match.")


def test_table_critique_peer_detector():
    assert detect_table_critique_peer(
        "Jordan, your rate-only thing ignores the $20 starting fee on my table."
    )


def test_teacher_fee_press_detector():
    assert detect_teacher_fee_press("okay but why do you think 20 won't matter")


def test_detect_peer_critique_turn_maya_on_jordan():
    transcript = [
        {
            "speaker_type": "student",
            "speaker_id": "jordan",
            "content": "Plan A is better because 0.10 is cheaper per text.",
        }
    ]
    role = detect_peer_critique_turn(
        "Maya find what's wrong in Jordan's claim",
        "maya",
        "phone_plans_linear_01",
        transcript,
    )
    assert role == "peer_critique"


def test_resolved_requires_crossover_unlocked():
    assert phase_after_breakthroughs(
        crossover=True,
        equations=True,
        teacher_confirmed=True,
        crossover_unlocked=False,
    ) == "nuanced"
    assert phase_after_breakthroughs(
        crossover=True,
        equations=True,
        teacher_confirmed=True,
        crossover_unlocked=True,
    ) == "resolved"


def test_jordan_fee_press_claim_lock_slice():
    block = claim_lock_block(
        phase="fight",
        plan="A",
        profile_id="jordan",
        acknowledged_fee=True,
        fee_press_count=2,
    )
    assert "$20" in block
    assert "x=100" in block.lower() or "solve" in block.lower()


def test_crossover_intuition_sets_flag():
    g = _fight_group()
    on_student_reply(
        g,
        "jordan",
        "Yeah but after enough texts Plan A eventually catches up.",
    )
    assert g.students["jordan"].crossover_intuition is True


def test_detect_peer_critique_help_table_after_fee():
    transcript = [
        {
            "speaker_type": "student",
            "speaker_id": "jordan",
            "content": "Yeah the $20 is there but 0.10 is still smaller per text.",
        }
    ]
    role = detect_peer_critique_turn(
        "maya help Jordan understand why heis wrong using the table",
        "maya",
        "phone_plans_linear_01",
        transcript,
    )
    assert role == "peer_critique"


def test_phase_nuanced_on_fee_uptake_when_unlocked():
    assert phase_after_breakthroughs(
        crossover=False,
        equations=False,
        teacher_confirmed=False,
        crossover_unlocked=True,
        fee_uptake=True,
    ) == "nuanced"


def test_soften_jordan_fee_press_mode():
    g = _fight_group()
    slot = g.students["jordan"]
    slot.behavior_profile["behavior_mode"] = "WRONG"
    slot.fee_press_count = 2
    from app.fight_progress import soften_jordan_fee_press_mode

    soften_jordan_fee_press_mode(g, slot)
    assert slot.behavior_profile["behavior_mode"] == "PARTIAL_ATTEMPT_THEN_STUCK"
