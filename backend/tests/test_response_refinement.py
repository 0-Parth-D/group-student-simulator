"""Tests for LLM-critique response refinement (mocked critic / student LLM)."""

from unittest.mock import patch

from app.models import Student
from app.response_refinement import (
    GenerationResult,
    build_expected_pack,
    build_llm_refinement_message,
    check_adult_register,
    check_discourse_progress,
    check_jordan_passive,
    check_crossover_gate,
    check_maya_overcomplete,
    critique_student_reply,
    generate_with_refinement,
    ocean_filler_guidance,
    rewrite_kid_voice,
)


def _maya() -> Student:
    return Student(
        student_id="Maya",
        Openness="Low",
        Conscientiousness="High",
        Extraversion="Low",
        Agreeableness="High",
        Neuroticism="High",
        construct_mastery={"linear_relationship": 2, "unit_rate": 2},
    )


def _jordan() -> Student:
    return Student(
        student_id="Jordan",
        Openness="Medium",
        Conscientiousness="High",
        Extraversion="High",
        Agreeableness="Low",
        Neuroticism="Low",
        construct_mastery={"linear_relationship": 3, "unit_rate": 3},
    )


def _profile(**overrides):
    base = {
        "behavior_mode": "PARTIAL_ATTEMPT_THEN_STUCK",
        "likely_correctness": "partially correct",
        "student_stack_level": 2,
        "target_stack_level": 3,
        "primary_construct": "linear_relationship",
        "help_seek_style": "anxious_helpless",
        "stall_active": False,
    }
    base.update(overrides)
    return base


def test_build_expected_pack_includes_personality_and_lp():
    pack = build_expected_pack(
        _profile(),
        student=_maya(),
        misconception={
            "id": "pr_compare_one_x_only",
            "prompt_cue": "You pick a few numbers and decide from those.",
        },
        turn_mode="math_scaffold",
    )
    assert "Low Extraversion" in pack["personality"] or "Extraversion=Low" in pack["personality"]
    assert "Maya" in pack["personality"]
    assert "Filler guidance" in pack["personality"]
    assert pack["learning"]
    assert "few numbers" in pack["misconception"] or "Standing" in pack["misconception"] or pack["misconception"]


def test_check_adult_register_catches_tutor_tone():
    fail = check_adult_register(
        "Plan A scales more slowly; therefore for any large volume Plan A is superior.",
        student=_maya(),
    )
    assert fail is not None
    assert fail["ok"] is False
    assert "tutor_tone" in fail["issues"] or "too_polished" in fail["issues"]


def test_check_adult_register_allows_hedged_kid_voice():
    ok = check_adult_register(
        "Wait… on my table at 50, Plan B was less though?",
        student=_maya(),
    )
    assert ok is None


def test_ocean_filler_guidance_differs_by_persona():
    maya = ocean_filler_guidance(_maya())
    jordan = ocean_filler_guidance(_jordan())
    assert "um" in maya.lower() or "kinda" in maya.lower()
    assert "okay so" in jordan.lower() or "wait" in jordan.lower()


def test_critique_ok_skips_content_rewrite_but_runs_voice_pass():
    draft = "Um… at 50 Plan B was cheaper on my table?"
    voiced = "Wait, at 50 Plan B was cheaper on my table?"
    critic_json = (
        '{"ok": true, "claim": {"pass": true, "detail": "ok"}, '
        '"personality": {"pass": true, "detail": "brief"}, '
        '"learning": {"pass": true, "detail": "ok"}, '
        '"misconception": {"pass": true, "detail": "ok"}, '
        '"issues": [], "brief": ""}'
    )
    with (
        patch("app.response_refinement.complete_chat", return_value=draft) as gen,
        patch("app.response_refinement.complete", return_value=critic_json),
        patch("app.response_refinement.rewrite_kid_voice", return_value=voiced) as voice,
    ):
        result = generate_with_refinement(
            "sys",
            [{"role": "user", "content": "hi"}],
            _profile(),
            student=_maya(),
            max_revisions=2,
            turn_mode="math_scaffold",
        )
        out, revisions = result
    assert isinstance(result, GenerationResult)
    assert out == voiced
    assert revisions == 0
    assert result.draft == draft
    assert result.final_reply == voiced
    assert result.critic is not None
    assert result.critic.get("ok") is True
    assert "pack" not in result.critic
    assert gen.call_count == 1
    assert voice.call_count == 1


def test_critique_fail_triggers_rewrite_then_voice():
    draft = (
        "Plan A scales up more slowly than Plan B because the per-text coefficient "
        "is lower, therefore for any large volume Plan A is obviously superior."
    )
    refined = "Wait… on my table at 50, Plan B was less though?"
    voiced = "Wait, on my table at 50 Plan B was less though?"
    critic_json = (
        '{"ok": false, '
        '"claim": {"pass": true, "detail": "ok"}, '
        '"personality": {"pass": false, "detail": "too long and tutor-like for Low E"}, '
        '"learning": {"pass": true, "detail": "ok"}, '
        '"misconception": {"pass": true, "detail": "ok"}, '
        '"issues": ["personality_overtalk", "tutor_tone"], '
        '"brief": "One short hedged sentence; quiet Maya voice."}'
    )
    with (
        patch(
            "app.response_refinement.complete_chat",
            side_effect=[draft, refined],
        ) as gen,
        patch("app.response_refinement.complete", return_value=critic_json),
        patch("app.response_refinement.rewrite_kid_voice", return_value=voiced) as voice,
    ):
        result = generate_with_refinement(
            "sys",
            [{"role": "user", "content": "explain"}],
            _profile(),
            student=_maya(),
            misconception={"prompt_cue": "table without crossover"},
            max_revisions=1,
            turn_mode="math_scaffold",
        )
        out, revisions = result
    assert out == voiced
    assert revisions == 1
    assert result.draft == draft
    assert result.final_reply == voiced
    assert result.critic is not None
    assert "tutor_tone" in result.critic.get("issues", [])
    assert "pack" not in result.critic
    assert gen.call_count == 2
    assert voice.call_count == 1
    refine_call = gen.call_args_list[1]
    hist = refine_call.args[2]
    assert any(
        "PERSONALITY TARGET" in (m.get("content") or "")
        for m in hist
        if m.get("role") == "user"
    )


def test_deterministic_adult_register_short_circuits_critic_llm():
    draft = "Therefore we can see that Plan A is essentially better."
    with patch("app.response_refinement.complete") as critic_llm:
        c = critique_student_reply(
            draft,
            _profile(),
            student=_maya(),
        )
    assert c["ok"] is False
    assert "tutor_tone" in c["issues"] or "too_polished" in c["issues"]
    critic_llm.assert_not_called()


def test_critique_unavailable_accepts_draft_then_voice():
    draft = "Some reply"
    voiced = "Um, some reply?"
    with (
        patch("app.response_refinement.complete_chat", return_value=draft),
        patch(
            "app.response_refinement.complete",
            side_effect=RuntimeError("model missing"),
        ),
        patch("app.response_refinement.rewrite_kid_voice", return_value=voiced),
    ):
        out, revisions = generate_with_refinement(
            "sys",
            [{"role": "user", "content": "x"}],
            _profile(),
            student=_maya(),
            max_revisions=2,
        )
    assert out == voiced
    assert revisions == 0


def test_refine_off_skips_voice_pass():
    draft = "ok"
    with (
        patch("app.response_refinement.complete_chat", return_value=draft),
        patch("app.response_refinement.REFINE_MODE", "off"),
        patch("app.response_refinement.rewrite_kid_voice") as voice,
    ):
        result = generate_with_refinement(
            "sys",
            [{"role": "user", "content": "x"}],
            _profile(),
            student=_maya(),
            max_revisions=2,
        )
    assert result.final_reply == draft
    assert result.draft == draft
    assert result.revisions == 0
    assert result.critic is None
    voice.assert_not_called()


def test_build_llm_refinement_message_includes_targets():
    critique = {
        "brief": "Too polished",
        "issues": ["tutor_tone"],
        "pack": {
            "personality": "Low E brief",
            "learning": "Stay partial",
            "misconception": "Show fee-blind slip",
        },
    }
    msg = build_llm_refinement_message(critique)
    assert "Too polished" in msg
    assert "Low E brief" in msg
    assert "Stay partial" in msg
    assert "fee-blind" in msg
    assert "fillers" in msg.lower()


def test_critique_student_reply_parses_json():
    raw = (
        '{"ok": false, "claim": {"pass": true, "detail": "ok"}, '
        '"personality": {"pass": false, "detail": "eager"}, '
        '"learning": {"pass": true, "detail": "x"}, '
        '"misconception": {"pass": true, "detail": "y"}, '
        '"issues": ["too_eager"], "brief": "Be quieter"}'
    )
    # Kid-voice draft so deterministic register gate does not short-circuit.
    with patch("app.response_refinement.complete", return_value=raw):
        c = critique_student_reply(
            "Wait I think Plan A wins? Easy.",
            _profile(),
            student=_maya(),
        )
    assert c["ok"] is False
    assert c["personality"]["pass"] is False
    assert "too_eager" in c["issues"]


def test_rewrite_kid_voice_returns_model_output():
    with patch(
        "app.response_refinement.complete",
        return_value='Wait, I think Plan B at 50?',
    ):
        out = rewrite_kid_voice(
            "Plan B is cheaper at fifty texts.",
            student=_maya(),
            turn_mode="math_scaffold",
        )
    assert "Wait" in out


def test_social_pack_drops_claim_and_misc_demand():
    pack = build_expected_pack(
        _profile(),
        student=_maya(),
        misconception={"prompt_cue": "fee-blind Plan A"},
        turn_mode="social",
    )
    assert "No misconception demand" in pack["misconception"]
    assert "No math" in pack["learning"] or "no Plan" in pack["learning"]


def test_social_critique_ignores_claim_and_misc_failures():
    raw = (
        '{"ok": false, "claim": {"pass": false, "detail": "no plan"}, '
        '"personality": {"pass": true, "detail": "ok"}, '
        '"learning": {"pass": true, "detail": "ok"}, '
        '"misconception": {"pass": false, "detail": "missing"}, '
        '"issues": ["claim_missing", "misconception_missing"], '
        '"brief": "State Plan B"}'
    )
    with patch("app.response_refinement.complete", return_value=raw):
        c = critique_student_reply(
            "I'm good, thanks!",
            {**_profile(), "implied_plan": "B"},
            student=_maya(),
            turn_mode="social",
            implied_plan="B",
        )
    assert c["ok"] is True
    assert c["claim"]["pass"] is True
    assert c["misconception"]["pass"] is True
    assert "claim_missing" not in c["issues"]


def test_social_refinement_message_forbids_plan_claims():
    msg = build_llm_refinement_message(
        {
            "brief": "Too formal",
            "issues": ["tutor_tone"],
            "pack": {
                "personality": "brief",
                "learning": "Social/off-topic turn: no math, no Plan A/B claim",
                "misconception": "No misconception demand on social/off-topic turns.",
            },
        }
    )
    assert "Do NOT introduce Plan" in msg
    assert "Keep the same core math claim" not in msg


def test_jordan_passive_blocked_in_wrong_mode():
    fail = check_jordan_passive(
        "I'm not sure what I'm supposed to do with that.",
        _jordan(),
        "WRONG",
        profile_id="jordan",
    )
    assert fail is not None
    assert "too_passive_jordan" in fail["issues"]


def test_jordan_passive_allowed_outside_wrong_mode():
    ok = check_jordan_passive(
        "I'm not sure what I'm supposed to do.",
        _jordan(),
        "CONFUSED_HELPSEEKING",
        profile_id="jordan",
    )
    assert ok is None


def test_crossover_blocked_before_teacher_unlock():
    fail = check_crossover_gate(
        "So 20 + 0.1x = 0.3x and x = 100 texts.",
        crossover_unlocked=False,
    )
    assert fail is not None
    assert "lp_overcomplete" in fail["issues"]


def test_crossover_allowed_after_teacher_unlock():
    ok = check_crossover_gate(
        "So 20 + 0.1x = 0.3x and x = 100 texts.",
        crossover_unlocked=True,
    )
    assert ok is None


def test_crossover_critique_integration_blocked():
    c = critique_student_reply(
        "So 20 + 0.1x = 0.3x and x = 100 texts.",
        _profile(behavior_mode="WRONG", crossover_unlocked=False),
        student=_jordan(),
        implied_plan="A",
        fight_phase="fight",
    )
    assert c["ok"] is False
    assert "lp_overcomplete" in c["issues"]


def test_discourse_progress_fails_repeat_without_peer_reference():
    prior = [
        "I made a table up to 50 texts and Plan B is cheaper at 50.",
        "I made a table up to 50 texts and Plan B is cheaper at 50.",
    ]
    fail = check_discourse_progress(
        "I made a table up to 50 texts and Plan B is cheaper at 50.",
        prior_peer_name="Jordan",
        prior_peer_text="Plan A is better because 0.10 is cheaper per text.",
        is_peer_turn=True,
        prior_replies=prior,
    )
    assert fail is not None
    assert "no_discourse_progress" in fail["issues"]


def test_discourse_progress_passes_with_peer_reference():
    ok = check_discourse_progress(
        "Jordan, but on my table at 50 Plan B was cheaper.",
        prior_peer_name="Jordan",
        prior_peer_text="Plan A is better because 0.10 is cheaper per text.",
        is_peer_turn=True,
        prior_replies=["I made a table up to 50 texts and Plan B is cheaper at 50."],
    )
    assert ok is None


def test_maya_overcomplete_blocks_x100_before_resolved():
    fail = check_maya_overcomplete(
        "Set them equal and x = 100 is the crossover point.",
        profile_id="maya",
        fight_phase="fight",
    )
    assert fail is not None
    assert "lp_overcomplete" in fail["issues"]


def test_maya_overcomplete_allows_table_numbers():
    ok = check_maya_overcomplete(
        "On my table Plan B is cheaper at 50 texts.",
        profile_id="maya",
        fight_phase="fight",
    )
    assert ok is None


def test_refinement_message_includes_move_hint_for_discourse():
    msg = build_llm_refinement_message(
        {
            "brief": "Repeat",
            "issues": ["no_discourse_progress"],
            "pack": {"personality": "x", "learning": "y", "misconception": "z"},
        }
    )
    assert "MOVE HINT" in msg


def test_discourse_fires_before_claim_on_peer_turn():
    c = critique_student_reply(
        "Plan B is always better in my table for sure at fifty.",
        {**_profile(), "profile_id": "maya", "implied_plan": "B"},
        student=_maya(),
        implied_plan="B",
        is_peer_turn=True,
        prior_peer_name="Jordan",
        prior_peer_text="Plan A wins because 0.10 is less per text.",
        prior_replies=[
            "Plan B is always better in my table for sure at fifty.",
        ],
        fight_phase="fight",
    )
    assert c["ok"] is False
    assert "no_discourse_progress" in c["issues"]
    assert "claim_flip" not in c["issues"]


def test_maya_critique_thin_blocks_table_repeat():
    from app.response_refinement import check_maya_critique_rich

    prior = ["On my table Plan B is cheaper at 50 texts for sure."]
    fail = check_maya_critique_rich(
        "On my table Plan B is cheaper at 50 texts for sure.",
        profile_id="maya",
        turn_role="peer_critique",
        prior_replies=prior,
    )
    assert fail is not None
    assert "maya_critique_thin" in fail["issues"]


def test_maya_critique_rich_allows_fee_contrast():
    from app.response_refinement import check_maya_critique_rich

    ok = check_maya_critique_rich(
        "Jordan, you ignored the $20 starting fee — Plan A costs more at first.",
        profile_id="maya",
        turn_role="peer_critique",
        prior_replies=["Plan B is cheaper at 50 on my table."],
    )
    assert ok is None


def test_voice_guard_keeps_peer_reference():
    draft = "Jordan, but the $20 fee matters on my table."
    voiced = "Wait, Plan B is cheaper at 50."
    critic_json = (
        '{"ok": true, "claim": {"pass": true, "detail": "ok"}, '
        '"personality": {"pass": true, "detail": "brief"}, '
        '"learning": {"pass": true, "detail": "ok"}, '
        '"misconception": {"pass": true, "detail": "ok"}, '
        '"issues": [], "brief": ""}'
    )
    with (
        patch("app.response_refinement.complete_chat", return_value=draft),
        patch("app.response_refinement.complete", return_value=critic_json),
        patch(
            "app.response_refinement.rewrite_kid_voice",
            return_value=voiced,
        ),
    ):
        result = generate_with_refinement(
            "sys",
            [{"role": "user", "content": "peer"}],
            {**_profile(), "profile_id": "maya"},
            student=_maya(),
            max_revisions=1,
            is_peer_turn=True,
            prior_peer_name="Jordan",
            prior_peer_text="Plan A is better because 0.10 is cheaper.",
            prior_replies=["Plan B is cheaper at 50 on my table."],
        )
    assert "Jordan" in result.final_reply or "$20" in result.final_reply


def test_check_reasoning_warrant_imported_in_pipeline():
    from app.response_refinement import check_reasoning_warrant

    assert check_reasoning_warrant("hello", turn_role="opening_attempt") is None


def test_critique_reasoning_press_skips_discourse_gate():
    from app.reasoning_warrant import warrant_expectation

    exp = warrant_expectation(
        profile_id="jordan",
        behavior_profile={
            "behavior_mode": "WRONG",
            "student_stack_level": 2,
            "primary_construct": "rate_comparison",
            "profile_id": "jordan",
        },
        student=_jordan(),
    )
    result = critique_student_reply(
        "Because 0.10 per text is less than 0.30.",
        {
            "behavior_mode": "WRONG",
            "student_stack_level": 2,
            "primary_construct": "rate_comparison",
            "profile_id": "jordan",
            "implied_plan": "A",
            "_warrant_expectation": exp,
        },
        student=_jordan(),
        turn_role="reasoning_press",
        implied_plan="A",
    )
    assert result.get("issues") != ["bare_claim"]
