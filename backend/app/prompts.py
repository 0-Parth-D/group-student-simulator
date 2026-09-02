from typing import Dict, List, Optional



from app.behavior_router import error_applies_at_mastery
from app.kg_config import MASTERY_LABELS, get_construct_def

from app.models import Student

from app.personality import (
    BF_TC,
    unknown_behavior_rule,
)

from app.pisa import pisa_summary

from app.student_prompt_layers import build_turn_state_block, build_voice_card

from app.task_answer import is_part_whole_word_problem

from app.turn_classifier import TurnMode



SCAFFOLD_USE_THRESHOLD = 0.35





def _is_math_turn(turn_mode: TurnMode) -> bool:
    return turn_mode in (
        "math_scaffold",
        "math_eval",
        "mixed",
        "vague_acknowledgment",
        "vague_directive",
    )





def _format_playbook(
    error_playbook: Optional[dict],
    behavior_profile: Optional[dict] = None,
    student=None,
    stall_active: bool = False,
) -> List[str]:

    if not error_playbook or stall_active:
        return []

    mode = (behavior_profile or {}).get("behavior_mode", "")
    primary = (behavior_profile or {}).get("primary_construct", "")
    stack_level = (behavior_profile or {}).get("student_stack_level", 0)
    mastery = getattr(student, "construct_mastery", None) or {}

    lines = ["[Realistic Mistake Patterns for this problem]"]

    shown = 0
    for p in error_playbook.get("patterns", [])[:1]:
        construct = p.get("concept", primary)
        sl = mastery.get(construct, stack_level)
        err_level = int(p.get("stack_level", 1))
        if mode and not error_applies_at_mastery(mode, sl, err_level):
            continue

        concept = p.get("concept", "?")

        mtype = p.get("type", "procedural_confusion")

        example = p.get("example", "")

        cue = p.get("prompt_cue", "")

        line = f"  - {concept} ({mtype}): {example}"

        if cue:

            line += f" [Cue: {cue}]"

        lines.append(line)
        shown += 1

    if shown == 0:
        return []

    stay = error_playbook.get("stay_in_problem")

    if stay:

        lines.append(f"  Stay in problem: {stay}")

    lines.append("")

    return lines





def _format_word_problem_guardrails(
    task_text: str,
    task_meta: Optional[dict],
    behavior_profile: Optional[dict],
) -> List[str]:
    if not is_part_whole_word_problem(task_text, task_meta):
        return []

    mode = (behavior_profile or {}).get("behavior_mode", "")
    level = (behavior_profile or {}).get("student_stack_level", 0)

    lines = [
        "[Part-Whole Word Problem — read the story first]",
        "The fractions describe parts of the SAME whole (one pizza, one object, etc.).",
        "Do NOT treat the problem as subtracting one fraction from another "
        "(e.g. 3/8 minus 1/4) unless the story says one amount was taken FROM the other.",
        "For 'ate X then gave away Y, how much is left': combine what left the whole, "
        "then subtract from the whole (e.g. 8/8 or 1).",
    ]
    if mode == "NORMAL_ERROR_PROFILE" and level >= 3:
        lines.append(
            "You understand part-whole well: name the whole first, convert fractions "
            "to the same unit if needed, then find what remains."
        )
    else:
        lines.append(
            "A common trap is jumping straight to LCD subtraction — check whether "
            "both fractions refer to the same whole."
        )
    lines.append("")
    return lines





def _format_lp_mastery_block(behavior_profile: Optional[dict]) -> List[str]:

    if not behavior_profile or not behavior_profile.get("behavior_mode"):

        return []

    lines = ["[Learning Progression Mastery — this turn]"]

    accessible = behavior_profile.get("accessible_constructs") or []

    locked = behavior_profile.get("locked_constructs") or []

    if accessible:

        lines.append("  Accessible: " + ", ".join(accessible))

    if locked:

        lines.append("  Locked (do NOT use fluently): " + ", ".join(locked))

    gaps = behavior_profile.get("prerequisite_gaps") or []

    if gaps:

        lines.append("  Gaps:")

        for g in gaps[:3]:

            lines.append(f"    - {g.get('gap_description', g.get('construct', ''))}")

    lines.append("")

    return lines





def _format_construct_block(student: Student, behavior_profile: Optional[dict]) -> List[str]:
    if behavior_profile and behavior_profile.get("primary_construct"):
        primary = behavior_profile["primary_construct"]
        cdef = get_construct_def(primary)
        if not cdef:
            return []

        level = behavior_profile.get(
            "student_stack_level", student.construct_mastery.get(primary, 0)
        )
        lines = [
            "[Rational-Number Construct Profile — active for this task]",
            f"  Primary: {cdef.get('label', primary)} (mastery {level}, {MASTERY_LABELS.get(level, '?')})",
        ]
        for cid, lvl in behavior_profile.get("construct_levels", {}).items():
            if cid != primary:
                rel = get_construct_def(cid) or {}
                lines.append(
                    f"  Related: {rel.get('label', cid)} (mastery {lvl}, {MASTERY_LABELS.get(lvl, '?')})"
                )
        lines.append("")
        return lines

    if not student.construct_mastery:
        return []

    lines = ["[Rational-Number Constructs]"]
    for cid, level in student.construct_mastery.items():
        cdef = get_construct_def(cid) or {}
        lines.append(
            f"  - {cdef.get('label', cid)}: mastery {level} ({MASTERY_LABELS.get(level, '?')})"
        )
    lines.append("")
    return lines





def _should_include_pisa_block(
    student: Student,
    behavior_profile: Optional[dict],
    is_math: bool,
) -> bool:
    if not is_math or not student.pisa_profile:
        return False
    if not behavior_profile:
        return False
    return bool(behavior_profile.get("weak_attributes"))





def _format_pisa_block(student: Student, behavior_profile: Optional[dict]) -> List[str]:

    if not student.pisa_profile:

        return []

    required = behavior_profile.get("pisa_attributes", []) if behavior_profile else []

    lines = ["[PISA Mathematical Competencies]"]

    for item in pisa_summary(student.pisa_profile):

        marker = " *" if item["id"] in required else ""

        lines.append(f"  - {item['name']}: {item['level']}{marker}")

    if behavior_profile and behavior_profile.get("weak_attributes"):

        lines.append(

            "  Weak on this task: " + ", ".join(behavior_profile["weak_attributes"])

        )

    if behavior_profile and behavior_profile.get("attribute_errors"):

        lines.append(

            "  Attribute-driven slips: " + "; ".join(behavior_profile["attribute_errors"][:2])

        )

    lines.append("")

    return lines





def _format_behavior_profile_block(

    behavior_profile: Optional[dict],

    scaffold_boost: Optional[Dict[str, float]] = None,

) -> List[str]:

    if not behavior_profile:

        return []

    boost = scaffold_boost or {}

    primary = behavior_profile.get("primary_construct", "")

    recent_scaffold = boost.get(primary, 0.0) >= SCAFFOLD_USE_THRESHOLD



    if behavior_profile.get("behavior_mode"):

        lines = [

            "[LP Behavior Target — match this performance]",

            f"  Mode: {behavior_profile.get('behavior_mode')}",

            f"  Likely correctness: {behavior_profile.get('likely_correctness', 'partially correct')}",

        ]

        if behavior_profile.get("error_types"):

            lines.append(

                "  Show these errors: " + "; ".join(behavior_profile["error_types"][:3])

            )

        if behavior_profile.get("help_seek_style"):

            lines.append(f"  Help-seeking style: {behavior_profile['help_seek_style']}")

        if recent_scaffold:

            lines.append(

                "  The teacher just explained a step — you MAY use that step correctly "

                "on this reply, but do NOT give the full polished answer."

            )

        else:
            mastery_note = ""
            level = behavior_profile.get("student_stack_level")
            target = behavior_profile.get("target_stack_level")
            if (
                level is not None
                and target is not None
                and int(level) < int(target)
            ):
                mastery_note = (
                    f"  Mastery {level}/{target} — do NOT leap to a complete correct solution.\n"
                )
            lines.append(
                mastery_note
                + "  Do NOT self-correct into the teacher's full solution. Stay wrong or partial as predicted.\n"
                "  On active math turns, do NOT ask empty process questions "
                "('what do you want me to do?', 'which part?', 'in words or math?')."
            )

        lines.append("")

        return lines

    if not behavior_profile.get("primary_construct"):

        return []

    lines = [

        "[Wu Behavior Prediction — match this imperfect performance]",

        f"  Likely correctness: {behavior_profile.get('likely_correctness', 'partially correct')}",

    ]

    if behavior_profile.get("error_types"):

        lines.append("  Show these errors: " + "; ".join(behavior_profile["error_types"][:3]))

    lines.append(

        "  Do NOT self-correct into the teacher's solution. Stay wrong or partial as predicted."

    )

    lines.append("")

    return lines





def _format_conversation_block(student: Student, turn_mode: TurnMode) -> List[str]:

    if turn_mode == "social":

        return [

            "[Conversation Mode — social turn]",

            "The teacher is making small talk or asking about you personally, NOT the math problem.",

            "Respond naturally as a middle school student.",

            unknown_behavior_rule(student, turn_mode),

            "",

        ]

    if turn_mode == "off_topic":

        return [

            "[Conversation Mode — off-topic turn]",

            "The teacher's message is unrelated to the math problem.",

            "Respond briefly as a student and steer back to the problem if asked.",

            "",

        ]

    if turn_mode == "mixed":

        return [

            "[Conversation Mode — mixed turn]",

            "The teacher combined personal chat with math. Brief social reply, then address math if asked.",

            "",

        ]

    return []





def student_prompt(
    student: Student,
    predicted_behavior: str,
    top_p_concepts: list,
    task_text: str = "",
    error_playbook: Optional[dict] = None,
    error_anchor: Optional[str] = None,
    behavior_profile: Optional[dict] = None,
    turn_mode: TurnMode = "math_scaffold",
    scaffold_boost: Optional[Dict[str, float]] = None,
    stall_active: bool = False,
    task_metadata: Optional[dict] = None,
) -> str:
    is_math = _is_math_turn(turn_mode) and turn_mode != "social"

    task_rule = ""
    if is_math and task_text:
        task_rule = (
            f"The assigned problem is: {task_text}\n"
            "Respond in that context. If the teacher asks unrelated math drills, stay unsure — "
            "do not showcase perfect mental math on off-topic problems."
        )

    anchor_block = []
    if error_anchor:
        anchor_block.extend([error_anchor, ""])

    stall_block = []
    if stall_active and is_math:
        mode = (behavior_profile or {}).get("behavior_mode", "")
        stall_lines = [
            "[STALL — HIGHEST PRIORITY — teacher gave no direction]",
            "The teacher only said something vague like 'okay' or 'go on' without a new step.",
            "Do NOT compute another step, repeat your answer, or state a final result.",
            "Ask what they want — e.g. 'What should I do next?' or 'Okay — which part?'",
        ]
        if mode == "WRONG":
            stall_lines.append(
                "WRONG-mode guessing is PAUSED this turn — stay confident in tone but only ask "
                "what step the teacher wants; do not do more math."
            )
        stall_lines.append("")
        stall_block = stall_lines

    voice_card = build_voice_card(
        student,
        behavior_profile if is_math else None,
        turn_mode,
        profile_id=(student.student_id or "").lower(),
        stall_active=stall_active,
    )
    turn_state_block = ""
    if is_math and behavior_profile:
        turn_state_block = build_turn_state_block(
            student,
            behavior_profile,
            error_playbook,
            turn_mode,
            stall_active=stall_active,
        )

    lines = [
        "[Role & Task Definition]",
        "You are a middle school student. Stay in character at all times.",
        "",
        *_format_conversation_block(student, turn_mode),
    ]

    if is_math:
        lines.extend(stall_block)
        mistake_follow = (
            "On this stall turn, do NOT use mistake patterns — only ask what the teacher wants."
            if stall_active
            else (
                "You mostly know this material — do not copy the major error patterns above."
                if behavior_profile
                and behavior_profile.get("behavior_mode") == "NORMAL_ERROR_PROFILE"
                and behavior_profile.get("student_stack_level", 0) >= 3
                else "Follow realistic mistake types from the patterns above when they apply to this step."
            )
        )
        lines.extend(
            [
                *_format_word_problem_guardrails(task_text, task_metadata, behavior_profile),
                *(
                    _format_pisa_block(student, behavior_profile)
                    if _should_include_pisa_block(student, behavior_profile, is_math)
                    else []
                ),
                *_format_playbook(
                    error_playbook, behavior_profile, student, stall_active=stall_active
                ),
                mistake_follow,
                "",
                *anchor_block,
                "[Stay on Task]",
                task_rule or "Focus on the tutoring problem, not unrelated math drills.",
                "",
            ]
        )
        if turn_state_block:
            lines.extend([turn_state_block, ""])

    if not is_math:
        traits = ""
        for t in ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]:
            lv = getattr(student, t)
            traits += f"  {t} ({lv}): {chr(59).join(BF_TC[t][lv])}\n"
        lines.extend(
            [
                "[Personality — BF-TC Traits]",
                traits,
                "",
            ]
        )

    if voice_card:
        lines.extend([voice_card, ""])

    lines.extend(
        [
            "[CLASSROOM FLOOR — applies to ALL profiles]",
            "You are a student talking to a teacher in school.",
            "Even if you are frustrated, disengaged, or uncooperative, you do NOT disrespect a teacher directly.",
            "You may be short, cold, or unenthusiastic — but never openly rude or hostile.",
            "",
        ]
    )

    if is_math:
        lines.extend(
            [
                "[NO SELF-DIRECTION — math turns]",
                "Do NOT propose the full solution or conclude the final answer yourself.",
                "You MAY ask what the teacher wants or show thinking aloud on the current step.",
                "ONLY respond to what the teacher just said.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "[Conversation guidance]",
                unknown_behavior_rule(student, turn_mode),
                "",
            ]
        )

    lines.extend(
        [
            "[Behaviour Constraint]",
            "Wait for the teacher. Reflect BOTH personality AND the turn type (social vs math).",
        ]
    )
    return "\n".join(lines)


