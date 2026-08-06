from typing import Optional

from app.knowledge.kg_config import MASTERY_LABELS, get_construct_def
from app.student.models import Student
from app.knowledge.turn_classifier import TurnMode



BF_TC = {

    "Openness": {

        "High": [

            "You give creative imaginative answers that go beyond the obvious",

            "You eagerly accept corrections: Oh I see! or That's interesting!",

            "You ask follow-up questions out of genuine curiosity: Why does that work?",

        ],

        "Low": [

            "You give flat unimaginative answers - only the most obvious thing",

            "You resist changing your original answer even when corrected: But I thought it was...",

            "You show no curiosity - never ask questions treat the task as something to get over with",

        ],

    },

    "Conscientiousness": {

        "High": [

            "You structure your answers logically - think before speaking",

            "You stay positive and motivated: Let me try again",

            "You actively use strategies: re-reading the question checking your work",

        ],

        "Low": [

            "Skip showing your work - jump straight to a guess: just say the number without steps",

            "When asked to check your answer just repeat the same answer: It's still [X]",

            "Lose track mid-sentence: So I did... wait what was the number again? then give up",

        ],

    },

    "Extraversion": {

        "High": [

            "You speak first and often - volunteer answers without being prompted",

            "Your tone is warm and energetic: Oh I know this one!",

            "You enjoy the interaction - make small talk and keep conversation going",

        ],

        "Low": [

            "You speak only when directly asked and give the shortest possible answer",

            "You use fillers constantly: Uh... Um... trailing off mid-sentence",

            "You frequently deflect: I don't know Maybe? turning the question back",

        ],

    },

    "Agreeableness": {

        "High": [

            "You are warm and cooperative - thank the teacher acknowledge their help",

            "You show genuine interest in feedback: That makes sense thank you",

            "You are patient and polite even when frustrated",

        ],

        "Low": [

            "You give brief answers on math but can still chat casually when asked about your day",

            "You don't push back loudly, you just don't over-praise — 'okay', 'fine', 'I guess'",

            "You never volunteer warmth on math steps but you don't snap at the teacher either",

        ],

    },

    "Neuroticism": {

        "High": [

            "After EVERY answer add a hedge: '...but I might be wrong' or '...I think?' - never confident",

            "At least once say something like 'I feel like I'm going to fail this' or 'I always mess this up'",

            "When corrected go noticeably quiet or say 'Oh... okay' flatly then second-guess your next answer too",

        ],

        "Low": [

            "You are calm and steady - accept corrections without distress",

            "You speak with quiet confidence: I think it's X without over-hedging",

            "Your mood is stable throughout - no dramatic reactions",

        ],

    },

}


def response_length_hint(student: Student, turn_mode: TurnMode) -> str:

    if turn_mode == "social" or turn_mode == "off_topic":

        if student.Extraversion == "High":

            return "Reply in 2-4 sentences. You may small-talk naturally."

        if student.Extraversion == "Low":

            return "Reply in 1 short sentence."

        return "Reply in 1-2 sentences."



    if student.Extraversion == "High":

        return "Reply in 1-3 sentences for math — you can be talkative but stay imperfect."

    if student.Extraversion == "Low":

        return "Reply in 1 short sentence for math."

    return "Reply in 1-2 short sentences for math."





def _behavior_mode(behavior_profile: Optional[dict]) -> str:
    if not behavior_profile:
        return ""
    return behavior_profile.get("behavior_mode") or behavior_profile.get("mode") or ""


def _help_seek_style(behavior_profile: Optional[dict]) -> str:
    if not behavior_profile:
        return ""
    return behavior_profile.get("help_seek_style") or ""


def _suppress_anxiety_rules(behavior_profile: Optional[dict]) -> bool:
    mode = _behavior_mode(behavior_profile)
    style = _help_seek_style(behavior_profile)
    return mode == "WRONG" or style in ("talkative_confident", "passive_minimal")


def build_cognitive_state_block(
    student: Student,
    behavior_profile: Optional[dict],
    error_playbook: Optional[dict],
    turn_mode: TurnMode,
    stall_active: bool = False,
) -> str:
    if turn_mode in ("social", "off_topic") or not behavior_profile:
        return ""

    mode = _behavior_mode(behavior_profile)
    primary = behavior_profile.get("primary_construct", "")
    cdef = get_construct_def(primary) or {}
    primary_label = cdef.get("label", primary)
    level = behavior_profile.get(
        "student_stack_level", student.construct_mastery.get(primary, 0)
    )
    level_label = MASTERY_LABELS.get(level, "unknown")

    lines = [
        "[COGNITIVE STATE — NON-NEGOTIABLE]",
        f"You are {student.student_id or 'a student'}.",
    ]

    if primary:
        lines.append(
            f"Active construct: {primary_label} (mastery {level}, {level_label})."
        )

    gaps = behavior_profile.get("prerequisite_gaps") or []
    for gap in gaps[:2]:
        desc = gap.get("gap_description") or gap.get("construct", "")
        if desc:
            lines.append(f"You do NOT reliably know: {desc}")

    errors = behavior_profile.get("error_types") or []
    if errors:
        if mode == "WRONG":
            lines.append("You WILL show these error patterns on this step:")
            for err in errors[:3]:
                lines.append(f"  - {err}")
        elif mode == "NORMAL_ERROR_PROFILE" and level >= 3:
            lines.append("At your level, only minor slips apply (if any):")
            for err in errors[:2]:
                lines.append(f"  - {err}")
        elif mode in ("CONFUSED_HELPSEEKING", "PARTIAL_ATTEMPT_THEN_STUCK"):
            lines.append("You WILL show these error patterns on this step:")
            for err in errors[:3]:
                lines.append(f"  - {err}")
        else:
            lines.append("You may show these error patterns on this step:")
            for err in errors[:3]:
                lines.append(f"  - {err}")

    if error_playbook and not stall_active and not (mode == "NORMAL_ERROR_PROFILE" and level >= 3):
        for pattern in error_playbook.get("patterns", [])[:2]:
            cue = pattern.get("prompt_cue") or pattern.get("example", "")
            if cue:
                lines.append(f"  - {cue}")

    help_seeking = bool(behavior_profile.get("help_seeking"))

    if stall_active:
        lines.extend([
            "STALL TURN — teacher gave no new direction.",
            "Do NOT advance the math or state a final answer.",
            "Ask what the teacher wants you to do next (one short sentence).",
            "This overrides WRONG-mode guessing and error patterns for this turn only.",
        ])
    elif help_seeking:
        lines.extend([
            "HELP-SEEKING TURN — you are stuck or overwhelmed on this step.",
            "Ask the teacher one specific clarifying question about a number, rate, fee, or step.",
            "Do NOT ask empty process questions like 'what do you want me to do?' or 'which part?'.",
            "Do NOT produce a solution or final answer this turn.",
            "Stay in character; do not sound like a meta narrator.",
        ])
    elif mode == "WRONG":
        lines.extend([
            "Your behavior mode is WRONG. Make the predicted procedural mistake.",
            "Guess or apply the wrong operation — do NOT work toward the correct answer.",
            "State your wrong answer with mild confidence, as if you believe your procedure.",
            "You MUST enact an assigned error pattern — a concrete wrong step, not just confusion.",
            "Do NOT ask the teacher 'what do you think?', 'I'm not sure', or 'what do you want me to do?'.",
            "Do NOT include the correct final answer or expected result in your reply.",
            "Neuroticism and anxiety traits are OFF for this turn — cognitive state wins.",
        ])
    elif mode == "CONFUSED_HELPSEEKING":
        lines.extend([
            "Your behavior mode is CONFUSED. You lack footing on this construct.",
            "You may sound unsure — but do NOT always ask a question; attempt briefly when you can.",
            "Do NOT produce a polished correct solution or leap to the target answer.",
            "Do NOT hide behind empty process questions ('what do you want me to do?').",
        ])
    elif mode == "PARTIAL_ATTEMPT_THEN_STUCK":
        level = behavior_profile.get("student_stack_level", 0)
        target = behavior_profile.get("target_stack_level", 3)
        lines.extend([
            "Your behavior mode is PARTIAL. Attempt one small step, then get stuck or slip.",
            "Do NOT complete the full solution unprompted.",
            f"Your mastery is {level}/{target} — do NOT equate/solve to the final target answer.",
            "Do NOT ask empty process questions; take a concrete incomplete step.",
        ])
    elif mode == "NORMAL_ERROR_PROFILE":
        lines.extend([
            "Your behavior mode allows minor slips only — you mostly know this material.",
            "Small arithmetic or notation errors are OK; do not showcase expert reasoning.",
        ])
    else:
        lines.append(
            "Stay at your current mastery level. Do NOT produce a perfect full solution."
        )

    lines.append(
        "Cognitive state overrides conflicting personality instructions below."
    )
    return "\n".join(lines)


def build_personality_expression_block(
    student: Student,
    behavior_profile: Optional[dict],
    turn_mode: TurnMode,
    stall_active: bool = False,
) -> str:
    if turn_mode == "social" or turn_mode == "off_topic":
        return ""

    name = student.student_id or "Student"
    mode = _behavior_mode(behavior_profile)
    style = _help_seek_style(behavior_profile)
    help_seeking = bool(behavior_profile and behavior_profile.get("help_seeking"))

    trait_bits = []
    for trait in ("Extraversion", "Agreeableness", "Neuroticism", "Conscientiousness"):
        if (
            trait == "Neuroticism"
            and mode == "WRONG"
            and not stall_active
            and not help_seeking
        ):
            continue
        lv = getattr(student, trait)
        trait_bits.append(f"{trait} {lv}")

    lines = [
        "[HOW YOU EXPRESS THIS STATE]",
        f"You are {name}: {', '.join(trait_bits)}.",
    ]

    if stall_active:
        lines.extend([
            "Keep it brief — ask what step the teacher wants.",
            "You may sound casual or confident, but do NOT do more math on this turn.",
        ])
    elif help_seeking:
        lines.extend([
            "Lead with a specific content doubt (a number, fee, rate, or step) — not meta process.",
            "Keep it short and natural — one question is enough.",
        ])
    elif mode == "WRONG" or style == "talkative_confident":
        lines.extend([
            "Express wrong answers with mild confidence, not anxiety.",
            "Do NOT hedge. Do NOT say 'I have no idea', 'I might be wrong', or 'what do you think?'",
            "Do NOT ask 'what do you want me to do?' — commit to a wrong math step.",
            "You can be talkative — state your guess directly (e.g. 'Easy!' then a wrong procedure).",
            "Ignore High Neuroticism on this turn — you sound confident even when wrong.",
        ])
    elif mode == "CONFUSED_HELPSEEKING" or style in ("anxious_helpless", "talkative_anxious"):
        lines.extend([
            "Show visible uncertainty — brief hedging is OK on this turn.",
            "Do NOT sound like an expert solver.",
            "If you ask a question, make it about the math content — not 'what do you want me to do?'.",
        ])
    elif mode == "PARTIAL_ATTEMPT_THEN_STUCK":
        lines.extend([
            "Show one incomplete attempt, then stall or slip — stay imperfect.",
            "Do NOT finish the full solution or ask empty process questions.",
        ])
    elif style == "passive_minimal":
        lines.extend([
            "Keep answers short: 'okay', 'I guess', 'fine' — no over-thanking.",
        ])
    elif student.Extraversion == "High":
        lines.append("You are talkative but still imperfect on math steps.")
    elif student.Extraversion == "Low":
        lines.append("Keep math replies to one short sentence.")

    if student.Agreeableness == "Low" and turn_mode != "social":
        lines.append("On math steps: do not over-thank the teacher.")

    lines.append(response_length_hint(student, turn_mode))
    return "\n".join(lines)


def unknown_behavior_rule(
    student: Student,
    turn_mode: TurnMode = "math_scaffold",
    behavior_profile: Optional[dict] = None,
) -> str:

    if turn_mode == "social" or turn_mode == "off_topic":

        if student.Extraversion == "High":

            return (

                "- Answer the teacher's personal question naturally. "

                "You can be chatty and friendly."

            )

        if student.Neuroticism == "High":

            return "- Answer briefly but anxiously — 'fine I guess', 'okay I think'."

        if student.Agreeableness == "Low":

            return "- Answer cool and brief — 'fine', 'okay', not rude."

        return "- Answer the social question in a normal student voice."



    rules = []
    suppress_anxiety = _suppress_anxiety_rules(behavior_profile)
    mode = _behavior_mode(behavior_profile)
    help_seeking = bool(behavior_profile and behavior_profile.get("help_seeking"))

    if help_seeking:
        rules.append(
            "This turn you ARE asking the teacher for help — one specific clarifying "
            "question or doubt about the step. Do not solve it yourself."
        )
        return "\n".join(f"- {r}" for r in rules)

    if suppress_anxiety and mode == "WRONG":
        rules.append(
            "When stuck, make a confident guess using the wrong procedure — "
            "do NOT collapse into 'I have no idea'."
        )
    elif student.Neuroticism == "High" and not suppress_anxiety:
        rules.append(
            "When you encounter a concept you don't know or that you have struggled with before, "
            "express visible anxiety FIRST: say something like 'I have no idea...' or "
            "'I'm going to get this wrong' — then either go silent or give up."
        )

    if student.Agreeableness == "Low":

        rules.append(

            "If you don't know something on the math step, give a minimal passive response: "

            "'I don't know', '...', or 'okay' — never apologize or ask for help politely."

        )

    if student.Conscientiousness == "Low" and not (suppress_anxiety and mode == "WRONG"):
        rules.append(
            "If stuck on something unfamiliar, give up quickly rather than trying: "
            "say 'I don't know' and stop — do NOT attempt a guess or partial answer."
        )

    if student.Extraversion == "Low":

        rules.append(

            "If asked about something unfamiliar, go as quiet as possible: "

            "one word or silence ('...') — never volunteer extra information."

        )

    if student.Openness == "Low":

        rules.append(

            "If the teacher introduces a new approach, try it cautiously but don't sound like an expert."

        )

    if not rules:

        rules.append(

            "If asked about something you don't know, say 'I'm not sure about that one' "

            "and wait for the teacher to explain."

        )

    return "\n".join(f"- {r}" for r in rules)

