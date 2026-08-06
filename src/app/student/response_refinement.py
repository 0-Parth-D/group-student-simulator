import logging
import re
from typing import Optional

from app.llm import complete_chat
from app.llm.roles import LLMRole
from app.knowledge.turn_classifier import TurnMode

logger = logging.getLogger(__name__)

EXPERT_MARKERS = [
    "the answer is",
    "therefore",
    "we can see that",
    "which equals",
    "so x =",
    "so y =",
    "it follows that",
    "hence",
    "in conclusion",
]

# Multi-step / target-complete leaps blocked when mastery is below target.
FULL_SOLUTION_MARKERS = [
    "set them equal",
    "set equal",
    "set the equations equal",
    "make them equal",
    "where they switch",
    "where they cost the same",
    "find the crossover",
    "solve for x",
    "so x =",
    "x equals",
    "x is just",
    "divide both sides",
]

CONFIDENT_PHRASES = [
    "easy!",
    "alright!",
    "no problem",
    "that's easy",
    "i know this",
    "obviously",
    "clearly",
    "definitely",
    "looks fine",
]

ANXIETY_PHRASES = (
    "i have no idea",
    "i don't know",
    "i dont know",
    "i'm not sure",
    "im not sure",
    "i might be wrong",
    "what do you think",
    "does that seem right",
    "i'm so confused",
    "im so confused",
    "this is too hard",
    "can we skip",
    "i'm lost",
    "im lost",
    "i still don't really know",
    "i dont really know",
)

# Allowed only on true teacher-stall turns. Banned on active math turns.
META_PROCESS_PHRASES = (
    "what do you want me to do",
    "what do you want us to do",
    "what do you want me",
    "what do you want us",
    "do you want me to",
    "what should i do with",
    "what do you want me to check",
    "which part",
    "what part",
    "which part of",
    "in words or",
    "words or math",
    "show the actual math",
    "explain it like in words",
    "what do you want me to do with it",
    "okay — what do you want",
    "what do you want me to do next",
)

STALL_OK_PHRASES = (
    "what should i do",
    "what do you want",
    "what part",
    "what next",
    "which part",
    "not sure what you want",
)

EQUATION_CHAIN = re.compile(
    r"\b\d+(?:\s*/\s*\d+)?\s*[=+\-]"
    r"|\b\d+\s*(?:plus|minus|times)\s*\d+"
    r"|\bx\s*="
    r"|\$\s*\d"
)

WRONG_PROCEDURE_CUES = (
    "plus",
    "minus",
    "times",
    "divide",
    "divided",
    "so it's",
    "so it is",
    "equals",
    "i got",
    "that makes",
    "easy",
    "just",
)


def _contains_correct_answer(text: str, avoid_answers: list) -> bool:
    lower = text.lower()
    for ans in avoid_answers:
        if ans and str(ans).lower() in lower:
            return True
    return False


def _error_surface_forms(behavior_profile: dict) -> list:
    """Wrong answers / phrases the student is *supposed* to show."""
    forms = []
    for key in ("error_labels", "error_types"):
        for item in behavior_profile.get(key) or []:
            if item:
                forms.append(str(item).lower())
    for ans in behavior_profile.get("wrong_answers") or []:
        if ans:
            forms.append(str(ans).lower())
    return forms


def _looks_like_wrong_enactment(response: str, behavior_profile: dict) -> bool:
    """True when the reply already surfaces an assigned error — not expert."""
    lower = response.lower()
    return any(form and form in lower for form in _error_surface_forms(behavior_profile))


def _mastery_below_target(behavior_profile: dict) -> bool:
    level = int(behavior_profile.get("student_stack_level") or 0)
    target = int(behavior_profile.get("target_stack_level") or 3)
    return level < target


def _has_full_solution_leap(response: str) -> bool:
    lower = response.lower()
    return any(marker in lower for marker in FULL_SOLUTION_MARKERS)


def is_too_expert(
    response: str,
    behavior_profile: dict,
    turn_mode: TurnMode = "math_scaffold",
    scaffold_boost: Optional[dict] = None,
) -> bool:
    """Heuristic check: response sounds like an expert or contradicts predicted errors."""
    if turn_mode in ("social", "off_topic"):
        return False

    if not response.strip():
        return False

    likely = behavior_profile.get("likely_correctness", "partially correct")
    if likely in ("mostly correct", "correct"):
        return False

    if behavior_profile.get("stall_active"):
        return False

    lower = response.lower()
    boost = scaffold_boost or {}
    primary = behavior_profile.get("primary_construct", "")
    recent_scaffold = boost.get(primary, 0.0) >= 0.35
    avoid = behavior_profile.get("avoid_answers") or []
    mode = behavior_profile.get("behavior_mode", "")
    below_target = _mastery_below_target(behavior_profile)

    if likely == "incorrect":
        if _contains_correct_answer(response, avoid):
            if recent_scaffold and behavior_profile.get("student_stack_level", 0) < 3:
                return False
            return True
        for phrase in CONFIDENT_PHRASES:
            if phrase in lower and _contains_correct_answer(response, avoid):
                return True

    if likely in ("incorrect", "partially correct"):
        for marker in EXPERT_MARKERS:
            if marker in lower:
                if recent_scaffold:
                    return False
                return True
        # Mastery gate: unfinished kids must not leap to a full target solution.
        if below_target and _has_full_solution_leap(response) and not recent_scaffold:
            return True
        if mode == "PARTIAL_ATTEMPT_THEN_STUCK" and _has_full_solution_leap(response):
            if not recent_scaffold:
                return True
        # A finished equation that lands on the correct answer, with no assigned
        # error surface form, is expert drift. Wrong enactments are allowed.
        if EQUATION_CHAIN.search(response) and likely == "incorrect":
            if _looks_like_wrong_enactment(response, behavior_profile):
                return False
            if _contains_correct_answer(response, avoid):
                if recent_scaffold:
                    return False
                return True
        if (
            EQUATION_CHAIN.search(response)
            and likely == "partially correct"
            and below_target
            and _contains_correct_answer(response, avoid)
            and not recent_scaffold
        ):
            return True

    return False


def is_too_anxious(
    response: str,
    behavior_profile: dict,
    turn_mode: TurnMode = "math_scaffold",
) -> bool:
    """WRONG-mode students should guess confidently, not hedge or help-seek."""
    if behavior_profile.get("stall_active"):
        return False
    if behavior_profile.get("help_seeking"):
        return False
    if turn_mode in ("social", "off_topic"):
        return False
    if not response.strip():
        return False

    mode = behavior_profile.get("behavior_mode", "")
    style = behavior_profile.get("help_seek_style", "")
    if mode != "WRONG" and style != "talkative_confident":
        return False

    lower = response.lower()
    if any(p in lower for p in ANXIETY_PHRASES):
        return True
    if lower.count("?") >= 2 and any(w in lower for w in ("think", "sure", "right")):
        return True
    if mode == "WRONG":
        has_confident = any(p in lower for p in CONFIDENT_PHRASES)
        if ("maybe" in lower or "i think" in lower) and not has_confident:
            return True
    return False


def is_meta_process_question(
    response: str,
    behavior_profile: dict,
    turn_mode: TurnMode = "math_scaffold",
) -> bool:
    """Active math turns must not hide behind empty process questions.

    True stall turns (teacher gave no direction) may ask what to do next.
    Help-seeking turns must ask about the *content*, not meta process.
    """
    if turn_mode in ("social", "off_topic"):
        return False
    if behavior_profile.get("stall_active"):
        return False
    if not response.strip():
        return False

    lower = response.lower().strip()
    if any(p in lower for p in META_PROCESS_PHRASES):
        return True
    # Bare process questions with almost no math content.
    if "?" in lower and not EQUATION_CHAIN.search(response):
        if any(
            p in lower
            for p in (
                "what do you want",
                "which part",
                "what part",
                "do you want me",
                "in words or",
            )
        ):
            return True
    return False


def missing_error_enactment(
    response: str,
    behavior_profile: dict,
    turn_mode: TurnMode = "math_scaffold",
) -> bool:
    """WRONG / forced-error modes must surface a mistake, not a clean hedge."""
    if turn_mode in ("social", "off_topic"):
        return False
    if behavior_profile.get("stall_active"):
        return False
    if behavior_profile.get("help_seeking"):
        return False
    if not response.strip():
        return False

    mode = behavior_profile.get("behavior_mode", "")
    likely = behavior_profile.get("likely_correctness", "")
    if mode != "WRONG" and likely != "incorrect":
        return False

    errors = _error_surface_forms(behavior_profile)
    if not errors:
        return False

    lower = response.lower()
    if _looks_like_wrong_enactment(response, behavior_profile):
        return False
    if any(cue in lower for cue in WRONG_PROCEDURE_CUES) and EQUATION_CHAIN.search(
        response
    ):
        # Has a concrete procedure attempt; good enough even if catalog string missed.
        return False
    # Hedging / empty confusion without enacting the assigned error.
    return True


def is_advancing_on_stall(
    response: str,
    behavior_profile: dict,
    turn_mode: TurnMode = "math_scaffold",
) -> bool:
    """On stall turns, student should ask for direction — not continue solving."""
    if not behavior_profile.get("stall_active"):
        return False
    if turn_mode in ("social", "off_topic"):
        return False
    if not response.strip():
        return False

    lower = response.lower()
    if any(p in lower for p in STALL_OK_PHRASES):
        if not EQUATION_CHAIN.search(response):
            return False
    if EQUATION_CHAIN.search(response):
        return True
    avoid = behavior_profile.get("avoid_answers") or []
    if _contains_correct_answer(response, avoid):
        return True
    if any(
        p in lower
        for p in ("the answer is", "that's the answer", "so it's", "equals", "so x =")
    ):
        return True
    return False


def build_stall_refinement_message(response: str) -> str:
    return (
        "[REVISION REQUIRED — teacher gave no direction; you advanced the solution]\n"
        "Do NOT compute another step or restate a final answer.\n"
        "Reply in one short sentence asking what the teacher wants — "
        "e.g. 'What should I do next?' or 'Okay — which part?'\n"
        "Revise ONLY your last student message.\n\n"
        f"Your previous reply:\n{response}"
    )


def build_meta_process_refinement_message(
    response: str, behavior_profile: dict
) -> str:
    mode = behavior_profile.get("behavior_mode", "")
    errors = behavior_profile.get("error_types") or behavior_profile.get(
        "error_labels", []
    )
    if mode == "WRONG":
        hint = (
            "State a wrong procedure confidently "
            f"(e.g. {errors[0] if errors else 'a realistic slip'})."
        )
    elif mode == "PARTIAL_ATTEMPT_THEN_STUCK":
        hint = "Take one small math step, then get stuck or slip — no meta questions."
    elif behavior_profile.get("help_seeking"):
        hint = (
            "Ask one *specific* content question "
            "(e.g. about a number, fee, rate, or step) — not 'what do you want me to do?'."
        )
    else:
        hint = "Attempt the current step or name a concrete point of confusion."
    return (
        "[REVISION REQUIRED — empty process question]\n"
        "Do NOT ask 'what do you want me to do?', 'which part?', or 'in words or math?'.\n"
        f"{hint}\n"
        "Stay in character as a middle-school student working the problem.\n"
        "Revise ONLY your last student message.\n\n"
        f"Your previous reply:\n{response}"
    )


def build_refinement_message(response: str, behavior_profile: dict) -> str:
    errors = behavior_profile.get("error_types") or behavior_profile.get("error_labels", [])
    weak = behavior_profile.get("weak_attributes", [])
    likely = behavior_profile.get("likely_correctness", "partially correct")
    mode = behavior_profile.get("behavior_mode", "")
    level = behavior_profile.get("student_stack_level", 0)
    target = behavior_profile.get("target_stack_level", 3)

    if mode == "WRONG":
        tone = (
            "Sound confident and direct — state a wrong procedure as if you believe it. "
            "No hedging, no asking the teacher for reassurance."
        )
    elif mode == "PARTIAL_ATTEMPT_THEN_STUCK":
        tone = (
            "Show only ONE incomplete step, then stall or slip. "
            "Do NOT finish the full solution or find the final target answer."
        )
    else:
        tone = "Use informal student language — brief and imperfect."

    mastery_note = ""
    if _mastery_below_target(behavior_profile):
        mastery_note = (
            f"Your mastery on this construct is {level}/{target} — "
            "do NOT leap to a complete correct solution.\n"
        )

    return (
        "[REVISION REQUIRED — your last reply was too expert or too complete]\n"
        f"Predicted performance: {likely}\n"
        f"{mastery_note}"
        f"Show these misconception/error patterns: {'; '.join(errors[:3]) or 'realistic confusion'}\n"
        f"Weak attributes to reflect: {', '.join(weak) or 'none'}\n"
        f"{tone}\n"
        "Do NOT give a polished full correct answer unless you only used one step the teacher just taught.\n"
        "Revise ONLY your last student message.\n\n"
        f"Your previous reply:\n{response}"
    )


def build_anxious_refinement_message(response: str, behavior_profile: dict) -> str:
    errors = behavior_profile.get("error_types") or behavior_profile.get("error_labels", [])
    return (
        "[REVISION REQUIRED — your last reply was too anxious or help-seeking for WRONG mode]\n"
        "You are a confident-but-wrong student. Re-state your answer with mild confidence.\n"
        "Do NOT say 'I have no idea', 'I'm not sure', 'what do you think', or similar.\n"
        "Start with mild confidence (e.g. 'Easy!' or 'I know this one!') then state a wrong procedure.\n"
        "Make the procedural mistake directly — e.g. subtract tops and bottoms, add across, etc.\n"
        f"Target error patterns: {'; '.join(errors[:2]) or 'wrong operation'}\n"
        "Do NOT include the correct final answer.\n"
        "Revise ONLY your last student message.\n\n"
        f"Your previous reply:\n{response}"
    )


def build_error_enactment_refinement_message(
    response: str, behavior_profile: dict
) -> str:
    errors = behavior_profile.get("error_types") or behavior_profile.get(
        "error_labels", []
    )
    return (
        "[REVISION REQUIRED — you did not enact the assigned mistake]\n"
        "You must SHOW a wrong procedure or wrong intermediate result, not just say you are confused.\n"
        f"Enact one of: {'; '.join(errors[:3]) or 'a realistic procedural slip'}\n"
        "State it as if you believe it. Keep middle-school voice.\n"
        "Revise ONLY your last student message.\n\n"
        f"Your previous reply:\n{response}"
    )


def generate_with_refinement(
    system: str,
    history: list,
    behavior_profile: dict,
    temperature: float | None = None,
    max_tokens: int = 200,
    max_revisions: int = 1,
    turn_mode: TurnMode = "math_scaffold",
    scaffold_boost: Optional[dict] = None,
    **_kwargs,
) -> tuple[str, int]:
    """Generate student reply with optional heuristic expert-tone revision."""
    if turn_mode in ("social", "off_topic"):
        max_tokens = max(max_tokens, 300)

    mode = behavior_profile.get("behavior_mode", "")
    if behavior_profile.get("stall_active"):
        max_revisions = max(max_revisions, 2)
    elif mode == "WRONG":
        max_revisions = max(max_revisions, 2)
    elif mode == "PARTIAL_ATTEMPT_THEN_STUCK":
        max_revisions = max(max_revisions, 2)
    else:
        max_revisions = max(max_revisions, 2)

    revisions = 0
    chat_kw = {"max_tokens": max_tokens}
    if temperature is not None:
        chat_kw["temperature"] = temperature
    response = complete_chat(LLMRole.STUDENT_REPLY, system, history, **chat_kw)

    while revisions < max_revisions:
        needs_stall_fix = is_advancing_on_stall(
            response, behavior_profile, turn_mode=turn_mode
        )
        needs_meta_fix = is_meta_process_question(
            response, behavior_profile, turn_mode=turn_mode
        )
        needs_expert_fix = is_too_expert(
            response,
            behavior_profile,
            turn_mode=turn_mode,
            scaffold_boost=scaffold_boost,
        )
        needs_anxious_fix = is_too_anxious(
            response, behavior_profile, turn_mode=turn_mode
        )
        needs_error_fix = missing_error_enactment(
            response, behavior_profile, turn_mode=turn_mode
        )
        if not (
            needs_stall_fix
            or needs_meta_fix
            or needs_expert_fix
            or needs_anxious_fix
            or needs_error_fix
        ):
            break

        if needs_stall_fix:
            logger.info("Student advanced on stall turn — refinement pass %d", revisions + 1)
            refine_msg = build_stall_refinement_message(response)
        elif needs_meta_fix:
            logger.info("Student used meta-process stall — refinement pass %d", revisions + 1)
            refine_msg = build_meta_process_refinement_message(
                response, behavior_profile
            )
        elif needs_anxious_fix:
            logger.info("Student reply too anxious — refinement pass %d", revisions + 1)
            refine_msg = build_anxious_refinement_message(response, behavior_profile)
        elif needs_error_fix:
            logger.info("Student missed error enactment — refinement pass %d", revisions + 1)
            refine_msg = build_error_enactment_refinement_message(
                response, behavior_profile
            )
        else:
            logger.info("Student reply too expert — refinement pass %d", revisions + 1)
            refine_msg = build_refinement_message(response, behavior_profile)

        refine_hist = history + [
            {"role": "assistant", "content": response},
            {"role": "user", "content": refine_msg},
        ]
        response = complete_chat(LLMRole.STUDENT_REPLY, system, refine_hist, **chat_kw)
        revisions += 1

    return response, revisions
