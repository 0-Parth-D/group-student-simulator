import json
import logging
import random
from copy import deepcopy
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.student.learning_profile import LearningProfile

from app.knowledge.kg_config import (
    MASTERY_LABELS,
    get_construct_def,
    get_prerequisites_of,
    load_knowledge_graph,
)
from app.llm import complete
from app.llm.roles import LLMRole

logger = logging.getLogger(__name__)

GRADING_RUBRIC = """Active construct: {construct_id} (mastery level {mastery}/3)
Expected answer: {expected_answer}
Expected error (if wrong): {expected_error}
Score 0: No relevant attempt or completely wrong procedure
Score 1: Correct procedure setup, wrong computation
Score 2: Correct answer reached, possibly with teacher help
Score 3: Correct answer, correct reasoning, unprompted
Student response: "{reply}"
Return ONLY JSON: {{"score": <0-3>, "error_type": "<subtract_across|wrong_lcd|correct|confused|other>"}}"""


def init_mastery_state(construct_mastery: Dict[str, int]) -> Dict[str, int]:
    """Session-local mastery map (0-3) seeded from student profile."""
    kg = load_knowledge_graph()
    state = {cid: 0 for cid in kg.get("constructs", {})}
    for cid, level in construct_mastery.items():
        if cid in state:
            state[cid] = max(0, min(3, int(level)))
    return state


def get_student_state(
    mastery_state: Dict[str, int],
    construct_id: str,
) -> dict:
    level = mastery_state.get(construct_id, 0)
    cdef = get_construct_def(construct_id) or {}
    return {
        "construct_id": construct_id,
        "mastery_level": level,
        "label": MASTERY_LABELS.get(level, "missing"),
        "construct_label": cdef.get("label", construct_id),
        "target_stack_level": cdef.get("target_stack_level", 3),
    }


def get_prerequisite_gaps(
    mastery_state: Dict[str, int],
    required_constructs: List[str],
) -> dict:
    """Traverse upstream prerequisites for required constructs; return gaps."""
    all_nodes = set(required_constructs)
    for cid in required_constructs:
        for pre in get_prerequisites_of(cid):
            all_nodes.add(pre)

    gaps = []
    accessible = []
    locked = []

    for cid in sorted(all_nodes):
        level = mastery_state.get(cid, 0)
        label = MASTERY_LABELS.get(level, "missing")

        if level >= 2:
            accessible.append(cid)
        else:
            locked.append(cid)

        if cid in required_constructs or cid in {
            pre for req in required_constructs for pre in get_prerequisites_of(req)
        }:
            if level <= 1:
                if level == 0:
                    desc = f"{cid} never observed — no foothold on this construct"
                else:
                    desc = f"{cid} at precursor level only — not grade-ready"
                prereqs = get_prerequisites_of(cid)
                bad_prereqs = [p for p in prereqs if mastery_state.get(p, 0) <= 1]
                if bad_prereqs and level >= 2:
                    desc = f"{cid} locked: prerequisites {bad_prereqs} not yet good"
                gaps.append(
                    {
                        "construct": cid,
                        "construct_label": (get_construct_def(cid) or {}).get("label", cid),
                        "current_mastery": level,
                        "label": label,
                        "required_mastery": 3,
                        "gap_description": desc,
                    }
                )

    return {
        "gaps": gaps,
        "accessible": accessible,
        "locked": locked,
        "required_constructs": list(required_constructs),
    }


def update_observation(
    mastery_state: Dict[str, int],
    construct_ids: List[str],
    reply_quality: str,
) -> Dict[str, int]:
    """Legacy fixed-step update; prefer update_mastery_state for LP sessions."""
    return update_mastery_state(
        mastery_state,
        construct_ids,
        reply_quality,
        learning_profile=None,
        scaffold_boost=None,
    )


def update_mastery_state(
    mastery_state: Dict[str, int],
    construct_ids: List[str],
    reply_quality: str,
    learning_profile: Optional["LearningProfile"] = None,
    scaffold_boost: Optional[Dict[str, float]] = None,
) -> Dict[str, int]:
    """Per-student mastery update with scaffold boost and learning profile."""
    updated = deepcopy(mastery_state)
    quality = reply_quality.lower().strip()
    boost = scaffold_boost or {}
    slip = learning_profile.slip_rate if learning_profile else 0.25
    retention = learning_profile.retention if learning_profile else 0.7
    max_gain = learning_profile.max_step_gain if learning_profile else 1

    for cid in construct_ids:
        if cid not in updated:
            continue
        current = updated[cid]
        latent = boost.get(cid, 0.0)
        gain = (
            learning_profile.effective_gain(cid)
            if learning_profile
            else 0.5
        )

        if quality == "correct":
            p_gain = min(1.0, gain + latent * 0.3)
            if random.random() < p_gain and current < 3:
                step = min(max_gain, 1)
                new_level = min(current + step, 3)
                updated[cid] = int(round(retention * new_level + (1 - retention) * current))
        elif quality == "partial":
            if latent >= 0.35 and current < 3 and random.random() < gain * 0.5:
                updated[cid] = min(current + 1, 3)
        elif quality in ("wrong", "confused"):
            if latent >= 0.5 and random.random() > slip:
                pass
            elif random.random() < slip and current > 0:
                updated[cid] = max(current - 1, 0)

    return updated


def _heuristic_reply_quality(
    student_reply: str,
    task_meta: dict,
    behavior_target: Optional[dict] = None,
) -> str:
    """Lightweight heuristic fallback for turn-level mastery updates."""
    lower = student_reply.lower().strip()
    if not lower:
        return "confused"

    shutdown = (
        "don't know",
        "dont know",
        "skip",
        "no idea",
        "where to start",
        "confus",
    )
    if any(p in lower for p in shutdown):
        return "confused"

    expected = (task_meta.get("expected_answer") or "").lower()
    if expected and expected in lower:
        return "correct"

    mode = (behavior_target or {}).get("behavior_mode") or (behavior_target or {}).get("mode")
    if mode == "CONFUSED_HELPSEEKING":
        return "confused"

    wrong_markers = ["4/3", "3/10", "3/7", "5:2", "4:5", "0.02"]
    if any(m in lower for m in wrong_markers):
        return "wrong"

    if mode in ("PARTIAL_ATTEMPT_THEN_STUCK", "WRONG"):
        return "wrong"

    return "partial"


def _score_to_quality(score: int, error_type: str) -> str:
    if score >= 3:
        return "correct"
    if score == 2:
        return "correct"
    if score == 1:
        return "partial"
    if (error_type or "").lower() in ("confused", "none", ""):
        return "confused"
    return "wrong"


def infer_reply_quality_semantic(
    student_reply: str,
    task_meta: dict,
    behavior_target: Optional[dict] = None,
    mastery_state: Optional[Dict[str, int]] = None,
) -> Optional[str]:
    """Rubric-prompted gpt-4o-mini grading (Phase 2). Returns None on failure."""
    construct_id = (behavior_target or {}).get("primary_construct") or ""
    if not construct_id:
        required = task_meta.get("required_constructs") or []
        construct_id = required[0] if required else "fraction_operations"

    mastery = (mastery_state or {}).get(construct_id, 0)
    if behavior_target:
        mastery = behavior_target.get("student_stack_level", mastery)

    expected_answer = task_meta.get("expected_answer") or ""
    expected_error = ""
    errors = (behavior_target or {}).get("error_types") or []
    if errors:
        expected_error = errors[0]
    elif task_meta.get("typical_errors"):
        expected_error = task_meta["typical_errors"][0].get("error", "")

    prompt = GRADING_RUBRIC.format(
        construct_id=construct_id,
        mastery=mastery,
        expected_answer=expected_answer or "unknown",
        expected_error=expected_error or "procedural error",
        reply=student_reply.replace('"', "'")[:600],
    )
    try:
        raw = complete(
            LLMRole.KG_GRADING,
            "You grade middle-school math tutoring responses. Return JSON only.",
            prompt,
        )
        payload = json.loads(raw)
        score = int(payload.get("score", 0))
        error_type = str(payload.get("error_type", "other"))
        return _score_to_quality(score, error_type)
    except Exception as exc:
        logger.warning("Semantic reply grading failed: %s", exc)
        return None


def infer_reply_quality(
    student_reply: str,
    task_meta: dict,
    behavior_target: Optional[dict] = None,
    mastery_state: Optional[Dict[str, int]] = None,
    use_semantic: bool = False,
) -> str:
    """Semantic rubric grading with heuristic fallback."""
    if use_semantic:
        semantic = infer_reply_quality_semantic(
            student_reply, task_meta, behavior_target, mastery_state
        )
        if semantic:
            return semantic
    return _heuristic_reply_quality(student_reply, task_meta, behavior_target)
