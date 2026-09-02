from dataclasses import asdict, dataclass
from typing import Dict, List, Literal, Optional

from app.kg_config import MASTERY_LABELS, get_construct_def, LP_PISA_TO_LEGACY
from app.knowledge_graph import get_prerequisite_gaps
from app.models import Student
from app.pisa import attribute_error_templates, weak_attributes

BehaviorMode = Literal[
    "CONFUSED_HELPSEEKING",
    "PARTIAL_ATTEMPT_THEN_STUCK",
    "WRONG",
    "NORMAL_ERROR_PROFILE",
]

MODE_RANK = {
    "CONFUSED_HELPSEEKING": 0,
    "WRONG": 1,
    "PARTIAL_ATTEMPT_THEN_STUCK": 2,
    "NORMAL_ERROR_PROFILE": 3,
}


def error_applies_at_mastery(
    mode: BehaviorMode, student_stack_level: int, error_stack_level: int
) -> bool:
    """Filter catalog errors so L3 students are not steered toward L1 misconceptions."""
    if mode == "NORMAL_ERROR_PROFILE" and student_stack_level >= 3:
        return error_stack_level >= 2
    if mode == "NORMAL_ERROR_PROFILE" and student_stack_level == 2:
        return error_stack_level >= 1
    return True


@dataclass
class BehaviorTarget:
    mode: BehaviorMode
    primary_construct: str
    stack_level: int
    error_types: List[str]
    help_seek_style: str
    prerequisite_gaps: List[dict]
    accessible_constructs: List[str]
    locked_constructs: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _help_seek_style(student: Student, mode: Optional[BehaviorMode] = None) -> str:
    # Confident-but-wrong personas (e.g. Jordan): WRONG mode uses direct guesses, not anxiety.
    if mode == "WRONG" and student.Extraversion == "High":
        return "talkative_confident"
    if student.Extraversion == "High" and student.Neuroticism == "High":
        return "talkative_anxious"
    if student.Extraversion == "High":
        return "talkative_confident"
    if student.Neuroticism == "High":
        return "anxious_helpless"
    if student.Agreeableness == "Low":
        return "passive_minimal"
    if student.Extraversion == "Low":
        return "one_word_silence"
    if student.Conscientiousness == "Low":
        return "give_up_quickly"
    return "hesitant_uncertain"


def _legacy_pisa_ids(task_meta: dict) -> List[str]:
    ids = []
    for pid in task_meta.get("pisa_attributes") or []:
        ids.append(LP_PISA_TO_LEGACY.get(pid, pid))
    return ids


def apply_mode_cap(mode: BehaviorMode, construct_id: str, mode_caps: Dict[str, BehaviorMode]) -> BehaviorMode:
    """Cap behavior mode after worked-example scaffolding (max PARTIAL, not NORMAL)."""
    cap = mode_caps.get(construct_id)
    if not cap:
        return mode
    if MODE_RANK.get(mode, 0) > MODE_RANK.get(cap, 99):
        return cap
    return mode


def cap_mode(mode_caps: Dict[str, BehaviorMode], construct_id: str, max_mode: BehaviorMode) -> Dict[str, BehaviorMode]:
    """Record a per-construct behavior ceiling."""
    updated = dict(mode_caps)
    current = updated.get(construct_id)
    if current is None or MODE_RANK.get(max_mode, 0) < MODE_RANK.get(current, 99):
        updated[construct_id] = max_mode
    return updated


def route_behavior(
    mastery_state: Dict[str, int],
    task_meta: dict,
    student: Student,
    misconceptions: Optional[List[dict]] = None,
    mode_caps: Optional[Dict[str, BehaviorMode]] = None,
) -> BehaviorTarget:
    required = task_meta.get("required_constructs") or []
    primary = required[0] if required else ""
    gap_info = get_prerequisite_gaps(mastery_state, required)
    gaps = gap_info["gaps"]

    required_levels = [mastery_state.get(c, 0) for c in required]
    has_missing = any(level == 0 for level in required_levels)
    has_bad = any(level == 1 for level in required_levels)
    has_partial = any(level == 2 for level in required_levels)
    all_good = required and all(mastery_state.get(c, 0) >= 3 for c in required)

    if has_missing:
        mode: BehaviorMode = "CONFUSED_HELPSEEKING"
    elif has_bad:
        mode = "WRONG"
    elif has_partial or gaps:
        mode = "PARTIAL_ATTEMPT_THEN_STUCK"
    elif all_good:
        mode = "NORMAL_ERROR_PROFILE"
    else:
        mode = "PARTIAL_ATTEMPT_THEN_STUCK"

    caps = mode_caps or {}
    if primary and primary in caps:
        mode = apply_mode_cap(mode, primary, caps)
    for cid in required:
        if cid in caps:
            mode = apply_mode_cap(mode, cid, caps)

    stack_level = mastery_state.get(primary, 0) if primary else 0
    error_types = []
    if misconceptions:
        for m in misconceptions:
            construct = m.get("construct_id", primary)
            sl = mastery_state.get(construct, stack_level)
            if error_applies_at_mastery(
                mode, sl, int(m.get("stack_level", 1))
            ):
                error_types.append(m.get("description", "")[:80])
            if len(error_types) >= 3:
                break
    elif task_meta.get("typical_errors"):
        for e in task_meta["typical_errors"]:
            construct = e.get("construct", primary)
            sl = mastery_state.get(construct, stack_level)
            if error_applies_at_mastery(
                mode, sl, int(e.get("stack_level", 1))
            ):
                error_types.append(e.get("error", "")[:80])
            if len(error_types) >= 3:
                break

    return BehaviorTarget(
        mode=mode,
        primary_construct=primary,
        stack_level=stack_level,
        error_types=error_types,
        help_seek_style=_help_seek_style(student, mode),
        prerequisite_gaps=gaps,
        accessible_constructs=gap_info["accessible"],
        locked_constructs=gap_info["locked"],
    )


def behavior_target_to_profile(target: BehaviorTarget, task_meta: dict, student: Student) -> dict:
    """Shape compatible with response_refinement and API responses."""
    likely_map = {
        "CONFUSED_HELPSEEKING": "incorrect",
        "WRONG": "incorrect",
        "PARTIAL_ATTEMPT_THEN_STUCK": "partially correct",
        "NORMAL_ERROR_PROFILE": "mostly correct",
    }
    primary = target.primary_construct
    cdef = get_construct_def(primary) or {}
    target_level = task_meta.get("target_stack_level", cdef.get("target_stack_level", 3))

    construct_levels = {
        cid: student.construct_mastery.get(cid, 0)
        for cid in task_meta.get("required_constructs", [])
    }

    legacy_pisa = _legacy_pisa_ids(task_meta)
    weak = weak_attributes(student.pisa_profile, legacy_pisa) if student.pisa_profile else []
    attr_errors = (
        attribute_error_templates(student.pisa_profile, legacy_pisa)
        if student.pisa_profile
        else []
    )

    return {
        "task_id": task_meta.get("task_id", ""),
        "primary_construct": primary,
        "target_stack_level": target_level,
        "student_stack_level": target.stack_level,
        "likely_correctness": likely_map.get(target.mode, "partially correct"),
        "behavior_mode": target.mode,
        "error_types": target.error_types,
        "error_labels": target.error_types,
        "pisa_attributes": (
            []
            if task_meta.get("task_id") == "phone_plans_linear_01"
            else task_meta.get("pisa_attributes", [])
        ),
        "weak_attributes": (
            []
            if task_meta.get("task_id") == "phone_plans_linear_01"
            else weak
        ),
        "attribute_errors": (
            []
            if task_meta.get("task_id") == "phone_plans_linear_01"
            else attr_errors[:4]
        ),
        "student_language": [],
        "construct_levels": construct_levels,
        "avoid_answers": [task_meta.get("expected_answer", "")]
        if task_meta.get("expected_answer")
        else [],
        "prerequisite_gaps": target.prerequisite_gaps,
        "accessible_constructs": target.accessible_constructs,
        "locked_constructs": target.locked_constructs,
        "help_seek_style": target.help_seek_style,
    }


def behavior_summary_text(target: BehaviorTarget, task_meta: dict) -> str:
    primary = target.primary_construct
    cdef = get_construct_def(primary) or {}
    name = cdef.get("label", primary)
    lines = [
        f"[LP Behavior Target — {target.mode}]",
        f"  Primary construct: {name} (mastery {target.stack_level}, {MASTERY_LABELS.get(target.stack_level, '?')})",
        f"  Task target stack level: {task_meta.get('target_stack_level', 3)}",
    ]
    if target.prerequisite_gaps:
        lines.append("  Prerequisite gaps:")
        for gap in target.prerequisite_gaps[:3]:
            lines.append(f"    - {gap.get('gap_description', gap.get('construct', ''))}")
    if target.error_types:
        lines.append("  Expected error patterns: " + "; ".join(target.error_types[:2]))
    lines.append(f"  Help-seeking style: {target.help_seek_style}")
    return "\n".join(lines)
