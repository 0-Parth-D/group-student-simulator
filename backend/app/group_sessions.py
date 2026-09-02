"""Teacher-facilitated group sessions (Model A): 1 teacher + 3 LP students.

Duplicates/adapts the 1:1 reply pipeline from sessions.py without modifying it.
"""

from __future__ import annotations

import logging
import re
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple

from app.behavior_router import (
    behavior_summary_text,
    behavior_target_to_profile,
    cap_mode,
    route_behavior,
)
from app.data import DEFAULT_TASK, get_profile, make_student
from app.demo_students import DEMO_STUDENTS, list_lp_profile_ids
from app.config import GENERATION_DEBUG
from app.eval_log import (
    build_eval_turn_record,
    build_generation_debug,
    learning_expected_behavior_text,
)
from app.group_context import awareness_block, trim_private_hist, wrap_user_message
from app.group_orchestrator import resolve_speakers_with_meta
from app.help_seeking import (
    build_help_seeking_prompt_block,
    decide_help_seeking,
    reply_looks_like_question,
    turns_since_help_seek,
)
from app.kg_config import MASTERY_LABELS
from app import kg_store
from app.learning_receptivity import (
    apply_scaffold_opportunity,
    receptivity_for_profile,
)
from app.fight_progress import (
    apply_listen_restraint,
    breakthrough_memory_block,
    claim_lock_block,
    decay_listen_restraint,
    detect_peer_critique_turn,
    detect_teacher_fee_press,
    late_turn_nudge_block,
    on_student_reply,
    on_teacher_message,
    should_use_claim_lock,
    soften_jordan_fee_press_mode,
    suppress_fight_misconception,
)
from app.fight_stance import (
    CLAIM_IDS,
    FE_SEED_LINES,
    is_fe_starter_fight,
    is_phone_plans_task,
    retrieve_for_group_fight,
)
from app.live_hud import PST_GROUP_BARRIERS
from app.llm_config import resolved_student_chat_params, student_max_tokens_for_turn
from app.misconception_store import (
    build_error_anchor,
    misconceptions_to_playbook,
    retrieve_error_instance,
    retrieve_misconceptions,
)
from app.models import Student
from app.personality import build_personality_export
from app.reasoning_warrant import (
    detect_teacher_reasoning_press,
    warrant_expectation,
)
from app.response_refinement import build_expected_pack, generate_with_refinement
from app.scaffold_detector import (
    apply_scaffold_boost,
    decay_scaffold_boost,
    detect_scaffold,
    scaffold_is_worked_example,
)
from app.session_shared import TEACHER_OPENER_TEMPLATE
from app.task_tagger import load_task_metadata, resolve_task_metadata
from app.turn_classifier import (
    classify_teacher_turn,
    mastery_update_turn,
)
from app.student_prompt_layers import (
    critique_move_block,
    peer_move_block,
    reasoning_press_block,
)
from app.turn_logic import (
    build_turn_context,
    low_agreeableness_reminder_pairs,
    prepare_peer_user_message,
    prepare_student_user_message,
    should_student_stall,
)

logger = logging.getLogger(__name__)

DEFAULT_GROUP_PROFILES = ["alex", "maya", "jordan"]
# PST game starter scenario (2-student conflicting claims). Phase 2 PST API reuses this.
FE_STARTER_PROFILES = ["maya", "jordan"]
DEFAULT_GROUP_TASK_ID = "phone_plans_linear_01"
MIN_GROUP_PROFILES = 2
MAX_GROUP_PROFILES = 3
ALLOWED_ORCHESTRATION = frozenset(
    {"teacher_directed", "teacher_directed_self_select"}
)

ReplySource = Literal["teacher", "peer"]


@dataclass
class StudentSlot:
    profile_id: str
    student: Student
    mastery_state: Dict[str, int] = field(default_factory=dict)
    behavior_profile: Dict = field(default_factory=dict)
    error_playbook: Dict = field(default_factory=dict)
    misconceptions: List[dict] = field(default_factory=list)
    predicted_behavior: str = ""
    student_hist: List[dict] = field(default_factory=list)
    turn_history: List[dict] = field(default_factory=list)
    scaffold_boost: Dict[str, float] = field(default_factory=dict)
    stall_active: bool = False
    eval_log: List[dict] = field(default_factory=list)
    concept_states: Dict[str, str] = field(default_factory=dict)
    mode_caps: Dict[str, str] = field(default_factory=dict)
    scaffolded_constructs: List[str] = field(default_factory=list)
    last_scaffold_events: List[dict] = field(default_factory=list)
    last_active_concepts: List[dict] = field(default_factory=list)
    last_turn_mode: str = ""
    receptivity: float = 0.5
    active_misc_id: str = ""
    implied_plan: str = ""
    claim_id: str = ""
    misc_pinned: bool = False
    mastery_fingerprint: str = ""
    solved_crossover: bool = False
    stated_equations: bool = False
    acknowledged_fee: bool = False
    crossover_intuition: bool = False
    table_critique_peer: bool = False
    fee_press_count: int = 0


@dataclass
class GroupSession:
    session_id: str
    task_text: str
    task_metadata: Dict = field(default_factory=dict)
    task_resolution: Dict = field(default_factory=dict)
    students: Dict[str, StudentSlot] = field(default_factory=dict)
    profile_ids: List[str] = field(default_factory=list)
    transcript: List[dict] = field(default_factory=list)
    turn: int = 0
    started: bool = False
    orchestration: str = "teacher_directed_self_select"
    enable_self_select: bool = True
    top_p_concepts: List[str] = field(default_factory=list)
    display_names: Dict[str, str] = field(default_factory=dict)
    last_observers: List[str] = field(default_factory=list)
    last_speak_decisions: List[dict] = field(default_factory=list)
    last_may_speak: List[str] = field(default_factory=list)
    last_mode: str = ""
    last_teacher_message: str = ""
    last_learning_events: List[dict] = field(default_factory=list)
    last_teaching_warning: bool = False
    last_pedagogical_move: str = ""
    orchestration_log: List[dict] = field(default_factory=list)
    fight_phase: str = "fight"
    breakthrough_bullets: List[str] = field(default_factory=list)
    listen_restraint: Dict[str, int] = field(default_factory=dict)
    focus_speaker: str = ""
    crossover_unlocked: bool = False


@dataclass
class _SlotPromptAdapter:
    """Thin adapter so build_turn_context can read slot + group fields."""

    student: Student
    predicted_behavior: str
    top_p_concepts: List[str]
    task_text: str
    error_playbook: Dict
    behavior_profile: Dict
    scaffold_boost: Dict[str, float]
    task_metadata: Dict
    mastery_state: Dict[str, int]
    messages: List[dict] = field(default_factory=list)


def _bank_task_description(task_id: str) -> str:
    for task in load_task_metadata().get("tasks", []):
        if task.get("task_id") == task_id:
            return (task.get("description") or "").strip()
    return ""


def _validate_profile_ids(profile_ids: Sequence[str]) -> List[str]:
    ids = [str(p).strip().lower() for p in profile_ids]
    n = len(ids)
    if n < MIN_GROUP_PROFILES or n > MAX_GROUP_PROFILES:
        raise ValueError(
            f"profile_ids must contain {MIN_GROUP_PROFILES}–{MAX_GROUP_PROFILES} "
            f"profiles (got {n})"
        )
    if len(set(ids)) != n:
        raise ValueError("profile_ids must be distinct")
    known = set(list_lp_profile_ids())
    unknown = [p for p in ids if p not in known]
    if unknown:
        raise ValueError(f"Unknown profile_id(s): {', '.join(unknown)}")
    return ids


def _display_name_for(profile_id: str) -> str:
    try:
        return get_profile(profile_id).get("name") or profile_id.capitalize()
    except KeyError:
        return profile_id.capitalize()


def _mastery_fingerprint(mastery: dict) -> str:
    return ",".join(f"{k}:{int(v)}" for k, v in sorted((mastery or {}).items()))


def _apply_slot_misconceptions(
    slot: StudentSlot,
    misconceptions: List[dict],
    task_meta: dict,
    *,
    pin: bool = False,
) -> None:
    slot.misconceptions = list(misconceptions or [])
    primary = slot.misconceptions[0] if slot.misconceptions else {}
    slot.active_misc_id = str(primary.get("id") or "")
    slot.implied_plan = str(primary.get("implied_plan") or "")
    slot.claim_id = CLAIM_IDS.get(slot.profile_id, slot.implied_plan)
    slot.misc_pinned = bool(pin)
    slot.error_playbook = misconceptions_to_playbook(slot.misconceptions[:1])
    bp = slot.behavior_profile or {}
    bp["active_misc_id"] = slot.active_misc_id
    bp["implied_plan"] = slot.implied_plan
    bp["claim_id"] = slot.claim_id
    barrier = PST_GROUP_BARRIERS.get(slot.profile_id) or {}
    bp["discourse_role"] = primary.get("discourse_role") or barrier.get(
        "discourse_role"
    )
    if primary.get("error_type"):
        bp["error_types"] = [primary["error_type"]]
        bp["error_labels"] = list(bp["error_types"])
    slot.behavior_profile = bp
    _ = task_meta


class GroupSessionStore:
    def __init__(self):
        self._sessions: Dict[str, GroupSession] = {}
        self._student_cache: Dict[str, Student] = {}

    def prepare_student(self, profile_id: str) -> Student:
        if profile_id not in self._student_cache:
            self._student_cache[profile_id] = make_student(profile_id)
        return deepcopy(self._student_cache[profile_id])

    def create(
        self,
        profile_ids: Optional[Sequence[str]] = None,
        task_text: Optional[str] = None,
        task_id: Optional[str] = None,
        orchestration: str = "teacher_directed_self_select",
        enable_self_select: Optional[bool] = None,
    ) -> GroupSession:
        ids = _validate_profile_ids(profile_ids or DEFAULT_GROUP_PROFILES)
        orch = (orchestration or "teacher_directed_self_select").strip()
        if orch not in ALLOWED_ORCHESTRATION:
            raise ValueError(
                "orchestration must be 'teacher_directed' or "
                "'teacher_directed_self_select' "
                f"(got {orchestration!r})"
            )
        self_select = (
            orch == "teacher_directed_self_select"
            if enable_self_select is None
            else bool(enable_self_select)
        )
        orch = (
            "teacher_directed_self_select"
            if self_select
            else "teacher_directed"
        )

        resolved_task_id = (task_id or "").strip() or DEFAULT_GROUP_TASK_ID
        text = (task_text or "").strip()
        if not text:
            text = _bank_task_description(resolved_task_id) or DEFAULT_TASK

        resolved = resolve_task_metadata(
            text, task_id=resolved_task_id, force_lp=True
        )
        task_meta = resolved["task_metadata"]
        required = task_meta.get("required_constructs") or []

        kg_store.seed_schema()
        students: Dict[str, StudentSlot] = {}
        display_names: Dict[str, str] = {}

        peer_picks: List[dict] = []
        for pid in ids:
            student = self.prepare_student(pid)
            display_names[pid] = _display_name_for(pid)
            mastery = kg_store.init_mastery_state(
                student.student_id, student.construct_mastery
            )
            target = route_behavior(mastery, task_meta, student)
            if is_phone_plans_task(task_meta.get("task_id") or resolved_task_id):
                misconceptions = retrieve_for_group_fight(
                    pid,
                    task_meta,
                    barrier=PST_GROUP_BARRIERS.get(pid),
                    peer_picks=peer_picks,
                    k=3,
                )
            else:
                misconceptions = retrieve_misconceptions(target, task_meta, k=3)
            target = route_behavior(mastery, task_meta, student, misconceptions)
            behavior_profile = behavior_target_to_profile(target, task_meta, student)
            slot = StudentSlot(
                profile_id=pid,
                student=student,
                mastery_state=mastery,
                behavior_profile=behavior_profile,
                predicted_behavior=behavior_summary_text(target, task_meta),
                concept_states={
                    cid: MASTERY_LABELS.get(mastery.get(cid, 0), "Unknown")
                    for cid in required
                },
                receptivity=receptivity_for_profile(pid),
                mastery_fingerprint=_mastery_fingerprint(mastery),
            )
            _apply_slot_misconceptions(
                slot,
                misconceptions,
                task_meta,
                pin=is_phone_plans_task(task_meta.get("task_id") or ""),
            )
            if slot.misconceptions:
                peer_picks.append(slot.misconceptions[0])
            students[pid] = slot

        group = GroupSession(
            session_id=str(uuid.uuid4()),
            task_text=text,
            task_metadata=task_meta,
            task_resolution=resolved["resolution"],
            students=students,
            profile_ids=ids,
            orchestration=orch,
            enable_self_select=self_select,
            top_p_concepts=list(required),
            display_names=display_names,
        )
        self._sessions[group.session_id] = group
        return group

    def get(self, session_id: str) -> GroupSession:
        group = self._sessions.get(session_id)
        if group is None:
            raise KeyError(f"Group session not found: {session_id}")
        return group

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def _refresh_behavior(self, group: GroupSession, slot: StudentSlot) -> None:
        target = route_behavior(
            slot.mastery_state,
            group.task_metadata,
            slot.student,
            slot.misconceptions,
            mode_caps=slot.mode_caps,
        )
        mastery_moved = _mastery_fingerprint(slot.mastery_state) != (
            slot.mastery_fingerprint or ""
        )
        re_retrieve = (not slot.misc_pinned) or mastery_moved
        if re_retrieve:
            if is_phone_plans_task((group.task_metadata or {}).get("task_id") or ""):
                peer_picks = [
                    s.misconceptions[0]
                    for pid, s in group.students.items()
                    if pid != slot.profile_id and s.misconceptions
                ]
                misconceptions = retrieve_for_group_fight(
                    slot.profile_id,
                    group.task_metadata,
                    barrier=PST_GROUP_BARRIERS.get(slot.profile_id),
                    peer_picks=peer_picks,
                    k=3,
                )
            else:
                misconceptions = retrieve_misconceptions(
                    target, group.task_metadata, k=3
                )
            _apply_slot_misconceptions(
                slot,
                misconceptions,
                group.task_metadata,
                pin=slot.misc_pinned
                and not mastery_moved
                and not suppress_fight_misconception(group, slot),
            )
            slot.mastery_fingerprint = _mastery_fingerprint(slot.mastery_state)
        target = route_behavior(
            slot.mastery_state,
            group.task_metadata,
            slot.student,
            slot.misconceptions,
            mode_caps=slot.mode_caps,
        )
        slot.behavior_profile = behavior_target_to_profile(
            target, group.task_metadata, slot.student
        )
        _apply_slot_misconceptions(
            slot,
            slot.misconceptions,
            group.task_metadata,
            pin=slot.misc_pinned,
        )
        slot.predicted_behavior = behavior_summary_text(target, group.task_metadata)
        required = group.task_metadata.get("required_constructs") or []
        slot.concept_states = {
            cid: MASTERY_LABELS.get(slot.mastery_state.get(cid, 0), "Unknown")
            for cid in required
        }

    def _record_opportunity(
        self, group: GroupSession, result, *, refresh: bool = True
    ) -> None:
        group.last_pedagogical_move = result.move
        if result.teaching_warning:
            group.last_teaching_warning = True
        event = result.to_dict()
        group.last_learning_events.append(event)
        if refresh:
            for gain in result.gains:
                slot = group.students.get(gain.profile_id)
                if slot:
                    self._refresh_behavior(group, slot)

    def _update_lp_state(
        self,
        group: GroupSession,
        slot: StudentSlot,
        student_reply: str,
        turn_mode: str,
        stall_active: bool = False,
        *,
        source: ReplySource = "teacher",
        help_seeking: bool = False,
    ) -> None:
        lp = slot.student.learning_profile
        if lp:
            slot.scaffold_boost = decay_scaffold_boost(
                slot.scaffold_boost, lp.forget_rate
            )

        from app.config import GROUP_TRACK_B

        # Track B owns group mastery via scaffold opportunities (not reply quality).
        if GROUP_TRACK_B:
            return

        # Legacy path when Track B is off: skip peer-sourced reply updates.
        if source == "peer":
            return

        if turn_mode in ("social", "off_topic") or stall_active or help_seeking:
            return
        if not mastery_update_turn(turn_mode):
            return

        required = group.task_metadata.get("required_constructs") or []
        quality = kg_store.infer_reply_quality(
            student_reply,
            group.task_metadata,
            slot.behavior_profile,
            mastery_state=slot.mastery_state,
        )
        slot.mastery_state = kg_store.update_observation(
            slot.student.student_id,
            slot.mastery_state,
            required,
            quality,
            learning_profile=lp,
            scaffold_boost=slot.scaffold_boost,
        )
        slot.student.construct_mastery = dict(slot.mastery_state)
        self._refresh_behavior(group, slot)

    def _transcript_classify_context(self, group: GroupSession) -> List[dict]:
        return [
            {
                "role": (
                    "teacher"
                    if e.get("speaker_type") == "teacher"
                    else "student"
                ),
                "content": e.get("content", ""),
            }
            for e in group.transcript[-6:]
        ]

    def _wrap_user_message(
        self,
        group: GroupSession,
        profile_id: str,
        core_msg: str,
        *,
        multi_round: bool = False,
        include_transcript: bool = True,
    ) -> str:
        return wrap_user_message(
            group,
            profile_id,
            core_msg,
            multi_round=multi_round,
            include_transcript=include_transcript,
        )

    def student_reply(
        self,
        group: GroupSession,
        profile_id: str,
        incoming: str,
        source: ReplySource,
        turn: int,
        *,
        allow_help_seeking: bool = True,
        multi_round: bool = False,
        game_flow: bool = False,
        expected_move: Optional[str] = None,
        peer_prev_name: str = "",
    ) -> Tuple[str, bool, List[dict], str, bool, bool]:
        slot = group.students[profile_id]
        vague_warning = False
        turn_role = ""

        if source == "peer":
            # Peer path: no teacher classification / stall; prompt as math turn.
            turn_mode = "peer"
            prompt_turn_mode = "math_scaffold"
            stall_active = False
            slot.stall_active = False
            slot.last_scaffold_events = []
            slot.last_turn_mode = turn_mode

            core_msg = incoming
            if not str(core_msg).startswith("["):
                core_msg = prepare_peer_user_message("Peer", incoming)
            user_msg = self._wrap_user_message(
                group,
                profile_id,
                core_msg,
                multi_round=multi_round,
                include_transcript=False,
            )
        else:
            context = self._transcript_classify_context(group)
            turn_mode = classify_teacher_turn(
                incoming, group.task_text, context=context
            )
            slot.last_turn_mode = turn_mode
            stall_active = should_student_stall(
                slot.turn_history, turn_mode, incoming
            )
            slot.stall_active = stall_active
            prompt_turn_mode = turn_mode

            if turn_mode == "math_scaffold" and scaffold_is_worked_example(incoming):
                primary = slot.behavior_profile.get("primary_construct", "")
                required = group.task_metadata.get("required_constructs") or []
                targets = [primary] if primary else required[:1]
                for cid in targets:
                    if cid and cid not in slot.scaffolded_constructs:
                        slot.scaffolded_constructs.append(cid)
                    if cid:
                        slot.mode_caps = cap_mode(
                            slot.mode_caps, cid, "PARTIAL_ATTEMPT_THEN_STUCK"
                        )

            lp = slot.student.learning_profile
            if (
                lp
                and turn_mode in ("math_scaffold", "math_eval", "mixed")
                and not stall_active
            ):
                events = detect_scaffold(
                    incoming, group.task_metadata, turn_mode
                )
                slot.last_scaffold_events = [e.to_dict() for e in events]
                primary = slot.behavior_profile.get("primary_construct", "")
                effective_gain = (
                    lp.effective_gain(primary) if primary else lp.gain_rate
                )
                slot.scaffold_boost = apply_scaffold_boost(
                    slot.scaffold_boost,
                    events,
                    effective_gain,
                    lp.scaffold_sensitivity,
                )
            else:
                slot.last_scaffold_events = []

            if (
                source == "teacher"
                and profile_id == "jordan"
                and detect_teacher_fee_press(incoming)
            ):
                slot.fee_press_count = int(getattr(slot, "fee_press_count", 0) or 0) + 1
                for cid in ("linear_relationship", "rate_comparison"):
                    cur = float(slot.scaffold_boost.get(cid, 0.0) or 0.0)
                    slot.scaffold_boost[cid] = min(1.0, cur + 0.3)

            turn_role = ""
            if source == "teacher":
                task_id = (group.task_metadata or {}).get("task_id", "")
                turn_role = (
                    detect_peer_critique_turn(
                        incoming,
                        profile_id,
                        task_id,
                        group.transcript,
                    )
                    or ""
                )
                if not turn_role and detect_teacher_reasoning_press(
                    incoming,
                    addressed_profile=profile_id,
                    task_id=task_id,
                ):
                    turn_role = "reasoning_press"

            core_msg, vague_warning = prepare_student_user_message(
                incoming, turn_mode, stall_active=stall_active
            )
            user_msg = self._wrap_user_message(
                group, profile_id, core_msg, multi_round=False
            )

        if (
            slot.student.Agreeableness == "Low"
            and (source == "peer" or turn_mode not in ("social", "off_topic"))
        ):
            for pair in low_agreeableness_reminder_pairs(turn):
                slot.student_hist.append(pair)

        slot.student_hist.append({"role": "user", "content": user_msg})

        active_misc = slot.misconceptions[0] if slot.misconceptions else None
        misc_id = (active_misc or {}).get("id", "")
        construct_id = slot.behavior_profile.get("primary_construct", "")
        fight_phase = getattr(group, "fight_phase", "fight")
        misc_suppressed = suppress_fight_misconception(group, slot)
        if misc_suppressed:
            active_misc = None
        retrieved_instance = None
        bp = slot.behavior_profile
        skip_error_anchor = (
            bp.get("behavior_mode") == "NORMAL_ERROR_PROFILE"
            and bp.get("student_stack_level", 0) >= 3
        )
        error_anchor = None
        if source == "teacher":
            if (
                active_misc
                and not skip_error_anchor
                and not misc_suppressed
                and turn_mode
                in (
                    "math_scaffold",
                    "math_eval",
                    "mixed",
                    "vague_acknowledgment",
                    "vague_directive",
                )
            ):
                retrieved_instance = retrieve_error_instance(
                    incoming,
                    misc_id,
                    construct_id=construct_id,
                    seed_claim=FE_SEED_LINES.get(profile_id, ""),
                )
            error_anchor = build_error_anchor(retrieved_instance, active_misc)
            if skip_error_anchor or stall_active:
                error_anchor = None
        else:
            # Peer: still allow a light error anchor from active misconception.
            if active_misc and not skip_error_anchor and not misc_suppressed:
                error_anchor = build_error_anchor(None, active_misc)

        adapter = _SlotPromptAdapter(
            student=slot.student,
            predicted_behavior=slot.predicted_behavior,
            top_p_concepts=list(group.top_p_concepts),
            task_text=group.task_text,
            error_playbook=slot.error_playbook,
            behavior_profile=slot.behavior_profile,
            scaffold_boost=slot.scaffold_boost,
            task_metadata=group.task_metadata,
            mastery_state=slot.mastery_state,
            messages=self._transcript_classify_context(group),
        )

        # Only teacher-addressed speakers may help-seek; at most one per round
        # (caller sets allow_help_seeking=False after the first help-seek).
        help_allow = allow_help_seeking and source == "teacher"
        if slot.misc_pinned and (slot.behavior_profile or {}).get(
            "behavior_mode"
        ) != "CONFUSED_HELPSEEKING":
            help_allow = False
        help_decision = decide_help_seeking(
            slot.student,
            slot.behavior_profile,
            turn_mode=prompt_turn_mode if source == "teacher" else "math_scaffold",
            stall_active=stall_active if source == "teacher" else False,
            turns_since_last=turns_since_help_seek(slot.turn_history, turn),
            has_misconceptions=bool(slot.misconceptions),
            allow=help_allow,
        )
        slot.behavior_profile["help_seeking"] = help_decision.triggered

        _, s_sys, active, _ = build_turn_context(
            adapter,
            incoming,
            turn_mode=prompt_turn_mode,
            stall_active=stall_active if source == "teacher" else False,
            error_anchor=error_anchor,
        )

        others = [
            group.display_names.get(pid, pid.capitalize())
            for pid in group.profile_ids
            if pid != profile_id
        ]
        self_name = group.display_names.get(profile_id, profile_id.capitalize())
        s_sys = (
            s_sys
            + "\n\n"
            + awareness_block(
                self_name,
                others,
                multi_round=multi_round and not game_flow,
                game_flow=game_flow,
            )
        )
        # Peer math reactions keep claim lock; social/stall teacher turns do not.
        use_claim_lock = bool(slot.implied_plan) and should_use_claim_lock(
            implied_plan=slot.implied_plan,
            phase=fight_phase,
            source=source,
            turn_mode=turn_mode,
            stall_active=stall_active,
        )
        if use_claim_lock:
            block = claim_lock_block(
                phase=fight_phase,
                plan=slot.implied_plan,
                profile_id=profile_id,
                acknowledged_fee=bool(getattr(slot, "acknowledged_fee", False)),
                fee_press_count=int(getattr(slot, "fee_press_count", 0) or 0),
            )
            s_sys = s_sys + "\n\n[CLAIM LOCK]\n" + block
        memory = breakthrough_memory_block(group, profile_id)
        if memory:
            s_sys = s_sys + "\n\n" + memory
        prior_peer_name = ""
        prior_peer_text = ""
        if source == "peer" or turn_role == "peer_critique":
            for entry in reversed(group.transcript or []):
                if entry.get("speaker_type") != "student":
                    continue
                sid = entry.get("speaker_id") or ""
                if sid == profile_id:
                    continue
                prior_peer_name = group.display_names.get(sid, sid.capitalize())
                prior_peer_text = entry.get("content") or ""
                break
        if source == "peer" and expected_move:
            prev = peer_prev_name
            if not prev:
                m = re.match(r"\[([^\]]+)\]:", incoming or "")
                if m:
                    prev = m.group(1)
            move_block = peer_move_block(
                expected_move,
                prev_name=prev,
            )
            if move_block:
                s_sys = s_sys + "\n\n" + move_block
        elif turn_role == "peer_critique":
            if not expected_move:
                expected_move = "evidence"
            critique_block = critique_move_block(
                peer_name=prior_peer_name or "Jordan",
                peer_text=prior_peer_text,
            )
            if critique_block:
                s_sys = s_sys + "\n\n" + critique_block
        if source == "teacher" and is_phone_plans_task(
            (group.task_metadata or {}).get("task_id", "")
        ):
            nudge = late_turn_nudge_block(
                profile_id=profile_id,
                teacher_msg=incoming if source == "teacher" else "",
                crossover_unlocked=bool(getattr(group, "crossover_unlocked", False)),
                fight_phase=fight_phase,
            )
            if nudge:
                s_sys = s_sys + "\n\n" + nudge
        if help_decision.triggered:
            s_sys = (
                s_sys
                + "\n\n"
                + build_help_seeking_prompt_block(
                    slot.student,
                    slot.behavior_profile,
                    group_context=True,
                )
            )
        slot.last_active_concepts = active

        soften_jordan_fee_press_mode(group, slot)

        warrant_exp = None
        if turn_role == "reasoning_press":
            warrant_exp = warrant_expectation(
                profile_id=profile_id,
                behavior_profile=slot.behavior_profile,
                student=slot.student,
                misconception=active_misc,
                fight_phase=fight_phase,
                crossover_unlocked=bool(getattr(group, "crossover_unlocked", False)),
                acknowledged_fee=bool(getattr(slot, "acknowledged_fee", False)),
                fee_press_count=int(getattr(slot, "fee_press_count", 0) or 0),
            )
            rp_block = reasoning_press_block(warrant_exp, student=slot.student)
            if rp_block:
                s_sys = s_sys + "\n\n" + rp_block

        max_tokens = student_max_tokens_for_turn(turn_mode)
        behavior_for_gen = {
            **slot.behavior_profile,
            "profile_id": profile_id,
            "crossover_unlocked": bool(getattr(group, "crossover_unlocked", False)),
            "stall_active": stall_active if source == "teacher" else False,
            "help_seeking": help_decision.triggered,
        }
        if warrant_exp is not None:
            behavior_for_gen["_warrant_expectation"] = warrant_exp
        # Refinement treats non-social/off_topic as math; "peer" is fine.
        gen_turn_mode = (
            prompt_turn_mode if source == "peer" else turn_mode
        )
        peer_revisions = 2 if (multi_round or game_flow) else 1
        active_misc = (
            slot.misconceptions[0] if slot.misconceptions else None
        )
        is_peer = source == "peer" or turn_role == "peer_critique"
        targets = None
        if GENERATION_DEBUG:
            targets = build_expected_pack(
                behavior_for_gen,
                student=slot.student,
                misconception=active_misc,
                turn_mode=gen_turn_mode,
                is_peer_turn=is_peer,
                turn_role=turn_role,
            )
            expected_eb = targets.get("learning") or ""
        else:
            expected_eb = learning_expected_behavior_text(
                behavior_for_gen,
                misconception=active_misc,
                turn_mode=gen_turn_mode,
                is_peer_turn=is_peer,
                student=slot.student,
                turn_role=turn_role,
            )

        llm_params = (
            resolved_student_chat_params(max_tokens) if GENERATION_DEBUG else None
        )
        prior_replies = [
            h.get("student_reply", "")
            for h in (slot.turn_history or [])
            if h.get("student_reply")
        ]
        gen = generate_with_refinement(
            s_sys,
            slot.student_hist,
            behavior_for_gen,
            max_tokens=max_tokens,
            max_revisions=peer_revisions,
            turn_mode=gen_turn_mode,
            scaffold_boost=slot.scaffold_boost,
            student=slot.student,
            misconception=active_misc if use_claim_lock and not misc_suppressed else None,
            is_peer_turn=is_peer,
            implied_plan=slot.implied_plan if use_claim_lock else "",
            prior_replies=prior_replies,
            fight_phase=fight_phase,
            expected_move=expected_move or "",
            turn_role=turn_role,
            prior_peer_name=prior_peer_name,
            prior_peer_text=prior_peer_text,
        )
        reply_text = gen.final_reply

        is_question = help_decision.triggered or reply_looks_like_question(reply_text)
        slot.behavior_profile.pop("help_seeking", None)

        generation_debug = None
        if GENERATION_DEBUG and targets is not None:
            generation_debug = build_generation_debug(
                system_prompt=s_sys,
                history=slot.student_hist,
                turn_mode=gen_turn_mode,
                is_peer_turn=is_peer,
                behavior_profile=behavior_for_gen,
                targets=targets,
                personality_export=build_personality_export(
                    slot.student, profile_id
                ),
                misconception=active_misc,
                draft=gen.draft,
                final_reply=gen.final_reply,
                revisions=gen.revisions,
                critic=gen.critic,
                llm_params=llm_params,
                retrieval={
                    "active_misc_id": slot.active_misc_id,
                    "implied_plan": slot.implied_plan,
                    "misc_ids": [m.get("id") for m in slot.misconceptions],
                    "pinned": slot.misc_pinned,
                },
            )

        slot.student_hist.append({"role": "assistant", "content": reply_text})
        eval_record = build_eval_turn_record(
            turn=turn,
            teacher_message=incoming,
            reply=reply_text,
            teacher_label=turn_mode,
            stall_active=stall_active if source == "teacher" else False,
            profile_id=profile_id,
            task_id=(group.task_metadata or {}).get("task_id", ""),
            behavior_profile=slot.behavior_profile,
            task_metadata=group.task_metadata,
            misconceptions=slot.misconceptions,
            mastery_state=slot.mastery_state,
            expected_behavior=expected_eb,
            vague_warning=vague_warning,
            help_seeking=help_decision.triggered,
            is_question=is_question,
            generation_debug=generation_debug,
        )
        eval_record["task_text"] = group.task_text
        eval_record["problem_type"] = (group.task_metadata or {}).get(
            "problem_type", ""
        )
        eval_record["source"] = source
        eval_record["help_seek_decision"] = help_decision.to_dict()
        slot.eval_log.append(eval_record)
        slot.turn_history.append(
            {
                "turn": turn,
                "teacher_label": turn_mode,
                "student_reply": reply_text,
                "stall_active": stall_active if source == "teacher" else False,
                "behavior_mode": slot.behavior_profile.get("behavior_mode"),
                "source": source,
                "help_seeking": help_decision.triggered,
                "is_question": is_question,
            }
        )

        group.transcript.append(
            {
                "turn": turn,
                "speaker_type": "student",
                "speaker_id": profile_id,
                "content": reply_text,
                "help_seeking": help_decision.triggered,
                "is_question": is_question,
            }
        )

        self._update_lp_state(
            group,
            slot,
            reply_text,
            turn_mode if source == "teacher" else "math_eval",
            stall_active if source == "teacher" else False,
            source=source,
            help_seeking=help_decision.triggered,
        )
        on_student_reply(
            group,
            profile_id,
            reply_text,
            teacher_msg=incoming if source == "teacher" else "",
        )
        from app.config import GROUP_TRACK_B

        if GROUP_TRACK_B and source == "peer" and reply_text.strip():
            peer_opp = apply_scaffold_opportunity(
                group,
                reply_text,
                speaker_id=profile_id,
                source="peer",
            )
            self._record_opportunity(group, peer_opp)
        trim_private_hist(slot)
        return (
            reply_text,
            vague_warning,
            active,
            turn_mode,
            help_decision.triggered,
            is_question,
        )

    def _reset_group_state(self, group: GroupSession) -> None:
        """Clear per-session turn state without generating opener replies."""
        for slot in group.students.values():
            slot.student_hist = []
            slot.turn_history = []
            slot.eval_log = []
            slot.stall_active = False
            slot.scaffold_boost = {}
            slot.last_scaffold_events = []
            slot.last_active_concepts = []
            slot.last_turn_mode = ""
            slot.solved_crossover = False
            slot.stated_equations = False
            slot.acknowledged_fee = False
            slot.crossover_intuition = False
            slot.table_critique_peer = False
            slot.fee_press_count = 0

        group.transcript = []
        group.turn = 0
        group.started = True
        group.last_observers = []
        group.last_speak_decisions = []
        group.last_may_speak = []
        group.last_mode = ""
        group.last_teacher_message = ""
        group.last_learning_events = []
        group.last_teaching_warning = False
        group.last_pedagogical_move = ""
        group.orchestration_log = []
        group.fight_phase = "fight"
        group.breakthrough_bullets = []
        group.listen_restraint = {}
        group.focus_speaker = ""
        group.crossover_unlocked = False

    def _ensure_started(self, group: GroupSession) -> None:
        """Activate session on first teacher message (game integration: no /start call)."""
        if not group.started:
            self._reset_group_state(group)

    def start(
        self, group: GroupSession
    ) -> Tuple[GroupSession, List[dict], bool, List[dict], str]:
        self._reset_group_state(group)

        opener = TEACHER_OPENER_TEMPLATE.format(task=group.task_text)
        group.turn = 1
        group.transcript.append(
            {
                "turn": 1,
                "speaker_type": "teacher",
                "speaker_id": "teacher",
                "content": opener,
            }
        )

        if is_fe_starter_fight(
            group.profile_ids, (group.task_metadata or {}).get("task_id") or ""
        ):
            replies: List[dict] = []
            speakers: List[str] = []
            for pid in group.profile_ids:
                seed = FE_SEED_LINES.get(pid)
                if not seed:
                    continue
                speakers.append(pid)
                group.transcript.append(
                    {
                        "turn": 1,
                        "speaker_type": "student",
                        "speaker_id": pid,
                        "content": seed,
                    }
                )
                slot = group.students[pid]
                slot.student_hist = [
                    {"role": "user", "content": opener},
                    {"role": "assistant", "content": seed},
                ]
                replies.append(
                    {
                        "speaker_id": pid,
                        "content": seed,
                        "help_seeking": False,
                        "is_question": False,
                    }
                )
            group.last_speak_decisions = [
                {
                    "profile_id": pid,
                    "will_speak": pid in speakers,
                    "score": 1.0 if pid in speakers else 0.0,
                    "reason": (
                        "frozen FE seed opener"
                        if pid in speakers
                        else "not selected for opener"
                    ),
                }
                for pid in group.profile_ids
            ]
            group.orchestration_log.append(
                {
                    "turn": group.turn,
                    "mode": "direct",
                    "speakers": speakers,
                    "observers": [],
                    "speak_decisions": list(group.last_speak_decisions),
                }
            )
            first_slot = group.students[speakers[0]] if speakers else None
            active = (
                first_slot.last_active_concepts if first_slot else []
            )
            return group, replies, False, active, "math_scaffold"

        first_id = group.profile_ids[0]
        reply, vague_warning, active, turn_mode, help_seeking, is_question = (
            self.student_reply(group, first_id, opener, "teacher", group.turn)
        )
        replies = [
            {
                "speaker_id": first_id,
                "content": reply,
                "help_seeking": help_seeking,
                "is_question": is_question,
            }
        ]
        group.last_speak_decisions = [
            {
                "profile_id": pid,
                "will_speak": pid == first_id,
                "score": 1.0 if pid == first_id else 0.0,
                "reason": (
                    "predictable first-roster opener"
                    if pid == first_id
                    else "not selected for opener"
                ),
            }
            for pid in group.profile_ids
        ]
        group.orchestration_log.append(
            {
                "turn": group.turn,
                "mode": "direct",
                "speakers": [first_id],
                "observers": [],
                "speak_decisions": list(group.last_speak_decisions),
            }
        )
        return group, replies, vague_warning, active, turn_mode

    def respond(
        self,
        group: GroupSession,
        teacher_msg: str,
        *,
        on_event=None,
    ) -> Tuple[GroupSession, List[dict], bool, List[dict], str, List[dict], str]:
        """Run a teacher turn. Optional ``on_event(dict)`` streams progress to the client.

        Event types:
          - speak_plan: speakers / may_speak / decisions (before any LLM reply)
          - thinking: profile_ids currently generating
          - reply: one finalized student reply (primary or game_flow)
          - done: emitted by the HTTP layer after this returns
        """
        self._ensure_started(group)

        def emit(event: dict) -> None:
            if on_event is None:
                return
            try:
                on_event(event)
            except Exception:
                logger.exception("respond on_event failed")

        group.turn += 1
        turn = group.turn
        group.transcript.append(
            {
                "turn": turn,
                "speaker_type": "teacher",
                "speaker_id": "teacher",
                "content": teacher_msg,
            }
        )

        on_teacher_message(group, teacher_msg)

        resolution = resolve_speakers_with_meta(
            teacher_msg,
            group,
            enable_self_select=group.enable_self_select,
        )
        speakers = apply_listen_restraint(
            group, list(resolution.speakers), teacher_msg
        )
        if speakers != resolution.speakers:
            kept = set(speakers)
            resolution.decisions = [
                {
                    **d,
                    "will_speak": d.get("profile_id") in kept
                    and d.get("will_speak", False),
                    "reason": (
                        f"{d.get('reason', '')}; listen restraint"
                        if d.get("profile_id") not in kept
                        and d.get("will_speak")
                        else d.get("reason", "")
                    ),
                }
                for d in resolution.decisions
            ]
        group.last_observers = list(resolution.constraints.must_not_speak)
        group.last_speak_decisions = list(resolution.decisions)
        group.last_may_speak = list(resolution.constraints.may_speak)
        group.last_mode = resolution.constraints.mode
        group.last_teacher_message = teacher_msg
        group.last_teaching_warning = False
        group.last_learning_events = []
        group.last_pedagogical_move = ""

        from app.config import GROUP_TRACK_B

        if GROUP_TRACK_B:
            teacher_opp = apply_scaffold_opportunity(
                group, teacher_msg, source="teacher"
            )
            self._record_opportunity(group, teacher_opp)

        # Early UX: who might speak — frontend can show thinking bubbles immediately.
        thinking_ids = list(
            dict.fromkeys(
                list(speakers)
                + [
                    d["profile_id"]
                    for d in group.last_speak_decisions
                    if d.get("will_speak") and d.get("profile_id")
                ]
                + list(resolution.constraints.may_speak or [])
            )
        )
        emit(
            {
                "type": "speak_plan",
                "speakers": list(speakers),
                "must_speak": list(resolution.constraints.must_speak),
                "may_speak": list(resolution.constraints.may_speak),
                "observers": list(group.last_observers),
                "speak_decisions": list(group.last_speak_decisions),
                "mode": resolution.constraints.mode,
                "thinking": thinking_ids,
            }
        )
        if thinking_ids:
            emit({"type": "thinking", "profile_ids": thinking_ids})

        replies: List[dict] = []
        vague_warning = False
        active: List[dict] = []
        turn_mode = ""
        help_seek_used = False

        # Classify once so multi-speaker primary waves stay social (not peer-math).
        teacher_turn_mode = classify_teacher_turn(
            teacher_msg,
            task_text=(group.task_metadata or {}).get("task_text", ""),
            context=list(group.transcript or []),
        )
        social_wave = teacher_turn_mode in ("social", "off_topic")

        reason_map = {
            d["profile_id"]: d.get("reason", "selected speaker")
            for d in group.last_speak_decisions
            if d.get("will_speak") and d.get("profile_id")
        }

        for i, pid in enumerate(speakers):
            emit({"type": "thinking", "profile_ids": [pid]})
            if i == 0 or social_wave:
                incoming = teacher_msg
                source: ReplySource = "teacher"
            else:
                prev = replies[-1]
                prev_name = group.display_names.get(
                    prev["speaker_id"], prev["speaker_id"].capitalize()
                )
                incoming = prepare_peer_user_message(prev_name, prev["content"])
                source = "peer"

            reply, vw, act, tm, help_seeking, is_question = self.student_reply(
                group,
                pid,
                incoming,
                source,
                turn,
                allow_help_seeking=not help_seek_used,
            )
            if help_seeking:
                help_seek_used = True
            reply_obj = {
                "speaker_id": pid,
                "content": reply,
                "help_seeking": help_seeking,
                "is_question": is_question,
                "source": "primary",
                "speak_reason": reason_map.get(pid, "primary speaker"),
            }
            replies.append(reply_obj)
            emit({"type": "reply", "kind": "primary", "reply": reply_obj})
            if i == 0:
                vague_warning = vw
                active = act
                turn_mode = tm
            elif not turn_mode:
                turn_mode = tm

        if social_wave and not turn_mode:
            turn_mode = teacher_turn_mode
        elif social_wave:
            # Prefer the pre-wave classify so logs stay social even if a slot drifted.
            turn_mode = teacher_turn_mode

        from app.config import GROUP_CONSTRAINT_PARSER

        group.orchestration_log.append(
            {
                "turn": turn,
                "teacher_message": teacher_msg,
                "constraint_parser": GROUP_CONSTRAINT_PARSER,
                "mode": resolution.constraints.mode,
                "turn_mode": turn_mode,
                "vague_warning": vague_warning,
                "must_speak": list(resolution.constraints.must_speak),
                "may_speak": list(resolution.constraints.may_speak),
                "speakers": list(speakers),
                "observers": list(group.last_observers),
                "speak_decisions": list(group.last_speak_decisions),
                "pedagogical_move": group.last_pedagogical_move,
                "teaching_warning": group.last_teaching_warning,
                "learning_events": list(group.last_learning_events),
                "receptivity": {
                    pid: group.students[pid].receptivity for pid in group.profile_ids
                },
            }
        )

        multi_round_replies: List[dict] = []
        stop_reason = ""
        from app.config import (
            GROUP_GAME_FLOW,
            GROUP_GAME_FLOW_MAX_ROUNDS,
            GROUP_MULTI_ROUND,
            GROUP_PEER_MAX_ROUNDS,
        )

        def _on_game_flow_item(item: dict) -> None:
            if item.get("_event") == "thinking":
                pids = [p for p in (item.get("profile_ids") or []) if p]
                if pids:
                    emit({"type": "thinking", "profile_ids": pids})
                return
            emit({"type": "reply", "kind": "game_flow", "reply": item})

        # Social/off-topic: greetings must not trigger peer claim restatement.
        if social_wave:
            stop_reason = "social_or_off_topic"
            group.orchestration_log[-1]["stop_reason"] = stop_reason
        elif GROUP_GAME_FLOW and GROUP_GAME_FLOW_MAX_ROUNDS > 0:
            from app.game_flow_orchestrator import run_game_flow_continuation

            multi_round_replies, stop_reason = run_game_flow_continuation(
                group,
                store=self,
                initial_reply_count=len(replies),
                on_reply=_on_game_flow_item if on_event else None,
            )
            if stop_reason:
                group.orchestration_log[-1]["stop_reason"] = stop_reason
        elif (
            GROUP_MULTI_ROUND
            and resolution.constraints.mode in ("discuss", "open", "critique")
            and GROUP_PEER_MAX_ROUNDS > 0
        ):
            from app.group_graph import run_peer_continuation

            multi_round_replies, stop_reason = run_peer_continuation(
                group, max_rounds=GROUP_PEER_MAX_ROUNDS, store=self
            )
            for reply_obj in multi_round_replies:
                emit({"type": "reply", "kind": "peer", "reply": reply_obj})

        decay_listen_restraint(group)

        return (
            group,
            replies,
            vague_warning,
            active,
            turn_mode,
            multi_round_replies,
            stop_reason,
        )

    def advance(
        self, group: GroupSession
    ) -> Tuple[GroupSession, List[dict], str]:
        """Run another peer-continuation block without a new teacher prompt."""
        self._ensure_started(group)
        from app.config import (
            GROUP_GAME_FLOW,
            GROUP_MULTI_ROUND,
            GROUP_PEER_MAX_ROUNDS,
            GROUP_PEER_NUDGE_ROUNDS,
        )

        if not group.transcript:
            raise ValueError("No transcript yet; send a teacher message first.")

        group.last_learning_events = []
        group.last_teaching_warning = False

        if GROUP_GAME_FLOW:
            from app.game_flow_orchestrator import run_game_flow_continuation

            nudge = GROUP_PEER_NUDGE_ROUNDS if GROUP_PEER_NUDGE_ROUNDS > 0 else GROUP_PEER_MAX_ROUNDS
            multi_round_replies, stop_reason = run_game_flow_continuation(
                group,
                store=self,
                initial_reply_count=0,
                max_rounds=nudge,
            )
            return group, multi_round_replies, stop_reason

        from app.group_graph import run_peer_continuation

        if not GROUP_MULTI_ROUND:
            raise ValueError("Multi-round peer continuation is disabled.")
        if GROUP_PEER_NUDGE_ROUNDS < 1 and GROUP_PEER_MAX_ROUNDS < 1:
            raise ValueError("Peer nudge/max rounds must be >= 1 to advance.")

        nudge = GROUP_PEER_NUDGE_ROUNDS if GROUP_PEER_NUDGE_ROUNDS > 0 else GROUP_PEER_MAX_ROUNDS
        multi_round_replies, stop_reason = run_peer_continuation(
            group, max_rounds=nudge, store=self
        )
        return group, multi_round_replies, stop_reason

    def export(self, group: GroupSession) -> dict:
        per_student = {}
        for pid in group.profile_ids:
            slot = group.students[pid]
            per_student[pid] = {
                "profile_id": pid,
                "display_name": group.display_names.get(pid, pid),
                "behavior_mode": slot.behavior_profile.get("behavior_mode"),
                "mastery_state": dict(slot.mastery_state),
                "predicted_behavior": slot.predicted_behavior,
                "eval_log": list(slot.eval_log),
                "personality": build_personality_export(slot.student, pid),
                "big_five": (DEMO_STUDENTS.get(pid) or {}).get("big_five") or {},
            }
        return {
            "session_id": group.session_id,
            "source": "group_session",
            "orchestration": group.orchestration,
            "enable_self_select": group.enable_self_select,
            "profile_ids": list(group.profile_ids),
            "task_text": group.task_text,
            "task_metadata": group.task_metadata,
            "task_resolution": group.task_resolution,
            "turn": group.turn,
            "transcript": list(group.transcript),
            "orchestration_log": list(group.orchestration_log),
            "students": per_student,
        }


group_session_store = GroupSessionStore()
