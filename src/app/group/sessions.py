"""Teacher-facilitated group sessions (Model A): 1 teacher + 3 LP students.

Duplicates/adapts the 1:1 reply pipeline from sessions.py without modifying it.
"""

from __future__ import annotations

import logging
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple

from app.student.behavior_router import (
    behavior_summary_text,
    behavior_target_to_profile,
    cap_mode,
    route_behavior,
)
from app.student.data import DEFAULT_TASK, get_profile, make_student
from app.student.demo_students import list_lp_profile_ids
from app.eval_log import build_eval_turn_record
from app.group.context import awareness_block, trim_private_hist, wrap_user_message
from app.group.orchestrator import resolve_speakers_with_meta
from app.student.help_seeking import (
    build_help_seeking_prompt_block,
    decide_help_seeking,
    reply_looks_like_question,
    turns_since_help_seek,
)
from app.knowledge.kg_config import MASTERY_LABELS
from app.knowledge import store as kg_store
from app.student.learning_receptivity import (
    apply_scaffold_opportunity,
    receptivity_for_profile,
)
from app.knowledge.misconception_store import (
    build_error_anchor,
    misconceptions_to_playbook,
    retrieve_error_instance,
    retrieve_misconceptions,
)
from app.student.models import Student
from app.student.response_refinement import generate_with_refinement
from app.knowledge.scaffold_detector import (
    apply_scaffold_boost,
    decay_scaffold_boost,
    detect_scaffold,
    scaffold_is_worked_example,
)
from app.student.sessions import TEACHER_OPENER_TEMPLATE
from app.knowledge.task_tagger import load_task_metadata, resolve_task_metadata
from app.knowledge.turn_classifier import classify_teacher_turn, mastery_update_turn
from app.knowledge.turn_logic import (
    build_turn_context,
    low_agreeableness_reminder_pairs,
    prepare_peer_user_message,
    prepare_student_user_message,
    should_student_stall,
)

logger = logging.getLogger(__name__)

DEFAULT_GROUP_PROFILES = ["jordan", "sam", "alex"]
DEFAULT_GROUP_TASK_ID = "phone_plans_linear_01"
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
    if len(ids) != 3:
        raise ValueError("profile_ids must contain exactly 3 profiles")
    if len(set(ids)) != 3:
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

        for pid in ids:
            student = self.prepare_student(pid)
            display_names[pid] = _display_name_for(pid)
            mastery = kg_store.init_mastery_state(
                student.student_id, student.construct_mastery
            )
            target = route_behavior(mastery, task_meta, student)
            misconceptions = retrieve_misconceptions(target, task_meta, k=3)
            target = route_behavior(mastery, task_meta, student, misconceptions)
            playbook = misconceptions_to_playbook(misconceptions)
            behavior_profile = behavior_target_to_profile(target, task_meta, student)
            students[pid] = StudentSlot(
                profile_id=pid,
                student=student,
                mastery_state=mastery,
                behavior_profile=behavior_profile,
                error_playbook=playbook,
                misconceptions=misconceptions,
                predicted_behavior=behavior_summary_text(target, task_meta),
                concept_states={
                    cid: MASTERY_LABELS.get(mastery.get(cid, 0), "Unknown")
                    for cid in required
                },
                receptivity=receptivity_for_profile(pid),
            )

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
        slot.misconceptions = retrieve_misconceptions(
            target, group.task_metadata, k=3
        )
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
        slot.error_playbook = misconceptions_to_playbook(slot.misconceptions)
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
    ) -> str:
        return wrap_user_message(
            group, profile_id, core_msg, multi_round=multi_round
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
    ) -> Tuple[str, bool, List[dict], str, bool, bool]:
        slot = group.students[profile_id]
        vague_warning = False

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
                group, profile_id, core_msg, multi_round=multi_round
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
                    incoming, misc_id, construct_id=construct_id
                )
            error_anchor = build_error_anchor(retrieved_instance, active_misc)
            if skip_error_anchor or stall_active:
                error_anchor = None
        else:
            # Peer: still allow a light error anchor from active misconception.
            if active_misc and not skip_error_anchor:
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
        help_decision = decide_help_seeking(
            slot.student,
            slot.behavior_profile,
            turn_mode=prompt_turn_mode if source == "teacher" else "math_scaffold",
            stall_active=stall_active if source == "teacher" else False,
            turns_since_last=turns_since_help_seek(slot.turn_history, turn),
            has_misconceptions=bool(slot.misconceptions),
            allow=allow_help_seeking and source == "teacher",
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
            + awareness_block(self_name, others, multi_round=multi_round)
        )
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

        max_tokens = 300 if turn_mode in ("social", "off_topic") else 200
        behavior_for_gen = {
            **slot.behavior_profile,
            "stall_active": stall_active if source == "teacher" else False,
            "help_seeking": help_decision.triggered,
        }
        # Refinement treats non-social/off_topic as math; "peer" is fine.
        gen_turn_mode = (
            prompt_turn_mode if source == "peer" else turn_mode
        )
        reply_text, _ = generate_with_refinement(
            s_sys,
            slot.student_hist,
            behavior_for_gen,
            max_tokens=max_tokens,
            max_revisions=1,
            turn_mode=gen_turn_mode,
            scaffold_boost=slot.scaffold_boost,
        )

        is_question = help_decision.triggered or reply_looks_like_question(reply_text)
        slot.behavior_profile.pop("help_seeking", None)

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
            vague_warning=vague_warning,
            help_seeking=help_decision.triggered,
            is_question=is_question,
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

    def start(
        self, group: GroupSession
    ) -> Tuple[GroupSession, List[dict], bool, List[dict], str]:
        for slot in group.students.values():
            slot.student_hist = []
            slot.turn_history = []
            slot.eval_log = []
            slot.stall_active = False
            slot.scaffold_boost = {}
            slot.last_scaffold_events = []
            slot.last_active_concepts = []
            slot.last_turn_mode = ""

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
        self, group: GroupSession, teacher_msg: str
    ) -> Tuple[GroupSession, List[dict], bool, List[dict], str, List[dict], str]:
        if not group.started:
            raise ValueError("Group session not started. Call start first.")

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

        resolution = resolve_speakers_with_meta(
            teacher_msg,
            group,
            enable_self_select=group.enable_self_select,
        )
        speakers = resolution.speakers
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

        replies: List[dict] = []
        vague_warning = False
        active: List[dict] = []
        turn_mode = ""
        help_seek_used = False

        for i, pid in enumerate(speakers):
            if i == 0:
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
            replies.append(
                {
                    "speaker_id": pid,
                    "content": reply,
                    "help_seeking": help_seeking,
                    "is_question": is_question,
                }
            )
            if i == 0:
                vague_warning = vw
                active = act
                turn_mode = tm
            elif not turn_mode:
                turn_mode = tm

        from app.config import GROUP_CONSTRAINT_PARSER

        group.orchestration_log.append(
            {
                "turn": turn,
                "teacher_message": teacher_msg,
                "constraint_parser": GROUP_CONSTRAINT_PARSER,
                "mode": resolution.constraints.mode,
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
        from app.config import GROUP_MULTI_ROUND, GROUP_PEER_MAX_ROUNDS

        if (
            GROUP_MULTI_ROUND
            and resolution.constraints.mode in ("discuss", "open", "critique")
            and GROUP_PEER_MAX_ROUNDS > 0
        ):
            from app.group.graph import run_peer_continuation

            multi_round_replies, stop_reason = run_peer_continuation(
                group, max_rounds=GROUP_PEER_MAX_ROUNDS, store=self
            )

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
        """Run another LangGraph peer-continuation block without a new teacher prompt."""
        if not group.started:
            raise ValueError("Group session not started. Call start first.")
        from app.config import GROUP_MULTI_ROUND, GROUP_PEER_MAX_ROUNDS, GROUP_PEER_NUDGE_ROUNDS
        from app.group.graph import run_peer_continuation

        if not GROUP_MULTI_ROUND:
            raise ValueError("Multi-round peer continuation is disabled.")
        if GROUP_PEER_NUDGE_ROUNDS < 1 and GROUP_PEER_MAX_ROUNDS < 1:
            raise ValueError("Peer nudge/max rounds must be >= 1 to advance.")
        if not group.transcript:
            raise ValueError("No transcript yet; send a teacher message first.")

        group.last_learning_events = []
        group.last_teaching_warning = False

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
