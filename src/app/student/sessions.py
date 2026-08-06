import logging
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.student.behavior_router import (
    behavior_summary_text,
    behavior_target_to_profile,
    cap_mode,
    route_behavior,
)
from app.student.data import DEFAULT_TASK, make_student
from app.knowledge.kg_config import MASTERY_LABELS, get_construct_def
from app.knowledge.task_tagger import resolve_task_metadata
from app.knowledge import store as kg_store
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
from app.knowledge.turn_classifier import classify_teacher_turn, mastery_update_turn
from app.knowledge.turn_logic import (
    build_turn_context,
    low_agreeableness_reminder_pairs,
    prepare_student_user_message,
    should_student_stall,
)
from app.student.help_seeking import (
    build_help_seeking_prompt_block,
    decide_help_seeking,
    reply_looks_like_question,
    turns_since_help_seek,
)
from app.eval_log import build_eval_turn_record

logger = logging.getLogger(__name__)

TEACHER_OPENER_TEMPLATE = (
    "Here's the problem: {task}\n"
    "Take a look at it — what do you think the first step should be?"
)


@dataclass
class ChatSession:
    session_id: str
    student: Student
    task_text: str
    predicted_behavior: str
    top_p_concepts: List[str]
    profile_id: str = ""
    concept_states: Dict[str, str] = field(default_factory=dict)
    error_playbook: Dict = field(default_factory=dict)
    behavior_profile: Dict = field(default_factory=dict)
    task_metadata: Dict = field(default_factory=dict)
    teacher_hist: List[dict] = field(default_factory=list)
    student_hist: List[dict] = field(default_factory=list)
    messages: List[dict] = field(default_factory=list)
    turn: int = 0
    started: bool = False
    last_active_concepts: List[dict] = field(default_factory=list)
    mastery_state: Dict[str, int] = field(default_factory=dict)
    misconceptions: List[dict] = field(default_factory=list)
    task_resolution: Dict = field(default_factory=dict)
    last_turn_mode: str = ""
    scaffold_boost: Dict[str, float] = field(default_factory=dict)
    last_scaffold_events: List[dict] = field(default_factory=list)
    turn_history: List[dict] = field(default_factory=list)
    stall_active: bool = False
    mode_caps: Dict[str, str] = field(default_factory=dict)
    scaffolded_constructs: List[str] = field(default_factory=list)
    eval_log: List[dict] = field(default_factory=list)


class SessionStore:
    def __init__(self):
        self._sessions: Dict[str, ChatSession] = {}
        self._student_cache: Dict[str, Student] = {}

    def prepare_student(self, profile_id: str) -> Student:
        if profile_id not in self._student_cache:
            self._student_cache[profile_id] = make_student(profile_id)
        return deepcopy(self._student_cache[profile_id])

    def create(
        self,
        profile_id: str,
        task_text: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> ChatSession:
        student = self.prepare_student(profile_id)
        task = (task_text or "").strip() or DEFAULT_TASK
        resolved = resolve_task_metadata(task, task_id=task_id or "", force_lp=True)
        session = self._create_lp_session(
            student,
            task,
            resolved["task_metadata"],
            task_resolution=resolved["resolution"],
            profile_id=profile_id,
        )
        self._sessions[session.session_id] = session
        return session

    def _refresh_behavior(self, session: ChatSession) -> None:
        target = route_behavior(
            session.mastery_state,
            session.task_metadata,
            session.student,
            session.misconceptions,
            mode_caps=session.mode_caps,
        )
        session.misconceptions = retrieve_misconceptions(
            target, session.task_metadata, k=3
        )
        target = route_behavior(
            session.mastery_state,
            session.task_metadata,
            session.student,
            session.misconceptions,
            mode_caps=session.mode_caps,
        )
        session.behavior_profile = behavior_target_to_profile(
            target, session.task_metadata, session.student
        )
        session.error_playbook = misconceptions_to_playbook(session.misconceptions)
        session.predicted_behavior = behavior_summary_text(target, session.task_metadata)
        required = session.task_metadata.get("required_constructs") or []
        session.concept_states = {
            cid: MASTERY_LABELS.get(session.mastery_state.get(cid, 0), "Unknown")
            for cid in required
        }

    def _create_lp_session(
        self,
        student: Student,
        task: str,
        task_meta: dict,
        task_resolution: Optional[dict] = None,
        profile_id: str = "",
    ) -> ChatSession:
        kg_store.seed_schema()
        mastery = kg_store.init_mastery_state(student.student_id, student.construct_mastery)
        target = route_behavior(mastery, task_meta, student)
        misconceptions = retrieve_misconceptions(target, task_meta, k=3)
        target = route_behavior(mastery, task_meta, student, misconceptions)
        playbook = misconceptions_to_playbook(misconceptions)
        behavior_profile = behavior_target_to_profile(target, task_meta, student)
        required = task_meta.get("required_constructs") or []

        session = ChatSession(
            session_id=str(uuid.uuid4()),
            student=student,
            task_text=task,
            predicted_behavior=behavior_summary_text(target, task_meta),
            top_p_concepts=required,
            profile_id=profile_id,
            concept_states={
                cid: MASTERY_LABELS.get(mastery.get(cid, 0), "Unknown") for cid in required
            },
            error_playbook=playbook,
            behavior_profile=behavior_profile,
            task_metadata=task_meta,
            mastery_state=mastery,
            misconceptions=misconceptions,
            task_resolution=task_resolution or {},
        )
        return session

    def get(self, session_id: str) -> ChatSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        return session

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def _student_reply(
        self,
        session: ChatSession,
        teacher_utt: str,
        turn: int,
    ) -> Tuple[str, bool, List[dict], str, bool, bool]:
        vague_warning = False
        context = [
            {"role": m.get("role", "unknown"), "content": m.get("content", "")}
            for m in session.messages[-6:]
        ]
        turn_mode = classify_teacher_turn(
            teacher_utt, session.task_text, context=context
        )
        session.last_turn_mode = turn_mode
        stall_active = should_student_stall(
            session.turn_history, turn_mode, teacher_utt
        )
        session.stall_active = stall_active

        if turn_mode == "math_scaffold" and scaffold_is_worked_example(teacher_utt):
            primary = session.behavior_profile.get("primary_construct", "")
            required = session.task_metadata.get("required_constructs") or []
            targets = [primary] if primary else required[:1]
            for cid in targets:
                if cid and cid not in session.scaffolded_constructs:
                    session.scaffolded_constructs.append(cid)
                if cid:
                    session.mode_caps = cap_mode(
                        session.mode_caps, cid, "PARTIAL_ATTEMPT_THEN_STUCK"
                    )

        lp = session.student.learning_profile
        if (
            lp
            and turn_mode in ("math_scaffold", "math_eval", "mixed")
            and not stall_active
        ):
            events = detect_scaffold(teacher_utt, session.task_metadata, turn_mode)
            session.last_scaffold_events = [e.to_dict() for e in events]
            primary = session.behavior_profile.get("primary_construct", "")
            effective_gain = lp.effective_gain(primary) if primary else lp.gain_rate
            session.scaffold_boost = apply_scaffold_boost(
                session.scaffold_boost,
                events,
                effective_gain,
                lp.scaffold_sensitivity,
            )
        else:
            session.last_scaffold_events = []

        if session.student.Agreeableness == "Low" and turn_mode not in ("social", "off_topic"):
            for pair in low_agreeableness_reminder_pairs(turn):
                session.student_hist.append(pair)

        user_msg, vague_warning = prepare_student_user_message(
            teacher_utt, turn_mode, stall_active=stall_active
        )
        session.student_hist.append({"role": "user", "content": user_msg})

        active_misc = session.misconceptions[0] if session.misconceptions else None
        misc_id = (active_misc or {}).get("id", "")
        construct_id = session.behavior_profile.get("primary_construct", "")
        retrieved_instance = None
        bp = session.behavior_profile
        skip_error_anchor = (
            bp.get("behavior_mode") == "NORMAL_ERROR_PROFILE"
            and bp.get("student_stack_level", 0) >= 3
        )
        if (
            active_misc
            and not skip_error_anchor
            and turn_mode in (
                "math_scaffold",
                "math_eval",
                "mixed",
                "vague_acknowledgment",
                "vague_directive",
            )
        ):
            retrieved_instance = retrieve_error_instance(
                teacher_utt, misc_id, construct_id=construct_id
            )
        error_anchor = build_error_anchor(retrieved_instance, active_misc)
        if skip_error_anchor or stall_active:
            error_anchor = None

        help_decision = decide_help_seeking(
            session.student,
            session.behavior_profile,
            turn_mode=turn_mode,
            stall_active=stall_active,
            turns_since_last=turns_since_help_seek(session.turn_history, turn),
            has_misconceptions=bool(session.misconceptions),
        )
        session.behavior_profile["help_seeking"] = help_decision.triggered

        _, s_sys, active, _ = build_turn_context(
            session,
            teacher_utt,
            turn_mode=turn_mode,
            stall_active=stall_active,
            error_anchor=error_anchor,
        )
        if help_decision.triggered:
            s_sys = (
                s_sys
                + "\n\n"
                + build_help_seeking_prompt_block(
                    session.student, session.behavior_profile
                )
            )
        session.last_active_concepts = active

        max_tokens = 300 if turn_mode in ("social", "off_topic") else 200
        behavior_for_gen = {
            **session.behavior_profile,
            "stall_active": stall_active,
            "help_seeking": help_decision.triggered,
        }
        student_reply, _ = generate_with_refinement(
            s_sys,
            session.student_hist,
            behavior_for_gen,
            max_tokens=max_tokens,
            max_revisions=1,
            turn_mode=turn_mode,
            scaffold_boost=session.scaffold_boost,
        )

        is_question = help_decision.triggered or reply_looks_like_question(
            student_reply
        )
        session.behavior_profile.pop("help_seeking", None)

        session.student_hist.append({"role": "assistant", "content": student_reply})
        eval_record = build_eval_turn_record(
            turn=turn,
            teacher_message=teacher_utt,
            reply=student_reply,
            teacher_label=turn_mode,
            stall_active=stall_active,
            profile_id=session.profile_id,
            task_id=(session.task_metadata or {}).get("task_id", ""),
            behavior_profile=session.behavior_profile,
            task_metadata=session.task_metadata,
            misconceptions=session.misconceptions,
            mastery_state=session.mastery_state,
            vague_warning=vague_warning,
            help_seeking=help_decision.triggered,
            is_question=is_question,
        )
        eval_record["task_text"] = session.task_text
        eval_record["problem_type"] = (session.task_metadata or {}).get("problem_type", "")
        eval_record["help_seek_decision"] = help_decision.to_dict()
        session.eval_log.append(eval_record)
        session.turn_history.append(
            {
                "turn": turn,
                "teacher_label": turn_mode,
                "student_reply": student_reply,
                "stall_active": stall_active,
                "behavior_mode": session.behavior_profile.get("behavior_mode"),
                "help_seeking": help_decision.triggered,
                "is_question": is_question,
            }
        )
        self._update_lp_state(
            session,
            student_reply,
            turn_mode,
            stall_active,
            help_seeking=help_decision.triggered,
        )
        return (
            student_reply,
            vague_warning,
            active,
            turn_mode,
            help_decision.triggered,
            is_question,
        )

    def _update_lp_state(
        self,
        session: ChatSession,
        student_reply: str,
        turn_mode: str,
        stall_active: bool = False,
        help_seeking: bool = False,
    ) -> None:
        lp = session.student.learning_profile
        if lp:
            session.scaffold_boost = decay_scaffold_boost(
                session.scaffold_boost, lp.forget_rate
            )

        if turn_mode in ("social", "off_topic") or stall_active or help_seeking:
            return
        if not mastery_update_turn(turn_mode):
            return

        required = session.task_metadata.get("required_constructs") or []
        quality = kg_store.infer_reply_quality(
            student_reply,
            session.task_metadata,
            session.behavior_profile,
            mastery_state=session.mastery_state,
        )
        session.mastery_state = kg_store.update_observation(
            session.student.student_id,
            session.mastery_state,
            required,
            quality,
            learning_profile=lp,
            scaffold_boost=session.scaffold_boost,
        )
        session.student.construct_mastery = dict(session.mastery_state)
        self._refresh_behavior(session)

    def start(self, session: ChatSession) -> Tuple[ChatSession, bool, List[dict], str]:
        session.teacher_hist = []
        session.student_hist = []
        session.messages = []
        session.turn = 0
        session.started = True
        session.eval_log = []

        opener = TEACHER_OPENER_TEMPLATE.format(task=session.task_text)
        session.teacher_hist.append({"role": "assistant", "content": opener})
        session.turn = 1

        (
            student_reply,
            vague_warning,
            active,
            turn_mode,
            help_seeking,
            is_question,
        ) = self._student_reply(session, opener, session.turn)
        session.teacher_hist.append({"role": "user", "content": student_reply})
        session.messages.append({"role": "teacher", "content": opener, "turn": 1})
        session.messages.append(
            {
                "role": "student",
                "content": student_reply,
                "turn": 1,
                "help_seeking": help_seeking,
                "is_question": is_question,
            }
        )
        return session, vague_warning, active, turn_mode

    def respond(
        self, session: ChatSession, teacher_msg: str
    ) -> Tuple[ChatSession, bool, List[dict], str]:
        if not session.started:
            raise ValueError("Session not started. Call start first.")

        session.teacher_hist.append({"role": "assistant", "content": teacher_msg})
        session.turn += 1
        (
            student_reply,
            vague_warning,
            active,
            turn_mode,
            help_seeking,
            is_question,
        ) = self._student_reply(session, teacher_msg, session.turn)
        session.teacher_hist.append({"role": "user", "content": student_reply})
        session.messages.append(
            {"role": "teacher", "content": teacher_msg, "turn": session.turn}
        )
        session.messages.append(
            {
                "role": "student",
                "content": student_reply,
                "turn": session.turn,
                "help_seeking": help_seeking,
                "is_question": is_question,
            }
        )
        return session, vague_warning, active, turn_mode


session_store = SessionStore()


def kg_summary(student: Student, limit: int = 12) -> List[dict]:
    items = []
    for cid, level in list(student.construct_mastery.items())[:limit]:
        cdef = get_construct_def(cid) or {}
        items.append(
            {
                "concept": cdef.get("label", cid),
                "state": MASTERY_LABELS.get(level, "Unknown"),
                "construct_id": cid,
                "mastery_level": level,
            }
        )
    return items
