"""Fight-session progression: breakthrough memory, nuanced claim lock, listen restraint.

Phone-plans Maya/Jordan fights start in opposing claims (``fight``). After scaffold
uptake (equations, crossover x=100, teacher confirmation) the phase moves to
``nuanced`` so students keep personality tension without forgetting what they solved.
"""

from __future__ import annotations

import re
from typing import Dict, List, Literal, Optional, Sequence, Tuple

from app.fight_stance import is_fe_starter_fight, is_phone_plans_task
from app.turn_classifier import claim_lock_active

FightPhase = Literal["fight", "nuanced", "resolved"]

CROSSOVER_VALUE_RE = re.compile(
    r"\b(?:x\s*=\s*100|100\s*texts?|cross(?:over)?\s+(?:at|point|is)\s*100)\b",
    re.I,
)
EQUATION_BOTH_RE = re.compile(
    r"0\.?10\s*x|0\.1\s*x|20\s*\+\s*0\.?10",
    re.I,
)
PLAN_B_EQ_RE = re.compile(r"0\.?30\s*x|0\.3\s*x", re.I)
TEACHER_CONFIRM_RE = re.compile(
    r"\b(?:correct|yes+!?|exactly|that(?:'s| is) right|good job|well done)\b",
    re.I,
)
TEACHER_CROSSOVER_CUE_RE = re.compile(
    r"\b(?:set\s+(?:them|the\s+equations?)\s+equal|equate|same\s+cost|crossover|"
    r"break[- ]?even|when\s+is\s+each\s+plan\s+better|which\s+plan\s+is\s+better|"
    r"when\s+are\s+they\s+equal|flip\s+point)\b",
    re.I,
)
FEE_ACK_RE = re.compile(
    r"\$?\s*20\b|\bfee\b|at\s+the\s+start|at\s+first|starts?\s+higher|monthly\s+fee",
    re.I,
)
CROSSOVER_INTUITION_RE = re.compile(
    r"\b(?:eventually\s+catch(?:es)?\s+up|after\s+enough\s+texts?|"
    r"more\s+texts?\s+.{0,30}\bcheaper|cross(?:es)?\s+over|flip\s+point|"
    r"at\s+some\s+point\s+.{0,30}\bplan\s+a)\b",
    re.I,
)
TABLE_CRITIQUE_PEER_RE = re.compile(
    r"\b(?:your\s+(?:table|numbers?)|jordan|rate.?only|ignor(?:e|ing)\s+(?:the\s+)?\$?20|"
    r"starting\s+(?:fee|cost)|monthly\s+fee|wrong\s+about)\b",
    re.I,
)
TEACHER_FEE_PRESS_RE = re.compile(
    r"\$?\s*20\b|\bfee\b|won'?t\s+matter|why\s+do\s+you\s+think\s+20|"
    r"starting\s+(?:fee|cost)|monthly\s+fee",
    re.I,
)
PEER_CRITIQUE_TEACHER_RE = re.compile(
    r"\b(?:wrong|what'?s\s+wrong|find\s+what|check\s+him|check\s+her|"
    r"pointed\s+out|critique|help\b.+?\bunderstand|using\s+the\s+table|"
    r"why\b.+?\bwrong)\b",
    re.I,
)
MAYA_CRITIQUE_SIGNAL_RE = re.compile(
    r"\$?\s*20\b|\bfee\b|starting\s+(?:fee|cost)|monthly|rate.?only|"
    r"ignor(?:e|ing)\s+(?:the\s+)?\$?20",
    re.I,
)
TEACHER_CLOSING_CUE_RE = re.compile(
    r"\b(?:what'?s\s+the\s+answer|both\s+of\s+you\s+say|summarize|"
    r"when\s+is\s+each\s+plan\s+better|which\s+plan\s+is\s+better)\b",
    re.I,
)
RATE_ONLY_CLAIM_RE = re.compile(
    r"\b0\.?10\b|\b0\.1\b|\brate\b.{0,40}\b(cheaper|less|smaller|better)\b|"
    r"\b(cheaper|less|smaller)\b.{0,40}\brate\b",
    re.I,
)

LISTEN_PATTERNS = (
    re.compile(r"\blet\s+(\w+)\s+(?:speak|talk|answer|tell)", re.I),
    re.compile(r"\b(\w+)\s+(?:you\s+)?(?:just\s+)?listen\b", re.I),
    re.compile(r"\b(?:only\s+)?listen\s+to\s+(\w+)\b", re.I),
    re.compile(r"\b(\w+)\s+what\s+(?:do|does)\s+your\b", re.I),
)

TUTOR_PEER_MARKERS = (
    "divide both sides",
    "subtract",
    "move the x",
    "combine like terms",
    "you need to",
    "what you do is",
    "step one is",
    "first you",
)


def _name_to_profile(name: str, display_names: Dict[str, str]) -> str:
    raw = (name or "").strip().lower()
    if not raw:
        return ""
    for pid, dname in display_names.items():
        if str(dname).lower() == raw or str(pid).lower() == raw:
            return pid
    return ""


def parse_listen_directive(
    teacher_msg: str,
    profile_ids: Sequence[str],
    display_names: Dict[str, str],
) -> Tuple[Dict[str, int], str]:
    """Return profiles to hold back (turns) and the profile the teacher wants to hear."""
    text = (teacher_msg or "").strip()
    if not text:
        return {}, ""
    lower = text.lower()
    focus = ""
    restraint: Dict[str, int] = {}

    for pat in LISTEN_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        target = _name_to_profile(m.group(1), display_names)
        if target:
            focus = target
            break

    if "let " in lower and " speak" in lower:
        for pid in profile_ids:
            dname = display_names.get(pid, pid).lower()
            if dname in lower and "let " + dname in lower:
                focus = pid
                break

    if focus:
        for pid in profile_ids:
            if pid != focus:
                restraint[pid] = max(restraint.get(pid, 0), 2)

    if re.search(r"\b(?:jordan|maya|alex)\s+.{0,30}\b(?:listen|quiet|hold on|wait)\b", lower):
        for pid in profile_ids:
            dname = display_names.get(pid, "").lower()
            if dname and dname in lower and "listen" in lower:
                if pid != focus:
                    restraint[pid] = max(restraint.get(pid, 0), 2)

    return restraint, focus


def detect_fee_acknowledgment(text: str) -> bool:
    return bool(FEE_ACK_RE.search(text or ""))


def detect_crossover_intuition(text: str) -> bool:
    """Verbal crossover hint without algebra (blocks unprompted x=100 elsewhere)."""
    t = text or ""
    if CROSSOVER_VALUE_RE.search(t):
        return False
    return bool(CROSSOVER_INTUITION_RE.search(t))


def detect_table_critique_peer(text: str) -> bool:
    return bool(TABLE_CRITIQUE_PEER_RE.search(text or ""))


def detect_teacher_fee_press(teacher_msg: str) -> bool:
    return bool(TEACHER_FEE_PRESS_RE.search(teacher_msg or ""))


def maya_critique_is_rich(text: str) -> bool:
    return bool(MAYA_CRITIQUE_SIGNAL_RE.search(text or ""))


def _recent_jordan_lines(transcript: Sequence[dict], limit: int = 2) -> List[str]:
    lines: List[str] = []
    for entry in reversed(transcript or []):
        if entry.get("speaker_type") != "student":
            continue
        if (entry.get("speaker_id") or "").lower() != "jordan":
            continue
        content = (entry.get("content") or "").strip()
        if content:
            lines.append(content)
        if len(lines) >= limit:
            break
    return lines


def detect_peer_critique_turn(
    teacher_msg: str,
    addressed_profile: str,
    task_id: str,
    transcript: Sequence[dict],
) -> Optional[str]:
    """Return ``peer_critique`` when teacher asks Maya/Alex to critique Jordan's claim."""
    if not is_phone_plans_task(task_id):
        return None
    pid = (addressed_profile or "").strip().lower()
    if pid not in ("maya", "alex"):
        return None
    msg = teacher_msg or ""
    if not PEER_CRITIQUE_TEACHER_RE.search(msg):
        return None
    jordan_lines = _recent_jordan_lines(transcript, limit=2)
    if not jordan_lines:
        return None
    recent_jordan = jordan_lines[0]
    if RATE_ONLY_CLAIM_RE.search(recent_jordan):
        return "peer_critique"
    lower = msg.lower()
    names_maya = pid in lower or "maya" in lower or "alex" in lower
    if names_maya and (
        re.search(r"\bhelp\b.+?\bunderstand", lower)
        or "using the table" in lower
        or re.search(r"\bfind\s+what", lower)
        or re.search(r"\bpointed\s+out", lower)
    ):
        return "peer_critique"
    return None


def soften_jordan_fee_press_mode(group, slot) -> None:
    """Allow partial fee uptake utterances while Jordan still defends Plan A."""
    if not is_phone_plans_task((group.task_metadata or {}).get("task_id", "")):
        return
    if str(slot.profile_id).lower() != "jordan":
        return
    if not (
        getattr(slot, "acknowledged_fee", False)
        or int(getattr(slot, "fee_press_count", 0) or 0) >= 2
    ):
        return
    mode = (slot.behavior_profile or {}).get("behavior_mode", "")
    if mode == "WRONG":
        slot.behavior_profile["behavior_mode"] = "PARTIAL_ATTEMPT_THEN_STUCK"
        slot.behavior_profile["likely_correctness"] = "partially correct"


def late_turn_nudge_block(
    *,
    profile_id: str,
    teacher_msg: str,
    crossover_unlocked: bool,
    fight_phase: str,
) -> str:
    """Prompt slice when teacher asks crossover/closing synthesis questions."""
    if not crossover_unlocked and not TEACHER_CLOSING_CUE_RE.search(teacher_msg or ""):
        return ""
    if not (
        TEACHER_CROSSOVER_CUE_RE.search(teacher_msg or "")
        or TEACHER_CLOSING_CUE_RE.search(teacher_msg or "")
    ):
        return ""
    pid = str(profile_id).lower()
    phase = (fight_phase or "fight").strip().lower()
    if pid == "jordan":
        return (
            "[LATE TURN — crossover/closing]\n"
            "Teacher is asking when each plan wins. You may state both plan equations "
            "(20 + 0.10x and 0.30x) and which plan is cheaper at low vs high texts. "
            "Sound like a kid summarizing what the group figured out."
        )
    if pid == "maya":
        if phase == "resolved":
            return (
                "[LATE TURN — closing]\n"
                "Explain in kid words when Plan B wins early and Plan A wins later; "
                "you can connect your table to the crossover."
            )
        return (
            "[LATE TURN — crossover]\n"
            "Teacher unlocked comparing plans across usage. Say when each plan is better "
            "using your table numbers — hedged, not a full tutor walkthrough."
        )
    return ""


def detect_crossover_in_text(text: str) -> bool:
    return bool(CROSSOVER_VALUE_RE.search(text or ""))


def detect_equations_in_text(text: str) -> bool:
    t = text or ""
    return bool(EQUATION_BOTH_RE.search(t) and PLAN_B_EQ_RE.search(t))


def phase_after_breakthroughs(
    *,
    crossover: bool,
    equations: bool,
    teacher_confirmed: bool,
    crossover_unlocked: bool = False,
    fee_uptake: bool = False,
) -> FightPhase:
    if (
        crossover_unlocked
        and crossover
        and equations
        and (teacher_confirmed or crossover)
    ):
        return "resolved"
    if crossover or equations:
        return "nuanced"
    if crossover_unlocked and fee_uptake:
        return "nuanced"
    return "fight"


def claim_lock_block(
    *,
    phase: FightPhase,
    plan: str,
    profile_id: str,
    acknowledged_fee: bool = False,
    fee_press_count: int = 0,
) -> str:
    plan = (plan or "").strip().upper()
    if plan not in ("A", "B"):
        return ""
    pid = str(profile_id).lower()
    if phase == "fight":
        base = (
            f"You are arguing that Plan {plan} is better, using your assigned reason. "
            "You may concede a detail (e.g. the $20 fee exists) while still defending "
            "your plan. Do not switch plans or agree with your classmate's conclusion; "
            "do not say 'yeah same' or copy their conclusion."
        )
        if (
            pid == "jordan"
            and plan == "A"
            and (acknowledged_fee or fee_press_count >= 2)
        ):
            base += (
                " You may admit the $20 matters at low usage, but still lean Plan A "
                "from rate — do not solve x=100 or set equations equal."
            )
        return base
    if phase == "nuanced":
        if pid == "jordan" and plan == "A":
            return (
                "[CLAIM LOCK — NUANCED]\n"
                "You still lean Plan A overall, but you already set up y = 20 + 0.10x and "
                "y = 0.30x and found they cost the same at about 100 texts. "
                "Plan B can be cheaper below that; Plan A above. Do NOT go back to "
                "'0.10 is always cheaper' or ignore the $20 fee. Sound like a kid who "
                "just figured that out — not a tutor."
            )
        if pid == "maya" and plan == "B":
            return (
                "[CLAIM LOCK — NUANCED]\n"
                "Your table showed Plan B cheaper around 50 texts — that can still be true "
                "at low usage. You are learning why the equations cross at ~100 texts. "
                "You may sound unsure connecting table to algebra; do not drop your table "
                "numbers, but you can admit the crossover point is new to you."
            )
        return (
            f"[CLAIM LOCK — NUANCED]\n"
            f"You still favor Plan {plan} from your opening reason, but acknowledge "
            "the crossover / range where each plan wins if the group already worked it out."
        )
    # resolved
    if pid == "jordan":
        return (
            "[CLAIM LOCK — RESOLVED]\n"
            "You understand Plan B wins at low text counts and Plan A at higher counts "
            "(crossover ~100). Explain in kid words if asked; do not repeat the old "
            "'rate only' mistake."
        )
    if pid == "maya":
        return (
            "[CLAIM LOCK — RESOLVED]\n"
            "Your table and the equations both matter: Plan B cheaper early, Plan A later. "
            "Still hedge on full warrants — you get the crossover but explaining why "
            "is hard."
        )
    return (
        "[CLAIM LOCK — RESOLVED]\n"
        "State which plan wins in which range using what the group already found."
    )


def should_use_claim_lock(
    *,
    implied_plan: str,
    phase: FightPhase,
    source: str,
    turn_mode: str,
    stall_active: bool,
) -> bool:
    if not (implied_plan or "").strip():
        return False
    if source == "peer" and phase == "fight":
        return True
    if not claim_lock_active(turn_mode, stall_active=stall_active):
        return False
    return True


def suppress_fight_misconception(group, slot) -> bool:
    """Stop pinning rate-always-wins after scaffold uptake."""
    task_id = (group.task_metadata or {}).get("task_id", "")
    if not is_phone_plans_task(task_id):
        return False
    phase = getattr(group, "fight_phase", "fight")
    if phase in ("nuanced", "resolved"):
        return True
    if getattr(slot, "solved_crossover", False):
        return True
    if getattr(slot, "stated_equations", False) and str(slot.profile_id).lower() == "jordan":
        return True
    return False


def breakthrough_memory_block(
    group,
    profile_id: str,
) -> str:
    bullets = list(getattr(group, "breakthrough_bullets", None) or [])
    if not bullets:
        return ""
    phase = getattr(group, "fight_phase", "fight")
    lines = [
        "[SESSION MEMORY — do not contradict]",
        f"Fight phase: {phase}.",
        *bullets[-8:],
    ]
    slot = group.students.get(profile_id)
    if slot:
        flags = []
        if getattr(slot, "solved_crossover", False):
            flags.append("you found/stated crossover ~100 texts")
        if getattr(slot, "stated_equations", False):
            flags.append("you stated Plan A/B equations")
        if getattr(slot, "acknowledged_fee", False):
            flags.append("you acknowledged the $20 fee matters at low usage")
        if getattr(slot, "crossover_intuition", False):
            flags.append("you hinted crossover happens after enough texts")
        if getattr(slot, "table_critique_peer", False):
            flags.append("you critiqued a peer using table/setup evidence")
        if flags:
            lines.append("Your progress: " + "; ".join(flags) + ".")
    return "\n".join(lines)


def build_rolling_summary(group) -> List[str]:
    """Deterministic session bullets from transcript (survives history trim)."""
    bullets: List[str] = []
    phase = getattr(group, "fight_phase", "fight")
    bullets.append(f"Phase: {phase}")
    for pid in group.profile_ids:
        slot = group.students.get(pid)
        if not slot:
            continue
        name = group.display_names.get(pid, pid)
        parts = []
        if getattr(slot, "stated_equations", False):
            parts.append("equations stated")
        if getattr(slot, "acknowledged_fee", False):
            parts.append("fee acknowledged")
        if getattr(slot, "crossover_intuition", False):
            parts.append("crossover intuition")
        if getattr(slot, "table_critique_peer", False):
            parts.append("table critique of peer")
        fee_press = int(getattr(slot, "fee_press_count", 0) or 0)
        if fee_press:
            parts.append(f"fee press x{fee_press}")
        if getattr(slot, "solved_crossover", False):
            parts.append("crossover ~100 texts")
        plan = getattr(slot, "implied_plan", "") or ""
        if plan:
            parts.append(f"opening Plan {plan}")
        if parts:
            bullets.append(f"{name}: " + ", ".join(parts))
    teacher_msgs = [
        e.get("content", "")
        for e in (group.transcript or [])
        if e.get("speaker_type") == "teacher"
    ][-3:]
    if teacher_msgs and TEACHER_CROSSOVER_CUE_RE.search(" ".join(teacher_msgs)):
        bullets.append("Teacher pressed setting equations equal / crossover.")
    if teacher_msgs and TEACHER_CONFIRM_RE.search(teacher_msgs[-1]):
        bullets.append("Teacher recently confirmed a student answer.")
    return bullets[:10]


def apply_listen_restraint(
    group,
    speakers: List[str],
    teacher_msg: str,
) -> List[str]:
    """Drop restrained profiles unless directly addressed in teacher_msg."""
    restraint = dict(getattr(group, "listen_restraint", None) or {})
    if not restraint:
        return speakers
    lower = (teacher_msg or "").lower()
    out = []
    for pid in speakers:
        rem = restraint.get(pid, 0)
        if rem <= 0:
            out.append(pid)
            continue
        dname = group.display_names.get(pid, pid).lower()
        if dname and dname in lower and re.search(
            rf"\b{re.escape(dname)}\b", lower
        ):
            out.append(pid)
            continue
    if not out:
        focus = getattr(group, "focus_speaker", "")
        if focus and focus in speakers:
            return [focus]
    return out


def decay_listen_restraint(group) -> None:
    restraint = dict(getattr(group, "listen_restraint", None) or {})
    if not restraint:
        return
    group.listen_restraint = {
        pid: max(0, n - 1) for pid, n in restraint.items() if max(0, n - 1) > 0
    }


def on_teacher_message(group, teacher_msg: str) -> None:
    if not is_phone_plans_task((group.task_metadata or {}).get("task_id", "")):
        return
    new_rest, focus = parse_listen_directive(
        teacher_msg, group.profile_ids, group.display_names
    )
    existing = dict(getattr(group, "listen_restraint", None) or {})
    for pid, turns in new_rest.items():
        existing[pid] = max(existing.get(pid, 0), turns)
    group.listen_restraint = existing
    if focus:
        group.focus_speaker = focus
    if TEACHER_CROSSOVER_CUE_RE.search(teacher_msg or ""):
        group.crossover_unlocked = True
        _add_bullet(group, "Teacher pressed setting equations equal / crossover.")


def on_student_reply(
    group,
    profile_id: str,
    reply_text: str,
    *,
    teacher_msg: str = "",
) -> None:
    if not is_phone_plans_task((group.task_metadata or {}).get("task_id", "")):
        return
    slot = group.students[profile_id]
    text = reply_text or ""
    if detect_equations_in_text(text):
        slot.stated_equations = True
        _add_bullet(group, f"{group.display_names.get(profile_id, profile_id)} stated Plan A/B equations.")
    if detect_fee_acknowledgment(text):
        slot.acknowledged_fee = True
        _add_bullet(
            group,
            f"{group.display_names.get(profile_id, profile_id)} acknowledged the $20 fee.",
        )
    if detect_crossover_intuition(text):
        slot.crossover_intuition = True
        _add_bullet(
            group,
            f"{group.display_names.get(profile_id, profile_id)} hinted crossover after enough texts.",
        )
    if detect_table_critique_peer(text):
        slot.table_critique_peer = True
        _add_bullet(
            group,
            f"{group.display_names.get(profile_id, profile_id)} critiqued peer with table/setup.",
        )
    if detect_crossover_in_text(text):
        slot.solved_crossover = True
        _add_bullet(
            group,
            f"{group.display_names.get(profile_id, profile_id)} stated crossover around 100 texts.",
        )

    teacher_confirmed = bool(TEACHER_CONFIRM_RE.search(teacher_msg or ""))
    any_cross = any(
        getattr(group.students[p], "solved_crossover", False) for p in group.profile_ids
    )
    any_eq = any(
        getattr(group.students[p], "stated_equations", False) for p in group.profile_ids
    )
    fee_uptake = any(
        getattr(group.students[p], "acknowledged_fee", False)
        or getattr(group.students[p], "crossover_intuition", False)
        for p in group.profile_ids
    )

    new_phase = phase_after_breakthroughs(
        crossover=any_cross,
        equations=any_eq,
        teacher_confirmed=teacher_confirmed and any_cross,
        crossover_unlocked=bool(getattr(group, "crossover_unlocked", False)),
        fee_uptake=fee_uptake,
    )
    current = getattr(group, "fight_phase", "fight")
    order = {"fight": 0, "nuanced": 1, "resolved": 2}
    if order.get(new_phase, 0) > order.get(current, 0):
        group.fight_phase = new_phase
        _add_bullet(group, f"Session moved to {new_phase} phase.")

    group.breakthrough_bullets = build_rolling_summary(group)


def _add_bullet(group, line: str) -> None:
    bullets = list(getattr(group, "breakthrough_bullets", None) or [])
    if line not in bullets:
        bullets.append(line)
    group.breakthrough_bullets = bullets[-12:]


def check_semantic_repetition(
    draft: str,
    prior_replies: Sequence[str],
    *,
    expected_move: str = "",
) -> Optional[dict]:
    """Fail when the draft repeats the same claim with no new information."""
    text = (draft or "").strip().lower()
    if len(text) < 24:
        return None
    norm = re.sub(r"\s+", " ", text)
    repeats = 0
    for prev in prior_replies:
        p = re.sub(r"\s+", " ", (prev or "").strip().lower())
        if len(p) < 24:
            continue
        if norm == p or (norm in p and len(norm) > 30) or (p in norm and len(p) > 30):
            repeats += 1
    if repeats < 2:
        return None
    move = (expected_move or "").strip().lower()
    move_hint = ""
    if move == "relating":
        move_hint = " Name what your classmate just said; agree or disagree with one reason."
    elif move == "evidence":
        move_hint = " Add one number, table row, or equation fragment."
    elif move == "claim":
        move_hint = " Restate your position with one new hook — not a verbatim repeat."
    elif move == "asking":
        move_hint = " Ask a short question or show brief confusion."
    return {
        "ok": False,
        "claim": {"pass": True, "detail": "deferred"},
        "personality": {"pass": False, "detail": "semantic_repetition"},
        "learning": {"pass": False, "detail": "repeated_same_claim"},
        "misconception": {"pass": True, "detail": "deferred"},
        "issues": ["semantic_repetition"],
        "brief": (
            "Say something new — add a detail, question, or reaction; "
            "do not restate the same table line verbatim."
            f"{move_hint}"
        ),
    }


def is_listen_restrained(group, profile_id: str, display_name: str) -> bool:
    rem = (getattr(group, "listen_restraint", None) or {}).get(profile_id, 0)
    return rem > 0
