from app.group.orchestrator import (
    parse_speak_constraints,
    resolve_speakers_with_meta,
)
from app.group.sessions import GroupSession, GroupSessionStore, StudentSlot
from app.student.models import Student


PROFILE_IDS = ["jordan", "sam", "alex"]
DISPLAY_NAMES = {"jordan": "Jordan", "sam": "Sam", "alex": "Alex"}


def _student(profile_id: str, extraversion: str, neuroticism: str) -> Student:
    return Student(
        student_id=DISPLAY_NAMES[profile_id],
        Openness="High",
        Conscientiousness="Low",
        Extraversion=extraversion,
        Agreeableness="High",
        Neuroticism=neuroticism,
    )


def _group() -> GroupSession:
    slots = {
        "jordan": StudentSlot(
            "jordan",
            _student("jordan", "High", "High"),
            mastery_state={"part_whole": 1},
            behavior_profile={
                "behavior_mode": "WRONG",
                "primary_construct": "part_whole",
            },
        ),
        "sam": StudentSlot(
            "sam",
            _student("sam", "Low", "High"),
            mastery_state={"part_whole": 3},
            behavior_profile={
                "behavior_mode": "NORMAL_ERROR_PROFILE",
                "primary_construct": "part_whole",
            },
        ),
        "alex": StudentSlot(
            "alex",
            _student("alex", "Low", "Low"),
            mastery_state={"part_whole": 3},
            behavior_profile={
                "behavior_mode": "NORMAL_ERROR_PROFILE",
                "primary_construct": "part_whole",
            },
        ),
    }
    return GroupSession(
        session_id="test-group",
        task_text="Maria ate pizza. How much is left?",
        task_metadata={"required_constructs": ["part_whole"]},
        students=slots,
        profile_ids=list(PROFILE_IDS),
        display_names=dict(DISPLAY_NAMES),
        started=True,
    )


def test_direct_address_selects_only_named_student():
    constraints = parse_speak_constraints(
        "Sam, what do you think?", PROFILE_IDS, DISPLAY_NAMES
    )
    assert constraints.must_speak == ["sam"]
    assert constraints.mode == "direct"


def test_discuss_excludes_student_told_to_watch():
    constraints = parse_speak_constraints(
        "Alex and Sam discuss; Jordan please watch",
        PROFILE_IDS,
        DISPLAY_NAMES,
    )
    assert constraints.may_speak == ["alex", "sam"]
    assert constraints.must_not_speak == ["jordan"]


def test_topic_only_name_is_not_directly_addressed():
    constraints = parse_speak_constraints(
        "Sam, explain why Jordan's method might be wrong",
        PROFILE_IDS,
        DISPLAY_NAMES,
    )
    assert constraints.must_speak == ["sam"]
    assert "jordan" not in constraints.must_speak


def test_wrong_extravert_prefers_disagreement_floor():
    resolution = resolve_speakers_with_meta(
        "Anyone disagree?",
        _group(),
        enable_self_select=True,
        constraint_parser="regex",
    )
    assert resolution.speakers[0] == "jordan"
    jordan = next(
        d for d in resolution.decisions if d["profile_id"] == "jordan"
    )
    assert jordan["will_speak"] is True
    assert "disagreement cue" in jordan["reason"]


def test_legacy_mode_keeps_first_roster_default():
    resolution = resolve_speakers_with_meta(
        "What should we do next?",
        _group(),
        enable_self_select=False,
        constraint_parser="regex",
    )
    assert resolution.speakers == ["jordan"]


def test_open_floor_downweights_student_who_just_spoke():
    group = _group()
    group.transcript = [
        {
            "turn": 1,
            "speaker_type": "student",
            "speaker_id": "jordan",
            "content": "My first idea.",
        },
        {
            "turn": 2,
            "speaker_type": "teacher",
            "speaker_id": "teacher",
            "content": "What should we do next?",
        },
    ]
    resolution = resolve_speakers_with_meta(
        "What should we do next?",
        group,
        enable_self_select=True,
        constraint_parser="regex",
    )
    jordan = next(d for d in resolution.decisions if d["profile_id"] == "jordan")
    assert "spoke last round" in jordan["reason"]

    clean = resolve_speakers_with_meta(
        "What should we do next?",
        _group(),
        enable_self_select=True,
        constraint_parser="regex",
    )
    clean_jordan = next(d for d in clean.decisions if d["profile_id"] == "jordan")
    assert jordan["score"] < clean_jordan["score"]


def test_observer_never_replies(monkeypatch):
    from app.group.orchestrator import parse_speak_constraints

    group = _group()
    store = GroupSessionStore()

    def fake_reply(group, profile_id, incoming, source, turn, **_kwargs):
        group.transcript.append(
            {
                "turn": turn,
                "speaker_type": "student",
                "speaker_id": profile_id,
                "content": f"{profile_id} reply",
            }
        )
        return f"{profile_id} reply", False, [], "math_scaffold", False, False

    monkeypatch.setattr(store, "student_reply", fake_reply)
    # Offline CI: don't call live hybrid LLM during unit tests.
    monkeypatch.setattr(
        "app.group.constraint_llm.resolve_constraint_parser",
        lambda name=None: parse_speak_constraints,
    )
    _, replies, vague_warning, active, turn_mode, multi, stop_reason = store.respond(
        group, "Alex and Sam discuss; Jordan please watch"
    )

    assert [reply["speaker_id"] for reply in replies] == ["alex", "sam"]
    assert group.last_observers == ["jordan"]
    assert vague_warning is False
    assert all(r["speaker_id"] != "jordan" for r in multi)

def test_resolve_constraint_parser_default_is_hybrid():
    from app.config import GROUP_CONSTRAINT_PARSER
    from app.group.constraint_llm import resolve_constraint_parser

    assert GROUP_CONSTRAINT_PARSER == "hybrid"
    assert resolve_constraint_parser().__name__ == "parse_speak_constraints_hybrid"


def test_resolve_constraint_parser_names():
    from app.group.constraint_llm import resolve_constraint_parser
    from app.group.orchestrator import parse_speak_constraints

    assert resolve_constraint_parser("regex") is parse_speak_constraints
    assert resolve_constraint_parser("hybrid").__name__ == "parse_speak_constraints_hybrid"
