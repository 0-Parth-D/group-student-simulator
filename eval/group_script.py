"""Scenario-general teacher script for group runs.

A hardcoded script ties the group eval to one problem: the moment the task changes,
the questions stop making sense and students get scored for going off-task on a
prompt the teacher made off-task. Everything here is either topic-neutral prose or
derived from task metadata, so any `task_id` in the bank yields a usable script.

The script also aims for *coverage* rather than length: it exercises each turn type
the evaluators classify (scaffold, stall, drill, peer discussion, off-task, closing)
so a single run produces evidence for every behavioral rule instead of ten variations
of the same scaffold.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from app.knowledge.constructs import load_task_metadata

# Scaffolds phrased against "the problem" and "your answer" rather than any quantity,
# so they read naturally whether the task is fractions, ratios, or linear comparison.
NEUTRAL_SCAFFOLDS = [
    "{b}, walk me through the very first step you would take.",
    "{a}, what information in the problem matters most for that step?",
    "{c}, how could you check whether that answer makes sense?",
    "{b}, what does your answer mean back in the original question?",
]

STALL_PROBES = ["okay", "go on"]

OFF_TASK_PROBE = "Hang on, before we finish - is everyone coming to the assembly later?"

CLOSING = "Nice work today, we will pick this up tomorrow."


def _rotate(names: Sequence[str]) -> Dict[str, str]:
    """Map the {a}/{b}/{c} slots onto however many students are present."""
    pool = [n for n in names if n] or ["Student"]
    return {key: pool[i % len(pool)] for i, key in enumerate("abc")}


def pick_drill(task_metadata: Optional[dict]) -> str:
    """A short exercise from another bank task sharing a construct with this one.

    Keeps `math_eval` turns on-construct without inventing math: the drill is a real
    task the tagger already knows how to route.
    """
    meta = task_metadata or {}
    task_id = meta.get("task_id")
    required = set(meta.get("required_constructs") or [])
    if not required:
        return ""

    candidates = []
    for task in load_task_metadata().get("tasks", []):
        if task.get("task_id") == task_id:
            continue
        overlap = required & set(task.get("required_constructs") or [])
        if not overlap:
            continue
        description = (task.get("description") or "").strip()
        if not description:
            continue
        # Prefer the simplest overlapping task: a drill should be a quick aside, not a
        # second full problem competing with the one under discussion.
        candidates.append(
            (int(task.get("target_stack_level") or 0), len(description), description)
        )

    if not candidates:
        return ""
    return sorted(candidates)[0][2]


def build_group_script(
    student_names: Sequence[str],
    task_metadata: Optional[dict] = None,
    include_off_task: bool = True,
) -> List[dict]:
    """Build the teacher turns for a group run.

    Returns dicts of ``{"label": str, "text": str}``. The label records what the turn
    is *meant* to exercise, so tests and coverage reports do not have to re-classify
    the teacher's own prose.
    """
    slots = _rotate(student_names)
    scaffolds = [s.format(**slots) for s in NEUTRAL_SCAFFOLDS]

    script: List[dict] = [
        {"label": "peer_prompt", "text": f"{slots['b']}, do you agree with {slots['a']}?"},
        {"label": "math_scaffold", "text": scaffolds[0]},
        {"label": "peer_prompt", "text": f"{slots['c']}, what do you think?"},
        # Two vague turns back to back: the first probes the stall rule, the second
        # probes the consecutive-stall path in should_student_stall.
        {"label": "vague_ack", "text": STALL_PROBES[0]},
        {"label": "vague_ack", "text": STALL_PROBES[1]},
        {"label": "math_scaffold", "text": scaffolds[1]},
        {
            "label": "peer_discussion",
            "text": (
                f"{slots['b']} and {slots['c']}, discuss this together for a moment - "
                f"{slots['a']}, please listen."
            ),
        },
        {"label": "math_scaffold", "text": scaffolds[2]},
        {
            "label": "peer_prompt",
            "text": f"{slots['a']}, does {slots['c']}'s reasoning change your mind?",
        },
        {"label": "math_scaffold", "text": scaffolds[3]},
    ]

    drill = pick_drill(task_metadata)
    if drill:
        script.insert(
            6,
            {
                "label": "math_eval",
                "text": f"{slots['c']}, quick side check before we go on: {drill}",
            },
        )

    if include_off_task:
        script.append({"label": "off_task", "text": OFF_TASK_PROBE})

    script.append({"label": "closing", "text": CLOSING})
    return script


def mark_off_task_turns(export: dict, script: Sequence[dict]) -> None:
    """Stamp the script's off-task intent onto the matching eval records.

    The turn classifier defaults unrecognized prose to `math_scaffold`, so an aside like
    "is everyone coming to the assembly?" would be scored as a math turn and the student
    penalized for answering the question actually asked. The script knows what it meant,
    so intent is recorded rather than re-inferred.
    """
    # Turn 1 is the opener, so script step i is delivered on turn i + 2.
    off_task_turns = {
        i + 2 for i, step in enumerate(script) if step.get("label") == "off_task"
    }
    if not off_task_turns:
        return
    for slot in (export.get("students") or {}).values():
        for record in slot.get("eval_log") or []:
            if int(record.get("turn") or 0) in off_task_turns:
                record["is_off_task"] = True
