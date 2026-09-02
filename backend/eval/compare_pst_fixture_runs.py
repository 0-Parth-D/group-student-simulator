#!/usr/bin/env python3
"""Score PST fixture runs: current backend replay vs FE-only / prior backend logs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
load_dotenv()

from eval.discourse_progress import score_export
from eval.run_pst_script_replay import FIXTURE_PATH, run_replay, load_fixture

# Verbatim transcripts from user session logs (same teacher script as fixture).
FE_ONLY_TRANSCRIPT: List[Dict[str, str]] = [
    {"speaker_type": "student", "speaker_id": "maya", "content": "I made a table up to 50 texts and Plan B is cheaper. Like at 50 it's $15 and Plan A is $25. So... Plan B."},
    {"speaker_type": "student", "speaker_id": "jordan", "content": "No way, it's Plan A. Ten cents a text is basically nothing, and Plan B charges triple that. Plan A is obviously better."},
    {"speaker_type": "teacher", "content": "canyon guys explain your reasoning for the claims"},
    {"speaker_type": "student", "speaker_id": "jordan", "content": "Plan A is 20 plus 0.1x and Plan B is 0.3x, so the coefficient on Plan A is smaller. That means each extra text costs less, so Plan A is better."},
    {"speaker_type": "student", "speaker_id": "maya", "content": "But in my table Plan B is lower every time we checked, like 10, 20, 30, 40, 50. At 50 it's $15 and Plan A is $25, so I said Plan B."},
    {"speaker_type": "teacher", "content": "Jordan what do you think maya is getting wrong"},
    {"speaker_type": "student", "speaker_id": "jordan", "content": "Her table is just checking small numbers, and that's slower than the equation. The coefficient is still smaller for Plan A, so I think that's what matters."},
    {"speaker_type": "student", "speaker_id": "maya", "content": "But my table still isn't wrong though, like at 10 to 50 Plan B really is less. I'm just saying what the numbers we checked say."},
    {"speaker_type": "teacher", "content": "Jordan what's your reasoning? maya can you check him?"},
    {"speaker_type": "student", "speaker_id": "jordan", "content": "I mean my reasoning is still that 0.1 is less than 0.3, so Plan A goes up slower. That's why I'm saying it's better."},
    {"speaker_type": "student", "speaker_id": "maya", "content": "But that doesn't make my table wrong, because from 10 to 50 Plan B is still less every time. I'm not saying forever, I'm just saying the numbers we checked."},
    {"speaker_type": "teacher", "content": "yes but maya find what are the wrong things in Jordan's claim"},
    {"speaker_type": "student", "speaker_id": "maya", "content": "I think he's wrong that Plan A is just better, because in the numbers we actually checked it isn't. Like 0.1 is smaller, yeah, but Plan A starts at 20, so from 10 to 50 Plan B still costs less."},
    {"speaker_type": "teacher", "content": "Jordan did you getit"},
    {"speaker_type": "student", "speaker_id": "jordan", "content": "I get what she's saying, but I still think the smaller coefficient is the main thing. The 20 matters at the start, I guess, but Plan A still goes up slower."},
    {"speaker_type": "teacher", "content": "okaybutwhydo you think 20 won't matter"},
    {"speaker_type": "student", "speaker_id": "jordan", "content": "Because after enough texts the 20 is just one fixed amount, but 0.3 keeps adding more every text. So eventually Plan B should pass it since it goes up faster."},
    {"speaker_type": "teacher", "content": "maya help Jordan understand why heis wrong using the table"},
    {"speaker_type": "student", "speaker_id": "maya", "content": "I mean, from the table he's wrong if he says Plan A is just better, because at 10, 20, 30, 40, 50 texts Plan B is lower every time. I'm just using the numbers we actually checked."},
    {"speaker_type": "teacher", "content": "Jordan did you get it?"},
    {"speaker_type": "student", "speaker_id": "jordan", "content": "Yeah, I get that in her table Plan B is lower for those small numbers. I'm just saying that doesn't prove it stays better once x gets bigger."},
    {"speaker_type": "teacher", "content": "socan you answer the question then? when is each plan better. maya?"},
    {"speaker_type": "student", "speaker_id": "maya", "content": "From my table, Plan B is better for 10 to 50 texts. I don't know the exact switch, but not just \"Plan A is always better.\""},
    {"speaker_type": "student", "speaker_id": "jordan", "content": "Just set them equal: 20 plus 0.1x = 0.3x, so x = 100. So Plan B is better under 100 texts, they're the same at 100, and Plan A is better over 100."},
    {"speaker_type": "teacher", "content": "okay can you explain maya why that would be thecase"},
    {"speaker_type": "student", "speaker_id": "maya", "content": "Um, because Plan A starts with that 20, so at first it costs more, but Plan B goes up faster each text. So like after enough texts it catches up to that 20 and then passes it."},
    {"speaker_type": "teacher", "content": "what's the answer both of you say"},
    {"speaker_type": "student", "speaker_id": "jordan", "content": "Plan B is better if you send less than 100 texts, they tie at 100, and Plan A is better if you send more than 100. So it's not just one plan always."},
    {"speaker_type": "student", "speaker_id": "maya", "content": "Yeah, I agree with that now. Plan B for less than 100, same at 100, and Plan A after that."},
]

PREV_BACKEND_TRANSCRIPT: List[Dict[str, str]] = [
    {"speaker_type": "student", "speaker_id": "maya", "content": "I made a table up to 50 texts and Plan B is cheaper. Like at 50 it's $15 and Plan A is $25. So... Plan B."},
    {"speaker_type": "student", "speaker_id": "jordan", "content": "No way, it's Plan A. Ten cents a text is basically nothing, and Plan B charges triple that. Plan A is obviously better."},
    {"speaker_type": "teacher", "content": "canyon guys explain your reasoning for the claims"},
    {"speaker_type": "student", "speaker_id": "maya", "content": "Um, at 50 texts Plan B was $15 and Plan A was $25, so Plan B cost less."},
    {"speaker_type": "student", "speaker_id": "jordan", "content": "Okay so 50 texts is barely anything, and if you go over that Plan A is way cheaper per text cause it's only $0.10 each."},
    {"speaker_type": "teacher", "content": "Jordan what do you think maya is getting wrong"},
    {"speaker_type": "student", "speaker_id": "jordan", "content": "Plan B charges $0.30 per text and Plan A is only $0.10 per text, so she's totally missing that Plan A's rate is way cheaper for each text she sends."},
    {"speaker_type": "student", "speaker_id": "maya", "content": "Um, my table has Plan B at $15 and Plan A at $25 for 50 texts, so Plan B is cheaper."},
    {"speaker_type": "teacher", "content": "Jordan what's your reasoning? maya can you check him?"},
    {"speaker_type": "student", "speaker_id": "jordan", "content": "Wait, are you asking me to actually write it down or just say it out loud?"},
    {"speaker_type": "student", "speaker_id": "maya", "content": "Um, I already got my table written down so."},
    {"speaker_type": "teacher", "content": "Just say it"},
    {"speaker_type": "student", "speaker_id": "jordan", "content": "Okay so Plan A charges $0.10 per text and Plan B charges $0.30 per text, so Plan A is way cheaper if you're sending a lot of texts."},
    {"speaker_type": "teacher", "content": "yes but maya find what are the wrong things in Jordan's claim"},
    {"speaker_type": "student", "speaker_id": "maya", "content": "I think he's leaving out the $20 fee, because my table has Plan A costing more at 50 texts."},
    {"speaker_type": "student", "speaker_id": "jordan", "content": "Wait, like the rate is still cheaper so who cares, $0.10 just beats $0.30 no matter what you do."},
    {"speaker_type": "teacher", "content": "Jordan did you getit"},
    {"speaker_type": "student", "speaker_id": "jordan", "content": "Okay so I'm not really sure what I'm supposed to do with that information."},
    {"speaker_type": "teacher", "content": "Didi you understant what Maya pointed out?"},
    {"speaker_type": "student", "speaker_id": "maya", "content": "Wait, like, are we still supposed to keep going or what?"},
    {"speaker_type": "teacher", "content": "Jordan, okaybutwhydo you think 20 won't matter"},
    {"speaker_type": "student", "speaker_id": "jordan", "content": "Okay so $0.10 versus $0.30 is like a huge gap per text, so at some point Plan A just catches up... I think."},
    {"speaker_type": "teacher", "content": "maya help Jordan understand why heis wrong using the table"},
    {"speaker_type": "student", "speaker_id": "maya", "content": "Um, that $20 fee makes Plan A start way higher I think."},
    {"speaker_type": "student", "speaker_id": "jordan", "content": "The $20 is like whatever, but Plan A is ten cents a text and Plan B is thirty cents, so that difference is gonna catch up way faster when you're actually texting people a normal amount."},
    {"speaker_type": "teacher", "content": "socan you answer the question then? when is each plan better. maya?"},
    {"speaker_type": "student", "speaker_id": "maya", "content": "Um, Plan B is cheaper if you text under 50 times, but idk exactly where it flips."},
    {"speaker_type": "teacher", "content": "okay can you explain maya why that would be thecase"},
    {"speaker_type": "student", "speaker_id": "maya", "content": "Um, my table shows Plan B costs less at 50, I just don't really know how to explain it past that."},
    {"speaker_type": "teacher", "content": "what's the answer both of you say"},
    {"speaker_type": "student", "speaker_id": "maya", "content": "Um, Plan B is better when you don't text that much."},
    {"speaker_type": "student", "speaker_id": "jordan", "content": "Okay so if you actually text a normal amount Plan A is way cheaper per text, like nobody barely texts anymore."},
]

METRIC_KEYS = (
    "peer_reference_rate",
    "repetition_streak",
    "jordan_passive_hits",
    "unprompted_crossover",
    "breakthrough_count",
    "fight_phase",
    "jordan_fee_uptake",
    "crossover_unlocked",
    "reasoning_press_turns",
    "warrant_on_press_rate",
    "bare_claim_on_press",
)


def _export_from_transcript(
    transcript: List[Dict[str, str]],
    *,
    label: str,
    fight_phase: str = "fight",
    crossover_unlocked: bool = False,
    breakthrough_count: int = 0,
    slot_flags: Dict[str, Any] | None = None,
) -> dict:
    return {
        "task_id": "phone_plans_linear_01",
        "display_names": {"maya": "Maya", "jordan": "Jordan"},
        "transcript": transcript,
        "fight_phase": fight_phase,
        "crossover_unlocked": crossover_unlocked,
        "breakthrough_bullets": [f"import:{label}"] * breakthrough_count,
        "slot_flags": slot_flags or {},
    }


def _infer_flags_from_transcript(transcript: List[Dict[str, str]]) -> Dict[str, Any]:
    text = " ".join(
        (e.get("content") or "")
        for e in transcript
        if e.get("speaker_type") == "student"
    ).lower()
    jordan_fee = bool(re.search(r"\$?\s*20\b|\bfee\b", text))
    crossover = bool(re.search(r"\bx\s*=\s*100\b|100\s*texts?", text))
    return {
        "jordan": {
            "behavior_mode": "WRONG",
            "student_stack_level": 2,
            "primary_construct": "rate_comparison",
            "acknowledged_fee": jordan_fee,
            "crossover_unlocked": crossover,
            "fight_phase": "nuanced" if crossover else "fight",
        },
        "maya": {
            "behavior_mode": "PARTIAL_ATTEMPT_THEN_STUCK",
            "student_stack_level": 2,
            "primary_construct": "rate_comparison",
            "fight_phase": "nuanced" if crossover else "fight",
        },
    }


def score_labeled_export(export: dict) -> Dict[str, Any]:
    flags = export.get("slot_flags") or _infer_flags_from_transcript(export.get("transcript") or [])
    export = {**export, "slot_flags": flags}
    metrics = score_export(export)
    critique = metrics.get("maya_critique") or {}
    return {
        **{k: metrics.get(k) for k in METRIC_KEYS},
        "maya_critique_rich": critique.get("maya_critique_rich", False),
        "found_critique_prompt": critique.get("found_critique_prompt", False),
    }


def compare_all(*, run_backend: bool = True) -> dict:
    rows: Dict[str, Dict[str, Any]] = {}

    fe_export = _export_from_transcript(
        FE_ONLY_TRANSCRIPT,
        label="fe_only",
        fight_phase="resolved",
        crossover_unlocked=True,
        breakthrough_count=4,
        slot_flags=_infer_flags_from_transcript(FE_ONLY_TRANSCRIPT),
    )
    rows["fe_only_log"] = score_labeled_export(fe_export)

    prev_export = _export_from_transcript(
        PREV_BACKEND_TRANSCRIPT,
        label="prev_backend",
        fight_phase="fight",
        crossover_unlocked=False,
        breakthrough_count=1,
        slot_flags=_infer_flags_from_transcript(PREV_BACKEND_TRANSCRIPT),
    )
    rows["prev_backend_log"] = score_labeled_export(prev_export)

    if run_backend:
        fixture = load_fixture(FIXTURE_PATH)
        replay = run_replay(fixture)
        export = replay["export"]
        export.setdefault("slot_flags", {})
        for pid in ("jordan", "maya"):
            export["slot_flags"].setdefault(pid, {})
            export["slot_flags"][pid].setdefault("behavior_mode", "WRONG" if pid == "jordan" else "PARTIAL_ATTEMPT_THEN_STUCK")
            export["slot_flags"][pid].setdefault("student_stack_level", 2)
            export["slot_flags"][pid].setdefault("primary_construct", "rate_comparison")
            export["slot_flags"][pid]["fight_phase"] = export.get("fight_phase", "fight")
            export["slot_flags"][pid]["crossover_unlocked"] = export.get("crossover_unlocked", False)
        rows["current_backend_replay"] = score_labeled_export(export)
        backend_payload = replay
    else:
        backend_payload = None

    return {"runs": rows, "backend_replay": backend_payload}


def _print_table(rows: Dict[str, Dict[str, Any]]) -> None:
    labels = list(rows.keys())
    print("\n=== PST fixture comparison (discourse_progress metrics) ===\n")
    header = f"{'metric':<28}" + "".join(f"{lab:>22}" for lab in labels)
    print(header)
    print("-" * len(header))
    all_keys = set()
    for r in rows.values():
        all_keys.update(r.keys())
    for key in METRIC_KEYS + ("maya_critique_rich", "found_critique_prompt"):
        if key not in all_keys:
            continue
        line = f"{key:<28}"
        for lab in labels:
            val = rows[lab].get(key, "")
            line += f"{str(val):>22}"
        print(line)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "results" / "pst_fixture_comparison.json"),
    )
    parser.add_argument(
        "--skip-backend",
        action="store_true",
        help="Score FE + prev backend logs only (no LLM replay)",
    )
    args = parser.parse_args()

    result = compare_all(run_backend=not args.skip_backend)
    _print_table(result["runs"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metrics_by_run": result["runs"]}
    if result.get("backend_replay"):
        payload["current_backend_replay_full"] = {
            "metrics": result["backend_replay"]["metrics"],
            "export": {
                "fight_phase": result["backend_replay"]["export"].get("fight_phase"),
                "crossover_unlocked": result["backend_replay"]["export"].get("crossover_unlocked"),
                "breakthrough_bullets": result["backend_replay"]["export"].get("breakthrough_bullets"),
                "transcript": result["backend_replay"]["export"].get("transcript"),
                "slot_flags": result["backend_replay"]["export"].get("slot_flags"),
            },
        }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
