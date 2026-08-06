#!/usr/bin/env python3
"""Run offline evaluation on a saved session log JSON file."""

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
load_dotenv()

from eval.behavioral_consistency import behavioral_consistency
from eval.believability import believability, linguistic_proxies
from eval.cognitive_fidelity import cognitive_fidelity
from eval.deterministic_checks import aggregate_deterministic, run_deterministic_checks
from eval.persona_stability import persona_stability


@dataclass
class EvalReport:
    cognitive_fidelity: float = 0.0
    behavioral_consistency: float = 0.0
    believability: float = 0.0
    persona_stability: dict = field(default_factory=dict)
    linguistic_proxies: dict = field(default_factory=dict)
    deterministic: dict = field(default_factory=dict)
    flagged_turns: List[dict] = field(default_factory=list)
    cognitive_scores: List[dict] = field(default_factory=list)
    behavioral_scores: List[dict] = field(default_factory=list)
    believability_scores: List[dict] = field(default_factory=list)
    deterministic_by_turn: List[List[dict]] = field(default_factory=list)


def _mean_score(items: List[dict], key: str) -> float:
    if not items:
        return 0.0
    vals = [float(item[key]) for item in items if isinstance(item.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def evaluate_session(
    session_log: List[dict],
    persona_sample_k: int = 10,
    skip_llm: bool = False,
    profile_id: Optional[str] = None,
) -> EvalReport:
    cognitive = []
    behavioral = []
    voice = []
    flagged = []
    det_by_turn = []

    for turn in session_log:
        if turn.get("role") not in ("student", None) and turn.get("type") != "math":
            if turn.get("teacher_label") not in ("math_scaffold", "math_eval"):
                continue

        det_results = run_deterministic_checks(turn)
        det_by_turn.append(det_results)
        for r in det_results:
            if not r.get("pass"):
                flagged.append({**turn, "failed_check": r})

        if skip_llm:
            continue

        if turn.get("type") == "math" or turn.get("teacher_label") in (
            "math_scaffold",
            "math_eval",
        ):
            score = cognitive_fidelity(turn)
            cognitive.append(score)
            if score.get("score", 0) < 3:
                flagged.append({**turn, "cognitive_score": score.get("score")})

        bscore = behavioral_consistency(turn)
        behavioral.append(bscore)
        if not bscore.get("consistent"):
            flagged.append({**turn, "behavioral_deviation": bscore.get("deviation")})

        vscore = believability(turn)
        voice.append(vscore)
        if float(vscore.get("score") or 0) < 3:
            flagged.append({**turn, "believability_score": vscore.get("score")})

    student_turns = [
        t
        for t in session_log
        if t.get("role") == "student" or t.get("type") == "student" or t.get("reply")
    ]
    sample = student_turns
    if len(student_turns) > persona_sample_k:
        sample = random.sample(student_turns, persona_sample_k)

    pid = profile_id or (student_turns[0].get("profile_id") if student_turns else None)
    persona = {} if skip_llm else persona_stability(sample, profile_id=pid)

    behavioral_rate = (
        sum(1.0 if b.get("consistent") else 0.0 for b in behavioral) / len(behavioral)
        if behavioral
        else 0.0
    )

    return EvalReport(
        cognitive_fidelity=_mean_score(cognitive, "score"),
        behavioral_consistency=behavioral_rate,
        believability=_mean_score(voice, "score"),
        persona_stability=persona,
        linguistic_proxies=linguistic_proxies(
            [t.get("reply") or "" for t in student_turns]
        ),
        deterministic=aggregate_deterministic(det_by_turn),
        flagged_turns=flagged,
        cognitive_scores=cognitive,
        behavioral_scores=behavioral,
        believability_scores=voice,
        deterministic_by_turn=det_by_turn,
    )


def evaluate_scenario(scenario: dict, skip_llm: bool = False) -> EvalReport:
    """Evaluate a scenario wrapper {scenario_id, profile_id, turns: [...]}."""
    turns = scenario.get("turns") or []
    return evaluate_session(
        turns,
        skip_llm=skip_llm,
        profile_id=scenario.get("profile_id"),
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved tutoring session log")
    parser.add_argument("log_path", help="Path to session log JSON (list of turns or scenario)")
    parser.add_argument("-o", "--output", help="Write report JSON to this path")
    parser.add_argument("--skip-llm", action="store_true", help="Deterministic checks only")
    args = parser.parse_args()

    path = Path(args.log_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "turns" in payload:
        report = evaluate_scenario(payload, skip_llm=args.skip_llm)
    elif isinstance(payload, dict) and "turns" in payload.get("scenarios", {}):
        report = evaluate_scenario(payload, skip_llm=args.skip_llm)
    else:
        session_log = payload.get("turns", payload) if isinstance(payload, dict) else payload
        report = evaluate_session(session_log, skip_llm=args.skip_llm)

    out = asdict(report)
    text = json.dumps(out, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
