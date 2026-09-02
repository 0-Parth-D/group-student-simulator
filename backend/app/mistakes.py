from typing import List

from app.construct_text import all_construct_cues, construct_cues


def playbook_summary(error_playbook: dict) -> dict:
    """Truncated playbook for API responses."""
    patterns = error_playbook.get("patterns", [])[:5]
    return {
        "patterns": patterns,
        "stay_in_problem": error_playbook.get("stay_in_problem", ""),
    }


def match_lp_constructs_by_keywords(
    teacher_utt: str, required_constructs: List[str]
) -> List[str]:
    """Match LP construct IDs from teacher utterance using KG construct cues."""
    utt = teacher_utt.lower().strip()
    matched = []
    pool = required_constructs or list(all_construct_cues().keys())
    for cid in pool:
        keywords = construct_cues(cid)
        if keywords and any(kw in utt for kw in keywords):
            matched.append(cid)
    return matched
