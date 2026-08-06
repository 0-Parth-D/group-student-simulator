from app.config import MAX_ACTIVE_CONCEPTS


def merge_concepts(*concept_lists) -> list:
    """Dedupe concept lists, preserving order."""
    merged = []
    for lst in concept_lists:
        for c in lst or []:
            if c not in merged:
                merged.append(c)
    return merged[:MAX_ACTIVE_CONCEPTS]
