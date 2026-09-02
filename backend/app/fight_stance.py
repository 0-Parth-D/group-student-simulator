"""Phone-plans PST fight: opposing claims, allowlists, and stance-gated retrieval."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

PHONE_PLANS_TASK_ID = "phone_plans_linear_01"
ALEX_FLAT_ADD_ID = "ur_rate_as_flat_add"
MAYA_TABLE_ID = "pr_table_missing_warrants"

JORDAN_PLAN_A_IDS = (
    "pr_lower_rate_always_wins",
    "pr_ignores_monthly_fee",
    "pr_slope_coefficient_only",
)

# Hard pins for the FE starter argument (Maya table/Plan B vs Jordan rate/Plan A).
FIGHT_ALLOWLIST: Dict[str, Dict[str, tuple]] = {
    PHONE_PLANS_TASK_ID: {
        "maya": (MAYA_TABLE_ID,),
        "jordan": JORDAN_PLAN_A_IDS,
        "alex": (ALEX_FLAT_ADD_ID,),
    }
}

# Parity with pst-training-game/src/data/scenario.js seededMessages.
FE_SEED_LINES: Dict[str, str] = {
    "maya": (
        "I made a table up to 50 texts and Plan B is cheaper. "
        "Like at 50 it's $15 and Plan A is $25. So... Plan B."
    ),
    "jordan": (
        "No way, it's Plan A. Ten cents a text is basically nothing, "
        "and Plan B charges triple that. Plan A is obviously better."
    ),
}

CLAIM_IDS = {
    "maya": "plan_b_table",
    "jordan": "plan_a_rate",
    "alex": "plan_a_flat_add",
}


def is_phone_plans_task(task_id: str) -> bool:
    return (task_id or "").strip() == PHONE_PLANS_TASK_ID


def is_fe_starter_fight(profile_ids: Sequence[str], task_id: str) -> bool:
    ids = {str(p).strip().lower() for p in profile_ids}
    return is_phone_plans_task(task_id) and ids >= {"maya", "jordan"} and "alex" not in ids


def fight_allowlist_for(profile_id: str, task_id: str) -> Optional[tuple]:
    by_profile = FIGHT_ALLOWLIST.get((task_id or "").strip())
    if not by_profile:
        return None
    return by_profile.get(str(profile_id).strip().lower())


def implied_plan_of(item: Optional[dict]) -> str:
    if not item:
        return ""
    return str(item.get("implied_plan") or "").strip().upper()


def catalog_by_id() -> Dict[str, dict]:
    from app.misconception_store import _all_misconceptions

    return {m.get("id", ""): m for m in _all_misconceptions() if m.get("id")}


def items_for_ids(ids: Iterable[str]) -> List[dict]:
    by_id = catalog_by_id()
    out = []
    for mid in ids:
        item = by_id.get(mid)
        if item:
            out.append(dict(item))
    return out


def soft_candidates_for_task(task_id: str, profile_id: str) -> List[dict]:
    """Catalog rows tagged for this task (Phase H); exclude Alex-only for Maya."""
    tid = (task_id or "").strip()
    pid = str(profile_id).strip().lower()
    rows = []
    for item in catalog_by_id().values():
        tasks = item.get("task_ids") or []
        if tid and tid not in tasks:
            continue
        roster = [str(r).lower() for r in (item.get("roster") or [])]
        if roster and pid not in roster:
            continue
        if pid == "maya" and item.get("id") == ALEX_FLAT_ADD_ID:
            continue
        rows.append(dict(item))
    return rows


def opposition_pick(
    ranked: List[dict],
    peer_plans: Sequence[str],
    *,
    profile_id: str = "",
) -> List[dict]:
    """Prefer a hit whose implied_plan is not already taken by a peer."""
    peers = {p.strip().upper() for p in peer_plans if p and p.strip().upper() not in ("", "UNDECIDED")}
    pid = str(profile_id).strip().lower()
    filtered = []
    for item in ranked:
        mid = item.get("id")
        if pid == "maya" and mid == ALEX_FLAT_ADD_ID:
            continue
        if pid == "maya" and mid == "pr_compare_one_x_only":
            continue
        plan = implied_plan_of(item)
        if plan in peers and plan not in ("", "UNDECIDED"):
            continue
        filtered.append(item)
    chosen = filtered or [
        i
        for i in ranked
        if not (pid == "maya" and i.get("id") in (ALEX_FLAT_ADD_ID, "pr_compare_one_x_only"))
    ]
    return chosen[:1] if chosen else ranked[:1]


def fight_query_text(
    profile_id: str,
    task_meta: dict,
    barrier: Optional[dict] = None,
) -> str:
    barrier = barrier or {}
    task_text = (
        (task_meta.get("description") or task_meta.get("task_text") or "")
        if task_meta
        else ""
    )
    seed = FE_SEED_LINES.get(str(profile_id).lower(), "")
    return " ".join(
        part
        for part in (
            task_text,
            barrier.get("summary") or "",
            barrier.get("discourse_role") or "",
            seed,
            f"student {profile_id}",
        )
        if part
    ).strip()


def retrieve_for_group_fight(
    profile_id: str,
    task_meta: dict,
    *,
    barrier: Optional[dict] = None,
    peer_picks: Optional[Sequence[dict]] = None,
    k: int = 3,
) -> List[dict]:
    """Stance-conditioned retrieve for group PST fights; falls back to metadata."""
    from app.config import USE_QDRANT
    from app.misconception_store import (
        _qdrant_catalog_retrieve_ids,
        _score_text_against_items,
    )

    task_id = (task_meta or {}).get("task_id") or ""
    pid = str(profile_id).strip().lower()
    allow = fight_allowlist_for(pid, task_id)
    if allow:
        pool = items_for_ids(allow)
    else:
        pool = soft_candidates_for_task(task_id, pid)

    peer_plans = [implied_plan_of(p) for p in (peer_picks or [])]
    query = fight_query_text(pid, task_meta or {}, barrier)

    ranked: List[dict] = []
    if allow:
        ranked = pool
    elif USE_QDRANT and pool:
        ids = [m.get("id") for m in pool if m.get("id")]
        hits = _qdrant_catalog_retrieve_ids(query, ids, k=max(k, len(ids)))
        if hits:
            ranked = hits
    if not ranked and pool:
        ranked = _score_text_against_items(query, pool, k=max(k, len(pool)))
    if not ranked:
        return []

    picked = opposition_pick(ranked, peer_plans, profile_id=pid)
    # Keep extras from the same allowlist that share implied_plan (Jordan's family).
    plan = implied_plan_of(picked[0]) if picked else ""
    extras = [
        item
        for item in ranked
        if item.get("id") != (picked[0].get("id") if picked else None)
        and (not plan or implied_plan_of(item) in (plan, "", "UNDECIDED"))
    ]
    out = (picked + extras)[:k]
    if not out and pool:
        return pool[:1]
    return out
