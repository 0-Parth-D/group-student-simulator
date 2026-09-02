"""Fight-aligned catalog tags and stance-gated retrieval."""

from app.fight_stance import (
    ALEX_FLAT_ADD_ID,
    FIGHT_ALLOWLIST,
    JORDAN_PLAN_A_IDS,
    MAYA_TABLE_ID,
    PHONE_PLANS_TASK_ID,
    fight_allowlist_for,
    implied_plan_of,
    items_for_ids,
    retrieve_for_group_fight,
)
from app.misconception_store import _all_misconceptions
from app.kg_config import load_misconception_catalog


def setup_function(_):
    load_misconception_catalog.cache_clear()
    _all_misconceptions.cache_clear()


def teardown_function(_):
    load_misconception_catalog.cache_clear()
    _all_misconceptions.cache_clear()


def test_maya_and_jordan_allowlists_oppose():
    maya_ids = fight_allowlist_for("maya", PHONE_PLANS_TASK_ID)
    jordan_ids = fight_allowlist_for("jordan", PHONE_PLANS_TASK_ID)
    assert maya_ids == (MAYA_TABLE_ID,)
    assert set(jordan_ids) == set(JORDAN_PLAN_A_IDS)
    maya = items_for_ids(maya_ids)[0]
    jordan = items_for_ids(jordan_ids)[0]
    assert implied_plan_of(maya) == "B"
    assert implied_plan_of(jordan) == "A"
    assert maya["id"] != jordan["id"]
    assert ALEX_FLAT_ADD_ID not in maya_ids


def test_alex_flat_add_tagged_alex_only():
    catalog = {
        m["id"]: m for m in load_misconception_catalog()["misconceptions"]
    }
    alex = catalog[ALEX_FLAT_ADD_ID]
    assert "alex" in [r.lower() for r in (alex.get("roster") or [])]
    maya_row = catalog[MAYA_TABLE_ID]
    assert maya_row["implied_plan"] == "B"
    assert "20.10" not in (maya_row.get("example_wrong_reply") or "")


def test_fe_seed_lines_match_pst_scenario_wording():
    from app.fight_stance import FE_SEED_LINES

    assert "$15" in FE_SEED_LINES["maya"] and "$25" in FE_SEED_LINES["maya"]
    assert "Plan B" in FE_SEED_LINES["maya"]
    assert "triple" in FE_SEED_LINES["jordan"]
    assert "Plan A is obviously better" in FE_SEED_LINES["jordan"]


def test_retrieve_for_group_fight_opposes_without_qdrant():
    task_meta = {
        "task_id": PHONE_PLANS_TASK_ID,
        "description": "Plan A $20+0.10x vs Plan B 0.30x",
        "required_constructs": [
            "rate_comparison",
            "linear_relationship",
            "unit_rate",
            "additive_vs_multiplicative",
            "symbolic_modeling",
        ],
    }
    maya = retrieve_for_group_fight(
        "maya",
        task_meta,
        barrier={"summary": "table missing warrants", "discourse_role": "missing_warrants"},
        peer_picks=[],
        k=1,
    )
    jordan = retrieve_for_group_fight(
        "jordan",
        task_meta,
        barrier={"summary": "rate always wins", "discourse_role": "procedural_imitation"},
        peer_picks=maya,
        k=1,
    )
    assert maya and maya[0]["id"] == MAYA_TABLE_ID
    assert jordan and jordan[0]["id"] in JORDAN_PLAN_A_IDS
    assert implied_plan_of(maya[0]) != implied_plan_of(jordan[0])
    assert maya[0]["id"] != ALEX_FLAT_ADD_ID


def test_fight_allowlist_registry():
    assert "maya" in FIGHT_ALLOWLIST[PHONE_PLANS_TASK_ID]
    assert "jordan" in FIGHT_ALLOWLIST[PHONE_PLANS_TASK_ID]
