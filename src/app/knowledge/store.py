"""Optional Neo4j backend for LP mastery state. Falls back to in-memory YAML graph when unset."""

import logging
from typing import Dict, List, Optional

from app.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from app.knowledge import graph as mem_kg
from app.knowledge.kg_config import load_knowledge_graph

logger = logging.getLogger(__name__)

_driver = None


def neo4j_enabled() -> bool:
    return bool(NEO4J_URI and NEO4J_PASSWORD)


def _get_driver():
    global _driver
    if _driver is not None:
        return _driver
    if not neo4j_enabled():
        return None
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        _driver = driver
        logger.info("Neo4j connected at %s", NEO4J_URI)
        return _driver
    except Exception as exc:
        logger.warning("Neo4j unavailable, using in-memory KG: %s", exc)
        return None


def seed_schema() -> bool:
    driver = _get_driver()
    if driver is None:
        return False

    kg = load_knowledge_graph()
    constructs = kg.get("constructs", {})
    prereqs = kg.get("prerequisites", [])

    with driver.session() as session:
        for cid, cdef in constructs.items():
            session.run(
                """
                MERGE (c:Construct {id: $id})
                SET c.label = $label, c.target_stack_level = $target
                """,
                id=cid,
                label=cdef.get("label", cid),
                target=cdef.get("target_stack_level", 3),
            )
        for row in prereqs:
            if len(row) >= 2:
                session.run(
                    """
                    MATCH (a:Construct {id: $from_id}), (b:Construct {id: $to_id})
                    MERGE (a)-[:PREREQUISITE_OF]->(b)
                    """,
                    from_id=row[0],
                    to_id=row[1],
                )
    return True


def init_mastery_state(student_id: str, construct_mastery: Dict[str, int]) -> Dict[str, int]:
    state = mem_kg.init_mastery_state(construct_mastery)
    driver = _get_driver()
    if driver is None:
        return state

    with driver.session() as session:
        session.run("MERGE (s:Student {id: $id})", id=student_id)
        for cid, level in state.items():
            session.run(
                """
                MATCH (s:Student {id: $sid}), (c:Construct {id: $cid})
                MERGE (s)-[o:OBSERVED_IN]->(c)
                SET o.level = $level
                """,
                sid=student_id,
                cid=cid,
                level=level,
            )
    return state


def get_prerequisite_gaps(mastery_state: Dict[str, int], required_constructs: List[str]) -> dict:
    return mem_kg.get_prerequisite_gaps(mastery_state, required_constructs)


def get_student_state(mastery_state: Dict[str, int], construct_id: str) -> dict:
    return mem_kg.get_student_state(mastery_state, construct_id)


def update_observation(
    student_id: str,
    mastery_state: Dict[str, int],
    construct_ids: List[str],
    reply_quality: str,
    learning_profile=None,
    scaffold_boost: Optional[Dict[str, float]] = None,
) -> Dict[str, int]:
    updated = mem_kg.update_mastery_state(
        mastery_state,
        construct_ids,
        reply_quality,
        learning_profile=learning_profile,
        scaffold_boost=scaffold_boost,
    )
    driver = _get_driver()
    if driver is None:
        return updated

    with driver.session() as session:
        for cid in construct_ids:
            if cid not in updated:
                continue
            session.run(
                """
                MATCH (s:Student {id: $sid})-[o:OBSERVED_IN]->(c:Construct {id: $cid})
                SET o.level = $level
                """,
                sid=student_id,
                cid=cid,
                level=updated[cid],
            )
    return updated


def infer_reply_quality(
    student_reply: str,
    task_meta: dict,
    behavior_target: Optional[dict] = None,
    mastery_state: Optional[Dict[str, int]] = None,
    use_semantic: bool = False,
) -> str:
    return mem_kg.infer_reply_quality(
        student_reply,
        task_meta,
        behavior_target,
        mastery_state=mastery_state,
        use_semantic=use_semantic,
    )
