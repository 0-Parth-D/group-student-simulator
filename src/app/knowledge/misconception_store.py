import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from app.student.behavior_router import BehaviorTarget
from app.config import QDRANT_PATH, USE_QDRANT
from app.knowledge.kg_config import load_misconception_catalog

logger = logging.getLogger(__name__)

MODE_TO_TAGS = {
    "CONFUSED_HELPSEEKING": ["CONFUSED"],
    "PARTIAL_ATTEMPT_THEN_STUCK": ["PARTIAL", "WRONG"],
    "WRONG": ["WRONG", "PARTIAL"],
    "NORMAL_ERROR_PROFILE": ["PARTIAL", "WRONG"],
}

CATALOG_COLLECTION = "misconceptions"
INSTANCES_COLLECTION = "misconception_instances"

_qdrant_client = None
_instances_ready = False
_catalog_ready = False

# Concrete error utterances (3+ per catalog misconception) for instance-index RAG.
MISCONCEPTION_INSTANCES = {
    "pw_larger_denominator_larger_fraction": [
        "1/4 is bigger because 4 is bigger than 3.",
        "The bottom number is bigger so the fraction has to be bigger.",
        "Four is more than three, so 1/4 wins.",
    ],
    "pw_adds_numerators_and_denominators": [
        "2 plus 1 is 3 and 5 plus 5 is 10, so 3/10.",
        "I added the tops and the bottoms: 3/10.",
        "You add everything — top and bottom — so I got 3/10.",
    ],
    "feq_subtract_across_bar": [
        "5 minus 1 is 4, 6 minus 3 is 3, answer is 4/3.",
        "I got 4 thirds... I just did 5 minus 1 on top and 6 minus 3 on the bottom.",
        "The answer is 4/3, I subtracted both parts.",
        "Easy! Top minus top, bottom minus bottom.",
    ],
    "feq_wrong_equivalent_fraction": [
        "LCD is 6. So 1/3 becomes 1/6. 5/6 minus 1/6 equals 4/6.",
        "I used 12 as the bottom because 6 times 3 is 12.",
        "I made both denominators 9 because that's between 6 and 3.",
        "I changed the bottom to 6 but forgot to fix the top.",
    ],
    "feq_no_simplification": [
        "5/6 minus 2/6 equals 3/6.",
        "I got 3/6 and stopped there.",
        "The answer is 3/6 — that looks right to me.",
    ],
    "rc_reverses_ratio_order": [
        "The ratio is 5:2 because there are 5 blue and 2 red.",
        "5 blue and 2 red, so 5:2.",
        "I wrote blue first so it's 5:2.",
    ],
    "rc_part_to_whole": [
        "2 red out of 7 total, so 2:7.",
        "There are 7 marbles total so the ratio is 2:7.",
        "I compared to the whole pile — 2:7.",
    ],
    "rc_fraction_label_only": [
        "2/5, like a fraction.",
        "It's 2 over 5.",
        "I just wrote it as a fraction 2/5.",
    ],
    "req_additive_scaling": [
        "2:3 then 4:5 because I added 2 to each.",
        "I added 2 to both sides so 4:5.",
        "The pattern is add 2 each time — 4:5.",
    ],
    "req_scales_numerator_only": [
        "I multiplied 2 by 2 to get 4. I left the 3 alone. So 4:3.",
        "Double the first number only — 4:3.",
        "4 on the left, 3 stays, so 4:3.",
    ],
    "ur_inverts_ratio": [
        "3 divided by 150 equals 0.02.",
        "I did hours divided by miles.",
        "150 over 3... wait is it 0.02?",
    ],
    "ur_no_unit_label": [
        "150 divided by 3 equals 50.",
        "The answer is 50.",
        "I got 50 but I'm not sure what it means.",
    ],
    "ur_no_attempt_shutdown": [
        "I don't even know where to start. Can we skip it?",
        "This is too confusing. I give up.",
        "I have no idea what to do on this one.",
    ],
    # Phone plans — LPF PRF constructs (additive/multiplicative, rates, linear, modeling)
    "ur_rate_as_flat_add": [
        "Plan A is just 20 dollars plus 10 cents, so like 20.10 total.",
        "You add the monthly fee and then add ten cents once — that's Plan A.",
        "I don't get multiplying by texts. Isn't it 20 plus 0.10?",
        "So Plan A costs twenty ten, basically.",
    ],
    "ur_cents_vs_dollars_slope": [
        "Plan A is 20 + 10x because each text is 10 cents.",
        "Plan B should be 30x — thirty cents times texts.",
        "I used 10 and 30 in the equations since the problem said 10 cents and 30 cents.",
        "Why would it be 0.1? It says ten cents, so 10x.",
    ],
    "pr_compare_one_x_only": [
        "At 50 texts Plan B is cheaper, so Plan B is always better.",
        "I checked 20 texts and Plan B won, so I'd pick Plan B.",
        "I made a little table up to 40 and Plan B stayed lower.",
        "For normal people who don't text a ton, Plan B is the answer.",
    ],
    "pr_slope_coefficient_only": [
        "It's 20 plus 0.1x. The 0.1 is just the coefficient.",
        "Plan A is 20+0.1x and Plan B is 0.3x — that's the formulas. I don't know what 0.1 means though.",
        "0.3 is the coefficient in front of x for Plan B.",
        "I can write the equations but I'm not sure what the decimals stand for in the story.",
    ],
    "pr_ignores_monthly_fee": [
        "Plan A is just 0.1 times texts, same idea as Plan B but cheaper per text.",
        "Both are like rate times texts — Plan A is 0.1x and Plan B is 0.3x.",
        "I ignored the 20 dollars; that seems separate from the texting cost.",
        "So Plan A is cheaper every text... I didn't put the monthly fee in the equation.",
    ],
    "pr_lower_rate_always_wins": [
        "Plan A is always better because ten cents is less than thirty cents per text.",
        "Lower rate wins, so Plan A, period.",
        "Why would anyone pick Plan B? It's three times more per text.",
        "Plan A has the better deal on texts so it's the better plan.",
    ],
    "pr_swaps_plan_rates": [
        "Plan A is 0.3x and Plan B is 20 plus 0.1x... I think I mixed them up but whatever.",
        "The one with no fee is 20 + 0.1x, right?",
        "Plan B has the monthly fee and Plan A is just 0.3 per text.",
        "I swapped which plan has the 20 dollar fee.",
    ],
    "pr_algebra_crossover_slip": [
        "20 plus 0.1x equals 0.3x, so x is 20 texts.",
        "I set them equal and got 50 texts somehow.",
        "Subtract 0.1x from both sides... I got x equals 20.",
        "They cost the same at like 10 texts I think.",
    ],
    "pr_fee_added_to_both": [
        "Both plans have like a 20 dollar fee, then Plan A adds 0.1x and Plan B adds 0.3x.",
        "I put 20 on Plan B too so it's fair.",
        "Plan B is 20 + 0.3x if they both charge a monthly fee.",
        "Every phone plan has a base fee, so both get +20.",
    ],
    "ur_per_month_not_per_text": [
        "Plan A is 20 plus another 0.10 for the month, and Plan B is just 0.30 for the month.",
        "The 10 cents and 30 cents are monthly charges, not per text.",
        "So Plan A costs 20.10 a month and Plan B costs 0.30 a month.",
        "I treated the rates like flat monthly add-ons.",
    ],
    "lr_assumes_proportional_doubling": [
        "If 50 texts costs $25 on Plan A, then 100 texts costs $50.",
        "Double the texts, double the bill — that's how rates work.",
        "I made a ratio table for Plan A: 50 texts to $25, so 200 texts is $100.",
        "It's proportional, so I can just scale it up.",
    ],
    "lr_line_starts_at_origin": [
        "Both lines start at zero and Plan A just goes up slower.",
        "Every graph starts at the corner, so both plans start at 0 dollars.",
        "At zero texts they both cost nothing, right?",
        "I drew two lines from the origin, one steeper than the other.",
    ],
}


@lru_cache(maxsize=1)
def _all_misconceptions() -> List[dict]:
    return load_misconception_catalog().get("misconceptions", [])


def _misc_by_id(misc_id: str) -> Optional[dict]:
    for item in _all_misconceptions():
        if item.get("id") == misc_id:
            return item
    return None


def _metadata_filter_retrieve(
    target: BehaviorTarget,
    task_meta: dict,
    k: int,
) -> List[dict]:
    required = set(task_meta.get("required_constructs") or [])
    if target.primary_construct:
        required.add(target.primary_construct)

    tags = MODE_TO_TAGS.get(target.mode, ["WRONG", "PARTIAL"])
    candidates = []

    for item in _all_misconceptions():
        cid = item.get("construct_id", "")
        if cid not in required:
            continue
        tag = item.get("behavior_tag", "")
        stack = item.get("stack_level", 1)
        score = 0
        if tag in tags:
            score += 3
        if abs(stack - max(target.stack_level, 1)) <= 1:
            score += 2
        if cid == target.primary_construct:
            score += 1
        if score > 0:
            candidates.append((score, item))

    candidates.sort(key=lambda x: (-x[0], x[1].get("id", "")))
    return [item for _, item in candidates[:k]]


_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        from app.config import EMBEDDER_MODEL

        logger.info("Loading sentence-transformer: %s", EMBEDDER_MODEL)
        _embedder = SentenceTransformer(EMBEDDER_MODEL)
    return _embedder


def _qdrant_vector_search(
    client,
    collection_name: str,
    query_vector: list,
    query_filter=None,
    limit: int = 1,
) -> list:
    """Vector search compatible with qdrant-client >=1.7 (query_points) and older search()."""
    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
        )
        return list(response.points or [])

    return client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=query_filter,
        limit=limit,
    )


def _get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client
    if not USE_QDRANT:
        return None

    try:
        from qdrant_client import QdrantClient
    except ImportError:
        logger.info("qdrant-client not installed; using metadata retrieval")
        return None

    storage = Path(QDRANT_PATH)
    storage.mkdir(parents=True, exist_ok=True)
    try:
        _qdrant_client = QdrantClient(path=str(storage))
        logger.info("Qdrant persistent client at %s", storage)
        return _qdrant_client
    except Exception as exc:
        msg = str(exc).lower()
        if "already accessed" in msg or "lock" in msg:
            logger.warning(
                "Qdrant storage locked by another process (e.g. running backend); "
                "using metadata retrieval fallback"
            )
        else:
            logger.warning("Qdrant unavailable; using metadata retrieval: %s", exc)
        return None


def _collection_exists(client, name: str) -> bool:
    try:
        client.get_collection(name)
        return True
    except Exception:
        return False


def build_instance_index(client=None, embed_fn=None) -> bool:
    """Build misconception_instances collection from MISCONCEPTION_INSTANCES."""
    global _instances_ready
    client = client or _get_qdrant_client()
    if client is None:
        return False

    if embed_fn is None:
        embed_fn = lambda texts: _get_embedder().encode(texts)

    try:
        from qdrant_client.models import Distance, PointStruct, VectorParams
    except ImportError:
        return False

    utterances = []
    for misc_id, lines in MISCONCEPTION_INSTANCES.items():
        for i, utt in enumerate(lines):
            utterances.append((misc_id, i, utt))

    if not utterances:
        return False

    texts = [u[2] for u in utterances]
    vectors = embed_fn(texts)
    if hasattr(vectors, "tolist"):
        vectors = vectors.tolist()

    dim = len(vectors[0])
    if not _collection_exists(client, INSTANCES_COLLECTION):
        client.create_collection(
            collection_name=INSTANCES_COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    else:
        client.delete_collection(INSTANCES_COLLECTION)
        client.create_collection(
            collection_name=INSTANCES_COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    points = []
    for (misc_id, i, utt), vector in zip(utterances, vectors):
        points.append(
            PointStruct(
                id=abs(hash(f"{misc_id}_{i}")) % (2**31),
                vector=vector,
                payload={"misc_id": misc_id, "utterance": utt},
            )
        )
    client.upsert(collection_name=INSTANCES_COLLECTION, points=points)
    _instances_ready = True
    logger.info("Built %s index (%d utterances)", INSTANCES_COLLECTION, len(points))
    return True


def _ensure_instance_index() -> Optional[object]:
    global _instances_ready
    client = _get_qdrant_client()
    if client is None:
        return None
    if _instances_ready and _collection_exists(client, INSTANCES_COLLECTION):
        return client
    if _collection_exists(client, INSTANCES_COLLECTION):
        _instances_ready = True
        return client
    if build_instance_index(client):
        return client
    return None


def _ensure_catalog_index(client) -> bool:
    """Index the catalog once per process; payloads go stale when the JSON is edited."""
    global _catalog_ready

    try:
        from qdrant_client.models import Distance, PointStruct, VectorParams
    except ImportError:
        return False

    if _catalog_ready:
        return True

    items = _all_misconceptions()
    if not items:
        return False

    if _collection_exists(client, CATALOG_COLLECTION):
        client.delete_collection(CATALOG_COLLECTION)

    embedder = _get_embedder()
    texts = [
        f"{m.get('description', '')} {m.get('prompt_cue', '')} {m.get('example_wrong_reply', '')}"
        for m in items
    ]
    vectors = embedder.encode(texts).tolist()

    client.create_collection(
        collection_name=CATALOG_COLLECTION,
        vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
    )

    points = []
    for idx, (item, vector) in enumerate(zip(items, vectors)):
        points.append(
            PointStruct(
                id=idx,
                vector=vector,
                payload={
                    "id": item.get("id", str(idx)),
                    "construct_id": item.get("construct_id", ""),
                    "stack_level": item.get("stack_level", 1),
                    "behavior_tag": item.get("behavior_tag", ""),
                    "description": item.get("description", ""),
                    "example_wrong_reply": item.get("example_wrong_reply", ""),
                    "prompt_cue": item.get("prompt_cue", ""),
                    "error_type": item.get("error_type", ""),
                    "source": item.get("source", ""),
                },
            )
        )
    client.upsert(collection_name=CATALOG_COLLECTION, points=points)
    _catalog_ready = True
    logger.info("Qdrant catalog index ready (%s misconceptions)", len(items))
    return True


def retrieve_error_instance(
    teacher_utterance: str,
    active_misc_id: Optional[str],
    construct_id: str = "",
) -> Optional[str]:
    """Per-turn retrieval of a concrete error utterance filtered by misconception id."""
    if not active_misc_id:
        return None

    fallback = MISCONCEPTION_INSTANCES.get(active_misc_id)
    catalog = _misc_by_id(active_misc_id)
    if not fallback and catalog:
        ex = catalog.get("example_wrong_reply")
        if ex:
            return ex

    client = _ensure_instance_index()
    if client is None:
        return fallback[0] if fallback else None

    try:
        from qdrant_client.models import FieldCondition, Filter, MatchValue
    except ImportError:
        return fallback[0] if fallback else None

    query = (
        f"Student weak in {construct_id} responding to teacher: {teacher_utterance}"
    )
    vector = _get_embedder().encode([query])[0].tolist()

    hits = _qdrant_vector_search(
        client,
        INSTANCES_COLLECTION,
        vector,
        query_filter=Filter(
            must=[FieldCondition(key="misc_id", match=MatchValue(value=active_misc_id))]
        ),
        limit=1,
    )
    if hits:
        return (hits[0].payload or {}).get("utterance")

    return fallback[0] if fallback else None


def _qdrant_catalog_retrieve(
    target: BehaviorTarget,
    task_meta: dict,
    k: int,
) -> Optional[List[dict]]:
    client = _get_qdrant_client()
    if client is None or not _ensure_catalog_index(client):
        return None

    try:
        from qdrant_client.models import FieldCondition, Filter, MatchAny
    except ImportError:
        return None

    required = list(task_meta.get("required_constructs") or [])
    if target.primary_construct and target.primary_construct not in required:
        required.append(target.primary_construct)
    if not required:
        return None

    tags = MODE_TO_TAGS.get(target.mode, ["WRONG", "PARTIAL"])
    query_text = (
        f"{target.mode} construct {target.primary_construct} "
        f"level {target.stack_level} " + " ".join(target.error_types[:2])
    )
    vector = _get_embedder().encode([query_text])[0].tolist()

    must = [FieldCondition(key="construct_id", match=MatchAny(any=required))]
    if tags:
        must.append(FieldCondition(key="behavior_tag", match=MatchAny(any=tags)))

    try:
        hits = _qdrant_vector_search(
            client,
            CATALOG_COLLECTION,
            vector,
            query_filter=Filter(must=must),
            limit=k,
        )
    except Exception as exc:
        logger.warning("Qdrant catalog search failed, using metadata fallback: %s", exc)
        return None
    if not hits:
        return None

    results = []
    for hit in hits:
        payload = dict(hit.payload or {})
        payload["score"] = hit.score
        results.append(payload)
    return results


def retrieve_misconceptions(
    target: BehaviorTarget,
    task_meta: dict,
    k: int = 3,
) -> List[dict]:
    """Retrieve misconceptions via Qdrant catalog index with metadata-filter fallback."""
    if USE_QDRANT:
        qdrant_hits = _qdrant_catalog_retrieve(target, task_meta, k)
        if qdrant_hits:
            return qdrant_hits
    return _metadata_filter_retrieve(target, task_meta, k)


def misconceptions_to_playbook(misconceptions: List[dict]) -> dict:
    patterns = []
    for m in misconceptions:
        patterns.append(
            {
                "concept": m.get("construct_id", "?"),
                "type": m.get("error_type", "procedural_confusion"),
                "example": m.get("example_wrong_reply", m.get("description", "")),
                "prompt_cue": m.get("prompt_cue", ""),
                "behavior_tag": m.get("behavior_tag", ""),
                "misc_id": m.get("id", ""),
                "stack_level": int(m.get("stack_level", 1)),
            }
        )
    return {
        "patterns": patterns,
        "stay_in_problem": (
            "Make mistakes only while working on the assigned problem. "
            "Use the error examples above — do not invent new error types."
        ),
    }


def build_error_anchor(
    retrieved_error_instance: Optional[str],
    active_misconception: Optional[dict],
) -> Optional[str]:
    """Few-shot error anchor for student prompt (Phase 3)."""
    if retrieved_error_instance:
        return (
            "[EXAMPLE OF YOUR ERROR — match this pattern in your response]\n"
            f'When asked a similar question, a student like you said:\n'
            f'"{retrieved_error_instance}"\n'
            "Make the same *type* of mistake, not necessarily the same numbers."
        )
    if active_misconception:
        cue = active_misconception.get("prompt_cue") or active_misconception.get("description", "")
        if cue:
            return f"[ERROR PATTERN]: {cue}"
    return None
