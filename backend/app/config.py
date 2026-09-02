import os
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent

CLAUDE_THINKING_MODELS = {"protected.Claude Sonnet 4.6", "protected.Claude Opus 4.1"}

TAMU_CHAT_API_KEY = os.getenv("TAMU_CHAT_API_KEY", "")
TAMU_CHAT_BASE_URL = os.getenv("TAMU_CHAT_BASE_URL", "https://chat-api.tamu.ai/openai")

MODEL = os.getenv("OPENAI_MODEL", "protected.gpt-5.4")
# Student replies via TAMU Chat API. Falls back to OPENAI_MODEL.
STUDENT_OPENAI_MODEL = os.getenv("STUDENT_OPENAI_MODEL") or MODEL
# Output caps for student replies (call sites use these instead of literals)
STUDENT_MAX_TOKENS_MATH = int(os.getenv("STUDENT_MAX_TOKENS_MATH", "200"))
STUDENT_MAX_TOKENS_SOCIAL = int(os.getenv("STUDENT_MAX_TOKENS_SOCIAL", "280"))
# Thinking-model floor for STUDENT_REPLY only (not judges). Default 384, not 2000.
STUDENT_THINKING_MIN_MAX_TOKENS = int(
    os.getenv("STUDENT_THINKING_MIN_MAX_TOKENS", "384")
)
# Optional fixed seed for student complete_chat; empty = random per call
STUDENT_LLM_SEED = os.getenv("STUDENT_LLM_SEED", "").strip()
# Fast/cheap model for turn tagging only (recommended: gpt-4o-mini on TAMU Chat).
# Empty → falls back to OPENAI_MODEL.
TURN_CLASSIFIER_MODEL = (
    os.getenv("TURN_CLASSIFIER_MODEL", "").strip()
    or os.getenv("OPENAI_MODEL", "protected.gpt-5.4")
)
# semantic_llm (default): hard gates → local embeddings → cheap LLM with rich context
# heuristic: keyword/discourse gates only (offline / no API)
# llm: always call classifier model except empty/ack-only hard gates
TURN_CLASSIFY_MODE = os.getenv("TURN_CLASSIFY_MODE", "semantic_llm").lower().strip()
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
REFINE_TEMP = float(os.getenv("REFINE_TEMPERATURE", "0.7"))

# Student reply backend: openai (default) | llama (local HF + optional LoRA)
STUDENT_LLM_BACKEND = os.getenv("STUDENT_LLM_BACKEND", "openai").lower()
LLAMA_BASE_MODEL = os.getenv(
    "LLAMA_BASE_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct"
)
LLAMA_ADAPTER_PATH = os.getenv("LLAMA_ADAPTER_PATH", "")
LLAMA_4BIT = os.getenv("LLAMA_4BIT", "true").lower() in ("1", "true", "yes")
LLAMA_DEVICE = os.getenv("LLAMA_DEVICE", "")  # cuda | mps | cpu | empty=auto
LLAMA_MAX_NEW_TOKENS = int(os.getenv("LLAMA_MAX_NEW_TOKENS", "200"))

# Pipeline hyperparameters (Wu et al.)
M = 5
N = 2
p = 3
L = 3
B = 2
DELTA = 0.9
NUM_TURNS = 8

EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL", "paraphrase-MiniLM-L6-v2")

# Learning progression / GraphRAG stack
USE_QDRANT = os.getenv("USE_QDRANT", "true").lower() in ("1", "true", "yes")
QDRANT_PATH = os.getenv("QDRANT_PATH", str(_BACKEND_ROOT / "qdrant_storage"))
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# Concept activation for interactive tutoring
BAD_CONCEPT_INJECT = int(os.getenv("BAD_CONCEPT_INJECT", "5"))
MAX_ACTIVE_CONCEPTS = int(os.getenv("MAX_ACTIVE_CONCEPTS", "8"))

# Embedding fallback for scaffold detection, so a scenario with no hand-written
# keywords still produces graded per-construct scaffold credit.
SCAFFOLD_SEMANTIC = os.getenv("SCAFFOLD_SEMANTIC", "true").lower() in ("1", "true", "yes")
SCAFFOLD_SEMANTIC_THRESHOLD = float(os.getenv("SCAFFOLD_SEMANTIC_THRESHOLD", "0.30"))

# Group facilitation constraint parser: hybrid (default) | llm | regex
# hybrid = LLM parse + regex must_not veto (Track C bake-off winner).
# regex kept only as hybrid veto / offline gold baseline (--parser regex).
GROUP_CONSTRAINT_PARSER = os.getenv("GROUP_CONSTRAINT_PARSER", "hybrid").lower().strip()

# Turn classifier: embedding prototypes for vague vs scaffold (optional)
TURN_CLASSIFY_SEMANTIC = os.getenv("TURN_CLASSIFY_SEMANTIC", "true").lower() in (
    "1",
    "true",
    "yes",
)
TURN_CLASSIFY_SEMANTIC_MARGIN = float(os.getenv("TURN_CLASSIFY_SEMANTIC_MARGIN", "0.05"))
LLM_VAGUE_MIN_CONFIDENCE = float(os.getenv("LLM_VAGUE_MIN_CONFIDENCE", "0.70"))

# Game-style orchestrator loop (mirrors pst-training-game); runs after primary reply(ies)
GROUP_GAME_FLOW = os.getenv("GROUP_GAME_FLOW", "true").lower() in ("1", "true", "yes")
GROUP_GAME_FLOW_MAX_ROUNDS = int(os.getenv("GROUP_GAME_FLOW_MAX_ROUNDS", "8"))
GROUP_GAME_FLOW_STUDENT_CAP = int(os.getenv("GROUP_GAME_FLOW_STUDENT_CAP", "4"))

# LangGraph volunteer peer continuation (discuss/open/critique fallback)
GROUP_MULTI_ROUND = os.getenv("GROUP_MULTI_ROUND", "true").lower() in ("1", "true", "yes")
GROUP_PEER_MAX_ROUNDS = int(os.getenv("GROUP_PEER_MAX_ROUNDS", "8"))
GROUP_PEER_NUDGE_ROUNDS = int(os.getenv("GROUP_PEER_NUDGE_ROUNDS", "3"))
GROUP_TRANSCRIPT_WINDOW = int(os.getenv("GROUP_TRANSCRIPT_WINDOW", "14"))
GROUP_PRIVATE_HIST_MAX = int(os.getenv("GROUP_PRIVATE_HIST_MAX", "16"))

# Comma-separated browser origins; empty = allow all (dev default)
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").strip()

# Track B: session-fixed scaffold receptivity + move gate (group only)
GROUP_TRACK_B = os.getenv("GROUP_TRACK_B", "true").lower() in ("1", "true", "yes")

# Student reply refinement: llm (critique+rewrite) | off
REFINE_MODE = (os.getenv("REFINE_MODE", "llm") or "llm").lower().strip()
REFINE_MAX_REVISIONS = int(os.getenv("REFINE_MAX_REVISIONS", "1"))

# Per-turn prompt/target/draft logs in eval_log exports (off by default)
GENERATION_DEBUG = os.getenv("GENERATION_DEBUG", "").lower() in ("1", "true", "yes")
GENERATION_DEBUG_HIST_MAX = int(os.getenv("GENERATION_DEBUG_HIST_MAX", "12"))

# Soft cap for group math system prompts (prompt-diet regression check)
GROUP_MATH_PROMPT_CHAR_SOFT_CAP = int(
    os.getenv("GROUP_MATH_PROMPT_CHAR_SOFT_CAP", "6500")
)
