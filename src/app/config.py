import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"

CLAUDE_THINKING_MODELS = {"protected.Claude Sonnet 4.6", "protected.Claude Opus 4.1"}

TAMU_CHAT_API_KEY = os.getenv("TAMU_CHAT_API_KEY", "")
TAMU_CHAT_BASE_URL = os.getenv("TAMU_CHAT_BASE_URL", "https://chat-api.tamu.ai/openai")

MODEL = os.getenv("OPENAI_MODEL", "protected.gpt-5.4")
STUDENT_OPENAI_MODEL = os.getenv("STUDENT_OPENAI_MODEL") or MODEL
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
REFINE_TEMP = float(os.getenv("REFINE_TEMPERATURE", "0.7"))

# Pipeline hyperparameters (Wu et al.)
M = 5
N = 2
p = 3
L = 3
B = 2
DELTA = 0.9
NUM_TURNS = 8

EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL", "paraphrase-MiniLM-L6-v2")

USE_QDRANT = os.getenv("USE_QDRANT", "true").lower() in ("1", "true", "yes")
QDRANT_PATH = os.getenv("QDRANT_PATH", str(REPO_ROOT / "data" / "qdrant_storage"))

# Optional Neo4j. Default path is the in-memory YAML knowledge graph.
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

BAD_CONCEPT_INJECT = int(os.getenv("BAD_CONCEPT_INJECT", "5"))
MAX_ACTIVE_CONCEPTS = int(os.getenv("MAX_ACTIVE_CONCEPTS", "8"))

SCAFFOLD_SEMANTIC = os.getenv("SCAFFOLD_SEMANTIC", "true").lower() in ("1", "true", "yes")
SCAFFOLD_SEMANTIC_THRESHOLD = float(os.getenv("SCAFFOLD_SEMANTIC_THRESHOLD", "0.30"))

GROUP_CONSTRAINT_PARSER = os.getenv("GROUP_CONSTRAINT_PARSER", "hybrid").lower().strip()

GROUP_MULTI_ROUND = os.getenv("GROUP_MULTI_ROUND", "true").lower() in ("1", "true", "yes")
GROUP_PEER_MAX_ROUNDS = int(os.getenv("GROUP_PEER_MAX_ROUNDS", "8"))
GROUP_PEER_NUDGE_ROUNDS = int(os.getenv("GROUP_PEER_NUDGE_ROUNDS", "3"))
GROUP_TRANSCRIPT_WINDOW = int(os.getenv("GROUP_TRANSCRIPT_WINDOW", "10"))
GROUP_PRIVATE_HIST_MAX = int(os.getenv("GROUP_PRIVATE_HIST_MAX", "12"))

GROUP_TRACK_B = os.getenv("GROUP_TRACK_B", "true").lower() in ("1", "true", "yes")
