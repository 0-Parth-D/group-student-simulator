#!/usr/bin/env python3
"""One-way sync: student-simulation backend -> group-student-simulator (clean backend)."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT.parent / "student-simulation" / "backend"
DST = ROOT / "backend"

APP_SKIP = frozenset({"sessions.py", "llama_infer.py", "main.py"})
EVAL_SKIP = frozenset({"run_battery.py", "run_eval.py", "behavioral_consistency.py"})


def _copy_tree(src_dir: Path, dst_dir: Path, *, skip_names: frozenset[str] = frozenset()) -> int:
    count = 0
    if not src_dir.is_dir():
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    for path in src_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src_dir)
        if rel.name in skip_names:
            continue
        if "__pycache__" in rel.parts:
            continue
        if rel.parts[:1] == ("results",) and rel.name != ".gitkeep":
            continue
        target = dst_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        count += 1
    return count


def _patch_group_sessions() -> None:
    path = DST / "app" / "group_sessions.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "from app.sessions import TEACHER_OPENER_TEMPLATE",
        "from app.session_shared import TEACHER_OPENER_TEMPLATE",
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    if not SRC.is_dir():
        raise SystemExit(f"SOURCE not found: {SRC}")
    n_app = _copy_tree(SRC / "app", DST / "app", skip_names=APP_SKIP)
    n_cfg = _copy_tree(SRC / "config", DST / "config")
    n_eval = _copy_tree(SRC / "eval", DST / "eval", skip_names=EVAL_SKIP)
    n_tests = _copy_tree(SRC / "tests", DST / "tests")
    n_docs = _copy_tree(SRC / "docs", DST / "docs")
    _patch_group_sessions()

    # Root docs
    for name in ("lpf-source-mapping.md",):
        src_doc = SRC.parent / "docs" / name
        if src_doc.is_file():
            (ROOT / "docs" / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_doc, ROOT / "docs" / name)

    # Env example: copy group-relevant tail from source
    src_env = SRC.parent / ".env.example"
    if src_env.is_file():
        shutil.copy2(src_env, ROOT / ".env.example")

    print(
        f"Synced from {SRC}\n"
        f"  app: {n_app} files (skipped {sorted(APP_SKIP)})\n"
        f"  config: {n_cfg}\n"
        f"  eval: {n_eval} (skipped {sorted(EVAL_SKIP)})\n"
        f"  tests: {n_tests}\n"
        f"  backend/docs: {n_docs}\n"
        f"  patched group_sessions -> session_shared"
    )


if __name__ == "__main__":
    main()
