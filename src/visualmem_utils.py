"""Shared helpers and constants for the visual-memory modules."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from PIL import Image
from google.genai import types


import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_PATH = "visual_memory.json"
DEFAULT_MODEL = "gemini-3.1-pro-preview"
AI_RESPONSE_LOG = "ai_responses.txt"


def query_gemini(client, text_prompt: str, input_images: list[str] = None, model_name: str ="gemini-3.1-pro-preview", system_prompt: str = "You are a helpful data creator."):
    """Query Gemini API with the given prompt."""
    contents = [text_prompt]
    if input_images:
        for img_path in input_images:
            img = Image.open(img_path)
            contents.append(img)

    last_err = None
    for attempt in range(1, 4):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                ),
            )
            return response.text
        except Exception as e:
            last_err = e
            print(f"Error querying Gemini (attempt {attempt}/3): {e}")
    print(f"Error querying Gemini: giving up after 3 attempts: {last_err}")
    return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uid(prefix: str = "id") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _img_id(path: str) -> str:
    """Deterministic image ID derived from a file path."""
    return f"img_{hashlib.sha256(path.encode()).hexdigest()[:12]}"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip().rstrip("."))


def _dedupe_strings(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        clean = value.strip()
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(clean)
    return out


def _normalize_face_visibility_score(value) -> int | float | None:
    """Return a numeric face visibility score clamped to 0..10."""
    if value is None or isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(score):
        return None
    score = min(10.0, max(0.0, score))
    return int(score) if score.is_integer() else score


def _face_visibility_score_value(value) -> float:
    score = _normalize_face_visibility_score(value)
    return -1.0 if score is None else float(score)


def _face_visibility_score_is_better(new_score, old_score) -> bool:
    normalized_new = _normalize_face_visibility_score(new_score)
    if normalized_new is None:
        return False
    return (
        _face_visibility_score_value(normalized_new)
        > _face_visibility_score_value(old_score)
    )


def _normalize_location_name(name: str | None) -> str:
    """Return a strict user-private location name for text matching."""
    clean = _norm(name or "")
    if not clean:
        return ""
    clean = re.sub(r"^(?:the\s+)?(?:user's|users|my|our)\s+", "", clean)
    clean = re.sub(r"^(?:the\s+)?user\s+", "", clean)
    clean = re.sub(r"^the\s+", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def _location_names(value) -> list[str]:
    """Normalize a string/list/dict location payload into location names."""
    if not value:
        return []
    if isinstance(value, dict):
        return _location_names(value.get("name"))
    if isinstance(value, str):
        name = _normalize_location_name(value)
        return [name] if name else []
    if isinstance(value, list):
        names: list[str] = []
        for item in value:
            names.extend(_location_names(item))
        return _dedupe_strings(names)
    return []

_UNKNOWN_REL_NAMES = {"", "unknown", "unknown person", "unnamed", "none", "null"}
_FATHER_ALIASES = {
    "dad",
    "father",
    "my dad",
    "my father",
    "the dad",
    "the father",
}
_MOTHER_ALIASES = {
    "mom",
    "mother",
    "mum",
    "my mom",
    "my mother",
    "my mum",
    "the mom",
    "the mother",
    "the mum",
}
_ROLE_ALIASES = {
    **{alias: "father" for alias in _FATHER_ALIASES},
    **{alias: "mother" for alias in _MOTHER_ALIASES},
    "brother": "brother",
    "my brother": "brother",
    "the brother": "brother",
    "sister": "sister",
    "my sister": "sister",
    "the sister": "sister",
    "sibling": "sibling",
    "my sibling": "sibling",
    "the sibling": "sibling",
    "neighbor": "neighbor",
    "my neighbor": "neighbor",
    "the neighbor": "neighbor",
    "friend": "friend",
    "my friend": "friend",
    "the friend": "friend",
    "coworker": "coworker",
    "co-worker": "coworker",
    "my coworker": "coworker",
    "my co-worker": "coworker",
    "colleague": "colleague",
    "my colleague": "colleague",
    "boss": "boss",
    "my boss": "boss",
    "manager": "manager",
    "my manager": "manager",
    "acquaintance": "acquaintance",
    "maintenance worker": "maintenance worker",
    "the maintenance worker": "maintenance worker",
    "receptionist": "receptionist",
    "the receptionist": "receptionist",
}
_ROLE_NAME_ALIASES = set(_ROLE_ALIASES)


def _canonical_relationship(rel: str | None) -> str:
    clean = _norm(rel or "")
    if clean in _ROLE_ALIASES:
        return _ROLE_ALIASES[clean]
    return clean or "unknown"


def _canonical_rel_name(name: str | None) -> str:
    clean = _norm(name or "")
    if clean in _ROLE_NAME_ALIASES or clean in _UNKNOWN_REL_NAMES:
        return "unknown"
    return (name or "").strip()

def _is_role_or_unknown_rel_name(name: str | None) -> bool:
    clean = _norm(name or "")
    return clean in _UNKNOWN_REL_NAMES or clean in _ROLE_NAME_ALIASES


def _safe_json_load(raw: str | None) -> dict | None:
    """Strip ```json fences and parse JSON. Returns None on failure."""
    if not raw:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _query_llm(*args, **kwargs):
    return query_gemini(*args, **kwargs)


def _log_ai_response(
    label: str,
    response: str | None,
    prompt: str = "",
    system_prompt: str = "",
    image_paths: list[str] | None = None,
) -> None:
    """Append an AI prompt + response (and any image paths) to the log file."""
    with open(AI_RESPONSE_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"[{_now()}] {label}\n")
        f.write(f"{'=' * 60}\n")
        # if system_prompt:
        #     f.write(f"[SYSTEM PROMPT]\n{system_prompt}\n\n")
        if prompt:
            f.write(f"[PROMPT]\n{prompt}\n\n")
        if image_paths:
            f.write("[IMAGE PATHS]\n")
            for p in image_paths:
                f.write(f"  - {p}\n")
            f.write("\n")
        f.write(f"[RESPONSE]\n{str(response)}\n")


def _empty_memory() -> dict:
    return {
        "last_updated": _now(),
        "main_user": {
            "name": None,
            "face_image": None,
            "face_image_visibility_score": None,
            "description": None,
            "objects": [],
            "pets": [],
            "private_locations": [],
            "facts": [],
            "confirmed": False,
        },
        "relationships": {},
        "images": {},
        "pending": {},
    }


def _sanitize(name: str) -> str:
    """Make a filename- and Qdrant-collection-safe slug from a user_id."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_") or "user"


def _render_user_facts(user_facts: list[dict]) -> str:
    """Concatenate fact ``statement`` fields into a single string."""
    parts = []
    for f in user_facts or []:
        stmt = (f.get("statement") or "").strip()
        if stmt:
            parts.append(stmt if stmt.endswith(".") else stmt + ".")
    return " ".join(parts)


def _render_conversation_context(messages: list[dict]) -> str:
    """Render all turns in a conversation as role-prefixed text."""
    lines = []
    for i, msg in enumerate(messages, start=1):
        role = str(msg.get("role") or "user").strip() or "user"
        content = str(msg.get("content") or "").strip()
        image_path = msg.get("image_path")

        if image_path:
            image_note = "<image>"
            content = f"{content}\n{image_note}" if content else image_note

        if content:
            lines.append(f"Turn {i} ({role}): {content}")

    return "\n".join(lines)