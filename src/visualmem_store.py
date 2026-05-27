"""Core VisualMemoryStore class and shared store helpers."""

from __future__ import annotations

import json
from pathlib import Path

from .visualmem_extraction import VisualMemoryExtractionMixin
from .visualmem_mutations import VisualMemoryMutationMixin
from .visualmem_pending import VisualMemoryPendingMixin
from .visualmem_search import VisualMemorySearchMixin
from .visualmem_utils import (
    DEFAULT_MEMORY_PATH,
    DEFAULT_MODEL,
    _dedupe_strings,
    _empty_memory,
    _norm,
    _now,
)

from .qdrant import QdrantManager


class VisualMemoryStore(
    VisualMemoryExtractionMixin,
    VisualMemorySearchMixin,
    VisualMemoryMutationMixin,
    VisualMemoryPendingMixin,
):
    """One user's visual memory.

    Owns the on-disk JSON file and (optionally) the LLM client and Qdrant
    collection used to extract and search facts.
    """

    def __init__(
        self,
        path: str | Path = DEFAULT_MEMORY_PATH,
        *,
        llm_client=None,
        model_name: str = DEFAULT_MODEL,
        qdrant_manager: "QdrantManager | None" = None,
    ):
        self.path = Path(path)
        self.memory_path = str(self.path)
        self.llm_client = llm_client
        self.model_name = model_name
        self.qdrant_manager = qdrant_manager

        self._data: dict = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                return data
            except (json.JSONDecodeError, KeyError):
                pass
        return _empty_memory()

    def reload(self) -> None:
        """Re-read `self._data` from disk (no-op if file is unchanged)."""
        self._data = self._load()

    def save(self) -> None:
        self._data["last_updated"] = _now()
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False))
        tmp.replace(self.path)

    @property
    def user(self) -> dict:
        return self._data["main_user"]

    @property
    def private_locations(self) -> list:
        return self.user.setdefault("private_locations", [])

    @property
    def relationships(self) -> dict:
        return self._data["relationships"]

    @property
    def images(self) -> dict:
        return self._data["images"]

    @property
    def pending(self) -> dict:
        return self._data["pending"]

    def _require_llm(self, op: str) -> None:
        if self.llm_client is None:
            raise RuntimeError(
                f"VisualMemoryStore.{op} requires llm_client; pass it to "
                "the VisualMemoryStore constructor."
            )

    def _scene_ownership(self, value: str | None) -> str:
        clean = _norm(value or "unknown")
        allowed = {
            "user_space",
            "other_person_space",
            "public_space",
            "unknown",
        }
        return clean if clean in allowed else "unknown"

    def _normalize_pending_types(self, values) -> list[str]:
        if not values:
            return []
        raw_values = values if isinstance(values, list) else [values]
        out: list[str] = []
        for value in raw_values:
            if not isinstance(value, str):
                continue
            clean = _norm(value)
            if clean in {"identity", "scene"} and clean not in out:
                out.append(clean)
        return out

    def _pending_types(self, pend: dict) -> list[str]:
        types = self._normalize_pending_types(pend.get("pending_types"))
        if not types:
            types = self._normalize_pending_types(pend.get("pending_type"))
        return types

    def _set_pending_types(self, pend: dict, pending_types: list[str]) -> None:
        normalized = self._normalize_pending_types(pending_types)
        pend["pending_types"] = normalized
        pend["pending_type"] = normalized[0] if normalized else None

    def _identity_candidates_from_context(self, candidates: list[str]) -> list[str]:
        return _dedupe_strings([
            c for c in candidates
            if isinstance(c, str) and c.strip()
        ])

    def _all_known_face_names(self) -> list[str]:
        names: list[str] = []
        if self.user.get("face_image"):
            names.append("User")
        names.extend(
            rel.get("name", "")
            for rel in self.relationships.values()
            if rel.get("face_image")
        )
        return _dedupe_strings(names)

    def _identity_names_for_pending(self, pend: dict) -> list[str]:
        if pend.get("load_all_known_faces"):
            return self._all_known_face_names()
        if "identity_candidates" in pend:
            names = list(pend.get("identity_candidates") or [])
        else:
            names = ["User"] + list(pend.get("named_people_in_context") or [])
        return _dedupe_strings(names)

    def _has_face_ref(self, name: str) -> bool:
        if name.lower().strip() == "user":
            return bool(self.user.get("face_image"))
        rel = self._find_rel_by_name(name)
        return bool(rel and rel.get("face_image"))

    def _identity_refs_available_for_pending(self, pend: dict) -> bool:
        names = self._identity_names_for_pending(pend)
        if pend.get("load_all_known_faces"):
            return bool(names)
        return bool(names) and any(self._has_face_ref(name) for name in names)

    def _build_identity_context_for_pending(
        self, pend: dict,
    ) -> tuple[str, list[str]]:
        return self.build_identity_context(self._identity_names_for_pending(pend))

    def _identity_resolution_note_for_pending(self, pend: dict) -> str:
        if "identity" not in self._pending_types(pend):
            return ""
        candidates = self._identity_names_for_pending(pend)
        if not candidates:
            return ""
        with_refs = [name for name in candidates if self._has_face_ref(name)]
        without_refs = [name for name in candidates if name not in with_refs]
        lines = [
            "IDENTITY RESOLUTION NOTE: This image was previously pending for identity.",
            "Candidate identities: " + ", ".join(candidates) + ".",
        ]
        if with_refs:
            lines.append(
                "Available face references: " + ", ".join(with_refs) + "."
            )
        if without_refs:
            lines.append(
                "No face reference is available for: "
                + ", ".join(without_refs)
                + "."
            )
        lines.append(
            "Use available face references to identify visible people. If there "
            "is one main visible person and they clearly do not match an "
            "available candidate reference, use the remaining plausible "
            "candidate from the context only when the candidate set is exhaustive."
        )
        return "\n".join(lines)

    def get_relationship(self, name: str) -> dict | None:
        return self._find_rel_by_name(name)

    def build_identity_context(
        self, names: list[str],
    ) -> tuple[str, list[str]]:
        """For a list of people, build a prompt-ready context block + the
        ordered list of face image paths to send to the LLM.

        Returns ``(context_text, face_image_paths)``.
        """
        lines: list[str] = []
        face_images: list[str] = []
        img_idx = 1

        for name in names:
            name_lower = name.lower().strip()
            if name_lower == "user":
                u = self.user
                if u.get("face_image"):
                    lines.append(
                        f"Image {img_idx}: Reference photo of USER "
                        f"({u.get('name') or 'User'})"
                    )
                    face_images.append(u["face_image"])
                    img_idx += 1
                else:
                    lines.append("USER: no face image available yet")
            else:
                rel = self._find_rel_by_name(name)
                if rel and rel.get("face_image"):
                    lines.append(
                        f"Image {img_idx}: Reference photo of {rel['name']} "
                        f"({rel['relationship']})"
                    )
                    face_images.append(rel["face_image"])
                    img_idx += 1
                elif rel:
                    lines.append(
                        f"{rel['name']} ({rel['relationship']}): known but "
                        "no face image"
                    )
                else:
                    lines.append(f"{name}: not in memory yet")

        return "\n".join(lines), face_images
