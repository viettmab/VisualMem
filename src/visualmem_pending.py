"""Pending-image resolution helpers for VisualMemoryStore."""

from __future__ import annotations

import random

from .prompts import SCENE_MATCH_PROMPT
from .visualmem_utils import (
    _dedupe_strings,
    _img_id,
    _location_names,
    _log_ai_response,
    _normalize_location_name,
    _query_llm as query_llm,
    _safe_json_load,
)


class VisualMemoryPendingMixin:
    def _sweep_pending_for_names(
        self,
        newly_confirmed_names: set[str],
        *,
        skip_fact_apply: bool = False,
    ) -> list[dict]:
        """Resolve any pending images whose required people are now known."""
        results: list[dict] = []
        pending_ids = list(self.pending.keys())

        for pend_id in pending_ids:
            if pend_id not in self.pending:
                continue  # already resolved by a recursive sweep
            pend = self.pending[pend_id]
            if "identity" not in self._pending_types(pend):
                continue
            needed_names = _dedupe_strings(
                list(pend.get("identity_candidates") or [])
                + list(pend.get("named_people_in_context") or [])
            )

            if (
                not pend.get("load_all_known_faces")
                and not newly_confirmed_names.intersection(
                    set(needed_names + ["User"])
                )
            ):
                continue

            if not self._identity_refs_available_for_pending(pend):
                continue  # still can't resolve

            results.append(
                self._clear_pending_type(
                    pend_id,
                    "identity",
                    skip_fact_apply=skip_fact_apply,
                )
            )

        self.save()
        return results

    def _sweep_pending_for_location(
        self,
        location: dict,
        *,
        skip_fact_apply: bool = False,
    ) -> list[dict]:
        """Resolve scene-pending images that match a confirmed location."""
        loc_name = _normalize_location_name(location.get("name"))
        if not loc_name:
            return []

        results: list[dict] = []
        for pend_id in list(self.pending.keys()):
            if pend_id not in self.pending:
                continue
            pend = self.pending[pend_id]
            if "scene" not in self._pending_types(pend):
                continue
            possible_locations = _location_names(pend.get("possible_location"))
            if loc_name not in possible_locations:
                continue
            if loc_name in _location_names(
                pend.get("scene_match_failed_locations")
            ):
                continue
            if not self._pending_scene_matches(location, pend):
                self._record_scene_match_failure(pend, loc_name)
                if self._all_possible_scene_locations_failed(pend):
                    results.append(self._clear_pending_type(
                        pend_id,
                        "scene",
                        skip_fact_apply=skip_fact_apply,
                    ))
                continue

            self._add_private_location(
                {"name": loc_name, "description": location.get("description", "")},
                image_id=_img_id(pend["image_path"]),
                image_path=pend["image_path"],
            )
            results.append(self._clear_pending_type(
                pend_id,
                "scene",
                skip_fact_apply=skip_fact_apply,
            ))

        self.save()
        return results

    def _try_resolve_pending_scene_with_existing_locations(
        self,
        pend_id: str,
        *,
        skip_fact_apply: bool = False,
    ) -> dict | None:
        """Try to clear a scene blocker using locations already in memory."""
        pend = self.pending.get(pend_id)
        if not pend or "scene" not in self._pending_types(pend):
            return None

        possible_locations = set(_location_names(pend.get("possible_location")))
        if not possible_locations:
            return None
        failed_locations = set(
            _location_names(pend.get("scene_match_failed_locations"))
        )

        candidates = [
            loc for loc in self.private_locations
            if _normalize_location_name(loc.get("name")) in possible_locations
            and _normalize_location_name(loc.get("name")) not in failed_locations
        ]
        if not candidates:
            return None

        for loc in candidates:
            loc_name = _normalize_location_name(loc.get("name"))
            if self._pending_scene_matches(loc, pend):
                self._add_private_location(
                    {
                        "name": loc_name,
                        "description": loc.get("description", ""),
                    },
                    image_id=_img_id(pend["image_path"]),
                    image_path=pend["image_path"],
                )
                return self._clear_pending_type(
                    pend_id,
                    "scene",
                    skip_fact_apply=skip_fact_apply,
                )
            self._record_scene_match_failure(pend, loc_name)

        if self._all_possible_scene_locations_failed(pend):
            return self._clear_pending_type(
                pend_id,
                "scene",
                skip_fact_apply=skip_fact_apply,
            )
        self.save()
        return {
            "pend_id": pend_id,
            "status": "pending",
            "pending_types": self._pending_types(pend),
            "scene_match_failed_locations": pend.get(
                "scene_match_failed_locations", []
            ),
        }

    def _all_possible_scene_locations_failed(self, pend: dict) -> bool:
        possible = set(_location_names(pend.get("possible_location")))
        if not possible:
            return False
        failed = set(_location_names(pend.get("scene_match_failed_locations")))
        return possible.issubset(failed)

    def _record_scene_match_failure(self, pend: dict, loc_name: str) -> None:
        loc_name = _normalize_location_name(loc_name)
        if not loc_name:
            return
        failed = _location_names(pend.get("scene_match_failed_locations"))
        if loc_name not in failed:
            failed.append(loc_name)
        pend["scene_match_failed_locations"] = failed
        pend["scene_resolution_note"] = self._scene_resolution_note(failed)

    def _scene_resolution_note(self, failed_locations: list[str]) -> str:
        names = _dedupe_strings([
            _normalize_location_name(name) for name in failed_locations
        ])
        if not names:
            return ""
        if len(names) == 1:
            loc_text = f"the user's {names[0]}"
        else:
            loc_text = (
                "the user's "
                + ", ".join(names[:-1])
                + f", or {names[-1]}"
            )
        return (
            "SCENE RESOLUTION NOTE: This image was compared with "
            f"{loc_text} and did not match. Do not treat this room as "
            f"{loc_text} or as a confirmed user-owned private location."
        )

    def _pending_scene_matches(self, location: dict, pend: dict) -> bool:
        image_paths = location.get("image_path") or []
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        image_paths = _dedupe_strings(image_paths)
        if len(image_paths) <= 3:
            reference_images = image_paths
        else:
            seed = (
                f"{location.get('name', '')}|"
                f"{pend.get('image_path', '')}|"
                f"{len(image_paths)}"
            )
            rng = random.Random(seed)
            reference_images = rng.sample(image_paths, 3)
        pending_image = pend.get("image_path")
        if not reference_images or not pending_image:
            return False

        prompt = (
            f"Candidate location name: {location.get('name', '')}\n"
            f"Known location description: {location.get('description', '')}\n\n"
            f"Images 1-{len(reference_images)} are confirmed reference images "
            "of this private location.\n"
            f"Image {len(reference_images) + 1} is the pending image to compare.\n"
            "If the pending image matches any one of the reference images, "
            "return same_location=true."
        )
        input_images = reference_images + [pending_image]
        raw = query_llm(
            client=self.llm_client,
            text_prompt=prompt,
            input_images=input_images,
            model_name=self.model_name,
            system_prompt=SCENE_MATCH_PROMPT,
        )
        _log_ai_response(
            "SCENE_MATCH",
            raw,
            prompt=prompt,
            system_prompt=SCENE_MATCH_PROMPT,
            image_paths=input_images,
        )
        match = _safe_json_load(raw)
        if not match:
            return False
        return bool(match.get("same_location"))

    def _clear_pending_type(
        self,
        pend_id: str,
        pending_type: str,
        *,
        skip_fact_apply: bool = False,
    ) -> dict:
        pend = self.pending.get(pend_id)
        if not pend:
            return {"pend_id": pend_id, "status": "missing"}

        if pending_type == "identity" and not pend.get("identity_resolution_note"):
            pend["identity_resolution_note"] = (
                self._identity_resolution_note_for_pending(pend)
            )
        remaining = [
            t for t in self._pending_types(pend)
            if t != pending_type
        ]
        self._set_pending_types(pend, remaining)
        if remaining:
            self.save()
            return {
                "pend_id": pend_id,
                "status": "partially_resolved",
                "cleared": pending_type,
                "remaining_pending_types": remaining,
            }

        pend_data = self.resolve_pending(pend_id)
        if not pend_data:
            return {"pend_id": pend_id, "status": "missing"}
        return self._extract_resolved_pending(
            pend_id,
            pend_data,
            skip_fact_apply=skip_fact_apply,
        )

    def _extract_resolved_pending(
        self,
        pend_id: str,
        pend_data: dict,
        *,
        skip_fact_apply: bool = False,
    ) -> dict:
        img_id = _img_id(pend_data["image_path"])
        identity_ctx, face_paths = self._build_identity_context_for_pending(
            pend_data
        )
        conversation_context = pend_data["conversation_context"]
        scene_note = (pend_data.get("scene_resolution_note") or "").strip()
        if scene_note:
            conversation_context = f"{conversation_context}\n\n{scene_note}"
        identity_note = (
            pend_data.get("identity_resolution_note")
            or self._identity_resolution_note_for_pending(pend_data)
        )
        if identity_note:
            conversation_context = f"{conversation_context}\n\n{identity_note}"
        extraction_result = self._extract_and_save(
            image_path=pend_data["image_path"],
            image_id=img_id,
            conversation_context=conversation_context,
            face_image_paths=face_paths,
            identity_context=identity_ctx,
            skip_fact_apply=skip_fact_apply,
        )
        self._replace_id_in_memory(pend_id, img_id)
        result = {
            "pend_id": pend_id,
            "image_id": img_id,
            "status": "resolved",
        }
        if extraction_result.get("changes"):
            result["changes"] = extraction_result["changes"]
        for key in ("pending_resolved", "scene_pending_resolved"):
            if key in extraction_result:
                result[key] = extraction_result[key]
        return result
