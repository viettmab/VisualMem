"""Confirmed-memory mutation helpers for VisualMemoryStore."""

from __future__ import annotations

from .visualmem_utils import (
    _ROLE_ALIASES,
    _ROLE_NAME_ALIASES,
    _canonical_rel_name,
    _canonical_relationship,
    _is_role_or_unknown_rel_name,
    _location_names,
    _norm,
    _normalize_face_visibility_score,
    _normalize_location_name,
    _now,
    _uid,
    _face_visibility_score_is_better,
)


class VisualMemoryMutationMixin:
    def add_confirmed_image(
        self,
        image_id: str,
        image_path: str,
        summary: str,
        subjects: list[dict],
        tags: list[str],
        conversation_context: str,
        extraction: dict,
    ) -> dict:
        """Add a fully confirmed image to main memory."""
        now = _now()
        changes = {
            "facts_added": [],
            "relationships_updated": [],
            "objects_added": [],
            "pets_added": [],
        }

        self.images[image_id] = {
            "image_path": image_path,
            "summary": summary,
            "photo_time": extraction.get("photo_time"),
            "subjects": subjects,
            "tags": tags,
            "timestamp": now,
            "conversation_context": conversation_context,
        }

        # User facts go through the unified merge pipeline at the call site
        # (so they can ADD/UPDATE/DELETE properly).
        changes["user_facts_pending"] = extraction.get("user_facts", [])

        for obj in extraction.get("objects_in_image", []):
            if obj.get("owner", "").lower() == "user":
                self._add_object(obj, image_id, image_path)
                changes["objects_added"].append(obj["name"])

        for pet in extraction.get("pets_in_image", []):
            if pet.get("owner", "").lower() == "user":
                self._add_pet(pet, image_id, image_path)
                changes["pets_added"].append(pet.get("name") or pet["type"])

        for mp in extraction.get("mentioned_people", []):
            self._add_relationship(mp, now)
            changes["relationships_updated"].append(mp["name"])

        # Pets/objects owned by others -> relationship facts
        for pet in extraction.get("pets_in_image", []):
            owner = pet.get("owner", "").lower()
            if owner and owner not in ("user", "unknown"):
                self._add_relationship_fact(
                    {
                        "person_name": pet["owner"],
                        "statement": (
                            f"{pet['owner']} has a {pet.get('type', 'pet')}"
                            + (f" named {pet['name']}" if pet.get("name") else "")
                        ),
                        "evidence": ["Visible in image"],
                        "confidence": 0.85,
                    },
                    image_id,
                )

        self.save()
        return changes

    def add_pending_image(
        self,
        pend_id: str,
        image_path: str,
        extraction: dict,
        conversation_context: str,
        reason: str,
        named_people: list[str],
        mentioned_from_context: list[dict] | None = None,
        pending_types: list[str] | None = None,
        possible_location: list[str] | None = None,
        scene_ownership: str | None = None,
        identity_candidates: list[str] | None = None,
        load_all_known_faces: bool = False,
    ) -> None:
        """Add an unconfirmed image to pending memory."""
        normalized_pending_types = self._normalize_pending_types(
            pending_types or ["identity"]
        )
        normalized_possible_location = _location_names(possible_location)
        normalized_identity_candidates = self._identity_candidates_from_context(
            identity_candidates or named_people
        )
        self.pending[pend_id] = {
            "image_path": image_path,
            "conversation_context": conversation_context,
            "reason_pending": reason,
            "named_people_in_context": named_people,
            "pending_type": (
                normalized_pending_types[0]
                if normalized_pending_types else None
            ),
            "pending_types": normalized_pending_types,
            "possible_location": normalized_possible_location,
            "scene_ownership": self._scene_ownership(scene_ownership),
            "identity_candidates": normalized_identity_candidates,
            "load_all_known_faces": bool(load_all_known_faces),
            "scene_match_failed_locations": [],
            "scene_resolution_note": "",
            "identity_resolution_note": "",
            "timestamp": _now(),
        }
        if mentioned_from_context:
            for mp in mentioned_from_context:
                self._add_relationship(mp, _now())
        if extraction:
            for mp in extraction.get("mentioned_people", []):
                self._add_relationship(mp, _now())
        self.save()

    def confirm_user(
        self,
        face_image_path: str,
        description: str,
        image_id: str,
        name: str | None = None,
        face_visibility_score=None,
    ) -> None:
        """Mark user identity as confirmed with a face image."""
        self.user["confirmed"] = True
        self.user["face_image"] = face_image_path
        normalized_score = _normalize_face_visibility_score(face_visibility_score)
        if normalized_score is not None:
            self.user["face_image_visibility_score"] = normalized_score
        self.user["description"] = description
        if name:
            self.user["name"] = name
        self.save()

    def resolve_pending(self, pend_id: str) -> dict | None:
        """Pop and return a pending image record."""
        return self.pending.pop(pend_id, None)

    def _add_fact(
        self, facts_list: list, fact: dict, image_id: str,
    ) -> None:
        stmt = fact.get("statement", "").strip()
        if not stmt:
            return
        for existing in facts_list:
            if _norm(existing["statement"]) == _norm(stmt):
                existing["confidence"] = min(
                    0.99,
                    existing["confidence"]
                    + (1 - existing["confidence"]) * 0.15,
                )
                if image_id and image_id not in existing.get("source_images", []):
                    existing.setdefault("source_images", []).append(image_id)
                return
        facts_list.append({
            "id": _uid("fact"),
            "statement": stmt,
            "evidence": fact.get("evidence", []),
            "confidence": fact.get("confidence", 0.5),
            "source_images": [image_id] if image_id else [],
        })

    def _add_object(
        self, obj: dict, image_id: str, image_path: str,
    ) -> None:
        name = obj.get("name", "").strip()
        if not name:
            return
        description = obj.get("description", "")
        for existing in self.user["objects"]:
            if (_norm(existing["name"]) == _norm(name)
                    and _norm(existing.get("description", "")) == _norm(description)):
                return
        self.user["objects"].append({
            "name": name,
            "description": description,
            "image_path": image_path,
            "first_seen": image_id,
        })

    def _add_pet(
        self, pet: dict, image_id: str, image_path: str,
    ) -> None:
        pet_name = pet.get("name", "").strip()
        pet_type = pet.get("type", "unknown")
        description = pet.get("description", "")
        for existing in self.user["pets"]:
            if pet_name != "unknown" and _norm(existing.get("name", "")) == _norm(pet_name):
                if description and not existing.get("description"):
                    existing["description"] = description
                return
            if (_norm(existing.get("type", "")) == _norm(pet_type)
                    and _norm(existing.get("breed", "")) == _norm(pet.get("breed", ""))):
                if pet_name and not existing.get("name"):
                    existing["name"] = pet_name
                if description and not existing.get("description"):
                    existing["description"] = description
                return
        self.user["pets"].append({
            "name": pet_name or None,
            "type": pet_type,
            "breed": pet.get("breed", ""),
            "description": description,
            "image_path": image_path,
            "first_seen": image_id,
        })

    def _add_private_location(
        self,
        location: dict | str | None,
        image_id: str,
        image_path: str,
    ) -> dict | None:
        """Add or update a user-owned private location by normalized name."""
        if isinstance(location, str):
            location = {"name": location, "description": ""}
        if not isinstance(location, dict):
            return None

        name = _normalize_location_name(location.get("name"))
        if not name:
            return None
        description = (location.get("description") or "").strip()

        for existing in self.private_locations:
            if _normalize_location_name(existing.get("name")) != name:
                continue
            existing["name"] = name
            existing_paths = existing.setdefault("image_path", [])
            if isinstance(existing_paths, str):
                existing_paths = [existing_paths]
                existing["image_path"] = existing_paths
            changed = False
            if image_path and image_path not in existing_paths:
                existing_paths.append(image_path)
                changed = True
            if description and not existing.get("description"):
                existing["description"] = description
                changed = True
            if not existing.get("first_seen"):
                existing["first_seen"] = image_id
                changed = True
            return (
                {"name": name, "location": existing}
                if changed else None
            )

        loc = {
            "name": name,
            "description": description,
            "image_path": [image_path] if image_path else [],
            "first_seen": image_id,
        }
        self.private_locations.append(loc)
        return {"name": name, "location": loc}

    def _add_relationship(self, mp: dict, now: str) -> None:
        name = _canonical_rel_name(mp.get("name"))
        if not name:
            return
        relationship = _canonical_relationship(mp.get("relationship", "unknown"))
        existing = None

        if _is_role_or_unknown_rel_name(name):
            for rel in self.relationships.values():
                if (
                    _is_role_or_unknown_rel_name(rel.get("name"))
                    and _canonical_relationship(rel.get("relationship")) == relationship
                ):
                    existing = rel
                    break
        else:
            existing = self._find_rel_by_name(name)

        if not existing and relationship in _ROLE_ALIASES.values():
            for rel in self.relationships.values():
                if (
                    _is_role_or_unknown_rel_name(rel.get("name"))
                    and _canonical_relationship(rel.get("relationship")) == relationship
                ):
                    existing = rel
                    break
        if existing:
            if (not _is_role_or_unknown_rel_name(name)
                    and _is_role_or_unknown_rel_name(existing.get("name"))):
                existing["name"] = name
            if relationship != "unknown" and existing.get("relationship", "unknown") in ("unknown", ""):
                existing["relationship"] = relationship
            elif relationship in _ROLE_ALIASES.values():
                existing["relationship"] = relationship
            self._merge_duplicate_relationships(existing)
        else:
            self.relationships[_uid("rel")] = {
                "name": name,
                "relationship": relationship,
                "face_image": None,
                "face_image_visibility_score": None,
                "facts": [],
                "source_images": [],
            }

    def _merge_duplicate_relationships(self, target: dict) -> None:
        """Merge role-only relationship records into `target`."""
        target_relationship = _canonical_relationship(target.get("relationship"))
        if target_relationship not in _ROLE_ALIASES.values():
            return

        target_id = next(
            (k for k, v in self.relationships.items() if v is target),
            None,
        )
        if not target_id:
            return

        for rel_id, rel in list(self.relationships.items()):
            if rel_id == target_id:
                continue
            if _canonical_relationship(rel.get("relationship")) != target_relationship:
                continue
            target_has_real_name = not _is_role_or_unknown_rel_name(target.get("name"))
            rel_has_real_name = not _is_role_or_unknown_rel_name(rel.get("name"))
            if target_has_real_name and rel_has_real_name:
                continue

            if rel_has_real_name and not target_has_real_name:
                target["name"] = rel["name"]
            target["relationship"] = target_relationship
            if (
                rel.get("face_image")
                and (
                    not target.get("face_image")
                    or _face_visibility_score_is_better(
                        rel.get("face_image_visibility_score"),
                        target.get("face_image_visibility_score"),
                    )
                )
            ):
                target["face_image"] = rel["face_image"]
                target["face_image_visibility_score"] = (
                    _normalize_face_visibility_score(
                        rel.get("face_image_visibility_score")
                    )
                )
            for source in rel.get("source_images", []):
                if source not in target.setdefault("source_images", []):
                    target["source_images"].append(source)
            for fact in rel.get("facts", []):
                if not any(
                    _norm(existing.get("statement", ""))
                    == _norm(fact.get("statement", ""))
                    for existing in target.setdefault("facts", [])
                ):
                    target["facts"].append(fact)
            for img in self.images.values():
                for subj in img.get("subjects", []):
                    if subj.get("ref_id") == rel_id:
                        subj["ref_id"] = target_id
                        if not _is_role_or_unknown_rel_name(target.get("name")):
                            subj["name"] = target["name"]
            del self.relationships[rel_id]

    def _add_relationship_fact(
        self, rf: dict, image_id: str,
    ) -> None:
        person_name = rf.get("person_name")
        name = _canonical_rel_name(person_name)
        stmt = rf.get("statement", "").strip()
        if not name or not stmt:
            return
        rel = self._find_rel_by_name(person_name or name)
        if not rel:
            return
        self._add_fact(rel["facts"], rf, image_id)

    def _find_rel_by_name(self, name: str) -> dict | None:
        lookup = _norm(name)
        for rel in self.relationships.values():
            if _norm(rel.get("name", "")) == lookup:
                return rel
        if lookup in _ROLE_NAME_ALIASES:
            target_relationship = _canonical_relationship(lookup)
            named_rel = None
            for rel in self.relationships.values():
                if _canonical_relationship(rel.get("relationship")) == target_relationship:
                    if not _is_role_or_unknown_rel_name(rel.get("name")):
                        return rel
                    named_rel = named_rel or rel
            return named_rel
        return None

    def _replace_id_in_memory(self, old_id: str, new_id: str) -> None:
        """Replace all occurrences of `old_id` with `new_id` in the
        ``source_images`` lists throughout the memory, then deduplicate.
        Used when a pending image becomes a confirmed image with a new id.
        """
        def _replace_and_dedup(sources: list) -> None:
            for i, s in enumerate(sources):
                if s == old_id:
                    sources[i] = new_id
            seen: set[str] = set()
            deduped: list[str] = []
            for s in sources:
                if s not in seen:
                    seen.add(s)
                    deduped.append(s)
            sources.clear()
            sources.extend(deduped)

        for fact in self.user.get("facts", []):
            _replace_and_dedup(fact.get("source_images", []))
        for rel in self.relationships.values():
            for fact in rel.get("facts", []):
                _replace_and_dedup(fact.get("source_images", []))
