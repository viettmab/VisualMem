"""LLM-driven image extraction flow for VisualMemoryStore."""

from __future__ import annotations

import json

from .prompts import (
    CONTEXT_ANALYSIS_PROMPT,
    EXTRACT_USER_PROMPT,
    VISUAL_EXTRACT_SYSTEM_PROMPT,
)
from .visualmem_utils import (
    _face_visibility_score_is_better,
    _img_id,
    _is_role_or_unknown_rel_name,
    _location_names,
    _log_ai_response,
    _normalize_face_visibility_score,
    _now,
    _query_llm as query_llm,
    _safe_json_load,
    _uid,
)


class VisualMemoryExtractionMixin:
    def process_image_turn(
        self,
        image_path: str,
        conversation_context: str = "",
        date: str | None = None,
        skip_fact_apply: bool = False,
    ) -> dict:
        """Run the full identity-first image pipeline against this store.

        Flow:
          1. Analyze context -> routing decision for identity refs
          2. Look up face refs in memory
          3. If routing says pending -> save to PENDING (no extraction)
          4. Otherwise extract WITH any selected ref images
          5. Update memory with named extraction
          6. For any newly confirmed person -> sweep pending images

        """
        self._require_llm("process_image_turn")

        image_path = str(image_path)
        image_id = _img_id(image_path)

        if image_id in self.images:
            return {"image_id": image_id, "status": "skipped"}
        if image_id in self.pending:
            return {"image_id": image_id, "status": "already_pending"}

        # Prepend the conversation date once so every downstream LLM call
        # (context analysis, extraction, fact merge) sees the same anchor
        # and can resolve relative time expressions like the text pipeline.
        if date:
            conversation_context = f"DATE: {date}\n{conversation_context}"

        # ── Step 1: context analysis ──────────────────────────────────
        ctx = self._analyze_context(conversation_context, image_path)
        _log_ai_response(
            "CONTEXT_ANALYSIS", json.dumps(ctx, indent=2),
            prompt=f'Conversation context: "{conversation_context}"',
            system_prompt=CONTEXT_ANALYSIS_PROMPT,
        )

        candidates = list(ctx.get("people_possibly_in_image") or [])
        user_confirmed_in_ctx = ctx.get("user_explicitly_confirmed", False)
        has_user = any(c.lower() == "user" for c in candidates)
        named_others = [c for c in candidates if c.lower() != "user"]
        load_all_known_faces = bool(ctx.get("load_all_known_faces", False))
        scene_ownership = self._scene_ownership(ctx.get("scene_ownership"))
        pending_types = self._normalize_pending_types(
            ctx.get("pending_types") or []
        )
        if ctx.get("pending", False) and not pending_types:
            pending_types.append("identity")
        context_possible_location = _location_names(ctx.get("possible_location"))
        context_private_location = _location_names(ctx.get("private_location"))
        has_scene_location_candidate = bool(
            context_possible_location or context_private_location
        )
        if (
            scene_ownership == "unknown"
            and has_scene_location_candidate
            and "scene" not in pending_types
        ):
            pending_types.append("scene")
        if scene_ownership != "unknown":
            pending_types = [t for t in pending_types if t != "scene"]
        if scene_ownership == "unknown" and not has_scene_location_candidate:
            pending_types = [t for t in pending_types if t != "scene"]
        if load_all_known_faces:
            pending_types = [t for t in pending_types if t != "identity"]
        pending_types = self._normalize_pending_types(pending_types)

        location_change = None
        scene_resolved: list[dict] = []
        if scene_ownership == "user_space":
            location_payload = ctx.get("private_location") or {}
            if not location_payload:
                possible_names = _location_names(ctx.get("possible_location"))
                if possible_names:
                    location_payload = {
                        "name": possible_names[0],
                        "description": "",
                    }
            location_change = self._add_private_location(
                location_payload,
                image_id=image_id,
                image_path=image_path,
            )
            if location_change:
                scene_resolved = self._sweep_pending_for_location(
                    location_change["location"],
                    skip_fact_apply=skip_fact_apply,
                )

        def _attach_location_results(result: dict) -> dict:
            if location_change:
                result.setdefault("changes", {}).setdefault(
                    "private_locations_updated", []
                ).append(location_change["name"])
            if scene_resolved:
                result["scene_pending_resolved"] = scene_resolved
            return result

        if pending_types:
            pend_id = _uid("pend")
            possible_location = context_possible_location
            if not possible_location:
                possible_location = context_private_location
            identity_candidates = self._identity_candidates_from_context(candidates)
            reason = ctx.get("pending_reason") or (
                "Scene ownership is unknown."
                if pending_types == ["scene"]
                else "Context analysis marked image pending."
            )
            self.add_pending_image(
                pend_id=pend_id,
                image_path=image_path,
                extraction={},
                conversation_context=conversation_context,
                reason=reason,
                named_people=named_others,
                mentioned_from_context=ctx.get("mentioned_people", []),
                pending_types=pending_types,
                possible_location=possible_location,
                scene_ownership=scene_ownership,
                identity_candidates=identity_candidates,
                load_all_known_faces=load_all_known_faces,
            )
            immediate_scene_result = (
                self._try_resolve_pending_scene_with_existing_locations(
                    pend_id,
                    skip_fact_apply=skip_fact_apply,
                )
                if "scene" in pending_types else None
            )
            if immediate_scene_result:
                immediate_scene_result.setdefault(
                    "image_id",
                    immediate_scene_result.get("image_id", pend_id),
                )
                return _attach_location_results(immediate_scene_result)
            return _attach_location_results({
                "image_id": pend_id,
                "status": "pending",
                "reason": reason,
                "pending_types": pending_types,
            })

        # Case A: exactly one named non-user person — save with that name
        if len(named_others) == 1 and not has_user and not user_confirmed_in_ctx and not load_all_known_faces:
            return _attach_location_results(self._extract_and_save(
                image_path, image_id, conversation_context,
                face_image_paths=[], identity_context="",
                forced_person_name=named_others[0],
                skip_fact_apply=skip_fact_apply,
            ))

        # Case B: user confirmed alone ("picture of me", no companions)
        if user_confirmed_in_ctx and not named_others and not load_all_known_faces:
            return _attach_location_results(self._extract_and_save(
                image_path, image_id, conversation_context,
                face_image_paths=[], identity_context="",
                confirm_user=True,
                skip_fact_apply=skip_fact_apply,
            ))

        # Case C: no people need identity matching
        if not candidates and not load_all_known_faces:
            return _attach_location_results(self._extract_and_save(
                image_path, image_id, conversation_context,
                face_image_paths=[], identity_context="",
                skip_fact_apply=skip_fact_apply,
            ))

        # ── Step 2: look up face references ───────────────────────────
        if load_all_known_faces:
            names_to_lookup = []
            if self.user.get("face_image"):
                names_to_lookup.append("User")
            names_to_lookup.extend(
                r["name"] for r in self.relationships.values() if r.get("face_image")
            )
        else:
            names_to_lookup = candidates

        deduped_names: list[str] = []
        seen_names: set[str] = set()
        for name in names_to_lookup:
            if not isinstance(name, str):
                continue
            clean = name.strip()
            if not clean:
                continue
            key = clean.lower()
            if key in seen_names:
                continue
            seen_names.add(key)
            deduped_names.append(clean)

        identity_context, face_image_paths = self.build_identity_context(
            deduped_names
        )

        # Extract, with ref images
        return _attach_location_results(self._extract_and_save(
            image_path, image_id, conversation_context,
            face_image_paths=face_image_paths,
            identity_context=identity_context,
            confirm_user=user_confirmed_in_ctx,
            skip_fact_apply=skip_fact_apply,
        ))

    def _analyze_context(
        self,
        conversation_context: str,
        image_path: str | None = None,
    ) -> dict:
        """Ask the LLM to analyze the context (and optional image) only for
        routing: identity refs, scene ownership, and pending blockers."""
        available_face_refs: list[str] = []
        if self.user.get("face_image"):
            available_face_refs.append("User")
        for rel in self.relationships.values():
            if not rel.get("face_image"):
                continue
            name = rel.get("name", "").strip()
            if not name:
                continue
            relationship = rel.get("relationship", "").strip()
            if relationship and relationship.lower() != "unknown":
                available_face_refs.append(f"{name} ({relationship})")
            else:
                available_face_refs.append(name)
        input_images = [image_path] if image_path else []
        prompt = (
            f'Conversation context: "{conversation_context}"\n\n'
            "Names with face references currently available in memory: "
            + (", ".join(available_face_refs) if available_face_refs else "none")
        )
        raw = query_llm(
            client=self.llm_client,
            text_prompt=prompt,
            input_images=input_images,
            model_name=self.model_name,
            system_prompt=CONTEXT_ANALYSIS_PROMPT,
        )
        empty = {
            "people_possibly_in_image": [],
            "mentioned_people": [],
            "load_all_known_faces": False,
            "user_explicitly_confirmed": False,
            "scene_ownership": "unknown",
            "scene_owner_name": None,
            "private_location": {},
            "possible_location": [],
            "pending": False,
            "pending_types": [],
            "pending_reason": "",
        }
        if raw is None:
            return empty
        return _safe_json_load(raw) or empty

    def _extract_and_save(
        self,
        image_path: str,
        image_id: str,
        conversation_context: str,
        face_image_paths: list[str],
        identity_context: str,
        forced_person_name: str | None = None,
        confirm_user: bool = False,
        skip_fact_apply: bool = False,
    ) -> dict:
        """Send ref images + target image to the LLM, parse the named
        extraction, and write everything to memory
        """
        input_images = face_image_paths + [image_path]
        target_idx = len(face_image_paths) + 1

        prompt = EXTRACT_USER_PROMPT.format(
            context=conversation_context, memory_block="",
        )
        if face_image_paths:
            prompt += (
                f"\n\nIDENTITY REFERENCES:\n{identity_context}\n"
                f"Image {target_idx}: TARGET image to analyze.\n\n"
                "Use the reference images to identify people in the target "
                "image. In scene_summary, use REAL NAMES (e.g., 'User and "
                "Maya at the co-working space'). For people_in_image, set "
                "label to the person's name."
            )
        elif forced_person_name:
            prompt += (
                f"\n\nThe person visible in this image is {forced_person_name}. "
                "Use this name in scene_summary and people_in_image label."
            )
        elif confirm_user:
            prompt += (
                "\n\nThe person in this image IS the user. Set is_user=true "
                "and label='User' in people_in_image. Use 'User' in "
                "scene_summary."
            )
        else:
            prompt += (
                "\n\nNo identity reference images are being provided for this turn. "
                "Perform extraction using only the target image and the conversation "
                "context."
            )

        raw = query_llm(
            client=self.llm_client,
            text_prompt=prompt,
            input_images=input_images,
            model_name=self.model_name,
            system_prompt=VISUAL_EXTRACT_SYSTEM_PROMPT,
        )
        _log_ai_response(
            "EXTRACTION", raw, prompt=prompt,
            system_prompt=VISUAL_EXTRACT_SYSTEM_PROMPT,
            image_paths=input_images,
        )
        if raw is None:
            raise RuntimeError("LLM returned no response.")

        extraction = _safe_json_load(raw)
        if extraction is None:
            raise RuntimeError(f"Failed to parse extraction JSON: {raw[:200]}")

        # Ensure relationships exist BEFORE we look them up while building
        # subjects, so _find_rel_by_name finds them.
        now = _now()
        for mp in extraction.get("mentioned_people", []):
            self._add_relationship(mp, now)
        if forced_person_name:
            if not self._find_rel_by_name(forced_person_name):
                self._add_relationship(
                    {"name": forced_person_name, "relationship": "unknown"},
                    now,
                )

        # Build the `subjects` list for the image record
        subjects: list[dict] = []
        user_found_in_extraction = any(
            p.get("is_user", False)
            for p in extraction.get("people_in_image", [])
        )
        for p in extraction.get("people_in_image", []):
            label = p.get("label", forced_person_name or "unknown")
            is_user = p.get("is_user", False)
            face_visibility_score = _normalize_face_visibility_score(
                p.get("face_visibility_score")
            )

            # confirm_user fallback: if the LLM didn't already tag anyone as
            # user but we expected just one person, treat them as the user.
            if confirm_user and not user_found_in_extraction and not is_user:
                if len(extraction.get("people_in_image", [])) == 1:
                    is_user = True

            if forced_person_name and not is_user:
                label = forced_person_name

            ref_id = "__user__" if is_user else None
            if is_user and _face_visibility_score_is_better(
                face_visibility_score,
                self.user.get("face_image_visibility_score"),
            ):
                self.user["face_image"] = image_path
                self.user["face_image_visibility_score"] = face_visibility_score
            if not is_user and label.lower() not in ("unknown", ""):
                rel = self._find_rel_by_name(label)
                if rel:
                    ref_id = next(
                        (k for k, v in self.relationships.items() if v is rel),
                        None,
                    )
                    if (
                        not rel.get("face_image")
                        or _face_visibility_score_is_better(
                            face_visibility_score,
                            rel.get("face_image_visibility_score"),
                        )
                    ):
                        rel["face_image"] = image_path
                        rel["face_image_visibility_score"] = face_visibility_score
                    if image_path not in rel.get("source_images", []):
                        rel.setdefault("source_images", []).append(image_path)

            subject = {
                "type": "person",
                "name": "User" if is_user else label,
                "ref_id": ref_id,
            }
            if face_visibility_score is not None:
                subject["face_visibility_score"] = face_visibility_score
            subjects.append(subject)

        for pet in extraction.get("pets_in_image", []):
            subjects.append({
                "type": "pet",
                "name": pet.get("name") or pet.get("type", "pet"),
            })

        # Embed and store in Qdrant
        if self.qdrant_manager:
            tags = extraction.get("tags", [])
            summary = extraction.get("scene_summary", "")
            embedding = self.qdrant_manager.embed_tags_and_summary(
                tags, summary,
            )
            subject_names = [s["name"] for s in subjects if s.get("name")]
            self.qdrant_manager.upsert_image(
                image_id=image_id,
                embedding=embedding,
                metadata={
                    "image_path": image_path,
                    "summary": summary,
                    "subjects": subject_names,
                    "tags": tags,
                },
            )

        # Confirm user if needed
        newly_confirmed_user = False
        if confirm_user and not self.user["confirmed"]:
            user_face_visibility_score = next(
                (
                    s.get("face_visibility_score")
                    for s in subjects
                    if s.get("type") == "person" and s.get("name") == "User"
                ),
                None,
            )
            self.confirm_user(
                face_image_path=image_path,
                description=extraction.get("scene_summary", ""),
                image_id=image_id,
                face_visibility_score=user_face_visibility_score,
            )
            newly_confirmed_user = True

        # Save the confirmed image
        summary = extraction.get("scene_summary", "")
        changes = self.add_confirmed_image(
            image_id=image_id,
            image_path=image_path,
            summary=summary,
            subjects=subjects,
            tags=extraction.get("tags", []),
            conversation_context=conversation_context,
            extraction=extraction,
        )

        # User facts → unified ADD/UPDATE/DELETE merge pipeline.
        # When skip_fact_apply=True, leave the pending facts in `changes` so
        # the caller can render them (e.g. into a text message) and route them
        # to an external memory backend instead of this store.
        if not skip_fact_apply:
            changes.pop("user_facts_pending", [])

        result = {
            "image_id": image_id,
            "status": "confirmed",
            "changes": changes,
        }

        # Sweep pending images for any newly confirmed person.
        newly_confirmed_names: set[str] = set()
        if newly_confirmed_user:
            newly_confirmed_names.add("User")
        for p in subjects:
            if p["type"] == "person" and p["name"] != "User":
                rel = self._find_rel_by_name(p["name"])
                if rel and rel.get("face_image") == image_path:
                    newly_confirmed_names.add(p["name"])
                    rel_name = rel.get("name")
                    if rel_name and not _is_role_or_unknown_rel_name(rel_name):
                        newly_confirmed_names.add(rel_name)

        if newly_confirmed_names and self.pending:
            result["pending_resolved"] = self._sweep_pending_for_names(
                newly_confirmed_names,
                skip_fact_apply=skip_fact_apply,
            )

        return result
