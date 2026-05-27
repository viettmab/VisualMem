"""Visual-memory search and answer generation."""

from __future__ import annotations

import json
import logging

from .prompts import QUESTION_ANALYSIS_PROMPT
from .visualmem_utils import (
    _dedupe_strings,
    _log_ai_response,
    _norm,
    _query_llm as query_llm,
    _safe_json_load,
)

logger = logging.getLogger(__name__)


class VisualMemorySearchMixin:
    def extract_keywords(self, question: str) -> list[str]:
        """Decompose `question` via the question-analysis prompt and flatten
        the result into a deduped list of search phrases. Falls back to
        ``[question]`` on any failure.
        """
        if self.llm_client is None:
            return [question]
        try:
            analysis = self._analyze_question(question)
        except Exception as e:
            logger.warning("extract_keywords: analysis failed (%s)", e)
            return [question]

        raw_phrases = (
            list(analysis.get("search_keywords") or [])
            + list(analysis.get("person_names") or [])
            + list(analysis.get("object_names") or [])
            + list(analysis.get("pet_keywords") or [])
            + ([analysis.get("photo_time")] if analysis.get("photo_time") else [])
        )
        seen: set[str] = set()
        out: list[str] = []
        for p in raw_phrases:
            if not isinstance(p, str):
                continue
            s = p.strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out or [question]

    def search_visual(
        self,
        question: str,
        question_images: list[str] | None = None,
    ) -> dict:
        self._require_llm("search_visual")

        q_analysis = self._analyze_question(question)
        _log_ai_response(
            "QUESTION_ANALYSIS", json.dumps(q_analysis, indent=2),
            prompt=question,
        )
        search_keywords = q_analysis.get("search_keywords", [])
        object_names = q_analysis.get("object_names", [])
        pet_keywords = q_analysis.get("pet_keywords", [])
        photo_time = q_analysis.get("photo_time")

        extra_context_lines, image_summaries, relevant_images = self._gather_visual_context(
            object_names=object_names,
            pet_keywords=pet_keywords,
            photo_time=photo_time,
            search_keywords=search_keywords,
        )

        info_block = (
            "Information from memory:" + "\n" + "\n".join(extra_context_lines)
            if extra_context_lines else ""
        )
        image_block = (
            "Relevant images:" + "\n" + "\n".join(image_summaries)
            if image_summaries else ""
        )
        context_block = (
            f"{info_block}\n\n"
            f"{image_block}\n\n"
        )
        return {
            "context_block": context_block,
            "image_paths": [],
            "relevant_images": relevant_images,
        }


    def _analyze_question(self, question: str) -> dict:
        """LLM analyzes the question to determine the search strategy.

        If the visual store has no known people, objects, or pets, there is
        nothing visual to search — return an empty/general result immediately
        without an LLM call.
        """
        empty = {
            "question_type": "general",
            "search_keywords": [],
            "person_names": [],
            "object_names": [],
            "pet_keywords": [],
            "photo_time": None,
        }

        known_people = [r["name"] for r in self.relationships.values()]
        known_objects = [
            f"{o.get('name', '')} ({o.get('description', '')})"
            for o in self.user.get("objects", [])
        ]
        known_pets = [
            (
                f"{p.get('name') or 'unnamed'} "
                f"({p.get('type', 'pet')}, {p.get('breed', '')}, "
                f"{p.get('description', '')})"
            )
            for p in self.user.get("pets", [])
        ]
        known_photo_times = _dedupe_strings([
            str(img.get("photo_time"))
            for img in self.images.values()
            if img.get("photo_time")
        ])

        if (
            not known_people
            and not known_objects
            and not known_pets
            and not self.images
            and not self.qdrant_manager
        ):
            return empty

        prompt = (
            f'Question: "{question}"\n\n'
            f"Known people: {', '.join(known_people) if known_people else 'none'}\n"
            f"Known objects: {', '.join(known_objects) if known_objects else 'none'}\n"
            f"Known pets: {', '.join(known_pets) if known_pets else 'none'}\n"
            f"Known photo_times: {', '.join(known_photo_times) if known_photo_times else 'none'}\n\n"
            "Analyze this question and return the search strategy JSON."
        )
        raw = query_llm(
            client=self.llm_client,
            text_prompt=prompt,
            input_images=[],
            model_name=self.model_name,
            system_prompt=QUESTION_ANALYSIS_PROMPT,
        )
        if raw is None:
            return empty
        return _safe_json_load(raw) or empty

    def _gather_visual_context(
        self,
        *,
        object_names: list[str],
        pet_keywords: list[str],
        photo_time: str | None,
        search_keywords: list[str],
    ) -> tuple[list[str], list[str], list[dict]]:
        """Return memory info lines, image summary lines, and image records.
        """
        extra_context_lines: list[str] = []
        image_summaries: list[str] = []
        relevant_images: list[dict] = []

        def _already_has_image(img_id: str) -> bool:
            return any(r.get("image_id") == img_id for r in relevant_images)

        def _img_text(img: dict) -> str:
            subjects = " ".join(
                s.get("name", "") for s in img.get("subjects", [])
            )
            return " ".join([
                img.get("summary", ""),
                " ".join(img.get("tags", [])),
                subjects,
                str(img.get("photo_time") or ""),
            ]).lower()

        def _photo_time_matches(img: dict) -> bool:
            if not photo_time:
                return True
            requested = _norm(str(photo_time))
            actual = _norm(str(img.get("photo_time") or ""))
            return bool(actual and (requested in actual or actual in requested))

        # ── Object search (text match only) ──────────────────────────
        user_objects = self.user.get("objects", [])
        matched_object_keys: set[tuple[str, str]] = set()

        for obj_name in object_names:
            obj_query = _norm(obj_name)
            if not obj_query:
                continue
            for obj in user_objects:
                obj_text = (
                    f"{obj.get('name', '')} {obj.get('description', '')}"
                ).strip()
                norm_obj_text = _norm(obj_text)
                norm_obj_name = _norm(obj.get("name", ""))
                if not (
                    obj_query in norm_obj_text
                    or (norm_obj_name and norm_obj_name in obj_query)
                ):
                    continue
                key = (_norm(obj.get("name", "")), _norm(obj.get("description", "")))
                if key in matched_object_keys:
                    continue
                matched_object_keys.add(key)
                if len(extra_context_lines) == 0:
                    extra_context_lines.append("User's objects:")
                extra_context_lines.append(
                    f"{obj.get('name', '')}: "
                    f"{obj.get('description', '')}"
                )

        # ── Pet search (substring) ───────────────────────────────────
        matched_pet_descriptions: set[str] = set()
        for kw in pet_keywords:
            pet_query = _norm(kw)
            if not pet_query:
                continue
            for pet in self.user.get("pets", []):
                pet_str = (
                    f"{pet.get('name', '')} {pet.get('type', '')} "
                    f"{pet.get('breed', '')} {pet.get('description', '')}"
                )
                norm_pet_str = _norm(pet_str)
                norm_pet_type = _norm(pet.get("type", ""))
                if not (
                    pet_query in norm_pet_str
                    or (norm_pet_type and norm_pet_type in pet_query)
                ):
                    continue
                description_key = _norm(pet.get("description") or pet_str)
                if description_key in matched_pet_descriptions:
                    continue
                if len(matched_pet_descriptions) == 0:
                    extra_context_lines.append("User's pets:")
                matched_pet_descriptions.add(description_key)
                extra_context_lines.append(
                    f"{pet.get('description', '')}"
                )

        # ── Image summary search (Qdrant + metadata substring) ───────
        query_terms = _dedupe_strings(
            list(search_keywords or [])
            + ([photo_time] if photo_time else [])
        )
        if self.qdrant_manager and query_terms:
            query_text = ", ".join(query_terms)
            for qr in self.qdrant_manager.search(query_text, top_k=10):
                img_id = qr.get("image_id")
                if not img_id:
                    continue
                img = self.images.get(img_id) or {
                    "image_path": qr.get("image_path"),
                    "summary": qr.get("summary", ""),
                    "subjects": qr.get("subjects", []),
                    "tags": qr.get("tags", []),
                    "photo_time": qr.get("photo_time"),
                }
                if not _already_has_image(img_id):
                    relevant_images.append({
                        "image_id": img_id,
                        "score": qr.get("score", 0),
                        **img,
                    })

        if photo_time:
            photo_time_images: list[dict] = []
            for img_id, img in self.images.items():
                if _photo_time_matches(img):
                    photo_time_images.append({
                        "image_id": img_id,
                        "score": 1.0,
                        **img,
                    })
            if photo_time_images:
                photo_ids = {
                    img.get("image_id") for img in photo_time_images
                }
                relevant_images = photo_time_images + [
                    img for img in relevant_images
                    if img.get("image_id") not in photo_ids
                ]

        for img_id, img in self.images.items():
            if not _photo_time_matches(img):
                continue
            text = _img_text(img)
            if not query_terms and photo_time:
                if not _already_has_image(img_id):
                    relevant_images.append({
                        "image_id": img_id,
                        "score": 1.0,
                        **img,
                    })
            elif any(_norm(term) in text for term in query_terms):
                if not _already_has_image(img_id):
                    relevant_images.append({
                        "image_id": img_id,
                        "score": 1.0,
                        **img,
                    })

        for idx, img in enumerate(relevant_images[:3], start=1):
            photo = img.get("photo_time") or "unknown"
            image_summaries.append(
                f"{idx}: "
                f"photo_time={photo}; "
                f"summary={img.get('summary', '')}; "
            )
        logger.debug("Num image summaries: %d", len(image_summaries))
        return extra_context_lines, image_summaries, relevant_images

    def search_images_by_subject(self, name: str) -> list[dict]:
        """Find images containing a specific person/pet/object name."""
        name_lower = name.lower()
        results = []
        for img_id, img in self.images.items():
            for subj in img.get("subjects", []):
                if name_lower in subj.get("name", "").lower():
                    results.append({"image_id": img_id, **img})
                    break
        return results