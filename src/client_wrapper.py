"""
VisualMemClient — per-user memory controller that wraps MemosApiOnlineClient.

Text turns go directly to the wrapped `MemosApiOnlineClient`. Image turns are
first processed through the per-user `VisualMemoryStore` with the full
conversation as context to extract objects, pets, relationships (which stay in
the store for visual-question answering) and user_facts. The user_facts are
concatenated into a sentence and appended to the image message's `content`,
then the (now text-only) message is forwarded to `MemosApiOnlineClient.add`
alongside regular text turns.
"""

from __future__ import annotations

import os
import re

from .qdrant import QdrantManager
from .visualmem_store import VisualMemoryStore
from .visualmem_utils import (
    DEFAULT_MODEL, 
    _sanitize, 
    _render_user_facts, 
    _render_conversation_context
)

class VisualMemClient:
    """Per-user image pipeline in front of a shared MemosApiOnlineClient."""

    def __init__(
        self,
        llm_client,
        memos_client,
        *,
        model_name: str = DEFAULT_MODEL,
        memory_dir: str = ".",
        memory_prefix: str = "visual_memory",
        collection_prefix: str = "visual_memory",
    ):
        """
        Args:
            llm_client: LLM client used for image extraction and question
                analysis (e.g. `google.genai.Client(...)`).
            memos_client: a `MemosApiOnlineClient` instance. All text-side
                storage and retrieval is delegated to this client.
            model_name: model name passed to `query_llm` calls.
            memory_dir: directory where per-user JSON files are written.
            memory_prefix: prefix for memory file names.
            collection_prefix: prefix for the per-user Qdrant collection.
        """
        self.llm_client = llm_client
        self.memos_client = memos_client
        self.model_name = model_name
        self.memory_dir = memory_dir
        self.memory_prefix = memory_prefix
        self.collection_prefix = collection_prefix

        os.makedirs(memory_dir, exist_ok=True)

        self._stores: dict[str, VisualMemoryStore] = {}

    # ── per-user state ────────────────────────────────────────────────

    def get_store(self, user_id: str) -> VisualMemoryStore:
        """Lazily build and cache the `VisualMemoryStore` for `user_id`."""
        if user_id in self._stores:
            return self._stores[user_id]

        slug = _sanitize(user_id)
        memory_path = os.path.join(
            self.memory_dir, f"{self.memory_prefix}_{slug}.json"
        )
        try:
            qm = QdrantManager(
                collection_name=f"{self.collection_prefix}_{slug}"
            )
        except Exception as e:
            print(f"  Qdrant unavailable for user={user_id} ({e}) — "
                  "semantic search disabled")
            qm = None
        store = VisualMemoryStore(
            path=memory_path,
            llm_client=self.llm_client,
            model_name=self.model_name,
            qdrant_manager=qm,
        )
        self._stores[user_id] = store
        return store

    def reset_user(self, user_id: str) -> None:
        """Wipe the on-disk memory file and Qdrant collection for `user_id`."""
        store = self.get_store(user_id)
        if os.path.exists(store.path):
            os.remove(store.path)
        if store.qdrant_manager is not None:
            try:
                store.qdrant_manager.clear_collection()
            except Exception as e:
                print(f"  WARNING: failed to clear collection for {user_id}: {e}")
        self._stores.pop(user_id, None)

    # ── add ───────────────────────────────────────────────────────────

    def add(
        self,
        messages: list[dict],
        user_id: str,
        conv_id: str | None = None,
        batch_size: int = 2,
        window_size: str = "full",
    ) -> dict:
        """Process `messages` into memory.

        Two-pass approach:
          1. **Pre-process images**: loop through `messages`, for every turn
             with an ``image_path``, run ``process_image_turn`` to extract
             objects/pets/relationships (stored in the visual store) and
             user_facts. ``window_size="2"`` uses only the image turn text as
             context; ``window_size="full"`` uses the full conversation.
          2. **Send everything to memos**: the now-fully-text message list is
             forwarded as one batch to ``MemosApiOnlineClient.add``.

        Per-turn timestamps are read from ``msg["chat_time"]``.

        Returns aggregated counters:
            {"text_messages": int, "image_turns": int}
        """
        if not messages:
            return {"text_messages": 0, "image_turns": 0}
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if window_size not in ("2", "full"):
            raise ValueError("window_size must be '2' or 'full'")

        store = self.get_store(user_id)
        image_turns = 0
        full_conversation_context = (
            _render_conversation_context(messages)
            if window_size == "full"
            else None
        )

        # ── Pass 1: pre-process images, enrich text with facts ────────
        enriched: list[dict] = []
        for msg in messages:
            image_path = msg.get("image_path")
            if image_path:
                if not os.path.exists(image_path):
                    print(f"  WARNING: image not found, skipping: {image_path}")
                    continue
                conversation_context = (
                    full_conversation_context
                    if window_size == "full"
                    else msg.get("content", "")
                )
                result = store.process_image_turn(
                    image_path=image_path,
                    conversation_context=conversation_context,
                    date=msg.get("chat_time"),
                    skip_fact_apply=True,
                )

                user_facts = (result.get("changes") or {}).get(
                    "user_facts_pending", []
                )
                fact_sentence = _render_user_facts(user_facts)
                content = msg.get("content") or ""
                if fact_sentence:
                    content = f"{content}\n{fact_sentence}" if content else fact_sentence

                image_turns += 1

                if not content:
                    continue
                out = {"role": msg.get("role", "user"), "content": content}
                if msg.get("chat_time") is not None:
                    out["chat_time"] = msg["chat_time"]
                enriched.append(out)
            else:
                if msg.get("content"):
                    out = {
                        "role": msg.get("role", "user"),
                        "content": msg["content"],
                    }
                    if msg.get("chat_time") is not None:
                        out["chat_time"] = msg["chat_time"]
                    enriched.append(out)
        
        # ── Pass 2: send everything to memos in one go ────────────────
        if enriched:
            self.memos_client.add(
                messages=enriched,
                user_id=user_id,
                conv_id=conv_id,
                batch_size=batch_size,
            )
            
    # ── search ────────────────────────────────────────────────────────
    def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 10,
        question_images: list[str] | None = None,
    ):
        store = self.get_store(user_id)
        visual_mem = store.search_visual(
            question=query,
            question_images=question_images,
        )
        fact_mem = self.memos_client.search(
            query=query,
            user_id=user_id,
            top_k=top_k,
        )

        return {
            **(visual_mem or {}),
            **(fact_mem or {}),
        }
