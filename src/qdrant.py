from dotenv import load_dotenv
import os

load_dotenv()

class QdrantManager:
    """
    Manages vector embeddings using Gemini embeddings and Qdrant for
    storage/search.

    - Gemini: encodes text into dense vectors
    - Qdrant: stores vectors with metadata, enables fast similarity search
    """

    def __init__(
        self,
        embedding_model: str = "gemini-embedding-2-preview",
        embedding_dim: int = 1536,
        collection_name: str = "visual_memory",
    ):
        # ── Gemini for embedding ──
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ImportError("Gemini required. Install: pip install google-genai") from exc

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import (
                Distance, VectorParams, PointStruct, Filter,
                FieldCondition, MatchValue,
            )
        except ImportError as exc:
            raise ImportError("Qdrant required. Install: pip install qdrant-client") from exc

        self._gemini = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self._genai_types = types
        self._embedding_model = embedding_model
        self._embed_dim = embedding_dim
        print(
            "Gemini embeddings: model=%s, dim=%d",
            embedding_model,
            self._embed_dim,
        )

        # ── Qdrant for vector storage ──
        try:
            
            self._qdrant_models = {
                "Distance": Distance,
                "VectorParams": VectorParams,
                "PointStruct": PointStruct,
                "Filter": Filter,
                "FieldCondition": FieldCondition,
                "MatchValue": MatchValue,
            }
            self.qdrant = QdrantClient(
                url=os.getenv("QDRANT_URL"), 
                api_key=os.getenv("QDRANT_API_KEY"),
            )
            self.collection_name = collection_name
            self._ensure_collection()
            print("Qdrant connected: url=%s, collection=%s",
                        os.getenv("QDRANT_URL"), collection_name)
        except ImportError:
            raise

    def _ensure_collection(self):
        """Create collection if it doesn't exist, or recreate if dimension mismatches."""
        collections = {c.name for c in self.qdrant.get_collections().collections}
        if self.collection_name in collections:
            info = self.qdrant.get_collection(self.collection_name)
            existing_dim = info.config.params.vectors.size
            if existing_dim != self._embed_dim:
                print(
                    "WARNING: Collection %s has dim=%d, expected %d. Recreating.",
                    self.collection_name, existing_dim, self._embed_dim,
                )
                self.qdrant.delete_collection(self.collection_name)
                collections.discard(self.collection_name)
        if self.collection_name not in collections:
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=self._qdrant_models["VectorParams"](
                    size=self._embed_dim,
                    distance=self._qdrant_models["Distance"].COSINE,
                ),
            )
            print("Created Qdrant collection: %s (dim=%d)",
                        self.collection_name, self._embed_dim)

        # Qdrant Cloud requires an explicit payload index before a field can
        # be used in a filter. Idempotent — Qdrant returns OK if it exists.
        try:
            self.qdrant.create_payload_index(
                collection_name=self.collection_name,
                field_name="type",
                field_schema="keyword",
            )
        except Exception as e:
            # Already-exists errors are fine; other errors we want to see.
            if "already exists" not in str(e).lower():
                print(f"WARNING: failed to create payload index on 'type': {e}")

    def _embed_with_task_type(self, text: str, task_type: str) -> list[float]:
        """Embed text with a Gemini task type chosen for the retrieval intent."""
        response = self._gemini.models.embed_content(
            model=self._embedding_model,
            contents=text,
            config=self._genai_types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self._embed_dim,
            ),
        )
        if getattr(response, "embeddings", None):
            return response.embeddings[0].values
        if getattr(response, "embedding", None):
            return response.embedding.values
        raise ValueError("Gemini embedding response did not contain vectors")

    def embed_query_text(self, text: str) -> list[float]:
        """Embed a search query for retrieval against stored image metadata."""
        return self._embed_with_task_type(text, "RETRIEVAL_QUERY")

    def embed_document_text(self, text: str) -> list[float]:
        """Embed stored text that should be retrieved later by search queries."""
        return self._embed_with_task_type(text, "RETRIEVAL_DOCUMENT")

    def embed_similarity_text(self, text: str) -> list[float]:
        """Embed text for symmetric text-to-text similarity comparisons."""
        return self._embed_with_task_type(text, "SEMANTIC_SIMILARITY")

    def embed_tags_and_summary(self, tags: list[str], summary: str) -> list[float]:
        """Create a combined embedding from tags + summary."""
        combined_parts = [summary.strip()] if summary and summary.strip() else []
        if tags:
            combined_parts.append(", ".join(tags))
        combined = "\n".join(combined_parts).strip()
        return self.embed_document_text(combined)

    def upsert_image(
        self,
        image_id: str,
        embedding: list[float],
        metadata: dict,
    ) -> None:
        """Store an image embedding + metadata in Qdrant."""
        # Use image_id hash as integer point ID
        point_id = abs(hash(image_id)) % (2**63)
        PointStruct = self._qdrant_models["PointStruct"]
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "type": "image",
                    "image_id": image_id,
                    **metadata,
                },
            )],
        )

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        filter_payload: dict | None = None,
    ) -> list[dict]:
        """
        Search Qdrant for images similar to query text.

        Returns list of {image_id, score, payload} dicts.
        """
        query_emb = self.embed_query_text(query_text)

        # Build filter if provided
        query_filter = None
        if filter_payload:
            Filter = self._qdrant_models["Filter"]
            FieldCondition = self._qdrant_models["FieldCondition"]
            MatchValue = self._qdrant_models["MatchValue"]
            conditions = []
            for key, value in filter_payload.items():
                conditions.append(FieldCondition(
                    key=key, match=MatchValue(value=value)
                ))
            query_filter = Filter(must=conditions)

        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_emb,
            limit=top_k,
            query_filter=query_filter,
        )

        return [
            {
                "image_id": hit.payload.get("image_id"),
                "score": hit.score,
                **hit.payload,
            }
            for hit in results.points
        ]

    def clear_collection(self) -> None:
        """Delete and recreate the collection (wipes all vectors)."""
        self.qdrant.delete_collection(self.collection_name)
        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=self._qdrant_models["VectorParams"](
                size=self._embed_dim,
                distance=self._qdrant_models["Distance"].COSINE,
            ),
        )
        print("Cleared Qdrant collection: %s", self.collection_name)

    def delete_image(self, image_id: str) -> None:
        """Delete an image from Qdrant by image_id."""
        point_id = abs(hash(image_id)) % (2**63)
        self.qdrant.delete(
            collection_name=self.collection_name,
            points_selector=[point_id],
        )
