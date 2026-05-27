import argparse
import json
import os

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time

from dotenv import load_dotenv
from tqdm import tqdm
from .paths import BASE_DIR, DEFAULT_INPUT


def resolve_image(path: str) -> str | None:
    """Return the absolute path for an image if it exists, else None."""
    full = os.path.join(BASE_DIR, path) if not os.path.isabs(path) else path
    if os.path.exists(full):
        return full
    return None

# ── per-frame search functions ───────────────────────────────────────

def ours_search(client, query, user_id, top_k):
    """VisualMemClient.search routes visual questions to answer_question
    and general questions to MemosApiOnlineClient.search."""
    start = time()
    result = client.search(
        query=query, user_id=user_id, top_k=top_k
    )

    # search() now returns a flat dict with keys from both visual_mem and
    # fact_mem merged together.  Always extract both sides.
    image_paths = result.get("image_paths", [])

    parts = []
    context_block = result.get("context_block", "")
    if context_block:
        parts.append(f"=== Visual Memory ===\n{context_block}")

    memories = result.get("text_mem", [{}])[0].get("memories", [])
    fact_text = "\n".join(m["memory"] for m in memories)
    pref = result.get("pref_string", "")
    if fact_text or pref:
        parts.append(f"=== Fact Memory ===\n{fact_text}\n{pref}".strip())

    context = "\n".join(parts)

    duration_ms = (time() - start) * 1000
    return context, image_paths, duration_ms


def memos_api_online_search(client, query, user_id, top_k):
    start = time()
    result = client.search(query=query, user_id=user_id, top_k=top_k)

    memories = result.get("text_mem", [{}])[0].get("memories", [])
    context = (
        "\n".join([i["memory"] for i in memories])
        + f"\n{result.get('pref_string', '')}"
    )

    duration_ms = (time() - start) * 1000
    return context, duration_ms

# ── dispatch ─────────────────────────────────────────────────────────

def search_query(client, query, user_id, frame, top_k=20):
    if frame == "ours":
        return ours_search(client, query, user_id, top_k)
    if frame == "memos":
        return memos_api_online_search(client, query, user_id, top_k)
    raise ValueError(f"Unknown frame: {frame}")


# ── client factory ───────────────────────────────────────────────────

def _build_client(frame, version, user_id):
    if frame == "ours":
        from google import genai
        from src.client_wrapper import VisualMemClient
        from src.client_text_backend import MemosApiOnlineClient

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY must be set for frame=ours")
        llm_client = genai.Client(api_key=api_key)
        
        memos_client = MemosApiOnlineClient()
        memory_dir = f"results/visualmem/{frame}-{version}/memories"
        os.makedirs(memory_dir, exist_ok=True)
        return VisualMemClient(
            llm_client=llm_client,
            memos_client=memos_client,
            memory_dir=memory_dir,
            memory_prefix="visual_memory",
            collection_prefix=f"visual_eval_{version}",
        )

    if frame == "memos":
        from src.client_text_backend import MemosApiOnlineClient

        return MemosApiOnlineClient()

    raise ValueError(f"Unknown frame: {frame}")


# ── per-persona processing ───────────────────────────────────────────

def load_existing_results(frame, version, persona_idx):
    result_path = (
        f"results/visualmem/{frame}-{version}/tmp/"
        f"{frame}_search_results_{persona_idx}.json"
    )
    if os.path.exists(result_path):
        try:
            with open(result_path) as f:
                return json.load(f), True
        except Exception as e:
            print(f"Error loading existing results for persona {persona_idx}: {e}")
    return {}, False


def process_persona(
    persona_idx, personas, frame, version, top_k=20, num_workers=1,
):
    persona_data = personas[persona_idx]
    persona_name = persona_data.get("name", "Unknown")
    events = persona_data.get("events", [])
    user_id = f"persona_{persona_idx}_{version}"
    persona_key = f"persona_{persona_idx}"

    existing_results, loaded = load_existing_results(frame, version, persona_idx)
    if loaded:
        print(f"Loaded existing results for persona {persona_idx}")
        return existing_results

    client = _build_client(frame, version, user_id)
    search_results = defaultdict(list)

    # Collect all questions across events
    qa_items = []
    for event in events:
        event_id = event.get("event_id")
        hidden_fact = event.get("question_hint", {}).get("hidden_fact", "")
        conversation_mode = event.get("conversation_mode", "")
        for question in event.get("questions", []):
            qa_items.append({
                "event_id": event_id,
                "hidden_fact": hidden_fact,
                "question": question,
                "conversation_mode": conversation_mode,
            })

    def process_qa(qa_item):
        q_data = qa_item["question"]
        query = q_data.get("question", "")
        if not query:
            return None

        question_image = q_data.get("image_path", None)

        if frame == "ours":
            context, image_paths, duration_ms = search_query(
                client, query, user_id, frame, top_k=top_k,
            )
        else:
            context, duration_ms = search_query(
                client, query, user_id, frame, top_k=top_k,
            )
            image_paths = []

        if not context:
            print(f"No context found for query: {query}")
            context = ""

        return {
            "event_id": qa_item["event_id"],
            "question_id": q_data.get("id"),
            "query": query,
            "conversation_mode": qa_item["conversation_mode"],
            "question_image": question_image,
            "choices": q_data.get("choices", {}),
            "ground_truth": q_data.get("ground_truth", ""),
            "hidden_fact": qa_item["hidden_fact"],
            "context": context,
            "image_paths": image_paths,
            "duration_ms": duration_ms,
        }

    futures = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for qa_item in qa_items:
            futures.append(executor.submit(process_qa, qa_item))

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Persona {persona_idx} ({persona_name})",
        ):
            result = future.result()
            if result:
                search_results[persona_key].append(result)

    os.makedirs(f"results/visualmem/{frame}-{version}/tmp/", exist_ok=True)
    out_path = (
        f"results/visualmem/{frame}-{version}/tmp/"
        f"{frame}_search_results_{persona_idx}.json"
    )
    with open(out_path, "w") as f:
        json.dump(dict(search_results), f, indent=2)
        print(f"Saved search results for persona {persona_idx}")

    return search_results


# ── driver ───────────────────────────────────────────────────────────

def main(frame, version="default", num_workers=1, top_k=20, input_path=DEFAULT_INPUT):
    load_dotenv()
    print(f"Loading personas from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        personas = json.load(f)
    num_personas = len(personas)

    os.makedirs(f"results/visualmem/{frame}-{version}/", exist_ok=True)
    all_search_results = defaultdict(list)

    for idx in range(num_personas):
        print(f"Processing persona {idx}...")
        persona_results = process_persona(
            idx, personas, frame, version, top_k, num_workers,
        )
        for key, results in persona_results.items():
            all_search_results[key].extend(results)

    out_path = f"results/visualmem/{frame}-{version}/{frame}_search_results.json"
    with open(out_path, "w") as f:
        json.dump(dict(all_search_results), f, indent=2)
        print("Saved all search results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lib",
        type=str,
        choices=[
            "ours",
            "memos",
        ],
        default="ours",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="default",
        help="Version identifier for saving results",
    )
    parser.add_argument(
        "--workers", type=int, default=5,
        help="Number of parallel workers per persona",
    )
    parser.add_argument(
        "--top_k", type=int, default=15,
        help="Number of results to retrieve",
    )
    parser.add_argument(
        "--input", type=str, default=DEFAULT_INPUT,
        help="Path to data.json",
    )
    args = parser.parse_args()
    main(args.lib, args.version, args.workers, args.top_k, args.input)
