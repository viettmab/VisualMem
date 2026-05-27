from __future__ import annotations

import argparse
import json
import os
import re

from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time

from dotenv import load_dotenv
from tqdm import tqdm
from .paths import BASE_DIR

from PIL import Image

def query_gemini_interleave(client, content_blocks: list[dict], model_name: str = "gemini-3.1-pro-preview") -> str:
    """Query Gemini API with interleaved text and image blocks.

    Each element of content_blocks must be one of:
        {"type": "text",  "text": "..."}
        {"type": "image", "image_path": "/abs/or/rel/path.jpg"}

    Blocks are sent to the model in their original order so the model can
    correlate each image with the text that surrounds it.
    """
    try:
        count = 0
        count_text = 0
        contents = []
        for block in content_blocks:
            if block["type"] == "text":
                contents.append(block["text"])
                count_text += 1
            elif block["type"] == "image":
                count += 1
                image_path = block["image_path"]
                img = Image.open(image_path)
                contents.append(img)
            else:
                raise ValueError(f"Unknown block type: {block['type']}")

        # print(f"Sending {count} images to {model_name}")
        # print(f"Sending {count_text} text to {model_name}")
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
        )
        return response.text
    except Exception as e:
        print(f"Error querying Gemini (interleave): {e}")
        return None

# ── helpers ──────────────────────────────────────────────────────────

def resolve_image(path: str) -> str | None:
    full = os.path.join(BASE_DIR, path) if not os.path.isabs(path) else path
    if os.path.exists(full):
        return full
    return None


def parse_response(response: str) -> tuple[str | None, str]:
    if not response:
        return None, ""
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```$", "", text.strip())
    try:
        data = json.loads(text)
        answer = str(data.get("final_answer", "")).strip().upper()
        reasoning = str(data.get("reasoning", ""))
        if answer in ("A", "B", "C", "D"):
            return answer, reasoning
    except (json.JSONDecodeError, AttributeError):
        pass
    match = re.search(r"\b([A-Da-d])\b", response)
    if match:
        return match.group(1).upper(), ""
    return None, ""


def build_question_prompt(search_result: dict) -> tuple[str, list[str]]:
    """Build MCQ prompt from a search result entry.

    Returns (prompt_text, question_image_paths).
    """
    q_text = search_result.get("query", "")
    q_choices = search_result.get("choices", {})
    context = search_result.get("context", "")

    text_choices: dict = {}
    image_choices: dict = {}
    question_images: list[str] = []

    for label, value in sorted(q_choices.items()):
        if isinstance(value, str) and value.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            image_choices[label] = value
        else:
            text_choices[label] = value

    choices_text = "\n".join(
        f"  {label}: {text}" for label, text in sorted(text_choices.items())
    )
    if image_choices:
        labels = ", ".join(sorted(image_choices.keys()))
        suffix = f"  (Choices {labels} are shown as images below)"
        choices_text = f"{choices_text}\n{suffix}" if choices_text else suffix
        for _, img_path in sorted(image_choices.items()):
            img = resolve_image(img_path)
            if img:
                question_images.append(img)

    # Question-level image goes first
    q_image = search_result.get("question_image")
    if q_image:
        img = resolve_image(q_image)
        if img:
            question_images.insert(0, img)

    full_question = (
        f"{q_text}\n\nChoices:\n{choices_text}\n\n"
        "Select the single best answer. "
        "Reply with ONLY a valid JSON object: "
        '{"final_answer": "<A, B, C, or D>", "reasoning": "<brief explanation>"}'
    )

    if context:
        full_question = f"Memory context:\n{context}\n\n{full_question}"

    return full_question, question_images


# ── LLM call ─────────────────────────────────────────────────────────

def answer_question(
    gemini_client: genai.Client,
    prompt: str,
    question_images: list[str] | None = None,
    memory_images: list[str] | None = None,
):
    """Call the LLM with the MCQ prompt, optionally including images."""
    content_blocks: list[dict] = [
        {
            "type": "text",
            "text": (
                "You are a helpful assistant with access to the user's memory. "
                "Answer the multiple-choice question based on the provided "
                "context and images."
            ),
        }
    ]

    # Memory-retrieved images first
    for img_path in (memory_images or [])[:10]:
        if os.path.exists(img_path):
            content_blocks.append({"type": "image", "image_path": img_path})
        else:
            print(f"  WARNING: memory image not found: {img_path}")

    # Question images (query image + image-based choices)
    for img_path in (question_images or []):
        if os.path.exists(img_path):
            content_blocks.append({"type": "image", "image_path": img_path})
        else:
            print(f"  WARNING: question image not found: {img_path}")

    content_blocks.append({"type": "text", "text": prompt})

    return query_gemini_interleave(
        gemini_client,
        content_blocks,
        model_name=os.getenv("CHAT_MODEL", "gemini-3.1-pro-preview"),
    )


# ── per-question processing ──────────────────────────────────────────

def process_qa(gemini_client, search_result):
    start = time()

    prompt, question_images = build_question_prompt(search_result)
    memory_images = search_result.get("image_paths") or []
    ground_truth = search_result.get("ground_truth", "").upper()

    try:
        raw_answer = answer_question(
            gemini_client, prompt,
            question_images=question_images,
            memory_images=memory_images,
        )
    except Exception as e:
        print(f"  [Error] Question {search_result.get('question_id')} failed: {e}")
        raw_answer = ""

    predicted, reasoning = parse_response(raw_answer)
    is_correct = predicted == ground_truth if predicted else False

    duration_ms = (time() - start) * 1000

    return {
        "event_id": search_result.get("event_id"),
        "question_id": search_result.get("question_id"),
        "query": search_result.get("query"),
        "choices": search_result.get("choices"),
        "ground_truth": ground_truth,
        "hidden_fact": search_result.get("hidden_fact", ""),
        "predicted": predicted,
        "reasoning": reasoning,
        "correct": is_correct,
        "raw_response": raw_answer,
        "search_context": search_result.get("context", ""),
        "memory_images": memory_images,
        "response_duration_ms": duration_ms,
        "search_duration_ms": search_result.get("duration_ms", 0),
    }


# ── persona processing ────────────────────────────────────────────────

def process_persona(persona_key, qa_results, api_key, position=0):
    from google import genai

    gemini_client = genai.Client(api_key=api_key)
    persona_responses = []
    totals = {"total": 0, "correct": 0, "skipped": 0}

    print(f"\nProcessing {persona_key} ({len(qa_results)} questions)...")
    for sr in tqdm(qa_results, desc=f"  {persona_key}", position=position, leave=True):
        r = process_qa(gemini_client, sr)
        persona_responses.append(r)

        if r["predicted"] is None:
            totals["skipped"] += 1
        else:
            totals["total"] += 1
            if r["correct"]:
                totals["correct"] += 1

    return persona_key, persona_responses, totals


def write_responses(response_path, frame, version, all_responses, totals):
    accuracy = totals["correct"] / totals["total"] if totals["total"] else 0.0
    output = {
        "frame": frame,
        "version": version,
        "total_questions": totals["total"],
        "correct": totals["correct"],
        "skipped": totals["skipped"],
        "accuracy": round(accuracy, 4),
        "persona_responses": all_responses,
    }
    with open(response_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


# ── driver ───────────────────────────────────────────────────────────

def main(frame, version="default", num_workers=1):
    search_path = f"results/visualmem/{frame}-{version}/{frame}_search_results.json"
    response_path = f"results/visualmem/{frame}-{version}/{frame}_responses.json"

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY must be set for Gemini responses")
    with open(search_path, "r", encoding="utf-8") as f:
        search_results = json.load(f)

    all_responses = {}
    totals = {"total": 0, "correct": 0, "skipped": 0}

    os.makedirs(os.path.dirname(response_path), exist_ok=True)

    persona_items = list(search_results.items())
    worker_count = max(1, min(num_workers, len(persona_items)))

    if worker_count == 1:
        for position, (persona_key, qa_results) in enumerate(persona_items):
            persona_key, persona_responses, persona_totals = process_persona(
                persona_key, qa_results, api_key, position=position
            )
            all_responses[persona_key] = persona_responses
            for key in totals:
                totals[key] += persona_totals[key]
            write_responses(response_path, frame, version, all_responses, totals)
    else:
        print(f"Running {len(persona_items)} personas with {worker_count} workers...")
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    process_persona, persona_key, qa_results, api_key, position
                ): persona_key
                for position, (persona_key, qa_results) in enumerate(persona_items)
            }
            for future in as_completed(futures):
                persona_key, persona_responses, persona_totals = future.result()
                all_responses[persona_key] = persona_responses
                for key in totals:
                    totals[key] += persona_totals[key]
                write_responses(response_path, frame, version, all_responses, totals)

    accuracy = totals["correct"] / totals["total"] if totals["total"] else 0.0
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"  Total questions: {totals['total']}")
    print(f"  Correct:         {totals['correct']}")
    print(f"  Skipped:         {totals['skipped']}")
    print(f"  Accuracy:        {accuracy:.2%}")
    print(f"{'=' * 60}")
    print(f"Results saved to: {response_path}")


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
        help="Version identifier for loading results",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of persona workers to run in parallel",
    )
    args = parser.parse_args()
    main(args.lib, args.version, args.workers)
