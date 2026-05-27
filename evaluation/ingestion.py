import argparse
import concurrent.futures
import json
import os
import time
import base64
import io

from dotenv import load_dotenv
from PIL import Image
from .paths import BASE_DIR, DEFAULT_INPUT

def encode_image(image_path, max_size=256, quality=75): 
    img = Image.open(image_path)
    img.thumbnail((max_size, max_size))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def resolve_image(path: str) -> str | None:
    full = os.path.join(BASE_DIR, path) if not os.path.isabs(path) else path
    if os.path.exists(full):
        return full
    print(f"  WARNING: Image not found: {full}")
    return None


def event_to_messages(conversation: list[dict], chat_time: str | None) -> list[dict]:
    messages: list[dict] = []
    for turn in conversation:
        msg: dict = {
            "role": turn.get("role", "user"),
            "content": turn.get("content", ""),
        }
        if chat_time is not None:
            msg["chat_time"] = chat_time
        raw_img = turn.get("image_path")
        if raw_img:
            resolved = resolve_image(raw_img)
            if resolved:
                msg["image_path"] = resolved
        messages.append(msg)
    return messages


def _strip_images(messages: list[dict]) -> list[dict]:
    """Drop image_path and any turns with no textual content — used when
    feeding a text-only backend. `chat_time` is preserved."""
    out: list[dict] = []
    for m in messages:
        if not m.get("content"):
            continue
        plain = {"role": m.get("role", "user"), "content": m["content"]}
        if m.get("chat_time") is not None:
            plain["chat_time"] = m["chat_time"]
        out.append(plain)
    return out


def _messages_with_base64_images(messages: list[dict]) -> list[dict]:
    """Convert messages for memos: image turns get their image encoded as
    base64 in the OpenAI multimodal content format. Text-only turns and
    ``chat_time`` are preserved as-is."""
    out: list[dict] = []
    for m in messages:
        image_path = m.get("image_path")
        if image_path:
            content_parts = []
            if m.get("content"):
                content_parts.append({"type": "text", "text": m["content"]})
            base64_image = encode_image(image_path)
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            })
            msg = {"role": m.get("role", "user"), "content": content_parts}
        else:
            if not m.get("content"):
                continue
            msg = {"role": m.get("role", "user"), "content": m["content"]}
        if m.get("chat_time") is not None:
            msg["chat_time"] = m["chat_time"]
        out.append(msg)
    return out


def ingest_event(client, event, frame, metadata, window_size="full"):
    date = event.get("date", "")
    event_id = event.get("event_id")
    user_id = metadata["user_id"]
    persona_idx = metadata["persona_idx"]
    event_conv_id = f"persona_{persona_idx}_event_{event_id}"

    print(f"Processing persona {persona_idx}, event {event_id} [{date}]")
    start_time = time.time()

    messages = event_to_messages(event.get("conversation", []), chat_time=date)
    if not messages:
        return 0.0

    if frame == "ours":
        batch_size = 2 if window_size == "2" else len(messages)
        client.add(
            messages=messages,
            user_id=user_id,
            conv_id=event_conv_id,
            batch_size=batch_size,
            window_size=window_size,
        )
    elif frame == "memos":
        memos_messages = _messages_with_base64_images(messages)
        if not memos_messages:
            return 0.0
        client.add(
            memos_messages,
            user_id,
            event_conv_id,
            batch_size=2,
        )
    else:
        raise ValueError(f"Unknown frame: {frame}")

    return round(time.time() - start_time, 2)


def _build_client(frame, version, user_id, *, reset=True):
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
        client = VisualMemClient(
            llm_client=llm_client,
            memos_client=memos_client,
            memory_dir=memory_dir,
            memory_prefix="visual_memory",
            collection_prefix=f"visual_eval_{version}",
        )
        if reset:
            client.reset_user(user_id)
        return client

    if frame == "memos":
        from src.client_text_backend import MemosApiOnlineClient

        return MemosApiOnlineClient()

    raise ValueError(f"Unknown frame: {frame}")


def process_persona(persona_idx, frame, personas, version, success_records, f, window_size):
    persona_data = personas[persona_idx]
    events = persona_data.get("events", [])
    persona_name = persona_data.get("name", "Unknown")
    user_id = f"persona_{persona_idx}_{version}"

    persona_record_prefix = f"{persona_idx}_"
    persona_has_success = any(
        record.startswith(persona_record_prefix) for record in success_records
    )
    client = _build_client(
        frame,
        version,
        user_id,
        reset=not persona_has_success,
    )

    print(
        f"Processing persona {persona_idx} ({persona_name}) — "
        f"{len(events)} events"
    )
    start_time = time.time()
    total_event_time = 0.0

    for event in events:
        event_id = event.get("event_id")
        record_key = f"{persona_idx}_{event_id}"
        if record_key in success_records:
            print(f"Event {record_key} already ingested")
            continue

        metadata = {"persona_idx": persona_idx, "user_id": user_id}
        event_time = ingest_event(client, event, frame, metadata, window_size)
        total_event_time += event_time
        print(
            f"Persona {persona_idx}, event {event_id} processed in "
            f"{event_time} seconds"
        )
        f.write(f"{record_key}\n")
        f.flush()

    elapsed = round(time.time() - start_time, 2)
    print(f"Persona {persona_idx} processed successfully in {elapsed} seconds")
    return elapsed


def main(
    frame,
    version="default",
    num_workers=4,
    input_path=DEFAULT_INPUT,
    window_size="full",
):
    load_dotenv()
    if window_size not in ("2", "full"):
        raise ValueError("window_size must be '2' or 'full'")
    print(f"Loading personas from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        personas = json.load(f)
    num_personas = len(personas)

    start_time = time.time()
    total_time = 0.0
    print(
        f"Starting processing for {num_personas} personas using "
        f"{num_workers} workers..."
    )
    os.makedirs(f"results/visualmem/{frame}-{version}/", exist_ok=True)
    success_records: list[str] = []
    record_file = f"results/visualmem/{frame}-{version}/success_records.txt"
    if os.path.exists(record_file):
        with open(record_file) as fh:
            for line in fh.readlines():
                success_records.append(line.strip())

    with (
        concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor,
        open(record_file, "a+") as f,
    ):
        futures = [
            executor.submit(
                process_persona,
                persona_idx,
                frame,
                personas,
                version,
                success_records,
                f,
                window_size,
            )
            for persona_idx in range(num_personas)
        ]
        for future in concurrent.futures.as_completed(futures):
            persona_time = future.result()
            total_time += persona_time

    average_time = total_time / max(num_personas, 1)
    minutes = int(average_time // 60)
    seconds = int(average_time % 60)
    print(
        f"The frame {frame} processed {num_personas} personas in average of "
        f"{minutes} minutes and {seconds} seconds per persona."
    )
    elapsed = round(time.time() - start_time, 2)
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"Total processing time: {minutes} minutes and {seconds} seconds.")


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
        help="Version identifier for saving results (e.g., 1010)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers to process personas",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help="Path to data.json",
    )
    parser.add_argument(
        "--window-size",
        type=str,
        choices=["2", "full"],
        default="full",
        help="Context window for --lib ours: '2' for per-image text, 'full' for full event context.",
    )
    args = parser.parse_args()

    main(args.lib, args.version, args.workers, args.input, args.window_size)
