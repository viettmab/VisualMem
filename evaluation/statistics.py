import argparse
import json
import os

from collections import defaultdict
from .paths import BASE_DIR, DEFAULT_INPUT


def _new_counter():
    return {
        "total_questions": 0,
        "answered": 0,
        "correct": 0,
        "skipped": 0,
    }


def _update_counter(counter, response):
    counter["total_questions"] += 1
    if response.get("predicted") is None:
        counter["skipped"] += 1
        return
    counter["answered"] += 1
    if response.get("correct"):
        counter["correct"] += 1


def _finalize_counter(counter):
    answered = counter["answered"]
    total = counter["total_questions"]
    counter["accuracy"] = round(counter["correct"] / answered, 4) if answered else 0.0
    counter["coverage"] = round(answered / total, 4) if total else 0.0
    return counter


def load_conversation_mode_lookup(input_path):
    """Build conversation_mode lookup for old response files.

    The phase2 file stores conversation_mode on each event. Response files
    include persona key and event_id, so the primary key is
    (persona_key, event_id). A secondary event_id-only lookup is kept for
    compatibility when event IDs are globally unique.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        personas = json.load(f)

    by_persona_event = {}
    event_modes = defaultdict(set)

    for persona_idx, persona in enumerate(personas):
        persona_key = f"persona_{persona_idx}"
        for event in persona.get("events", []):
            event_id = event.get("event_id")
            if event_id is None:
                continue
            mode = event.get("conversation_mode", "") or "unknown"
            by_persona_event[(persona_key, event_id)] = mode
            event_modes[event_id].add(mode)

    by_event = {
        event_id: next(iter(modes))
        for event_id, modes in event_modes.items()
        if len(modes) == 1
    }
    return by_persona_event, by_event


def get_conversation_mode(response, persona_key, by_persona_event, by_event):
    if response.get("conversation_mode"):
        return response["conversation_mode"]

    event_id = response.get("event_id")
    if (persona_key, event_id) in by_persona_event:
        return by_persona_event[(persona_key, event_id)]
    return by_event.get(event_id, "unknown")


def compute_statistics(responses, input_path):
    by_persona_event, by_event = load_conversation_mode_lookup(input_path)

    overall = _new_counter()
    by_mode = defaultdict(_new_counter)
    by_persona = defaultdict(_new_counter)
    missing_mode = []

    for persona_key, persona_responses in responses.get("persona_responses", {}).items():
        for response in persona_responses:
            mode = get_conversation_mode(
                response, persona_key, by_persona_event, by_event
            )
            if mode == "unknown":
                missing_mode.append(
                    {
                        "persona": persona_key,
                        "event_id": response.get("event_id"),
                        "question_id": response.get("question_id"),
                    }
                )

            _update_counter(overall, response)
            _update_counter(by_mode[mode], response)
            _update_counter(by_persona[persona_key], response)

    return {
        "frame": responses.get("frame"),
        "version": responses.get("version"),
        "overall": _finalize_counter(overall),
        "by_conversation_mode": {
            mode: _finalize_counter(stats)
            for mode, stats in sorted(by_mode.items())
        },
        "by_persona": {
            persona: _finalize_counter(stats)
            for persona, stats in sorted(by_persona.items())
        },
        "missing_conversation_mode": missing_mode,
    }


def print_summary(stats):
    overall = stats["overall"]
    print("\nOverall")
    print(f"  Total questions: {overall['total_questions']}")
    print(f"  Answered:        {overall['answered']}")
    print(f"  Correct:         {overall['correct']}")
    print(f"  Skipped:         {overall['skipped']}")
    print(f"  Accuracy:        {overall['accuracy']:.2%}")
    print(f"  Coverage:        {overall['coverage']:.2%}")

    print("\nBy conversation_mode")
    for mode, values in stats["by_conversation_mode"].items():
        print(
            f"  {mode}: "
            f"accuracy={values['accuracy']:.2%}, "
            f"coverage={values['coverage']:.2%}, "
            f"correct={values['correct']}, "
            f"answered={values['answered']}, "
            f"total={values['total_questions']}, "
            f"skipped={values['skipped']}"
        )

    missing = stats["missing_conversation_mode"]
    if missing:
        print(f"\nWARNING: missing conversation_mode for {len(missing)} responses")


def main(frame, version="default", input_path=DEFAULT_INPUT, output_path=None):
    response_path = f"results/visualmem/{frame}-{version}/{frame}_responses.json"
    if output_path is None:
        output_path = f"results/visualmem/{frame}-{version}/{frame}_statistics.json"

    with open(response_path, "r", encoding="utf-8") as f:
        responses = json.load(f)

    stats = compute_statistics(responses, input_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print_summary(stats)
    print(f"\nStatistics saved to: {output_path}")


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
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help="Path to data.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path for the statistics JSON",
    )
    args = parser.parse_args()
    main(args.lib, args.version, args.input, args.output)
