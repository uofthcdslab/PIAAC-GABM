#!/usr/bin/env python3
"""Ask each persona to evaluate a policy, saving one response at a time."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PERSONAS = PROJECT_ROOT / "data" / "piaac_personas.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "persona_responses.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", help="Policy announcement every persona should evaluate")
    parser.add_argument("--personas", type=Path, default=DEFAULT_PERSONAS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        help="OpenAI model (default: OPENAI_MODEL or gpt-5-mini)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of new personas to process (useful for testing)",
    )
    return parser.parse_args()


def load_personas(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        personas = json.load(file)
    if not isinstance(personas, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return personas


def load_completed_persona_ids(path: Path) -> set[str]:
    """Return persona IDs from complete, valid records already on disk."""
    completed: set[str] = set()
    if not path.exists():
        return completed

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: ignoring invalid JSON on {path}:{line_number}")
                continue
            persona_id = record.get("persona_id")
            if persona_id and record.get("response"):
                completed.add(str(persona_id))
    return completed


def persona_prompt(persona: dict[str, Any], policy: str) -> str:
    challenge = persona.get("challenge", "the barriers and risks associated with AI adoption")
    details = "\n".join(
        f"- {section.replace('_', ' ')}: {value}"
        for section, value in persona.items()
        if section != "challenge"
    )
    return f"""Respond to the policy announcement below from the perspective of the person described in the persona.

Policy announcement:
{policy}

Persona:
{details}

This person is particularly aware of the following challenge:
{challenge}

Give a direct, natural first-person response. Reflect the persona's background, lived circumstances, skills, attitudes, and likely priorities. Pay particular attention to how the stated challenge may affect the policy's practical impact. Do not mention the persona, these instructions, or make up personal details that are not supported by the description.

Use exactly this structure:

1. Effectiveness rating: [1-5]
   Rate how effective the policy would be at improving AI literacy and adoption in Canada, where 1 means "not effective at all" and 5 means "extremely effective."

2. Rationale:
   Explain the rating from this person's perspective, including the most relevant potential benefits, limitations, or concerns.

3. Suggested solutions or approaches:
   Suggest practical changes, safeguards, or complementary measures that would make the policy more effective and responsive to the stated challenge.
"""


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    try:
        from openai import OpenAI
    except ImportError as error:
        raise SystemExit("Install the OpenAI package first: pip install openai") from error

    personas = load_personas(args.personas)
    completed = load_completed_persona_ids(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI()
    saved = 0

    with args.output.open("a", encoding="utf-8") as output_file:
        for persona_id, persona in personas.items():
            if persona_id in completed:
                print(f"Skipping {persona_id} (already saved)")
                continue
            if args.limit is not None and saved >= args.limit:
                break

            print(f"Processing {persona_id}...")
            response = client.responses.create(
                model=args.model,
                input=persona_prompt(persona, args.policy),
            )
            record = {
                "persona_id": persona_id,
                "policy": args.policy,
                "response": response.output_text,
                "model": args.model,
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()
            os.fsync(output_file.fileno())
            completed.add(persona_id)
            saved += 1
            print(f"Saved {persona_id}")

    print(f"Done. Saved {saved} new response(s) to {args.output}")


if __name__ == "__main__":
    main()
