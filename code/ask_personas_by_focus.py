import argparse
import json
from pathlib import Path
from typing import Any
from random import seed
from time import time

seed(42)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PERSONAS = PROJECT_ROOT / "data" / "piaac_personas.json"
PERSONA_CHALLENGES = PROJECT_ROOT / "outputs" / "persona_challenges.json"

FOCUS_PROMPTS = {
    "literacy": {
        "outcome": (
            "improving Canadians' ability to understand, critically assess, "
            "and appropriately use AI"
        ),
        "label": "AI literacy",
    },
    "adoption": {
        "outcome": (
            "increasing the responsible uptake and use of AI by people and "
            "organizations in Canada"
        ),
        "label": "responsible AI adoption",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask each persona to evaluate every policy for one outcome."
    )
    parser.add_argument(
        "--focus",
        required=True,
        choices=FOCUS_PROMPTS,
        help="Outcome to evaluate: AI literacy or responsible AI adoption.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory (default: outputs/<focus>).",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.6-terra",
        help="OpenAI model to use (default: gpt-5.6-terra).",
    )
    return parser.parse_args()


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


def persona_prompt(
    persona: dict[str, Any], policy: str, focus: str, assigned_challenges: str
) -> str:
    focus_prompt = FOCUS_PROMPTS[focus]
    details = "\n".join(
        f"- {section.replace('_', ' ')}: {value}"
        for section, value in persona.items()
    )
    return f"""Respond to the policy announcement below from the perspective of the person described in the persona.

Policy announcement:
{policy}

Persona:
{details}

This person has been assigned the following challenges about the Canadian AI landscape:
{assigned_challenges}

Give a direct, natural first-person response. Reflect the persona's background, lived circumstances, skills, attitudes, and likely priorities. Pay particular attention to how the assigned challenges, if any, may affect the policy's practical impact. Do not mention the persona, these instructions, or make up personal details that are not supported by the description.

Use exactly this structure:

1. Effectiveness rating: [1-5]
   Rate how effective the policy would be at {focus_prompt["outcome"]}, where 1 means "not effective at all" and 5 means "extremely effective."

2. Rationale:
   Explain the rating from this person's perspective. Focus specifically on {focus_prompt["label"]}, including the most relevant potential benefits, limitations, or concerns.

3. Suggested solutions or approaches:
   Suggest practical changes, safeguards, or complementary measures that would make the policy more effective at improving {focus_prompt["label"]} and responsive to the stated challenge.

Factual-grounding requirements:

- Distinguish between facts and personal judgments.
- Use the supplied policy announcement and evidence brief for factual claims.
- Do not invent statistics, program details, eligibility rules, costs,
  implementation timelines, or expected outcomes.
- If a relevant fact is missing or uncertain, search for reliable,
  up-to-date information before answering.
- Prefer official Canadian government sources and other primary sources.
- Cite the source immediately after any externally verified factual claim.
- If reliable information cannot be found, explicitly say that the
  information is uncertain or unavailable.
- The effectiveness rating is a personal judgment and should be based on
  the persona's circumstances plus the verified evidence.
- Do not invent experiences, beliefs, or circumstances not contained in
  the persona.
"""

def main() -> None:
    args = parse_args()
    output_dir = args.output or PROJECT_ROOT / "outputs" / args.focus
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from openai import OpenAI
    except ImportError as error:
        raise SystemExit("Install the OpenAI package first: pip install openai") from error

    with open(PROJECT_ROOT / "data" / "policies.json", "r", encoding="utf-8") as file:
        policies = json.load(file)

    with open(DEFAULT_PERSONAS, "r", encoding="utf-8") as file:
        personas = json.load(file)

    with open(PROJECT_ROOT / "data" / "challenges.json", "r", encoding="utf-8") as file:
        challenges = json.load(file)

    with open(PERSONA_CHALLENGES, "r", encoding="utf-8") as file:
        persona_challenges = json.load(file)

    client = OpenAI()
    for persona_id, persona in personas.items():
        output_path = output_dir / f"{persona_id}.json"
        if output_path.exists():
            print(f"Skipping {persona_id} (already saved)")
            continue

        print(f"Processing {persona_id} for {args.focus}...")
        start = time()
        records = []
        assigned_challenges = persona_challenges[persona_id]
        challenge_text = "\n".join(
            f"- {challenge}: {challenges[challenge]}"
            for challenge in assigned_challenges
        ) or "- No specific challenge assigned."

        for policy_id, policy in policies.items():
            print(f"asking response to {policy_id}")
            response = client.responses.create(
                model=args.model,
                tools=[{"type": "web_search"}],
                input=persona_prompt(persona, policy, args.focus, challenge_text),
            )
            records.append(
                {
                    "policy": policy_id,
                    "focus": args.focus,
                    "response": response.output_text,
                }
            )

        with output_path.open("w", encoding="utf-8") as fout:
            json.dump(records, fout, indent=4, ensure_ascii=False)
        print(
            f"Done. Saved {persona_id} responses for {args.focus}. "
            f"It took {time() - start} seconds."
        )


if __name__ == "__main__":
    main()
    
