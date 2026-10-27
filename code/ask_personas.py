import json
from pathlib import Path
from random import seed
from openai import OpenAI
from time import time

seed(42)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PERSONAS = PROJECT_ROOT / "data" / "piaac_personas.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "single"


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


def persona_prompt(persona, policy):
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

This person is particularly aware of the following challenge about the Canadian AI landscape:
{challenge}

Give a direct, natural first-person response. Reflect the persona's background, lived circumstances, skills, attitudes, and likely priorities. Pay particular attention to how the stated challenge may affect the policy's practical impact. Do not mention the persona, these instructions, or make up personal details that are not supported by the description.

Use exactly this structure:

1. Effectiveness rating: [1-5]
   Rate how effective the policy would be at improving AI literacy and adoption in Canada, where 1 means "not effective at all" and 5 means "extremely effective."

2. Rationale:
   Explain the rating from this person's perspective, including the most relevant potential benefits, limitations, or concerns.

3. Suggested solutions or approaches:
   Suggest practical changes, safeguards, or complementary measures that would make the policy more effective and responsive to the stated challenge.

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

with open(PROJECT_ROOT / "data" / "policies.json", "r", encoding="utf-8") as file:
    policies = json.load(file)

with open(PROJECT_ROOT / "data" / "piaac_personas.json", "r", encoding="utf-8") as file:
    personas = json.load(file)

for persona_id, persona in personas.items():
    # print(f"Persona ID: {persona_id}")
    if Path(DEFAULT_OUTPUT / f"{persona_id}.json").exists():
        print(f"Skipping {persona_id} (already saved)")
        continue
    client = OpenAI()
    print(f"Processing {persona_id}...")

    start = time()
    records = []

    for policy_id, policy in policies.items():
        print(f"asking response to {policy_id}")
        response = client.responses.create(
            model='gpt-4o-mini',
            tools=[{"type": "web_search"}],
            input=persona_prompt(persona, policy),
        )
        record = {
            "policy": policy_id,
            "response": response.output_text,
        }
        records.append(record)

    with open(DEFAULT_OUTPUT/f"{persona_id}.json", "w", encoding="utf-8") as fout:
        json.dump(records, fout, indent=4)
    print(f"Done. Saved {persona_id} new response(s). It took {time()-start} seconds.")
