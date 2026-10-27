import json
from pathlib import Path
from random import randint, sample, seed


PROJECT_ROOT = Path(__file__).resolve().parent.parent
seed(42)

with open(PROJECT_ROOT / "data" / "piaac_personas.json", encoding="utf-8") as file:
    personas = json.load(file)

with open(PROJECT_ROOT / "data" / "challenges.json", encoding="utf-8") as file:
    challenges = json.load(file)


persona_challenges = {}
challenge_names = list(challenges.keys())
for persona_id in personas:
    number_of_challenges = randint(0, len(challenge_names))
    persona_challenges[persona_id] = sample(challenge_names, number_of_challenges)

output = PROJECT_ROOT / "outputs" / "persona_challenges.json"
with open(output, "w", encoding="utf-8") as file:
    json.dump(persona_challenges, file, indent=4)

print(f"Wrote {len(persona_challenges)} assignments to {output}")
