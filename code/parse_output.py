import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from openpyxl import Workbook

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "single"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "parsed_responses.xlsx"

RESPONSE_PATTERN = re.compile(
    r"""
    (?:\#{1,6}\s*)?\*{0,2}(?:1\.\s*)?\*{0,2}
    (?:Effectiveness\s+rating|Évaluation\s+de\s+l'efficacité)
    \s*\*{0,2}\s*:\s*(?:\*{0,2}\s*){0,2}
    (?P<rating>[1-5])\*{0,2}.*?
    (?:\#{1,6}\s*)?\*{0,2}(?:2\.\s*)?\*{0,2}
    (?:Rationale|Raisonnement)
    \s*\*{0,2}\s*:\s*(?:\*{0,2}\s*){0,2}
    (?P<rationale>.*?)
    (?:\#{1,6}\s*)?\*{0,2}(?:3\.\s*)?\*{0,2}
    (?:Suggested\s+solutions\s+or\s+approaches|Solutions\s+ou\s+approches\s+suggérées)
    \s*\*{0,2}\s*:\s*(?:\*{0,2}\s*){0,2}
    (?P<suggestions>.*?)\s*$
    """,
    re.IGNORECASE | re.MULTILINE | re.DOTALL | re.VERBOSE,
)


def parse_response(response):
    match = RESPONSE_PATTERN.search(response)
    if match is None:
        raise ValueError(f"Unexpected response format: {response[:100]!r}")

    return (
        int(match.group("rating")),
        " ".join(match.group("rationale").splitlines()).strip(),
        " ".join(match.group("suggestions").splitlines()).strip(),
    )


def excel_sheet_name(policy, used_names):
    cleaned = re.sub(r"[\\/?*:[\]]", " ", policy).strip() or "Policy"
    name = cleaned[:31]
    suffix = 2
    while name.lower() in used_names:
        marker = f" ({suffix})"
        name = f"{cleaned[:31 - len(marker)]}{marker}"
        suffix += 1
    used_names.add(name.lower())
    return name


def main():
    responses_by_policy = defaultdict(list)

    for agent_fname in sorted(os.listdir(DEFAULT_INPUT)):
        if not agent_fname.startswith("PIAAC"):
            continue

        agent_id = Path(agent_fname).stem
        with open(DEFAULT_INPUT / agent_fname, "r", encoding="utf-8") as fin:
            agent_responses = json.load(fin)

        for item in agent_responses:
            policy = item["policy"]
            try:
                rating, rationale, suggestions = parse_response(item["response"])
            except ValueError as error:
                print(f"Skipping {agent_fname} / {policy}: {error}", file=sys.stderr)
                continue

            responses_by_policy[policy].append(
                {
                    "agent_id": agent_id,
                    "rating": rating,
                    "rationale": rationale,
                    "suggestions": suggestions,
                }
            )

    workbook = Workbook()
    workbook.remove(workbook.active)
    used_names = set()

    for policy, responses in responses_by_policy.items():
        sheet = workbook.create_sheet(excel_sheet_name(policy, used_names))
        sheet.append(["Agent ID", "Rating", "Rationale", "Suggestions"])
        for response in responses:
            sheet.append(
                [
                    response["agent_id"],
                    response["rating"],
                    response["rationale"],
                    response["suggestions"],
                ]
            )

    workbook.save(DEFAULT_OUTPUT)

    response_count = sum(map(len, responses_by_policy.values()))
    print(f"Saved {response_count:,} responses to {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
