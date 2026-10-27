"""Run a reproducible LLooM analysis over the policy-response workbook.

The safe default is an inventory-only run.  API-backed concept discovery or
scoring only starts when ``--mode discover`` or ``--mode full`` is supplied
together with ``--yes``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd


TEXT_COLUMNS = ("Rationale", "Suggestions")
REQUIRED_COLUMNS = ("Agent ID", "Rating", *TEXT_COLUMNS)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def slugify(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return value or "policy"


def clean_policy_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    cleaned = df.loc[:, REQUIRED_COLUMNS].copy()
    cleaned["Agent ID"] = cleaned["Agent ID"].astype("string")
    cleaned["Rating"] = pd.to_numeric(cleaned["Rating"], errors="coerce")
    for column in TEXT_COLUMNS:
        cleaned[column] = cleaned[column].astype("string").str.strip()
        cleaned.loc[cleaned[column].isin(["", "<NA>"]), column] = pd.NA
    return cleaned


def stratified_sample(
    df: pd.DataFrame, text_column: str, sample_size: int, random_state: int
) -> pd.DataFrame:
    usable = df.dropna(subset=["Agent ID", text_column]).copy()
    if len(usable) <= sample_size:
        return usable.reset_index(drop=True)

    # Preserve the rating distribution while ensuring small rating groups are
    # represented.  The second sample makes the final size exact.
    pieces: list[pd.DataFrame] = []
    for rating, group in usable.groupby("Rating", dropna=False):
        target = max(1, round(sample_size * len(group) / len(usable)))
        pieces.append(group.sample(min(target, len(group)), random_state=random_state))
    sampled = pd.concat(pieces).drop_duplicates(subset=["Agent ID"])
    if len(sampled) > sample_size:
        sampled = sampled.sample(sample_size, random_state=random_state)
    elif len(sampled) < sample_size:
        remaining = usable.loc[~usable.index.isin(sampled.index)]
        sampled = pd.concat(
            [
                sampled,
                remaining.sample(
                    min(sample_size - len(sampled), len(remaining)),
                    random_state=random_state,
                ),
            ]
        )
    return sampled.reset_index(drop=True)


def workbook_inventory(path: Path) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    cleaned: dict[str, pd.DataFrame] = {}
    inventory: list[dict[str, Any]] = []
    for sheet_name, raw_df in sheets.items():
        df = clean_policy_frame(raw_df)
        cleaned[sheet_name] = df
        inventory.append(
            {
                "policy": sheet_name,
                "rows": len(df),
                "unique_agent_ids": int(df["Agent ID"].nunique()),
                "ratings_present": sorted(
                    int(value) for value in df["Rating"].dropna().unique()
                ),
                "rationales_present": int(df["Rationale"].notna().sum()),
                "suggestions_present": int(df["Suggestions"].notna().sum()),
            }
        )
    return cleaned, inventory


def concept_records(session: Any, policy: str, field: str) -> list[dict[str, Any]]:
    records = []
    for concept_id, concept in session.concepts.items():
        record = concept.to_dict()
        record.update({"policy": policy, "field": field, "concept_id": concept_id})
        records.append(record)
    return records


async def analyze_field(
    *,
    policy: str,
    field: str,
    full_df: pd.DataFrame,
    output_dir: Path,
    mode: str,
    sample_size: int,
    max_concepts: int,
    batch_size: int,
    random_state: int,
) -> dict[str, Any]:
    import text_lloom.workbench as wb

    sampled = stratified_sample(full_df, field, sample_size, random_state)
    lloom_df = sampled.loc[:, ["Agent ID", "Rating", field]].copy()
    session = wb.lloom(df=lloom_df, text_col=field, id_col="Agent ID")

    seed = (
        "reasons the respondent believes the policy will or will not improve "
        "AI literacy and adoption"
        if field == "Rationale"
        else "concrete recommendations for improving the policy"
    )
    await session.gen(seed=seed, debug=False)
    await session.select_auto(max_concepts=max_concepts)

    run_dir = output_dir / slugify(policy) / field.lower()
    run_dir.mkdir(parents=True, exist_ok=True)
    concepts = concept_records(session, policy, field)
    (run_dir / "concepts.json").write_text(
        json.dumps(concepts, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(concepts).to_csv(run_dir / "concepts.csv", index=False)
    sampled.to_csv(run_dir / "induction_sample.csv", index=False)
    session.save(folder=str(run_dir), file_name="lloom_session")

    result: dict[str, Any] = {
        "policy": policy,
        "field": field,
        "induction_documents": len(sampled),
        "concepts_generated": len(concepts),
        "mode": mode,
    }
    if mode == "full":
        scoring_df = full_df.dropna(subset=["Agent ID", field]).loc[
            :, ["Agent ID", "Rating", field]
        ]
        scores = await session.score(
            df=scoring_df,
            batch_size=batch_size,
            get_highlights=True,
            debug=False,
        )
        scores.insert(0, "policy", policy)
        scores.insert(1, "field", field)
        scores.to_csv(run_dir / "document_concept_scores.csv", index=False)
        session.export_df().to_csv(run_dir / "concept_summary.csv", index=False)
        result["scored_documents"] = len(scoring_df)
        result["score_rows"] = len(scores)
    return result


async def run(args: argparse.Namespace) -> None:
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    sheets, inventory = workbook_inventory(input_path)

    selected_names = list(sheets)
    if args.sheets:
        requested = {name.strip() for name in args.sheets.split(",") if name.strip()}
        missing = requested.difference(sheets)
        if missing:
            raise ValueError(f"Unknown sheet(s): {', '.join(sorted(missing))}")
        selected_names = [name for name in sheets if name in requested]

    selected_inventory = [row for row in inventory if row["policy"] in selected_names]
    print(pd.DataFrame(selected_inventory).to_string(index=False))
    if args.mode == "inventory":
        return

    if not args.yes:
        raise SystemExit(
            "LLM calls were not started. Re-run with --yes after reviewing the inventory."
        )
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for the default LLooM models.")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "inventory.json").write_text(
        json.dumps(selected_inventory, indent=2), encoding="utf-8"
    )
    manifest = {
        "input": str(input_path),
        "mode": args.mode,
        "sample_size": args.sample_size,
        "max_concepts": args.max_concepts,
        "batch_size": args.batch_size,
        "random_state": args.random_state,
        "analyses": [],
    }
    for policy in selected_names:
        for field in TEXT_COLUMNS:
            print(f"\nAnalyzing {policy!r} / {field} ...", flush=True)
            result = await analyze_field(
                policy=policy,
                field=field,
                full_df=sheets[policy],
                output_dir=output_dir,
                mode=args.mode,
                sample_size=args.sample_size,
                max_concepts=args.max_concepts,
                batch_size=args.batch_size,
                random_state=args.random_state,
            )
            manifest["analyses"].append(result)
            (output_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Induce and score LLooM concepts for each policy sheet."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "parsed_responses.xlsx",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "lloom_analysis",
    )
    parser.add_argument(
        "--mode", choices=("inventory", "discover", "full"), default="inventory"
    )
    parser.add_argument(
        "--sheets", help="Comma-separated exact sheet names; defaults to all sheets."
    )
    parser.add_argument("--sample-size", type=int, default=250)
    parser.add_argument("--max-concepts", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--yes", action="store_true", help="Required to authorize API-backed work."
    )
    args = parser.parse_args()
    if args.sample_size < 1 or args.max_concepts < 1 or args.batch_size < 1:
        parser.error("sample-size, max-concepts, and batch-size must be positive")
    return args


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
