import csv
import math
from pathlib import Path
from typing import Any

import pandas as pd
import xlsxwriter

DATA_DIR = Path(__file__).parent.parent / "data"

def clean(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def parse_scheme(text):
    """Parse codebook strings such as '1: Male; 2: Female'."""
    result: dict[str, str] = {}
    for item in clean(text).split("; "):
        if ": " not in item:
            continue
        code, label = item.split(": ", 1)
        code = code.strip()
        result[code] = label.strip()
        if code.startswith("."):
            result[code.lower()] = label.strip()
    return result


def number_or_text(value: str):
    value = value.strip()
    if not value:
        return ""
    try:
        if value.lstrip("+-").isdigit():
            return int(value)
        return float(value)
    except ValueError:
        return value


def create_readable_file():
    DATA_FILE = DATA_DIR / "prgcanp2.csv"
    CODEBOOK_FILE = DATA_DIR / "piaac-cy2-international-codebook.xlsx"
    OUTPUT_FILE = DATA_DIR / "piaac_cycle2.xlsx"
    SHEET_NAME = "Data"

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    codebook = pd.read_excel(CODEBOOK_FILE, sheet_name="PUF")
    metadata = {clean(row["Variable"]): row for _, row in codebook.iterrows()}

    value_maps: dict[str, dict[str, str]] = {}
    headers: dict[str, str] = {}
    for variable, row in metadata.items():
        headers[variable] = f"{variable} — {clean(row['Label'])}"
        mapping: dict[str, str] = {}
        mapping.update(parse_scheme(row.get("Value Scheme Detailed")))
        mapping.update(parse_scheme(row.get("Missing Scheme Detailed: SPSS")))
        mapping.update(parse_scheme(row.get("Missing Scheme Detailed: SAS")))
        value_maps[variable] = mapping

    workbook = xlsxwriter.Workbook(
        OUTPUT_FILE,
        {
            "constant_memory": True,
            "strings_to_formulas": False,
            "strings_to_urls": False,
        },
    )
    worksheet = workbook.add_worksheet(SHEET_NAME)
    worksheet.freeze_panes(1, 1)

    with DATA_FILE.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source, delimiter=";")
        variables = next(reader)
        worksheet.write_row(0, 0, [headers.get(v, v) for v in variables])

        maps = [value_maps.get(variable, {}) for variable in variables]
        last_row = 0
        for row_number, row in enumerate(reader, start=1):
            output_row: list[int | float | str] = []
            for value, mapping in zip(row, maps):
                label = mapping.get(value)
                if label is None and value.startswith("."):
                    label = mapping.get(value.lower())
                output_row.append(f"{value} — {label}" if label else number_or_text(value))
            worksheet.write_row(row_number, 0, output_row)
            last_row = row_number

    worksheet.autofilter(0, 0, last_row, len(variables) - 1)
    workbook.close()


def create_weighted_sample():
    DATA_FILE = DATA_DIR / "piaac_cycle2.xlsx"
    df = pd.read_excel(DATA_FILE)
    full_sample = df.sample(n=3200, replace=True, weights="SPFWT0 — Final full sample weight", random_state=42)
    full_sample.to_excel(DATA_DIR / "piaac_cycle2_weighted_sample.xlsx", index=False)

create_readable_file()
create_weighted_sample()