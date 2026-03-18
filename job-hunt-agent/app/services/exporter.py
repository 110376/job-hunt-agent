from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path

from app.schemas import JobRecord


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^\w\-\u4e00-\u9fff]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-") or "jobs"


def export_records(records: list[JobRecord], output_dir: str, role_name: str) -> tuple[str, str]:
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix = f"{_slugify(role_name)}-{now}"

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    json_path = output / f"{prefix}.json"
    csv_path = output / f"{prefix}.csv"

    payload = [record.model_dump() for record in records]
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "title",
        "company",
        "location",
        "salary",
        "tech_tags",
        "requirements",
        "source",
        "job_url",
    ]
    # Use UTF-8 BOM so Windows Excel can detect UTF-8 correctly on direct open.
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in payload:
            row = dict(record)
            row["tech_tags"] = ", ".join(row.get("tech_tags", []))
            writer.writerow(row)

    return str(json_path), str(csv_path)
