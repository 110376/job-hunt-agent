from __future__ import annotations

from pathlib import Path
import shutil
import uuid

from app.schemas import JobRecord
from app.services.exporter import export_records


def test_export_records_writes_json_and_csv() -> None:
    local_tmp = Path(".pytest_local_tmp") / f"exporter-{uuid.uuid4().hex}"
    local_tmp.mkdir(parents=True, exist_ok=True)

    records = [
        JobRecord(
            title="机器学习工程师(校招)",
            company="Bar",
            location="上海",
            salary="18k-26k",
            tech_tags=["机器学习", "llm"],
            requirements="熟悉深度学习",
            source="liepin",
            job_url="https://example.com/job/1",
        )
    ]

    json_path, csv_path = export_records(
        records=records,
        output_dir=str(local_tmp),
        role_name="AI Engineer",
    )

    try:
        assert Path(json_path).exists()
        assert Path(csv_path).exists()
    finally:
        shutil.rmtree(local_tmp, ignore_errors=True)


def test_export_records_csv_has_utf8_bom_for_excel() -> None:
    local_tmp = Path(".pytest_local_tmp") / f"exporter-bom-{uuid.uuid4().hex}"
    local_tmp.mkdir(parents=True, exist_ok=True)

    records = [
        JobRecord(
            title="算法工程师（校招）",
            company="测试公司",
            location="上海",
            salary="18k-26k",
            tech_tags=["机器学习", "llm"],
            requirements="熟悉深度学习",
            source="liepin",
            job_url="https://example.com/job/2",
        )
    ]

    _json_path, csv_path = export_records(
        records=records,
        output_dir=str(local_tmp),
        role_name="AI Engineer",
    )

    try:
        raw = Path(csv_path).read_bytes()
        assert raw.startswith(b"\xef\xbb\xbf")
    finally:
        shutil.rmtree(local_tmp, ignore_errors=True)
