import contextlib
import csv
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

CSV_FIELDNAMES = [
    "filename", "filepath", "file_type",
    "camera_make", "camera_model", "lens",
    "iso", "shutter_speed", "aperture", "focal_length",
    "date_shot", "gps_lat", "gps_lon",
    "description", "tags_str", "quality_score", "quality_reasoning",
    "preview_path", "status", "error_message",
]


class OutputWriter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.csv_path = output_dir / "results.csv"
        self.json_path = output_dir / "results.json"
        self.previews_dir = output_dir / "previews"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.previews_dir.mkdir(exist_ok=True)

    def get_previews_dir(self) -> Path:
        return self.previews_dir

    def load_processed_filenames(self) -> set[str]:
        if not self.csv_path.exists():
            return set()
        processed = set()
        try:
            with open(self.csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("status") == "ok" and row.get("filename"):
                        processed.add(row["filename"])
        except Exception as e:
            logger.warning("Could not read %s for resume: %s", self.csv_path.name, e)
        return processed

    def is_already_processed(self, filename: str, processed: set[str]) -> bool:
        return filename in processed

    def append_record(self, record: dict) -> None:
        self._append_csv(record)
        self._append_json(record)

    def _append_csv(self, record: dict) -> None:
        write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(record)

    def _append_json(self, record: dict) -> None:
        data = self._load_json()

        json_record = {k: v for k, v in record.items() if k != "tags_str"}
        data.append(json_record)

        # Write to a sibling file and rename over the target. Writing in place
        # truncates results.json first, so an interrupt mid-write left a partial
        # file that the next run failed to parse and silently restarted from
        # empty -- losing every result recorded so far. os.replace is atomic on
        # the same filesystem, so the file on disk is always a complete list.
        tmp_path = self.json_path.with_name(self.json_path.name + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.json_path)

    def _load_json(self) -> list:
        if not self.json_path.exists():
            return []
        try:
            with open(self.json_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            # Don't silently drop the old results -- keep them next to the new
            # file so nothing is destroyed by a bad parse.
            salvage = self.json_path.with_name(self.json_path.name + ".corrupt")
            logger.warning("Could not parse %s (%s); moving it to %s", self.json_path.name, e, salvage.name)
            with contextlib.suppress(OSError):
                os.replace(self.json_path, salvage)
            return []
        return data if isinstance(data, list) else []
