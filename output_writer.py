import csv
import json
from pathlib import Path

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
                    if row.get("status") == "ok":
                        processed.add(row["filename"])
        except Exception:
            pass
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
        if self.json_path.exists():
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = []
        else:
            data = []

        json_record = {k: v for k, v in record.items() if k != "tags_str"}
        data.append(json_record)

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
