import argparse
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
import openai
import os
from tqdm import tqdm

from exif_reader import extract_exif
from output_writer import OutputWriter
from preview_generator import PreviewGenerationError, generate_preview
from vision_analyzer import VisionAnalysisError, analyze_photo

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".tif", ".tiff", ".rw2", ".arw", ".cr3", ".nef"}

RAW_EXTENSIONS = {".rw2", ".arw", ".cr3", ".nef"}
TIFF_EXTENSIONS = {".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="photo-ai-toolkit: analyze photos with AI and extract EXIF metadata"
    )
    parser.add_argument("--input", required=True, help="Path to folder with photos")
    parser.add_argument("--output", required=True, help="Path to output folder for results")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess all photos, ignoring previously processed files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pipeline without making OpenAI API calls (for testing)",
    )
    return parser.parse_args()


def classify_file_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in RAW_EXTENSIONS:
        return "RAW"
    if ext in TIFF_EXTENSIONS:
        return "TIFF"
    return "JPEG"


def discover_photos(input_dir: Path) -> list[Path]:
    photos = []
    for p in sorted(input_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            photos.append(p)
    return photos


def process_photo(
    photo_path: Path,
    writer: OutputWriter,
    processed_filenames: set[str],
    client: openai.OpenAI,
    dry_run: bool,
) -> dict:
    filename = photo_path.name
    file_type = classify_file_type(photo_path)

    if writer.is_already_processed(filename, processed_filenames):
        logger = logging.getLogger(__name__)
        logger.info("Skipping already processed: %s", filename)
        return {
            "filename": filename,
            "filepath": str(photo_path.resolve()),
            "file_type": file_type,
            "status": "skipped",
            "error_message": None,
        }

    logger = logging.getLogger(__name__)
    logger.info("Processing: %s", filename)

    record = {
        "filename": filename,
        "filepath": str(photo_path.resolve()),
        "file_type": file_type,
        "camera_make": None,
        "camera_model": None,
        "lens": None,
        "iso": None,
        "shutter_speed": None,
        "aperture": None,
        "focal_length": None,
        "date_shot": None,
        "gps_lat": None,
        "gps_lon": None,
        "description": None,
        "tags": [],
        "tags_str": None,
        "quality_score": None,
        "quality_reasoning": None,
        "preview_path": None,
        "status": "error",
        "error_message": None,
    }

    exif = extract_exif(photo_path, file_type)
    record.update(exif)

    try:
        preview_path = generate_preview(
            photo_path, file_type, writer.get_previews_dir()
        )
        record["preview_path"] = str(preview_path)
    except PreviewGenerationError as e:
        record["error_message"] = f"Preview error: {e}"
        logger.error("Preview failed for %s: %s", filename, e)
        writer.append_record(record)
        return record

    try:
        analysis = analyze_photo(preview_path, client, dry_run=dry_run)
        record["description"] = analysis["description"]
        record["tags"] = analysis["tags"]
        record["tags_str"] = "; ".join(analysis["tags"])
        record["quality_score"] = analysis["quality_score"]
        record["quality_reasoning"] = analysis["quality_reasoning"]
        record["status"] = "ok"
    except VisionAnalysisError as e:
        record["error_message"] = f"Vision API error: {e}"
        logger.error("Vision analysis failed for %s: %s", filename, e)
        writer.append_record(record)
        return record

    writer.append_record(record)

    if not dry_run:
        time.sleep(1)

    return record


def print_summary(records: list[dict]) -> None:
    ok = [r for r in records if r.get("status") == "ok"]
    skipped = [r for r in records if r.get("status") == "skipped"]
    errors = [r for r in records if r.get("status") == "error"]

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total processed : {len(ok)}")
    print(f"Skipped         : {len(skipped)}")
    print(f"Errors          : {len(errors)}")

    if ok:
        scores = [r["quality_score"] for r in ok if r.get("quality_score") is not None]
        if scores:
            avg = sum(scores) / len(scores)
            print(f"Average quality : {avg:.1f}/1000")

        top8 = sorted(ok, key=lambda r: r.get("quality_score") or 0, reverse=True)[:8]
        print("\nTop-8 photos by quality:")
        for i, r in enumerate(top8, 1):
            score = r.get("quality_score", "?")
            desc = (r.get("description") or "")[:60]
            print(f"  {i}. [{score}/1000] {r['filename']}")
            if desc:
                print(f"       {desc}")

        stories = sorted(
            [r for r in ok if (r.get("quality_score") or 0) >= 700],
            key=lambda r: r.get("quality_score") or 0,
            reverse=True,
        )
        print("\nRECOMMENDED FOR STORIES (score 700+):")
        if stories:
            for r in stories:
                print(f"  [{r['quality_score']}/1000] {r['filename']}")
        else:
            print("  None met the threshold.")
    else:
        print("No photos were successfully processed.")

    print("=" * 50)


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "processing.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    setup_logging(output_dir)
    logger = logging.getLogger(__name__)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        print("Error: OPENAI_API_KEY not found in environment or .env file", file=sys.stderr)
        sys.exit(1)

    client = openai.OpenAI(api_key=api_key or "dry-run-key")

    photos = discover_photos(input_dir)
    if not photos:
        print(f"No supported photo files found in: {input_dir}")
        sys.exit(0)

    logger.info("Found %d photo(s) in %s", len(photos), input_dir)
    if args.dry_run:
        logger.info("DRY RUN mode: no API calls will be made")

    writer = OutputWriter(output_dir)
    processed_filenames = set() if args.force else writer.load_processed_filenames()

    if processed_filenames:
        logger.info("Loaded %d previously processed filename(s) for skip check", len(processed_filenames))

    records = []
    with tqdm(photos, desc="Analyzing photos", unit="photo") as pbar:
        for photo_path in pbar:
            pbar.set_postfix(file=photo_path.name[:30])
            try:
                record = process_photo(
                    photo_path, writer, processed_filenames, client, dry_run=args.dry_run
                )
                records.append(record)
            except Exception as e:
                logger.error("Unexpected error processing %s: %s", photo_path.name, e, exc_info=True)
                records.append({
                    "filename": photo_path.name,
                    "filepath": str(photo_path.resolve()),
                    "file_type": classify_file_type(photo_path),
                    "status": "error",
                    "error_message": f"Unexpected: {e}",
                })

    print_summary(records)
    logger.info("Results saved to: %s", output_dir)


if __name__ == "__main__":
    main()
