"""The command line. Subcommands, batch actions, and a hard rule about deleting.

    analyze      measure, score, route, write reports and a plan
    report       rebuild reports from a stored run (filter, sort, localise)
    reclassify   redo routing with different thresholds -- no re-analysis
    quarantine   carry out a plan (requires --apply; dry run otherwise)
    restore      undo a quarantine operation
    purge        permanently delete, behind four separate gates
    export       build a marketplace-ready package
    override     record a manual decision that future runs must respect
    profiles     list the built-in calibration profiles

Two conventions run through all of it. Anything that touches the filesystem is a
dry run unless `--apply` is passed, and prints what it would do instead. And
`purge` is the only operation that removes data; it needs a typed phrase, a
minimum age, an unlocked directory, and a manifest entry per file.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import bootstrap
import i18n
import layout
import overrides as overrides_module
import pipeline
import preflight
import quarantine as quarantine_module
import reports
import stock_metadata
import workspace
from calibration import BUILTIN_PROFILES, CalibrationProfile, resolve
from reports import AssetRecord
from scoring import RouteClass

SORT_KEYS = {
    "score": lambda r: r.scores.get("routing_score", 0),
    "potential": lambda r: r.scores.get("post_edit_potential", 0),
    "current": lambda r: r.scores.get("current_quality", 0),
    "stock": lambda r: r.scores.get("stock_potential", 0),
    "portfolio": lambda r: r.scores.get("portfolio_potential", 0),
    "confidence": lambda r: r.confidence,
    "gain": lambda r: r.expected_gain,
    "filename": lambda r: r.filename,
}


# Exit codes, named so a script can tell the two apart: a configuration that
# was never going to work, and an analysis that started and then broke.
PREFLIGHT_EXIT = 2
SEMANTIC_FAILED_EXIT = 3


def setup_logging(log_dir: Path, verbose: bool = False) -> None:
    """File plus stdout, with a redacting filter on every handler.

    The filter is not optional. An API key in a traceback ends up in
    processing.log, which is the file a user attaches to a bug report.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.FileHandler(log_dir / "processing.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
    redactor = reports.RedactingFilter()
    for handler in handlers:
        handler.addFilter(redactor)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


# --- analyze ----------------------------------------------------------------


def cmd_analyze(args: argparse.Namespace) -> int:
    """The product command: full local + Stage 2 + Stage 3, or nothing.

    The order below is the fix for a specific failure. A run once discovered
    299 assets, checksummed and decoded every one of them, generated previews,
    measured them, migrated the output directory to a new layout -- and then
    made its first API call and learned the configured model was not available
    to the key. No report, the previous report already moved, and every minute
    of it spent on work that could not be used.

    So nothing here touches a photograph or the output directory until the
    model has answered a request with an image in it.
    """
    return _analyze(args, semantic=True)


def cmd_measure(args: argparse.Namespace) -> int:
    """Local measurement only. A developer tool, and labelled as one.

    Kept because the deterministic half of this pipeline is genuinely useful on
    its own for debugging, and deleting it would make the technical filters
    untestable against a real archive without spending money. It is not
    `analyze`, it does not claim to be a photo analysis, and its output says so.
    """
    return _analyze(args, semantic=False)


def _analyze(args: argparse.Namespace, *, semantic: bool) -> int:
    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    if not input_dir.is_dir():
        print(f"Error: input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    language = i18n.normalise(args.lang)

    # --- nothing above this line has touched a file, and nothing below it runs
    # --- until the model has proved it works.
    if semantic:
        model = bootstrap.resolve_model(args.model)
        result = preflight.run(model)
        print(preflight.format_result(result), flush=True)
        if not result.ok:
            print(preflight.format_failure(result), file=sys.stderr)
            return PREFLIGHT_EXIT
        print(flush=True)
    else:
        model = None
        print(
            "Local measurement only. This is not a photo analysis: no content "
            "check, no artistic read, and no photograph can be ranked as a top "
            "one. Use `analyze` for that.",
            flush=True,
        )

    space = workspace.Workspace(output_dir)
    moved = workspace.migrate(output_dir)
    space.create()
    setup_logging(space.internal, args.verbose)
    if moved:
        print(f"Tidied {len(moved)} item(s) from an earlier run into {workspace.INTERNAL}/")

    options = pipeline.PipelineOptions(
        input_dir=input_dir,
        output_dir=output_dir,
        quarantine_dir=Path(args.quarantine).resolve() if args.quarantine else None,
        language=language,
        profile_name=args.profile,
        profile_path=Path(args.profile_file).resolve() if args.profile_file else None,
        semantic=semantic,
        stage3=semantic and not getattr(args, "no_stage3", False),
        semantic_model=model,
        include_video=not args.no_video,
        video_samples=args.video_samples,
        force=args.force,
        limit=args.limit,
        copyright_holder=args.copyright or "",
        darkroom=args.darkroom,
        darkroom_renderer=args.renderer,
        shadow_mode=not args.no_shadow_mode,
        insights_scope=getattr(args, "insights_scope", "new"),
    )

    printed = {"n": 0, "analysed": 0, "reused": 0, "announced": False}

    def progress(filename: str, index: int, total: int, reused: bool = False) -> None:
        # Progressive, and deliberately plain: a progress bar hides the name of
        # the file that is about to crash the run.
        #
        # Reused assets are counted, not listed. Printing 47 filenames under a
        # heading that says "analyzing" is how an incremental run came to look
        # exactly like a full one.
        printed["n"] = index
        if reused:
            printed["reused"] += 1
            return
        printed["analysed"] += 1
        if not printed["announced"]:
            printed["announced"] = True
            print("\nAnalyzing new assets:", flush=True)
        print(f"  [{printed['analysed']:>4}] {filename}", flush=True)

    if semantic and not args.yes:
        stop = _confirm_cost(input_dir, model, options)
        if stop is not None:
            return stop

    print(f"Analyzing {input_dir}")
    try:
        result = pipeline.run(options, progress=progress)
    except bootstrap.SemanticCredentialsMissing:
        print(i18n.t("creds.error", language), file=sys.stderr)
        return PREFLIGHT_EXIT
    except bootstrap.SemanticUnavailable as e:
        # The preflight passed and the API failed anyway -- a revoked key
        # mid-run, a network drop. The previous report stays exactly as it was.
        print(f"\nThe analysis stopped: {reports.redact(str(e))}", file=sys.stderr)
        print("The previous report was left untouched.", file=sys.stderr)
        return SEMANTIC_FAILED_EXIT
    if result.cancelled:
        print("Run cancelled; partial results follow.")

    store = overrides_module.OverrideStore(space.internal / overrides_module.OVERRIDES_NAME)
    applied = overrides_module.apply_to(result.records, store)
    if applied:
        print(f"Applied {applied} manual override(s).")
        # The summary is built during the run, before overrides exist. Printing
        # it unchanged reported the tool's own conclusion as the outcome and
        # silently contradicted the class stored against each asset.
        result.summary = reports.summarise(result.records)
        result.planned_operations = [
            op
            for op in result.planned_operations
            if _still_trash(op.asset_id, result.records)
        ]
        overrides_module.resolve_observations(
            result.records, store, space.internal / "model_monitoring.json"
        )

    try:
        _write_outputs(
            result, space, language,
            insights_scope=getattr(args, "insights_scope", "new"),
        )
    except workspace.PublishError as e:
        print(f"\nThe run did not produce a complete result: {e}", file=sys.stderr)
        print("The previous report was left untouched.", file=sys.stderr)
        return SEMANTIC_FAILED_EXIT

    print(reports.format_summary(result.summary, language, expert=args.expert))
    _print_run_tally(result, space)

    if result.planned_operations:
        print()
        print(quarantine_module.summarise_plan(result.planned_operations, language))
    return 0


def _still_trash(asset_id: str, records: list[AssetRecord]) -> bool:
    """A user who rescued a file from trash must not still see it in the plan."""
    record = next((r for r in records if r.asset_id == asset_id), None)
    return record is not None and record.route_class == RouteClass.TRASH.value


def _write_outputs(
    result: pipeline.RunResult,
    space: workspace.Workspace,
    language: str,
    *,
    insights_scope: str = "new",
) -> None:
    """Build the run in staging, validate it, then swap it in.

    Nothing a photographer opens is touched until the whole run is on disk and
    complete. A failure half-way leaves the previous report, the previous
    category folders and the previous recipes exactly as they were, which is the
    difference between a run that failed and a run that cost you the last good
    one.
    """
    import batches
    import insights as insights_module
    import recipe_export
    import simple_report

    space.create()
    reports_dir = space.reports
    staged = workspace.staging_dir(space, result.run_id or "run")

    recipes = recipe_export.export_all(
        result.records, result.measurements, staged / workspace.RECIPES, language=language
    )
    counts = workspace.build_category_farm(
        result.records, workspace.Workspace(staged)
    )

    simple_report.write(
        result.records,
        staged / workspace.REPORT_NAME,
        language=language,
        insights_link=workspace.INSIGHTS_NAME,
    )

    manifest = batches.latest(space.internal / batches.MANIFEST_NAME)
    scoped, actual_scope = batches.scope_records(result.records, manifest, insights_scope)
    collection = insights_module.build(scoped, result.measurements)
    insights_module.write(
        collection,
        staged / workspace.INSIGHTS_NAME,
        language=language,
        report_link=workspace.REPORT_NAME,
        scope=actual_scope,
        total_stored=len(result.records),
    )

    published = workspace.publish(space, staged)

    # Everything below is diagnostics, written where it has always been written.
    # It is not part of the transaction: `.internal/` is not what a failed run
    # would damage, and holding the report back on a CSV error would be worse
    # than writing an imperfect one.
    reports.write_json(result.records, reports_dir / "analysis.json", summary=result.summary)
    reports.write_csv(result.records, reports_dir / "analysis.csv")
    reports.write_html(
        result.records, reports_dir / "full_report.html", summary=result.summary, language=language
    )
    reports.write_json_atomic(collection.to_dict(), reports_dir / "insights.json")

    layout.write_record_manifest(result.records, reports_dir)
    layout.build_class_farm(result.records, space.routing_views)
    trash = layout.write_record_delete_candidates(result.records, reports_dir)

    previews = [
        (r.filename, Path(r.preview_path))
        for r in result.records
        if r.route_class == RouteClass.TRASH.value and r.preview_path and Path(r.preview_path).exists()
    ]
    layout.build_contact_sheet(previews, reports_dir / "contact_sheet_delete.jpg")
    layout.build_comparison_sheet(
        _comparison_rows(result.records, language),
        reports_dir / "contact_sheet_duplicates.jpg",
        language=language,
        semantic_ran=result.semantic_completed,
    )

    print()
    print(f"  {space.report}")
    print(f"  {space.insights}")
    print("  " + ", ".join(f"{name}/ {n}" for name, n in _folder_counts(counts).items()))
    if recipes["written"]:
        print(f"  {workspace.RECIPES}/ {len(recipes['written'])} edit recipe(s)")
    if trash["count"]:
        print(f"  {trash['count']} file(s) proposed for removal (nothing has been moved)")
    logging.getLogger(__name__).debug("Published %s", ", ".join(published))


# Above this many photographs the run stops and asks, because the difference
# between 20 files and 300 is the difference between a few cents and a few
# dollars -- and the number nobody sees before it is spent is the one that
# causes the argument afterwards.
CONFIRM_ABOVE = 60


def _confirm_cost(input_dir: Path, model: str, options) -> int | None:
    """Count what is new, price it, and ask. Returns an exit code to stop, or None.

    Counting happens with checksums only: no photograph is decoded to produce
    this estimate, so saying no costs nothing but the file walk.
    """
    import batches
    import media

    assets = media.discover(
        input_dir,
        follow_symlinks=options.follow_symlinks,
        exclude=media.excluded_roots(options.output_dir, options.resolved_quarantine()),
    )
    if not options.include_video:
        assets = [a for a in assets if a.kind is not media.MediaKind.VIDEO]
    if options.limit:
        assets = assets[: options.limit]

    cache = pipeline.AnalysisCache(options.internal / pipeline.CACHE_NAME)
    history = batches.read_all(options.internal / batches.MANIFEST_NAME)
    fresh, modified, reused = batches.classify_assets(
        assets, cache, model, seen_keys=batches.seen_keys(history)
    )
    billable = len(fresh) + len(modified)

    print(f"\nAssets discovered: {len(assets)}")
    print(f"  New:      {len(fresh)}")
    if modified:
        print(f"  Modified: {len(modified)}")
    print(f"  Reused:   {len(reused)}  (no API calls, no charge)")

    if billable == 0:
        print("\nNothing new to analyse.")
        return None
    if billable < CONFIRM_ABOVE:
        return None

    estimate = bootstrap.estimate_cost(model, billable)
    print(
        f"\n{billable} photographs will be sent to {model}, costing roughly "
        f"${estimate:.2f}. This is an estimate, not a quote."
    )
    try:
        answer = input("Continue? [y/N] ").strip().lower()
    except EOFError:
        answer = ""
    if answer not in ("y", "yes"):
        print("Stopped. Nothing was sent and nothing was changed.")
        return 0
    return None


def _print_run_tally(result: pipeline.RunResult, space: workspace.Workspace) -> None:
    """What the run actually did, in the terms it cost.

    Separate from the collection summary on purpose: one describes the
    photographs, this describes the work. A person who has just waited fifteen
    minutes is owed both.
    """
    calls = result.llm_calls or {}
    total_calls = sum(calls.values())
    lines = [
        "",
        "Run",
        f"  New assets analysed:      {len(result.new_assets)}",
        f"  Modified re-analysed:     {len(result.modified_assets)}",
        f"  Unchanged reused:         {len(result.reused_assets)}",
    ]
    if result.semantic_requested:
        lines += [
            f"  LLM calls:                {total_calls}"
            + (f"  ({_call_breakdown(calls)})" if total_calls else ""),
            f"  Content check completed:  {sum(1 for r in result.records if r.semantic_present)}",
            f"  Artistic read completed:  {result.stage3_completed}",
        ]
        if result.stage3_failed:
            lines.append(f"  Artistic read failed:     {result.stage3_failed}")
    failures = sum(1 for r in result.records if r.status != "ok")
    if failures:
        lines.append(f"  Failed to analyse:        {failures}")
    lines.append(f"  Report:                   {space.report}")
    print("\n".join(lines))


def _call_breakdown(calls: dict) -> str:
    return ", ".join(f"{name} {count}" for name, count in sorted(calls.items()))


def _folder_counts(counts: dict[str, int]) -> dict[str, int]:
    return {workspace.CATEGORY_DIRS[k]: v for k, v in counts.items()}


# --- report -----------------------------------------------------------------


def _comparison_rows(records: list[AssetRecord], language: str) -> list[dict]:
    """Pair every duplicate candidate with the frame that beat it."""
    best_by_cluster = {r.cluster_id: r for r in records if r.best_in_cluster}

    rows: list[dict] = []
    for record in records:
        if record.route_class not in (
            RouteClass.DUPLICATE_CANDIDATE.value,
            RouteClass.TRASH.value,
        ):
            continue
        if not record.preview_path:
            continue
        best = best_by_cluster.get(record.cluster_id)
        rows.append(
            {
                "label": record.filename,
                "candidate_preview": record.preview_path,
                "best_label": best.filename if best and best is not record else "",
                "best_preview": best.preview_path if best and best is not record else "",
                "margin": f"{record.cluster_margin:g}",
                "reason": _localise_first_reason(record, language),
            }
        )
    return rows


def _localise_first_reason(record: AssetRecord, language: str) -> str:
    from scoring import Reason

    for payload in record.reason_keys or []:
        return Reason(
            payload.get("key", ""), payload.get("params") or {}, payload.get("text", "")
        ).localise(language)
    return record.reasons[0] if record.reasons else ""


def _load_records(path: Path) -> list[AssetRecord]:
    rows, _ = reports.read_json(path)
    known = set(AssetRecord.__dataclass_fields__)
    return [AssetRecord(**{k: v for k, v in row.items() if k in known}) for row in rows]


def _filter(records: list[AssetRecord], args: argparse.Namespace) -> list[AssetRecord]:
    out = records
    if args.media:
        out = [r for r in out if r.media_type == args.media]
    if args.route_class:
        wanted = set(args.route_class)
        out = [r for r in out if r.route_class in wanted]
    if args.route:
        out = [r for r in out if r.route == args.route]
    if args.genre:
        out = [r for r in out if r.genre == args.genre]
    if args.marketplace:
        out = [
            r
            for r in out
            if any(m.get("platform_id") == args.marketplace and m.get("eligible") for m in r.marketplaces)
        ]
    if args.min_score is not None:
        out = [r for r in out if r.scores.get("routing_score", 0) >= args.min_score]
    if args.min_potential is not None:
        out = [r for r in out if r.scores.get("post_edit_potential", 0) >= args.min_potential]
    if args.min_confidence is not None:
        out = [r for r in out if r.confidence >= args.min_confidence]
    if args.needs_release:
        out = [r for r in out if "needs_model_release" in r.tags]
    if args.cluster:
        out = [r for r in out if r.cluster_id == args.cluster]
    if args.duplicates_only:
        out = [r for r in out if r.cluster_size > 1]
    return out


def cmd_report(args: argparse.Namespace) -> int:
    analysis = Path(args.analysis).resolve()
    if not analysis.exists():
        print(f"Error: no analysis at {analysis}", file=sys.stderr)
        return 1

    language = i18n.normalise(args.lang)
    records = _filter(_load_records(analysis), args)
    key = SORT_KEYS.get(args.sort, SORT_KEYS["score"])
    records.sort(key=key, reverse=args.sort != "filename")

    summary = reports.summarise(records)
    if args.format in ("table", "all"):
        _print_table(records, language, args.limit)
        print(reports.format_summary(summary, language))
    out_dir = analysis.parent
    if args.format in ("json", "all"):
        print(f"JSON: {reports.write_json(records, out_dir / 'filtered.json', summary=summary)}")
    if args.format in ("csv", "all"):
        print(f"CSV:  {reports.write_csv(records, out_dir / 'filtered.csv')}")
    if args.format in ("html", "all"):
        print(f"HTML: {reports.write_html(records, out_dir / 'filtered.html', summary=summary, language=language)}")
    return 0


def _print_table(records: list[AssetRecord], language: str, limit: int | None) -> None:
    header = (
        f"{'file':<28}{'class':<16}{'now':>5}{'pot':>5}{'gain':>6}"
        f"{'stock':>7}{'port':>6}{'conf':>6}  problems"
    )
    print()
    print(header)
    print("-" * len(header))
    for record in records[: limit or len(records)]:
        scores = record.scores
        blockers = record.issues.get("unrecoverable") or []
        partial = record.issues.get("partially_fixable") or []
        note = blockers[0] if blockers else (partial[0] if partial else "")
        print(
            f"{record.filename[:27]:<28}"
            f"{i18n.t(f'class.{record.route_class}', language)[:15]:<16}"
            f"{scores.get('current_quality', 0):>5}"
            f"{scores.get('post_edit_potential', 0):>5}"
            f"{record.expected_gain:>+6}"
            f"{scores.get('stock_potential', 0):>7}"
            f"{scores.get('portfolio_potential', 0):>6}"
            f"{record.confidence:>6}  {note[:40]}"
        )


# --- reclassify -------------------------------------------------------------


def cmd_reclassify(args: argparse.Namespace) -> int:
    analysis = Path(args.analysis).resolve()
    calibration = resolve(args.profile, Path(args.profile_file) if args.profile_file else None)
    changes = pipeline.reclassify(analysis, calibration)
    moved = [c for c in changes if c["changed"]]

    print(f"Re-routed {len(changes)} asset(s) using {calibration.fingerprint}")
    print(f"{len(moved)} changed class:\n")
    for change in moved[: args.limit or 50]:
        print(f"  {change['filename']:<30} {change['previous_class']:>16} -> {change['route_class']}")
    if not moved:
        print("  (no class changed)")
    print("\nNo pixels were decoded and no tokens were spent.")
    return 0


# --- quarantine / restore / purge -------------------------------------------


def cmd_quarantine(args: argparse.Namespace) -> int:
    analysis = Path(args.analysis).resolve()
    records = _load_records(analysis)
    input_root = Path(args.input).resolve() if args.input else None
    quarantine_dir = Path(args.quarantine).resolve()

    quarantine = quarantine_module.Quarantine(
        quarantine_dir, source_roots=[input_root] if input_root else []
    )
    moves = []
    for record in records:
        if record.route_class != RouteClass.TRASH.value:
            continue
        if record.proposed_action == "excluded_by_user":
            continue
        # The exact group recorded at analysis time, never a glob. Globbing
        # `stem.*` re-derives the group from whatever is on disk *now*: it
        # sweeps in unrelated files that happen to share a stem -- an edited
        # `P1042675.tif`, a `P1042675.mp4` from a different shoot -- and misses
        # any sidecar named differently. Both directions move files nobody
        # decided anything about.
        files = [Path(p) for p in (record.all_files or [record.source_path])]
        moves.append(
            quarantine_module.PlannedMove(
                asset_id=record.asset_id,
                files=[p for p in files if p.exists()],
                destination_dir=quarantine_dir,
                reason="; ".join(record.reasons[:1]) or "below every threshold",
                route_class=record.route_class,
                scores=record.scores,
                states=record.file_states,
                evidence=record.evidence,
            )
        )

    planned = quarantine.plan(moves)
    if not args.apply:
        print(quarantine_module.summarise_plan(planned, i18n.normalise(args.lang)))
        return 0

    results = quarantine.apply(planned, dry_run=False)
    moved = sum(1 for r in results if r.status == "moved")
    skipped = sum(1 for r in results if r.status == "skipped")
    failed = [r for r in results if r.status == "failed"]
    print(f"Moved {moved}, skipped {skipped}, failed {len(failed)}.")
    for failure in failed:
        print(f"  FAILED {failure.source}: {failure.error}")
    print(f"Manifest: {quarantine.manifest.path}")
    print("Restore anything with:  python cli.py restore --quarantine <dir> --apply")
    return 1 if failed else 0


def cmd_darkroom(args: argparse.Namespace) -> int:
    """Render edit suggestions for a stored run. Writes only under suggestions/."""
    import darkroom

    analysis = Path(args.analysis).resolve()
    records = _load_records(analysis)
    out_dir = Path(args.output).resolve() if args.output else analysis.parent.parent

    shown = 0
    for record in records:
        if not record.edit_recipes:
            continue
        print(darkroom.format_report(record))
        shown += 1
        if args.limit and shown >= args.limit:
            break
    if not shown:
        print(
            "No edit recipes in this run. Re-run `analyze --darkroom` to generate them "
            "(about a second per frame)."
        )
    print(f"\nSuggestions live under {out_dir / 'suggestions'}; no original was touched.")
    return 0


def cmd_apply_recipe(args: argparse.Namespace) -> int:
    """Write a recipe beside the RAW. Dry run by default; refuses to clobber."""
    import media
    from edit_schema import read_recipe
    from exporters import adobe_xmp

    recipe = read_recipe(Path(args.recipe))
    raw_path = Path(args.raw).resolve()
    if not raw_path.exists():
        print(f"Error: {raw_path} does not exist", file=sys.stderr)
        return 1

    current = media.checksum_file(raw_path)
    plan = adobe_xmp.plan_apply(recipe, raw_path, current_checksum=current, force=args.force)

    if plan.stale:
        print(
            f"Refusing: this recipe was computed from different contents.\n"
            f"  recipe checksum : {recipe.source_checksum[:16]}\n"
            f"  file checksum   : {current[:16]}\n"
            "Re-analyse the file before applying anything to it.",
            file=sys.stderr,
        )
        return 2

    if plan.exists:
        print(f"An existing sidecar is already at {plan.target}")
        if plan.diff:
            print("Differences that would be written:")
            for line in plan.diff:
                print(line)
        if plan.would_overwrite:
            print(
                "\nRefusing to overwrite your own edits. Re-run with --force if that is "
                "genuinely what you want.",
                file=sys.stderr,
            )
            return 2

    if not args.apply:
        print(f"\nWould write {plan.target}. Nothing has been written; re-run with --apply.")
        return 0

    plan.target.write_text(adobe_xmp.to_adobe_xmp(recipe), encoding="utf-8")
    print(f"Wrote {plan.target}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    """The active-learning session: the questions worth five minutes."""
    import active_learning
    import preference_model
    from preference_store import PreferenceStore

    analysis = Path(args.analysis).resolve()
    records = _load_records(analysis)
    store = PreferenceStore(analysis.parent.parent / "preferences.jsonl")
    model = preference_model.fit(store)

    questions = active_learning.propose(records, model, limit=args.limit or 12)
    print(active_learning.format_session(questions))
    print(f"{store.count()} decision(s) recorded so far.")
    print("Answer with:  python cli.py record --signal <signal> --winner <id> --loser <id>")
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    """Record one decision the photographer made."""
    from preference_store import Decision, PreferenceStore

    store = PreferenceStore(Path(args.store).resolve())
    decision = store.record(
        Decision(
            signal=args.signal,
            winner=args.winner or "",
            loser=args.loser or "",
            asset_id=args.asset or "",
            answer=args.answer or "",
            genre=args.genre or "",
            camera=args.camera or "",
            note=args.note or "",
        )
    )
    print(f"Recorded {decision.signal} (weight {decision.weight:.1f}). {store.count()} total.")
    return 0


def cmd_policy(args: argparse.Namespace) -> int:
    """What the tool would automate, and what is holding each gate shut."""
    import selective_policy

    records = _load_records(Path(args.analysis).resolve())
    buckets: dict[str, int] = {}
    for record in records:
        buckets[record.decision_bucket or "unassigned"] = (
            buckets.get(record.decision_bucket or "unassigned", 0) + 1
        )

    total = max(len(records), 1)
    needs_human = sum(
        count for bucket, count in buckets.items()
        if bucket in ("manual_review", "curatorial_review")
    )
    print(f"{len(records)} asset(s)\n")
    for bucket, count in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"  {bucket:<28}{count:>6}  {count / total:>6.1%}")
    print(f"\n  needs a full human decision : {needs_human} ({needs_human / total:.1%})")
    print(f"  handled without one         : {1 - needs_human / total:.1%}")
    acting = sum(
        1 for r in records
        if r.decision_bucket == selective_policy.Bucket.SAFE_QUARANTINE_CANDIDATE.value
    )
    print(f"  acts on files               : {acting}")

    held = [r for r in records if r.policy_evidence and r.abstained]
    if held:
        print("\nWhy the tool held back (most common):")
        counts: dict[str, int] = {}
        for record in held:
            counts[record.policy_evidence[0]] = counts.get(record.policy_evidence[0], 0) + 1
        for reason, count in sorted(counts.items(), key=lambda kv: -kv[1])[:6]:
            print(f"  {count:>5}  {reason}")
    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    """False-trash rate, drift and calibration. Switches automation off itself."""
    from model_monitoring import Monitor

    monitor = Monitor(Path(args.state).resolve())
    report = monitor.evaluate()
    monitor.save()

    print(f"false-trash rate      : {report['false_trash_rate']:.3%} over {report['resolved_cases']} resolved")
    print(f"drift (out of dist.)  : {report['drift']:.1%}")
    print(f"worst calibration gap : {report['worst_calibration_gap']:.3f}")
    print(f"automation            : {'ON' if report['automation_enabled'] else 'OFF'}")
    if report["disabled_reason"]:
        print(f"  reason: {report['disabled_reason']}")
    for problem in report["problems"]:
        print(f"  PROBLEM: {problem}")
    if args.enable:
        ok, message = monitor.enable_automation(holdout_checks=args.holdout)
        monitor.save()
        print(f"\nenable: {'granted' if ok else 'refused'} -- {message}")
    return 0


def cmd_trash(args: argparse.Namespace) -> int:
    """Carry out delete_plan.json in Python. Moves to Trash, never unlinks."""
    plan = Path(args.plan).resolve()
    if not plan.exists():
        print(f"Error: no plan at {plan}", file=sys.stderr)
        return 1

    report = layout.move_to_trash(plan, dry_run=not args.apply, trash_dir=Path(args.trash) if args.trash else None)
    if not args.apply:
        print(f"{report['planned']} file(s) would move to {report['trash']}.")
        for skip in report["skipped"][:20]:
            print(f"  skip {skip['file']}: {skip['why']}")
        print("\nNothing has been moved. Re-run with --apply.")
        return 0

    print(f"Moved {report['moved']} of {report['planned']} file(s) to {report['trash']}.")
    for skip in report["skipped"]:
        print(f"  skip {skip['file']}: {skip['why']}")
    print("Nothing was permanently deleted; empty the Trash yourself when happy.")
    return 0


def cmd_restore(args: argparse.Namespace) -> int:
    quarantine = quarantine_module.Quarantine(Path(args.quarantine).resolve())
    planned = quarantine.restore(args.operation, dry_run=True)
    if not args.apply:
        print(f"{len(planned)} file(s) would be restored to their original paths.")
        for op in planned[:20]:
            print(f"  {op.destination} -> {op.source}")
        print("\nNothing has been moved. Re-run with --apply.")
        return 0

    results = quarantine.restore(args.operation, dry_run=False)
    restored = sum(1 for r in results if r.status == "restored")
    print(f"Restored {restored} of {len(results)}.")

    # A restore is the loudest correction available: the tool proposed removing
    # this file and the photographer went and undid it. Recording it is what
    # gives the monitor a real false-trash rate instead of one over an empty set.
    if restored and args.monitor:
        from model_monitoring import Monitor
        from preference_store import Decision, PreferenceStore, Signal

        monitor = Monitor(Path(args.monitor).resolve())
        store = PreferenceStore(Path(args.monitor).resolve().parent / "preferences.jsonl")
        for op in results:
            if op.status != "restored":
                continue
            monitor.resolve(op.asset_id, "restored")
            store.record(
                Decision(signal=Signal.RESTORED_FROM_QUARANTINE.value, asset_id=op.asset_id,
                         tool_said="trash", answer="keep", note=op.reason)
            )
        report = monitor.evaluate()
        monitor.save()
        print(f"Recorded {restored} correction(s). False-trash rate now "
              f"{report['false_trash_rate']:.3%} over {report['resolved_cases']} resolved.")
        if not report["automation_enabled"] and report["disabled_reason"]:
            print(f"Automation switched off: {report['disabled_reason']}")
    for op in results:
        if op.status != "restored":
            print(f"  {op.status}: {op.destination} ({op.error})")
    return 0


def cmd_purge(args: argparse.Namespace) -> int:
    quarantine = quarantine_module.Quarantine(Path(args.quarantine).resolve())
    try:
        report = quarantine.purge(
            confirmation=args.confirm or "",
            older_than_days=args.older_than,
            dry_run=not args.apply,
        )
    except ValueError:
        print(
            "Refusing to purge.\n"
            f"  This permanently deletes quarantined files older than {args.older_than} days.\n"
            f"  There is no undo. Re-run with:  --confirm '{quarantine_module.PURGE_CONFIRMATION}' --apply",
            file=sys.stderr,
        )
        return 2
    except quarantine_module.OperationLocked as e:
        print(f"Refusing to purge: {e}", file=sys.stderr)
        return 2

    if not args.apply:
        print(
            f"{report['eligible']} file(s), {report['bytes'] / 1_048_576:.1f} MB would be "
            f"PERMANENTLY deleted. Nothing has been removed."
        )
        return 0
    print(f"Permanently deleted {report['purged']} file(s).")
    return 0


# --- export -----------------------------------------------------------------


def cmd_export(args: argparse.Namespace) -> int:
    analysis = Path(args.analysis).resolve()
    records = _load_records(analysis)
    out_dir = Path(args.output).resolve() / args.platform
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = [
        r
        for r in records
        if r.route_class in (RouteClass.STOCK_STANDARD.value, RouteClass.STOCK_STRONG.value, RouteClass.FLAGSHIP.value)
        and any(m.get("platform_id") == args.platform and m.get("eligible") for m in r.marketplaces)
    ]
    if not selected:
        print(f"No asset is currently eligible for {args.platform}.")
        return 0

    rows = []
    checklist = []
    for record in selected:
        meta = stock_metadata.StockMetadata(**{
            k: v for k, v in (record.stock_metadata or {}).items()
            if k in stock_metadata.StockMetadata.__dataclass_fields__
        })
        rows.append((record.filename, meta))
        if meta.model_release_required:
            checklist.append(f"{record.filename}: model release required")
        if meta.property_release_required:
            checklist.append(f"{record.filename}: property release required")
        if meta.logo_warning:
            checklist.append(f"{record.filename}: {meta.logo_warning}")
        stock_metadata.write_xmp_sidecar(meta, out_dir / record.filename)

    csv_path = stock_metadata.write_submission_csv(rows, out_dir / "submission.csv")
    (out_dir / "release_checklist.txt").write_text(
        "\n".join(checklist) or "No releases required for this batch.\n", encoding="utf-8"
    )
    (out_dir / "README.txt").write_text(
        f"""Marketplace package: {args.platform}
{len(selected)} asset(s).

This is an EXPORT PACKAGE, not an upload. Nothing has been submitted anywhere.

  submission.csv         metadata for each file, in upload order
  <name>.xmp             IPTC/XMP sidecar per file
  release_checklist.txt  what still needs paperwork

To submit: upload the original files through the platform's own contributor
portal and attach submission.csv where the platform supports a metadata import.
Verify the platform's current requirements first -- they change, and the rules
this package was built from were last checked on the date recorded in
data/marketplace_rules.json.
""",
        encoding="utf-8",
    )
    print(f"Package for {args.platform}: {len(selected)} asset(s) -> {out_dir}")
    print(f"  {csv_path}")
    if checklist:
        print(f"  {len(checklist)} item(s) need releases; see release_checklist.txt")
    return 0


# --- override ---------------------------------------------------------------


def cmd_override(args: argparse.Namespace) -> int:
    analysis = Path(args.analysis).resolve()
    records = _load_records(analysis)
    store = overrides_module.OverrideStore(analysis.parent.parent / overrides_module.OVERRIDES_NAME)

    if args.list:
        for override in store.all():
            print(
                f"  {override.filename:<30} {override.tool_said or '?':>16} -> "
                f"{override.route_class or ('excluded' if override.excluded else '?')}"
                + (f"  ({override.note})" if override.note else "")
            )
        if not len(store):
            print("  (no manual overrides recorded)")
        return 0

    matches = [r for r in records if r.filename == args.filename or r.asset_id == args.filename]
    if not matches:
        print(f"Error: {args.filename} is not in {analysis}", file=sys.stderr)
        return 1

    record = matches[0]
    if args.clear:
        removed = store.remove(record.asset_id)
        store.save()
        print("Removed." if removed else "There was no override for that asset.")
        return 0

    override = overrides_module.capture(record)
    override.route_class = args.set_class
    override.genre = args.set_genre
    override.marketplaces = args.set_marketplace or []
    override.excluded = args.exclude
    override.note = args.note or ""
    store.set(override)
    store.save()
    print(
        f"Recorded: {record.filename} {record.route_class} -> "
        f"{args.set_class or ('excluded' if args.exclude else record.route_class)}"
    )
    print(f"Future runs will respect this. Stored in {store.path}")
    return 0


def cmd_profiles(args: argparse.Namespace) -> int:
    for name, factory in BUILTIN_PROFILES.items():
        profile = factory()
        marker = "" if profile.is_fitted else "  [provisional -- not fitted to labelled data]"
        print(f"  {name:<18} {profile.media:<6} v{profile.version}{marker}")
        print(f"    {profile.notes}")
    if args.dump:
        path = Path(args.dump)
        factory = BUILTIN_PROFILES.get(args.name or "default-photo")
        if factory is None:
            print(f"Unknown profile {args.name}", file=sys.stderr)
            return 1
        print(f"\nWrote {factory().save(path)}")
        print("Edit the thresholds, then pass it with --profile-file.")
    return 0


def cmd_validate_profile(args: argparse.Namespace) -> int:
    profile = CalibrationProfile.load(Path(args.path))
    print(f"{profile.name} v{profile.version} ({profile.media})")
    print(f"  fitted: {profile.is_fitted}")
    print("  weights:")
    for key, value in sorted(profile.normalised_weights().items()):
        print(f"    {key:<24}{value:.3f}")
    print("  thresholds:")
    for key, value in sorted(profile.thresholds.items()):
        print(f"    {key:<24}{value}")
    return 0


# --- argument wiring --------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="photo-ai-toolkit",
        description="Cull, assess and route photos and video by realistic post-edit potential.",
    )
    parser.add_argument("--lang", default="en", choices=list(i18n.SUPPORTED), help="UI language")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_shared_analysis_arguments(parser) -> None:
        """Everything both the product command and the developer one take."""
        parser.add_argument("--input", required=True)
        parser.add_argument("--output", required=True)
        parser.add_argument(
            "--quarantine", help="quarantine directory (default: <output>/.internal/quarantine)"
        )
        parser.add_argument(
            "--profile", choices=list(BUILTIN_PROFILES), help="built-in calibration profile"
        )
        parser.add_argument("--profile-file", help="path to a calibration profile JSON")
        parser.add_argument(
            "--expert",
            action="store_true",
            help="print the route classes, stock counters and release status as well",
        )
        parser.add_argument("--no-video", action="store_true")
        parser.add_argument("--video-samples", type=int, default=9)
        parser.add_argument("--force", action="store_true", help="ignore the analysis cache")
        parser.add_argument("--limit", type=int)
        parser.add_argument("--copyright", help="copyright holder for generated metadata")
        parser.add_argument(
            "--darkroom", action="store_true",
            help="render edit suggestions (about a second per frame)",
        )
        parser.add_argument("--renderer", help="darkroom engine: builtin, darktable, rawtherapee")
        parser.add_argument(
            "--no-shadow-mode", action="store_true",
            help="let the policy act instead of only recording what it would do",
        )

    analyze = sub.add_parser(
        "analyze",
        help="the full analysis: local measurement + content check + artistic read",
    )
    add_shared_analysis_arguments(analyze)
    analyze.add_argument(
        "--model",
        default=None,
        help=(
            f"overrides {bootstrap.MODEL_VAR}, which overrides the default "
            f"({bootstrap.DEFAULT_SEMANTIC_MODEL}). Whatever is chosen is verified "
            "against your account before any photograph is opened, and never "
            "silently replaced"
        ),
    )
    analyze.add_argument(
        "--no-stage3",
        action="store_true",
        help="run the content check but not the artistic read (nothing can reach TOP)",
    )
    analyze.add_argument(
        "-y", "--yes",
        action="store_true",
        help="do not ask before spending on a large batch",
    )
    analyze.add_argument(
        "--insights-scope",
        choices=["new", "all"],
        default="new",
        help=(
            "which photographs the insights page describes: the ones this run "
            "analysed (default), or everything ever stored in this output directory"
        ),
    )
    analyze.set_defaults(func=cmd_analyze)

    measure = sub.add_parser(
        "measure",
        help=(
            "DEVELOPER TOOL: local measurement only, no content check and no "
            "artistic read. Not a photo analysis, and its output says so"
        ),
    )
    add_shared_analysis_arguments(measure)
    measure.set_defaults(func=cmd_measure)

    report = sub.add_parser("report", help="filter, sort and re-render a stored run")
    report.add_argument("--analysis", required=True)
    report.add_argument("--format", default="table", choices=["table", "json", "csv", "html", "all"])
    report.add_argument("--sort", default="score", choices=list(SORT_KEYS))
    report.add_argument("--media", choices=["photo", "video"])
    report.add_argument("--route-class", action="append", choices=[c.value for c in RouteClass])
    report.add_argument("--route", choices=["commercial", "editorial"])
    report.add_argument("--genre")
    report.add_argument("--marketplace")
    report.add_argument("--min-score", type=int)
    report.add_argument("--min-potential", type=int)
    report.add_argument("--min-confidence", type=int)
    report.add_argument("--needs-release", action="store_true")
    report.add_argument("--cluster")
    report.add_argument("--duplicates-only", action="store_true")
    report.add_argument("--limit", type=int)
    report.set_defaults(func=cmd_report)

    reclassify = sub.add_parser("reclassify", help="redo routing with new thresholds, free")
    reclassify.add_argument("--analysis", required=True)
    reclassify.add_argument("--profile", choices=list(BUILTIN_PROFILES))
    reclassify.add_argument("--profile-file")
    reclassify.add_argument("--limit", type=int)
    reclassify.set_defaults(func=cmd_reclassify)

    quarantine = sub.add_parser("quarantine", help="move trash-class files (dry run by default)")
    quarantine.add_argument("--analysis", required=True)
    quarantine.add_argument("--quarantine", required=True)
    quarantine.add_argument("--input", help="source root, to fence the operation")
    quarantine.add_argument("--apply", action="store_true", help="actually move the files")
    quarantine.set_defaults(func=cmd_quarantine)

    darkroom_cmd = sub.add_parser("darkroom", help="show edit suggestions from a stored run")
    darkroom_cmd.add_argument("--analysis", required=True)
    darkroom_cmd.add_argument("--output")
    darkroom_cmd.add_argument("--limit", type=int)
    darkroom_cmd.set_defaults(func=cmd_darkroom)

    apply_cmd = sub.add_parser("apply-recipe", help="write a recipe beside the RAW (dry run)")
    apply_cmd.add_argument("--recipe", required=True)
    apply_cmd.add_argument("--raw", required=True)
    apply_cmd.add_argument("--apply", action="store_true")
    apply_cmd.add_argument("--force", action="store_true", help="overwrite an existing sidecar")
    apply_cmd.set_defaults(func=cmd_apply_recipe)

    ask = sub.add_parser("ask", help="the questions worth answering, most informative first")
    ask.add_argument("--analysis", required=True)
    ask.add_argument("--limit", type=int)
    ask.set_defaults(func=cmd_ask)

    record_cmd = sub.add_parser("record", help="record one decision for the personal model")
    record_cmd.add_argument("--store", required=True)
    record_cmd.add_argument("--signal", required=True)
    record_cmd.add_argument("--winner")
    record_cmd.add_argument("--loser")
    record_cmd.add_argument("--asset")
    record_cmd.add_argument("--answer")
    record_cmd.add_argument("--genre")
    record_cmd.add_argument("--camera")
    record_cmd.add_argument("--note")
    record_cmd.set_defaults(func=cmd_record)

    policy = sub.add_parser("policy", help="what would be automated, and what is holding it back")
    policy.add_argument("--analysis", required=True)
    policy.set_defaults(func=cmd_policy)

    monitor = sub.add_parser("monitor", help="false-trash rate, drift and calibration")
    monitor.add_argument("--state", required=True)
    monitor.add_argument("--enable", action="store_true")
    monitor.add_argument("--holdout", type=int, default=0)
    monitor.set_defaults(func=cmd_monitor)

    trash = sub.add_parser("trash", help="carry out delete_plan.json (dry run unless --apply)")
    trash.add_argument("--plan", required=True)
    trash.add_argument("--trash", help="target directory (default: ~/.Trash)")
    trash.add_argument("--apply", action="store_true")
    trash.set_defaults(func=cmd_trash)

    restore = sub.add_parser("restore", help="undo a quarantine operation")
    restore.add_argument("--quarantine", required=True)
    restore.add_argument("--operation", help="operation id; default is everything still quarantined")
    restore.add_argument("--apply", action="store_true")
    restore.add_argument("--monitor", help="monitoring state to record the correction in")
    restore.set_defaults(func=cmd_restore)

    purge = sub.add_parser("purge", help="PERMANENTLY delete quarantined files")
    purge.add_argument("--quarantine", required=True)
    purge.add_argument("--older-than", type=int, default=quarantine_module.DEFAULT_PURGE_AGE_DAYS)
    purge.add_argument("--confirm", help=f"must be exactly: {quarantine_module.PURGE_CONFIRMATION}")
    purge.add_argument("--apply", action="store_true")
    purge.set_defaults(func=cmd_purge)

    export = sub.add_parser("export", help="build a marketplace-ready package")
    export.add_argument("--analysis", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--platform", required=True)
    export.set_defaults(func=cmd_export)

    override = sub.add_parser("override", help="record a manual decision")
    override.add_argument("--analysis", required=True)
    override.add_argument("filename", nargs="?", default="")
    override.add_argument("--set-class", choices=[c.value for c in RouteClass])
    override.add_argument("--set-genre")
    override.add_argument("--set-marketplace", action="append")
    override.add_argument("--exclude", action="store_true", help="exclude from future analysis")
    override.add_argument("--note")
    override.add_argument("--clear", action="store_true")
    override.add_argument("--list", action="store_true")
    override.set_defaults(func=cmd_override)

    profiles = sub.add_parser("profiles", help="list or dump calibration profiles")
    profiles.add_argument("--name")
    profiles.add_argument("--dump", help="write the profile to this path for editing")
    profiles.set_defaults(func=cmd_profiles)

    validate = sub.add_parser("validate-profile", help="show a profile as it will be applied")
    validate.add_argument("path")
    validate.set_defaults(func=cmd_validate_profile)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Before anything reads an environment variable. The whole point of a single
    # bootstrap is that no subcommand has to remember to do this.
    bootstrap.load_project_environment()
    args = build_parser().parse_args(argv)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        logging.getLogger().handlers[0].addFilter(reports.RedactingFilter())
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted. Nothing was moved or deleted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
