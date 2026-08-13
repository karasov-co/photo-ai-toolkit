"""User-facing strings, in English and Russian.

The repository had no localisation layer before this feature, so this is a small
deliberate one rather than a framework: a flat catalogue, a lookup that falls
back to English, and a rule that nothing user-facing is written as a literal
anywhere else.

Two things that are *not* translated, on purpose:

- **Machine-readable values.** Route classes, issue codes and tags keep their
  English identifiers in JSON and CSV, because those are keys other software
  reads. Only their labels are localised.
- **Score dimension names in exports.** Same reason.

Coverage is enforced by a test rather than by discipline: every key in the
English catalogue must exist in the Russian one, so adding a string in one
language and forgetting the other fails the suite.
"""

from __future__ import annotations

DEFAULT_LANGUAGE = "en"
SUPPORTED = ("en", "ru")

STRINGS: dict[str, dict[str, str]] = {
    "en": {
        # headings
        "summary.title": "COLLECTION SUMMARY",
        "summary.total": "Total assets",
        "summary.photos": "Photos",
        "summary.videos": "Videos",
        "summary.failed": "Failed to analyze",
        "summary.low_confidence": "Low confidence",
        "summary.recoverable_space": "Space recoverable from quarantine",
        "summary.top_genres": "Top genres",
        "summary.strongest": "Strongest assets",
        "summary.categories": "What these photographs are:",
        "summary.route_classes": "What can be done with them:",
        "summary.top_photos": "Top photos",
        "summary.no_top_photos": (
            "Nothing in this run reached TOP. That is a normal result for a shoot, "
            "and it is not a failure of the analysis."
        ),
        "misc.not_for_stock": "Not for stock",
        # --- the default report ---
        "report.title": "Your photographs",
        "report.lede": (
            "Ranked by how good each photograph can realistically become after a normal "
            "edit -- not by how the untouched file looks right now."
        ),
        "report.potential": "after editing",
        "report.current": "technical quality now",
        "report.empty_section": "Nothing here.",
        "report.no_top": (
            "Nothing reached the top this time: the strongest photograph scored {best}, "
            "and this section starts at {threshold}. That is a normal result for a "
            "shoot, and the photographs below it are still the best ones you have."
        ),
        "report.expert": "Expert details",
        "report.expert_note": (
            "The full numbers behind each decision. Nothing here is needed to use the "
            "report; the complete data is in .internal/reports/analysis.json."
        ),
        "report.recipe_ready": "Edit recipe written to edit_recipes/",
        "report.insights_link": "What this collection says about your photography",
        "report.uplift_unvalidated": (
            "The gain after editing is an estimate from an internal metric that has "
            "not been checked against a labelled set."
        ),
        "report.footer": (
            "Scores are suggestions, not verdicts. Nothing has been moved, changed or "
            "deleted: the folders hold links to your original files."
        ),
        "category.note.TOP": "The strongest work here. Edit these first.",
        "category.note.GOOD_STOCK": "Good photographs that also work as stock material.",
        "category.note.GOOD_EDITORIAL": (
            "Good photographs of real places and moments. Sellable as editorial — "
            "news, travel, documentary — rather than as advertising."
        ),
        "category.note.GOOD_PERSONAL": "Good photographs worth keeping and printing.",
        "category.note.NEEDS_DECISION": "Genuinely borderline. A quick look settles them.",
        "category.note.WEAK": (
            "Kept, not deleted. Blinks, missed moments, accidental frames and weaker "
            "takes of a shot you already have."
        ),
        "expert.file": "File",
        "expert.potential": "After editing",
        "expert.current": "Technical now",
        "expert.category": "Category",
        "expert.route_class": "Route class",
        "expert.stage3_delta": "Artistic read",
        "expert.stock_blockers": "Stock notes",
        # --- photographer insights ---
        "insights.title": "What this collection says about your photography",
        "insights.lede": (
            "Patterns across all {total} photographs. Every line below names the number "
            "and the files it came from -- nothing here is general advice."
        ),
        "insights.lede_new": (
            "Insights based on {total} newly analyzed photographs, out of {stored} "
            "in this collection. The rest were analysed in an earlier run and have "
            "not changed; use --insights-scope all to include them."
        ),
        "insights.back": "Back to the photographs",
        "insights.genres": "What you shoot best",
        "insights.genres_lead": "Ranked by how well each did, not by how many you took: {genres}",
        "insights.genres_evidence": "{genre} scored highest against your own average",
        "insights.habits": "Your visual habits",
        "insights.technical": "What you do reliably well",
        "insights.artistic": "What the artistic read keeps finding",
        "insights.weaknesses": "What is costing you frames",
        "insights.improvements": "The three things worth changing next",
        "insights.inspiration": "Worth looking at",
        "insights.inspiration_note": (
            "Names to look up in the genres you shoot most. No commentary, because a "
            "one-line summary of somebody's life work is usually wrong."
        ),
        "insights.footer": (
            "Based only on this collection. A different shoot would produce different "
            "patterns, and one weak run is not a verdict on anybody's photography."
        ),
        "summary.top_gap": (
            "(nothing reached TOP: the best photograph scored {best}, and TOP needs {threshold})"
        ),
        "summary.semantic_partial": (
            "*** {count} of {total} files have NO content check: their genre, faces and "
            "trademarks are unverified and no portrait gate was applied to them ***"
        ),
        "summary.stage3_completed": "Artistic read completed",
        # categories
        "category.TOP": "Top",
        "category.GOOD_STOCK": "Good — stock",
        "category.GOOD_EDITORIAL": "Good — editorial",
        "category.GOOD_PERSONAL": "Good — personal",
        "category.NEEDS_DECISION": "Needs decision",
        "category.WEAK": "Weak",
        "category.top": "final score {value}: a completed artistic read and no critical defect",
        "category.weak": "final score {value} is below the keep threshold",
        "category.good_stock": "final score {value}, no release needed and commercially usable",
        "category.good_editorial": "final score {value}: a place and a moment, sellable as editorial",
        "category.good_personal": "final score {value}: worth keeping, but not stock material",
        "category.needs_decision": "on the boundary at {value} and the analysis is unsure",
        "category.defect.bad_expression": "the expression failed: {detail}",
        "category.defect.eyes_closed": "the subject's eyes are closed: {detail}",
        "category.defect.accidental_frame": "not a photograph anybody meant to take: {detail}",
        "category.defect.dead_moment": "nothing is happening: {detail}",
        "category.defect.no_subject": "no legible subject: {detail}",
        "category.defect.inferior_duplicate": "a better frame of this exists: {detail}",
        "category.defect.unrecoverable": "unrecoverable: {detail}",
        "summary.clusters": "Duplicate clusters",
        "summary.missing_releases": "Missing releases",
        "summary.marketplace_ready": "Marketplace-ready",
        # classes
        "class.trash": "Trash / reject",
        "class.review": "Needs manual review",
        "class.archive_only": "Keep in the archive",
        "class.duplicate_candidate": "Similar frame, compare by hand",
        "class.stock_standard": "Usable stock",
        "class.stock_strong": "Strong stock",
        "class.flagship": "Flagship / portfolio",
        # routes
        "route.commercial": "Commercial",
        "route.editorial": "Editorial only",
        # scores
        "score.current_quality": "Current quality",
        "score.recoverability": "Recoverability",
        "score.post_edit_potential": "Realistic post-edit potential",
        "score.aesthetic_potential": "Aesthetic potential",
        "score.stock_potential": "Stock potential",
        "score.portfolio_potential": "Portfolio potential",
        "score.legal_readiness": "Legal readiness",
        "score.uniqueness": "Uniqueness",
        "score.confidence": "Confidence",
        "score.routing_score": "Overall",
        # issue groups
        "issues.fixable": "Fixable problems",
        "issues.partially_fixable": "Partially fixable problems",
        "issues.unrecoverable": "Unrecoverable problems",
        "issues.none": "None detected",
        # recipe
        "recipe.title": "Suggested edit",
        "recipe.expected_gain": "Expected realistic improvement",
        "recipe.none": "No edit needed",
        # actions
        "action.proposed": "Proposed action",
        "action.none": "Leave in place",
        "action.quarantine": "Move to quarantine",
        "action.review": "Hold for manual review",
        "action.dry_run": "Nothing has been moved. Re-run with --apply to carry this out.",
        "action.approved": "Applied",
        # warnings
        "warn.release_required": "A model or property release is required",
        "warn.logo": "Readable trademark present",
        "warn.provenance": "Provenance is undeclared",
        "warn.disclaimer": (
            "Scores are recommendations, not guarantees of artistic quality, "
            "marketplace acceptance, or sales. Release and IP detection is advisory "
            "and does not replace legal review. Marketplace policies change and must "
            "be re-verified."
        ),
        # The same caution without the licensing vocabulary. A photographer
        # sorting their own holiday photographs is not making a legal decision
        # and should not be handed a legal disclaimer.
        "warn.disclaimer_simple": (
            "These are suggestions, not verdicts -- a score is one opinion about a "
            "photograph. Nothing has been moved, changed or deleted: the folders "
            "hold links to your original files."
        ),
        # misc
        "misc.duplicate_of": "Weaker duplicate of",
        "misc.best_in_cluster": "Best of {n} similar frames",
        "misc.strengths": "Strengths",
        "misc.reasons": "Why this class",
        "misc.marketplaces": "Recommended marketplaces",
        "misc.none": "none",
        "misc.usable_segment": "Usable segment",
        "misc.poster_frame": "Best frame",
        # credentials and analysis mode
        "creds.environment": "Semantic credentials: found in environment",
        "creds.dotenv": "Semantic credentials: loaded from {path}",
        "creds.missing": "Semantic credentials: missing",
        "creds.local_only": (
            "No API key found: running local measurement only. Without the content check "
            "and the artistic read, no photograph can be categorised as TOP and release "
            "status stays unverified. Set OPENAI_API_KEY in .env for the full analysis."
        ),
        "creds.error": (
            "Error: --semantic was requested but OPENAI_API_KEY was not found.\n"
            "Add OPENAI_API_KEY to the .env file in the project root and run again.\n"
            "Example: OPENAI_API_KEY=your_key_here"
        ),
        "creds.failed": "Error: the semantic pass could not run: {reason}",
        "creds.fallback_hint": (
            "Re-run with --allow-semantic-fallback to accept a local-only result instead."
        ),
        "mode.title": "Analysis mode",
        "mode.local_only": "local-only",
        "mode.local_and_semantic": "local + semantic",
        "mode.local_only_after_semantic_failure": "local-only after semantic failure",
        "mode.banner": "NOBODY LOOKED AT THESE PHOTOGRAPHS",
        "mode.banner_detail": (
            "Only the files were measured. Without the content and artistic passes, "
            "nothing can be ranked as a top photograph and the subject of each frame "
            "is unknown rather than 'other'."
        ),
        "summary.technically_usable": "Technically usable, needs checking",
        "summary.fully_checked": "Fully checked and ready to export",
        "summary.not_semantically_checked": "Not checked by semantic analysis",
        "summary.technically_usable_help": (
            "Past the technical thresholds. Content, faces, logos and releases may not "
            "have been checked."
        ),
        "summary.fully_checked_help": (
            "Everything checked and exportable: content, faces, logos, releases and "
            "marketplace metadata. Requires the semantic pass."
        ),
        "summary.release_status": "Release status",
        "summary.release_unchecked": "not checked",
        "summary.export_blocked_reason": (
            "Nothing is export-ready because content, faces, logos and metadata "
            "have not been checked."
        ),
        # plan
        "plan.would_move": "{count} file(s) would move, {mb} MB:",
        "plan.nothing_to_move": "Nothing to move.",
        "plan.dry_run": "Nothing has been moved. Re-run with --apply to carry this out.",
        # reasons, localised for display; JSON keeps the English codes
        "reason.unrecoverable": "unrecoverable problems: {detail}",
        "reason.release_unchecked": (
            "release status not checked (no semantic pass): commercial stock is blocked "
            "until faces and trademarks have actually been checked"
        ),
        "reason.release_blocked": (
            "{present} present: a release is required, so commercial stock is blocked"
        ),
        "reason.low_confidence": "confidence {value} is below {threshold}: a human should decide",
        "reason.below_trash": (
            "realistic post-edit potential {value} is below {threshold}, but nothing about "
            "this frame is unrecoverable: keeping it"
        ),
        "reason.stock_strong": "stock potential {value} is strong",
        "reason.stock_standard": "stock potential {value} is usable after the suggested edit",
        "reason.hero_blocked": "held back from flagship: {detail}",
        "reason.flagship": (
            "portfolio potential {value} clears the absolute floor and ranks near the top "
            "of its genre"
        ),
        "reason.archive": (
            "recoverable ({value} potential) but below the stock floor ({threshold}): keep "
            "for the archive or decide by hand"
        ),
        "reason.duplicate_candidate": (
            "a sharper frame exists in this group ({margin} points higher); compare them by hand"
        ),
        "reason.duplicate_unchecked": (
            "a sharper frame exists in this group ({margin} points higher), but no content "
            "check ran -- only sharpness separates them, so compare them by hand"
        ),
        "reason.short_clip": "only {seconds}s long: too short for most marketplaces",
        # contact sheet
        "sheet.candidate": "candidate",
        "sheet.cluster_best": "best of group",
        "sheet.margin": "difference",
        "sheet.semantic_ran": "semantic analysis ran",
        "sheet.semantic_missing": "NO semantic analysis -- check by hand",
    },
    "ru": {
        "summary.title": "СВОДКА ПО КОЛЛЕКЦИИ",
        "summary.total": "Всего файлов",
        "summary.photos": "Фотографий",
        "summary.videos": "Видео",
        "summary.failed": "Не удалось проанализировать",
        "summary.low_confidence": "Низкая уверенность",
        "summary.recoverable_space": "Освободится после карантина",
        "summary.top_genres": "Основные жанры",
        "summary.strongest": "Сильнейшие кадры",
        "summary.categories": "Что это за фотографии:",
        "summary.route_classes": "Что с ними можно делать:",
        "summary.top_photos": "Лучшие фотографии",
        "summary.no_top_photos": (
            "Ни один кадр в этом запуске не попал в TOP. Для съёмки это нормальный "
            "результат, а не сбой анализа."
        ),
        "misc.not_for_stock": "Не для стока",
        "report.title": "Ваши фотографии",
        "report.lede": (
            "Отсортировано по тому, насколько сильной фотография реально может стать "
            "после обычной обработки, а не по тому, как выглядит необработанный файл."
        ),
        "report.potential": "после обработки",
        "report.current": "техническое качество файла",
        "report.empty_section": "Здесь пусто.",
        "report.no_top": (
            "В этот раз сюда никто не попал: лучшая фотография набрала {best}, а раздел "
            "начинается с {threshold}. Для съёмки это нормально, и кадры ниже — всё "
            "равно лучшее, что у вас есть."
        ),
        "report.expert": "Подробности для специалистов",
        "report.expert_note": (
            "Полные цифры за каждым решением. Для работы с отчётом они не нужны; "
            "все данные лежат в .internal/reports/analysis.json."
        ),
        "report.recipe_ready": "Рецепт обработки сохранён в edit_recipes/",
        "report.insights_link": "Что эта съёмка говорит о вашей фотографии",
        "report.uplift_unvalidated": (
            "Прирост после обработки — оценка внутренней метрики, не проверенная "
            "на размеченном наборе."
        ),
        "report.footer": (
            "Оценки — это подсказки, а не приговор. Ничего не перемещено, не изменено "
            "и не удалено: в папках лежат ссылки на ваши оригиналы."
        ),
        "category.note.TOP": "Самое сильное здесь. Начните обработку с них.",
        "category.note.GOOD_STOCK": "Хорошие фотографии, которые годятся и для стока.",
        "category.note.GOOD_EDITORIAL": (
            "Хорошие снимки реальных мест и моментов. Продаются как редакционные — "
            "новости, travel, документалистика, — а не как реклама."
        ),
        "category.note.GOOD_PERSONAL": "Хорошие фотографии — оставить и напечатать.",
        "category.note.NEEDS_DECISION": "Действительно на грани. Одного взгляда хватит.",
        "category.note.WEAK": (
            "Сохранены, не удалены. Моргания, упущенные моменты, случайные кадры и "
            "слабые дубли того, что у вас уже есть."
        ),
        "expert.file": "Файл",
        "expert.potential": "После обработки",
        "expert.current": "Технически сейчас",
        "expert.category": "Категория",
        "expert.route_class": "Класс маршрута",
        "expert.stage3_delta": "Художественный разбор",
        "expert.stock_blockers": "Заметки по стоку",
        "insights.title": "Что эта съёмка говорит о вашей фотографии",
        "insights.lede": (
            "Закономерности по всем {total} фотографиям. Каждая строка ниже называет "
            "число и файлы, из которых оно получено, — общих советов здесь нет."
        ),
        "insights.lede_new": (
            "Выводы по {total} новым фотографиям из {stored} в этой коллекции. "
            "Остальные анализировались в прошлый раз и не менялись; чтобы учесть "
            "и их, запустите с --insights-scope all."
        ),
        "insights.back": "Назад к фотографиям",
        "insights.genres": "Что у вас получается лучше всего",
        "insights.genres_lead": "По результату, а не по количеству снятого: {genres}",
        "insights.genres_evidence": "{genre} — выше вашего собственного среднего",
        "insights.habits": "Ваши визуальные привычки",
        "insights.technical": "Что вы стабильно делаете хорошо",
        "insights.artistic": "Что раз за разом находит художественный разбор",
        "insights.weaknesses": "Что стоит вам кадров",
        "insights.improvements": "Три вещи, которые стоит изменить",
        "insights.inspiration": "Стоит посмотреть",
        "insights.inspiration_note": (
            "Имена для поиска в тех жанрах, которые вы снимаете чаще всего. Без "
            "комментариев: пересказать чью-то работу одной строкой почти всегда значит соврать."
        ),
        "insights.footer": (
            "Только по этой съёмке. Другая съёмка даст другие закономерности, и один "
            "слабый запуск — не приговор ничьей фотографии."
        ),
        "summary.top_gap": (
            "(в TOP не попал никто: лучшая фотография набрала {best}, а для TOP нужно {threshold})"
        ),
        "summary.semantic_partial": (
            "*** У {count} из {total} файлов НЕТ проверки содержимого: жанр, лица и "
            "торговые марки не проверены, портретные правила к ним не применялись ***"
        ),
        "summary.stage3_completed": "Художественный разбор выполнен",
        "category.TOP": "Лучшие",
        "category.GOOD_STOCK": "Хорошие — сток",
        "category.GOOD_EDITORIAL": "Хорошие — редакционные",
        "category.GOOD_PERSONAL": "Хорошие — для себя",
        "category.NEEDS_DECISION": "Нужно решение",
        "category.WEAK": "Слабые",
        "category.top": "итоговая оценка {value}: художественный разбор выполнен, критических дефектов нет",
        "category.weak": "итоговая оценка {value} ниже порога сохранения",
        "category.good_stock": "итоговая оценка {value}, релиз не нужен, коммерчески пригодно",
        "category.good_editorial": "итоговая оценка {value}: место и момент, годится для редакционного использования",
        "category.good_personal": "итоговая оценка {value}: стоит оставить, но не для стока",
        "category.needs_decision": "на границе ({value}), и разбор не уверен",
        "category.defect.bad_expression": "неудачное выражение лица: {detail}",
        "category.defect.eyes_closed": "у героя закрыты глаза: {detail}",
        "category.defect.accidental_frame": "кадр снят случайно: {detail}",
        "category.defect.dead_moment": "мёртвый момент, ничего не происходит: {detail}",
        "category.defect.no_subject": "нет читаемого сюжета: {detail}",
        "category.defect.inferior_duplicate": "есть кадр этого же лучше: {detail}",
        "category.defect.unrecoverable": "неисправимо: {detail}",
        "summary.clusters": "Групп дублей",
        "summary.missing_releases": "Не хватает релизов",
        "summary.marketplace_ready": "Готово к загрузке на сток",
        "class.trash": "Брак / в отказ",
        "class.review": "Нужен ручной просмотр",
        "class.archive_only": "Оставить в архиве",
        "class.duplicate_candidate": "Похожий кадр, сравнить вручную",
        "class.stock_standard": "Пригодно для стока",
        "class.stock_strong": "Сильный сток",
        "class.flagship": "Флагман / портфолио",
        "route.commercial": "Коммерческое",
        "route.editorial": "Только редакционное",
        "score.current_quality": "Текущее качество",
        "score.recoverability": "Запас на обработку",
        "score.post_edit_potential": "Реальный потенциал после обработки",
        "score.aesthetic_potential": "Художественный потенциал",
        "score.stock_potential": "Стоковый потенциал",
        "score.portfolio_potential": "Потенциал для портфолио",
        "score.legal_readiness": "Юридическая готовность",
        "score.uniqueness": "Неповторимость",
        "score.confidence": "Уверенность оценки",
        "score.routing_score": "Итог",
        "issues.fixable": "Исправимые проблемы",
        "issues.partially_fixable": "Частично исправимые проблемы",
        "issues.unrecoverable": "Неисправимые проблемы",
        "issues.none": "Не обнаружено",
        "recipe.title": "Рекомендуемая обработка",
        "recipe.expected_gain": "Ожидаемый реальный прирост",
        "recipe.none": "Обработка не требуется",
        "action.proposed": "Предлагаемое действие",
        "action.none": "Оставить на месте",
        "action.quarantine": "Переместить в карантин",
        "action.review": "Отложить на ручной просмотр",
        "action.dry_run": "Ничего не перемещено. Запустите с --apply, чтобы выполнить.",
        "action.approved": "Выполнено",
        "warn.release_required": "Требуется релиз модели или собственности",
        "warn.logo": "В кадре читаемый товарный знак",
        "warn.provenance": "Происхождение файла не задекларировано",
        "warn.disclaimer_simple": (
            "Это подсказки, а не приговор: оценка — всего лишь одно мнение о "
            "фотографии. Ничего не перемещено, не изменено и не удалено — в папках "
            "лежат ссылки на ваши оригиналы."
        ),
        "warn.disclaimer": (
            "Оценки — это рекомендации, а не гарантия художественного качества, "
            "приёмки на стоке или продаж. Определение релизов и прав носит "
            "справочный характер и не заменяет юридическую проверку. Правила "
            "стоков меняются, их нужно перепроверять."
        ),
        "misc.duplicate_of": "Слабее дубля",
        "misc.best_in_cluster": "Лучший из {n} похожих кадров",
        "misc.strengths": "Сильные стороны",
        "misc.reasons": "Почему такой класс",
        "misc.marketplaces": "Рекомендуемые стоки",
        "misc.none": "нет",
        "misc.usable_segment": "Пригодный фрагмент",
        "misc.poster_frame": "Лучший кадр",
        "creds.environment": "Ключ для semantic-анализа: найден в окружении",
        "creds.dotenv": "Ключ для semantic-анализа: загружен из {path}",
        "creds.missing": "Ключ для semantic-анализа: не найден",
        "creds.local_only": (
            "API-ключ не найден: выполняется только локальный анализ. Без проверки "
            "содержимого и художественного разбора ни одна фотография не может попасть "
            "в TOP, а статус релизов остаётся непроверенным. Укажите OPENAI_API_KEY "
            "в .env для полного анализа."
        ),
        "creds.error": (
            "Ошибка: включён --semantic, но OPENAI_API_KEY не найден.\n"
            "Добавьте OPENAI_API_KEY в файл .env в корне проекта и повторите запуск.\n"
            "Пример: OPENAI_API_KEY=your_key_here"
        ),
        "creds.failed": "Ошибка: semantic-анализ не выполнен: {reason}",
        "creds.fallback_hint": (
            "Запустите с --allow-semantic-fallback, чтобы принять результат "
            "только локального анализа."
        ),
        "mode.title": "Режим анализа",
        "mode.local_only": "только локальный",
        "mode.local_and_semantic": "локальный + semantic",
        "mode.local_only_after_semantic_failure": "только локальный после сбоя semantic",
        "mode.banner": "НА ЭТИ ФОТОГРАФИИ НИКТО НЕ СМОТРЕЛ",
        "mode.banner_detail": (
            "Измерены только файлы. Без разбора содержания и художественного разбора "
            "ни один кадр не может попасть в лучшие, а сюжет каждого кадра "
            "неизвестен, а не «other»."
        ),
        "summary.technically_usable": "Технически пригодно, нужна проверка",
        "summary.fully_checked": "Полностью проверено и готово к экспорту",
        "summary.not_semantically_checked": "Не проверено semantic-анализом",
        "summary.technically_usable_help": (
            "Прошло технические пороги. Содержание, лица, логотипы и релизы могли "
            "не проверяться."
        ),
        "summary.fully_checked_help": (
            "Проверено полностью и готово к экспорту: содержание, лица, логотипы, "
            "релизы и метаданные. Требует semantic-анализа."
        ),
        "summary.release_status": "Статус релизов",
        "summary.release_unchecked": "не проверен",
        "summary.export_blocked_reason": (
            "Ничего не готово к экспорту, потому что содержание, лица, логотипы "
            "и метаданные не проверены."
        ),
        "plan.would_move": "Будет перемещено файлов: {count}, {mb} МБ:",
        "plan.nothing_to_move": "Перемещать нечего.",
        "plan.dry_run": "Ничего не перемещено. Запустите с --apply, чтобы выполнить.",
        "reason.unrecoverable": "неисправимые проблемы: {detail}",
        "reason.release_unchecked": (
            "статус релизов не проверен (semantic-анализ не выполнялся): коммерческий сток "
            "заблокирован, пока лица и товарные знаки не проверены"
        ),
        "reason.release_blocked": (
            "в кадре {present}: нужен релиз, поэтому коммерческий сток заблокирован"
        ),
        "reason.low_confidence": "уверенность {value} ниже {threshold}: решение за человеком",
        "reason.below_trash": (
            "реальный потенциал после обработки {value} ниже {threshold}, но ничего "
            "неисправимого в кадре нет: оставляем"
        ),
        "reason.stock_strong": "стоковый потенциал {value} — высокий",
        "reason.stock_standard": "стоковый потенциал {value} пригоден после обработки",
        "reason.hero_blocked": "не пропущен во флагманы: {detail}",
        "reason.flagship": (
            "потенциал для портфолио {value} проходит абсолютный порог и входит в топ "
            "своего жанра"
        ),
        "reason.archive": (
            "поправимо (потенциал {value}), но ниже стокового порога ({threshold}): "
            "оставить в архиве или решить вручную"
        ),
        "reason.duplicate_candidate": (
            "в этой группе есть более резкий кадр (на {margin} баллов выше); "
            "сравните их вручную"
        ),
        "reason.duplicate_unchecked": (
            "в этой группе есть более резкий кадр (на {margin} баллов выше), но содержание "
            "не проверялось — их различает только резкость, сравните вручную"
        ),
        "reason.short_clip": "длина всего {seconds} с: слишком коротко для большинства стоков",
        "sheet.candidate": "кандидат",
        "sheet.cluster_best": "лучший в группе",
        "sheet.margin": "разница",
        "sheet.semantic_ran": "semantic-анализ выполнен",
        "sheet.semantic_missing": "semantic-анализ НЕ выполнялся — проверьте вручную",
    },
}


def t(key: str, language: str = DEFAULT_LANGUAGE, **kwargs) -> str:
    """Look up a string, falling back to English and then to the key itself.

    Returning the key rather than raising is deliberate: a missing translation
    should degrade a label, not end a run that has already spent money on
    analysis.
    """
    catalogue = STRINGS.get(language) or STRINGS[DEFAULT_LANGUAGE]
    text = catalogue.get(key) or STRINGS[DEFAULT_LANGUAGE].get(key) or key
    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, IndexError):
            return text
    return text


def normalise(language: str | None) -> str:
    """Accepts 'ru', 'ru_RU', 'ru-RU'. Anything unknown becomes English."""
    if not language:
        return DEFAULT_LANGUAGE
    base = str(language).replace("_", "-").split("-")[0].lower()
    return base if base in SUPPORTED else DEFAULT_LANGUAGE


def missing_keys(language: str) -> set[str]:
    """Which English keys this language has not translated. Used by the tests."""
    return set(STRINGS[DEFAULT_LANGUAGE]) - set(STRINGS.get(language, {}))
