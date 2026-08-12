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
        "mode.banner": "SEMANTIC ANALYSIS DID NOT RUN",
        "mode.banner_detail": (
            "Content, faces, logos and release status were not checked. Genres are "
            "unknown, not 'other'."
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
        "mode.banner": "SEMANTIC-АНАЛИЗ НЕ ВЫПОЛНЯЛСЯ",
        "mode.banner_detail": (
            "Содержание, лица, логотипы и статус релизов не проверены. "
            "Жанр неизвестен, а не «other»."
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
