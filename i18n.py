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
