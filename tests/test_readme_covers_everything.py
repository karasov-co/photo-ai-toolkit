"""The README must document every command and every flag the CLI accepts.

Asserted rather than promised. A README goes stale the moment a flag is added
and nobody notices, and the first person to notice is a stranger who tried the
flag that was not there. This test is the only thing standing between the two.
"""

from __future__ import annotations

import argparse
import pathlib

import cli

README = pathlib.Path(__file__).resolve().parent.parent / "README.md"


def _subcommands():
    parser = cli.build_parser()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return parser, dict(action.choices)
    raise AssertionError("the CLI has no subcommands any more")


def test_every_command_is_in_the_readme():
    _, commands = _subcommands()
    text = README.read_text(encoding="utf-8")
    missing = [name for name in commands if f"`{name}`" not in text]
    assert not missing, f"undocumented commands: {missing}"


def test_every_flag_is_in_the_readme():
    parser, commands = _subcommands()
    text = README.read_text(encoding="utf-8")

    missing = []
    for name, sub in commands.items():
        for action in sub._actions:
            for option in action.option_strings:
                if option in ("-h", "--help"):
                    continue
                if option not in text:
                    missing.append(f"{name} {option}")
    for action in parser._actions:
        for option in action.option_strings:
            if option not in ("-h", "--help") and option not in text:
                missing.append(f"(global) {option}")

    assert not missing, f"undocumented flags: {sorted(set(missing))}"


def test_the_readme_does_not_document_flags_that_do_not_exist():
    """The other direction. A flag that was removed and left in the README
    sends somebody to a usage error with a copied command."""
    parser, commands = _subcommands()
    real = {"-h", "--help", "--version"}
    for sub in list(commands.values()) + [parser]:
        for action in sub._actions:
            real.update(action.option_strings)

    text = README.read_text(encoding="utf-8")
    # Only inside backticks: prose like "--apply is the only thing" is fine,
    # a code-formatted flag is a claim that you can type it.
    import re

    claimed = set(re.findall(r"`(--[a-z0-9-]+)[^`]*`", text))
    invented = sorted(flag for flag in claimed if flag not in real)
    assert not invented, f"the README documents flags the CLI does not have: {invented}"


def test_the_screenshot_is_actually_there():
    """A README whose first image 404s on GitHub is worse than no image."""
    text = README.read_text(encoding="utf-8")
    import re

    for path in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text):
        if path.startswith("http"):
            continue
        assert (README.parent / path).is_file(), f"missing image: {path}"


def test_the_price_in_the_readme_matches_the_pricing_file():
    """The two used to disagree by a factor of three, which is how a quote
    becomes a complaint."""
    import re

    import llm_provider

    entry = llm_provider.load_pricing()["models"]["grok-4.6"]
    # Whitespace-normalised: the price is bold and the line wraps between the
    # number and the unit, which is a formatting detail and not a claim.
    text = re.sub(r"\s+", " ", README.read_text(encoding="utf-8"))
    assert f"${entry['usd_per_100_photos']:.2f} per 100 photographs" in text


# --- prose against code, everywhere it can be checked -------------------------
#
# The commit that added the tests above was about a docstring that had gone
# stale. Three days later scoring.py's own docstring still described the
# editorial branch, the release logic and a `legal_readiness` dimension, all of
# which had been deleted. One module having a drift test is not a policy.

REMOVED_VOCABULARY = (
    "legal_readiness",
    "model release",
    "editorial-only",
    "editorial only",
    "cannot reach commercial stock",
)


def _module_docstrings():
    import importlib
    import pkgutil

    root = pathlib.Path(__file__).resolve().parent.parent
    for path in sorted(root.glob("*.py")):
        if path.stem in ("conftest", "setup"):
            continue
        try:
            module = importlib.import_module(path.stem)
        except Exception:  # pragma: no cover - a module that needs an argument
            continue
        if module.__doc__:
            yield path.name, module.__doc__
    del pkgutil


def test_no_module_docstring_still_describes_the_release_logic():
    offenders = []
    for name, doc in _module_docstrings():
        lowered = doc.lower()
        for phrase in REMOVED_VOCABULARY:
            # An explicit "this is gone" sentence is the point of keeping the
            # history in the prose; a description in the present tense is not.
            if phrase in lowered and not any(
                marker in lowered for marker in ("used to", "are all\ngone", "is gone", "no longer")
            ):
                offenders.append(f"{name}: {phrase}")
    assert not offenders, f"docstrings describe deleted behaviour: {offenders}"


def test_the_scoring_docstring_lists_the_dimensions_that_exist():
    import scoring

    doc = scoring.__doc__
    fields = set(scoring.AssetScores().to_dict())
    listed = {line.split()[1] for line in doc.splitlines()
              if line.startswith("    ") and len(line.split()) > 2 and len(line.split()[0]) == 1}
    assert listed == fields, f"docstring lists {listed - fields}, missing {fields - listed}"
    assert f"{len(fields)} numbers per asset" in doc.replace("Ten", "10").replace("Nine", "9")
