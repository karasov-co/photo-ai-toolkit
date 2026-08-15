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
