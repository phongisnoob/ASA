"""Documentation must match the implementation: commands parse, links resolve, catalogues are complete."""

import re
import shlex
from pathlib import Path

import pytest

from asa_m2.cli import build_parser

ROOT = Path(__file__).resolve().parents[2]
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]


def fenced_blocks(text: str):
    for m in re.finditer(r"```(\w*)\n(.*?)```", text, re.DOTALL):
        yield m.group(1), m.group(2)


def documented_commands() -> list[tuple[str, str]]:
    found = []
    for path in DOCS:
        for lang, body in fenced_blocks(path.read_text(encoding="utf-8")):
            if lang not in {"bash", "sh", "shell", "console"}:
                continue
            joined = re.sub(r"\\\n\s*", " ", body)
            for line in joined.splitlines():
                line = line.split("  #")[0].strip()
                if line.startswith("asa-m2 "):
                    found.append((path.name, line.split(" > ")[0].strip()))  # drop shell redirection only
    return found


COMMANDS = documented_commands()


def test_the_docs_actually_contain_commands():
    assert len(COMMANDS) >= 20, "command extraction found too few examples; the regex or the docs changed"
    assert {c.split()[1] for _, c in COMMANDS} >= {"check", "g2p", "prepare-corpus", "validate-corpus", "align-mfa",
                                                    "inspect-textgrid", "export", "evaluate-boundaries"}


@pytest.mark.parametrize("source,command", COMMANDS)
def test_every_documented_command_parses_against_the_real_cli(source, command):
    argv = shlex.split(command)[1:]
    ns = build_parser().parse_args(argv)  # raises SystemExit(2) on an unknown command/option/missing argument
    assert callable(ns.func), f"{source}: {command}"


def test_prose_references_to_commands_exist():
    sub = next(a for a in build_parser()._actions if a.dest == "command").choices
    for path in DOCS:
        for name in re.findall(r"`asa-m2 ([a-z][a-z-]+)", path.read_text(encoding="utf-8")):
            assert name in sub, f"{path.name} mentions a non-existent command `asa-m2 {name}`"


def slug(heading: str) -> str:
    s = heading.strip().lower()
    s = re.sub(r"[^\w\- ]", "", s)
    return s.replace(" ", "-")


@pytest.mark.parametrize("path", DOCS, ids=lambda p: p.name)
def test_relative_links_and_anchors_resolve(path):
    text = path.read_text(encoding="utf-8")
    anchors = {slug(m.group(2)) for m in re.finditer(r"^(#{1,6})\s+(.*)$", text, re.MULTILINE)}
    for target in re.findall(r"\]\(([^)\s]+)\)", text):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        file_part, _, anchor = target.partition("#")
        if file_part:
            resolved = (path.parent / file_part).resolve()
            assert resolved.exists(), f"{path.name}: broken link {target}"
            if anchor and resolved.suffix == ".md":
                other = {slug(m.group(2)) for m in re.finditer(r"^(#{1,6})\s+(.*)$", resolved.read_text(encoding="utf-8"), re.MULTILINE)}
                assert anchor in other, f"{path.name}: anchor #{anchor} not found in {file_part}"
        else:
            assert anchor in anchors, f"{path.name}: anchor #{anchor} not found"


def test_documented_file_paths_exist():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for rel in set(re.findall(r"(?:examples|scripts|configs|docs|tests)/[\w./-]+\.(?:py|csv|txt|dict|yaml|json|md)", text)):
        assert (ROOT / rel).exists(), f"README mentions missing file {rel}"


def test_quality_flag_catalogue_matches_source():
    codes: set[str] = set()
    for p in (ROOT / "src" / "asa_m2").rglob("*.py"):
        s = p.read_text(encoding="utf-8")
        codes |= set(re.findall(r'(?:_flag\(\s*|code=)"([a-z_]+)"', s))
    codes |= {"no_textgrid_produced", "textgrid_unusable"}
    doc = (ROOT / "docs" / "data_contract.md").read_text(encoding="utf-8")
    missing = sorted(c for c in codes if f"`{c}`" not in doc and c not in {"phone_too_short", "phone_too_long", "word_too_short", "word_too_long"})
    assert not missing, f"flag codes missing from docs/data_contract.md: {missing}"
    for c in ("phone_too_short/long", "word_too_short/long"):
        assert c in doc


def test_exit_codes_documented_match_code():
    from asa_m2 import exceptions as ex
    text = (ROOT / "docs" / "troubleshooting.md").read_text(encoding="utf-8")
    for const, code in (("EXIT_CONFIG", 3), ("EXIT_DATA", 4), ("EXIT_EXTERNAL", 5), ("EXIT_ALIGNMENT", 6), ("EXIT_OPTIONAL_DEP", 7)):
        assert getattr(ex, const) == code
        assert re.search(rf"^\| {code} \|", text, re.MULTILINE), f"exit code {code} missing from troubleshooting table"


def test_readme_reports_numbers_that_the_build_report_will_also_use():
    """The README's headline test count must be updated by the build (see scripts/count_tests.py)."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert re.search(r"\*\*\d+ passed\*\*", text)
