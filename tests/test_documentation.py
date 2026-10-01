from pathlib import Path
import re
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN = sorted(
    p
    for p in ROOT.rglob("*.md")
    if not any(
        part.startswith(".") or part in {"runs", "build"} for part in p.relative_to(ROOT).parts
    )
)
LINK = re.compile(r"\]\(([^)\s]+)\)|srcset=\"([^\"]+)\"|src=\"([^\"]+)\"")


@pytest.mark.parametrize("path", MARKDOWN, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_local_links_resolve(path):
    for match in LINK.finditer(path.read_text(encoding="utf-8")):
        target = next(group for group in match.groups() if group)
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        assert (path.parent / target.split("#")[0]).exists(), f"{path.name}: {target}"


def test_every_figure_has_light_and_dark_variants():
    figures = ROOT / "datasets" / "figures"
    svgs = list(figures.rglob("*.svg"))
    assert len(svgs) >= 16
    for svg in svgs:
        stem, theme = svg.stem.rsplit("-", 1)
        other = "dark" if theme == "light" else "light"
        assert theme in {"light", "dark"} and (svg.parent / f"{stem}-{other}.svg").is_file()
        tags = {element.tag.rsplit("}", 1)[-1] for element in ET.parse(svg).getroot().iter()}
        assert not tags & {"image", "foreignObject"}, f"{svg.name} is not fully vector"


def test_readme_pictures_offer_both_themes():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for picture in re.findall(r"<picture>(.*?)</picture>", readme, flags=re.S):
        assert "prefers-color-scheme: dark" in picture and "prefers-color-scheme: light" in picture


def test_public_tree_passes_privacy_audit():
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from verify import audit, checksums

    findings, _ = audit(ROOT)
    assert findings == []
    import json

    assert checksums(ROOT) == json.loads((ROOT / "datasets/checksums.json").read_text())


def test_privacy_audit_detects_planted_values(tmp_path):
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from verify import audit

    private = "/" + "Users/someone/project"
    temp = "/" + "var/folders/ab/xyz"
    email = "person" + "@" + "example.org"
    (tmp_path / "notes.md").write_text(f"{private}\n{temp}\n{email}\n")
    findings, scanned = audit(tmp_path)
    assert scanned == 1
    assert {f["category"] for f in findings} == {
        "private-home-path",
        "machine-temp-path",
        "email-address",
    }
