"""Download reference data. Currently: the go-basic ontology.

UniProt/GOA training-label downloads are intentionally not automated yet — the
exact organism scope and evidence-code filtering are decisions to make against
docs/02-data-sources.md before pulling. This module handles what is unambiguous.
"""

from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path

from viral_annotation.config import DATA_DIR, GO_BASIC_URL, GO_OBO_PATH

# Some mirrors (e.g. current.geneontology.org) reject the default urllib
# User-Agent with HTTP 403, so we send an explicit one.
_USER_AGENT = "viral-annotation/0.0.1 (SBIR DPA26BZ03-DV014; +https://www.uniprot.org)"


def _fetch(url: str, dest: Path) -> None:
    """Stream a URL to disk with an explicit User-Agent."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out)


def download_go_basic(dest: Path = GO_OBO_PATH, force: bool = False) -> Path:
    """Fetch go-basic.obo to `dest`. No-op if it already exists unless force."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"[skip] {dest} already exists ({dest.stat().st_size:,} bytes). "
              f"Use force=True to re-download.")
        return dest
    print(f"[get ] {GO_BASIC_URL}\n       -> {dest}")
    _fetch(GO_BASIC_URL, dest)
    print(f"[done] {dest.stat().st_size:,} bytes")
    return dest


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (`va-download-go` / `python -m ...download`)."""
    argv = sys.argv[1:] if argv is None else argv
    force = "--force" in argv
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_go_basic(force=force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
