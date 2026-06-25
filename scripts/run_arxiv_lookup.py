"""Run a visible KamiKnows arXiv metadata lookup.

This script is intentionally separate from extraction. It only retrieves and
prints metadata from arXiv. It does not call an LLM, parse PDFs, or save a
training dataset.

Examples:

    python scripts/run_arxiv_lookup.py --id 2301.00001
    python scripts/run_arxiv_lookup.py --query 'cat:hep-ex AND calorimeter' --max-results 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the script runnable with: python scripts/run_arxiv_lookup.py
# without requiring an editable install during this early tutorial phase.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kamiknows.ingestion.arxiv_downloader import (  # noqa: E402
    ArxivIngestionError,
    fetch_arxiv_metadata_by_id,
    search_arxiv_metadata,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve minimal metadata from the arXiv API."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--id", help="arXiv ID, URL, or arXiv: identifier.")
    source.add_argument("--query", help="arXiv search query, e.g. 'cat:hep-ex'.")
    parser.add_argument(
        "--max-results",
        type=int,
        default=1,
        help="Maximum number of search results when using --query. Default: 1",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP timeout in seconds. Default: 30",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        if args.id:
            result: dict | list[dict] = fetch_arxiv_metadata_by_id(
                args.id,
                timeout_seconds=args.timeout,
            )
        else:
            result = search_arxiv_metadata(
                args.query,
                max_results=args.max_results,
                timeout_seconds=args.timeout,
            )
    except (ArxivIngestionError, RuntimeError, ValueError) as exc:
        print(f"KamiKnows arXiv lookup failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
