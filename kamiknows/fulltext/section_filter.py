"""Optional experimental removal of acknowledgments and bibliography before chunking."""

import re


EXCLUDED_HEADINGS = {
    "acknowledgment", "acknowledgments", "acknowledgement", "acknowledgements",
    "bibliography", "references", "reference", "reference list", "literature cited",
}
# These are unexpanded bibliography-command arguments in the current parsed corpus.
# Match a complete trailing marker, never occurrences of 'references' in prose.
TRAILING_MARKER = re.compile(r"(?<!\S)(?:refs|biblio\s+spphys)\s*$")


def filter_sections(sections):
    """Return copies of retained sections and auditable exclusion records.

    Heading matching is exact after punctuation/number normalization. Bibliography
    embedded inside an acknowledgments section is removed with that section.
    Unlabelled bibliography inside scientific sections is not guessed from prose.
    """
    kept, audit = [], []
    for index, original in enumerate(sections):
        section = dict(original)
        heading = str(section.get("heading") or "")
        normalized = re.sub(r"[^a-z]+", " ", heading.casefold()).strip()
        identity = {"section_index": index, "section_id": section.get("section_id"),
                    "heading": heading}
        text = str(section.get("text") or "")
        if normalized in EXCLUDED_HEADINGS:
            audit.append({**identity, "action": "exclude_section",
                          "reason": "acknowledgments_or_bibliography_heading",
                          "removed_text": text, "removed_words": len(text.split())})
            continue
        match = TRAILING_MARKER.search(text)
        if match:
            section["text"] = text[:match.start()].rstrip()
            audit.append({**identity, "action": "remove_trailing_marker",
                          "reason": "unexpanded_bibliography_marker",
                          "removed_text": match.group(),
                          "removed_words": len(match.group().split())})
        kept.append(section)
    return kept, audit
