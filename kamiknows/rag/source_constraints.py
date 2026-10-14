"""Vincoli sulle fonti esplicitamente nominate nel testo di una domanda.

Questo modulo non usa expected_arxiv_ids, non calcola embedding e non chiama LLM.
Il chiamante estrae gli ID dalla domanda, filtra i candidati PRIMA del ranking e
infine bilancia i risultati. Senza ID espliciti resta disponibile tutto il corpus.
Le funzioni restituiscono copie o indici: non modificano i dati ricevuti.
"""

from __future__ import annotations

import re
from typing import Any


SOURCE_POLICY_VERSION = "explicit_arxiv_sources_v1"

# Formato moderno (1705.02355) e storico (hep-ex/9901001), con versione opzionale.
# Il riconoscimento è sintattico: non verifica online che il paper esista.
_ID = r"(?:\d{4}\.\d{4,5}|[a-z][a-z.-]*/\d{7})(?:v[1-9]\d*)?"
_REFERENCE = re.compile(
    rf"(?<![\w./])(?:arxiv\s*:\s*|https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/)?"
    rf"(?P<id>{_ID})(?:\.pdf)?(?![\w]|\.\d)",
    re.IGNORECASE,
)
_VERSION = re.compile(r"v[1-9]\d*$", re.IGNORECASE)


class SourceConstraintError(ValueError):
    """Il vincolo richiesto non può essere soddisfatto dai risultati disponibili."""


def plan_source_constraints(chunks, question_text, top_k, source_policy="none"):
    """Prepara filtro e diagnostica condivisi da confronto, QA e loader.

    Se manca anche una sola fonte richiesta, non forniamo un contesto parziale
    come se bastasse: il record resta presente con contesto vuoto e stato esplicito.
    """
    if source_policy not in {"none", "explicit_arxiv"}:
        raise SourceConstraintError(f"Unknown source policy: {source_policy}")
    if top_k < 1:
        raise SourceConstraintError("top_k must be positive")
    ids = extract_requested_arxiv_ids(question_text) if source_policy == "explicit_arxiv" else []
    diagnostics = source_constraint_diagnostics(chunks, ids)
    diagnostics["source_policy"] = source_policy
    diagnostics["top_k"] = top_k
    if source_policy == "none":
        diagnostics["policy"] = "none"
    if len(ids) > top_k:
        diagnostics["status"] = "source_budget_insufficient"
    indices = filter_candidate_indices(chunks, ids)
    if diagnostics["status"] != "available":
        indices = []
    diagnostics["expected_result_count"] = min(top_k, len(indices))
    return indices, ids, diagnostics


def _normalize_id(value: str) -> str:
    """Accetta un ID, un prefisso arXiv o un URL abs/pdf completo."""
    match = _REFERENCE.fullmatch(value.strip())
    if match is None:
        raise SourceConstraintError(f"Invalid arXiv reference: {value!r}")
    return match.group("id").lower()


def _requested_ids(values: list[str]) -> list[str]:
    # Elimina duplicati senza cambiare l'ordine. Un ID senza versione include già
    # tutte le versioni: non richiediamo due posti per paper e versione dello stesso.
    ids = list(dict.fromkeys(_normalize_id(value) for value in values))
    unversioned = {value for value in ids if not _VERSION.search(value)}
    return [value for value in ids
            if not _VERSION.search(value) or _VERSION.sub("", value) not in unversioned]


def extract_requested_arxiv_ids(question_text: str) -> list[str]:
    """Estrae ID dal testo della domanda, NON dalle etichette del benchmark.

    Esempio: 'Compare arXiv:1705.02355v2 and 2106.05285v3' restituisce due ID.
    Un nome come 'CaloGAN' da solo non viene interpretato come un vincolo: per
    associare nomi e fonti servirebbe una politica distinta e non ambigua.
    """
    return _requested_ids([match.group("id") for match in _REFERENCE.finditer(question_text)])


def _matches(chunk: dict[str, Any], requested_id: str) -> bool:
    """Versione esplicita: corrispondenza esatta. Senza versione: stesso paper."""
    # paper_id è un identificativo interno: non lo usiamo come sostituto di arxiv_id.
    try:
        source_id = _normalize_id(str(chunk.get("arxiv_id") or ""))
    except SourceConstraintError:
        return False
    if _VERSION.search(requested_id):
        return source_id == requested_id
    return _VERSION.sub("", source_id) == requested_id


def filter_candidate_indices(
    chunks: list[dict[str, Any]], requested_ids: list[str],
) -> list[int]:
    """Restituisce gli indici ammessi, mantenendo l'ordine originale.

    Applicare questi STESSI indici a chunks e chunk_vectors evita di associare
    un testo all'embedding di un altro testo. Una fonte assente non attiva mai
    un fallback al corpus completo; con zero fonti richieste, invece, ammette tutto.
    """
    ids = _requested_ids(requested_ids)
    if not ids:
        return list(range(len(chunks)))
    return [index for index, chunk in enumerate(chunks)
            if any(_matches(chunk, source) for source in ids)]


def source_constraint_diagnostics(
    chunks: list[dict[str, Any]], requested_ids: list[str],
) -> dict[str, Any]:
    """Descrive i candidati disponibili per manifest, log e gestione delle assenze."""
    ids = _requested_ids(requested_ids)
    counts = {source: sum(_matches(chunk, source) for chunk in chunks) for source in ids}
    missing = [source for source, count in counts.items() if count == 0]
    return {
        "policy": SOURCE_POLICY_VERSION,
        "mode": "requested_sources" if ids else "unrestricted",
        "requested_arxiv_ids": ids,
        "candidate_count": len(filter_candidate_indices(chunks, ids)),
        "candidate_counts_by_source": counts,
        "missing_requested_sources": missing,
        "status": "requested_source_unavailable" if missing else "available",
    }


def select_balanced_results(
    ranked_results: list[dict[str, Any]], requested_ids: list[str], top_k: int,
) -> list[dict[str, Any]]:
    """Seleziona risultati già ordinati dal retriever, dal migliore al peggiore.

    Con più fonti, riserva il miglior chunk di ciascuna e riempie i posti restanti
    seguendo il ranking. Poi ripristina l'ordine originale tra i chunk selezionati
    e riassegna ranghi/citazioni. Non confronta punteggi TF-IDF con punteggi MiniLM.

    Passare una lista di candidati sufficientemente ampia, NON soltanto il top-k
    globale: una fonte richiesta potrebbe trovarsi oltre quei primi risultati.
    Se manca una fonte o top_k è insufficiente, solleva un errore esplicito che il
    chiamante dovrà registrare. Non inventa passaggi per riempire i posti mancanti.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise SourceConstraintError("top_k must be a positive integer")
    ids = _requested_ids(requested_ids)
    if len(ids) > top_k:
        raise SourceConstraintError("top_k is smaller than the number of requested sources")

    candidates = []
    seen = set()
    for result in ranked_results:
        if ids and not any(_matches(result, source) for source in ids):
            continue
        chunk_id = result.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise SourceConstraintError("ranked results must contain nonempty chunk_id values")
        if chunk_id not in seen:
            candidates.append(result)
            seen.add(chunk_id)

    selected = set()
    for source in ids:
        best = next((i for i, result in enumerate(candidates) if _matches(result, source)), None)
        if best is None:
            raise SourceConstraintError(f"Requested source absent from ranked candidates: {source}")
        selected.add(best)
    for index in range(len(candidates)):
        if len(selected) >= top_k:
            break
        selected.add(index)

    output = []
    for rank, index in enumerate(sorted(selected), start=1):
        # Copia superficiale: modifichiamo solo campi scalari, non il testo o i metadati.
        result = dict(candidates[index])
        result.update(rank=rank, citation_id=f"C{rank}", citation_label=f"[C{rank}]")
        if "retrieved_rank" in result:
            result["retrieved_rank"] = rank
        output.append(result)
    return output
