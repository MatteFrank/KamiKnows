import pytest
from kamiknows.rag.qa_grounded_v1 import build_grounded_qa_prompt_v1


def test_synthesis_changes_instructions_without_changing_schema_or_evidence():
    context={"question":"Compare the studies.","retrieved_chunks":[
        {"citation_label":"[C1]","chunk_id":"c","text":"Exact source text."}],
        "source_constraint":{"source_policy":"explicit_arxiv","requested_arxiv_ids":["1705.02355v2"],"status":"available"}}
    old=build_grounded_qa_prompt_v1(context)
    new=build_grounded_qa_prompt_v1(context,"synthesis_v2")
    marker="Return only valid JSON with this exact shape:"
    assert old.split(marker,1)[1]==new.split(marker,1)[1]
    assert 'Requested sources: 1705.02355v2' in new
    assert 'A comparison does not need to appear verbatim' in new
    assert 'If only part of the question is supported' in new
    assert 'Do not leave answer empty' in new
    assert old==build_grounded_qa_prompt_v1(context,"grounded_json_v1")
    with pytest.raises(ValueError):build_grounded_qa_prompt_v1(context,"unknown")
