"""Verifica i vincoli sulla domanda con ranking deliberatamente fuorviante."""
import copy
import pytest

from kamiknows.rag.source_constraints import (
    extract_requested_arxiv_ids, filter_candidate_indices, plan_source_constraints,
    select_balanced_results, SourceConstraintError,
)
from kamiknows.rag.retriever_v1 import TfidfRetrieverV1
from kamiknows.rag.qa_grounded_v1 import retrieve_contexts_v1, build_grounded_qa_prompt_v1
from scripts.compare_retrievers_v1 import retrieve_for_question


def corpus():
    return [{'chunk_id': str(i), 'arxiv_id': source, 'paper_id': source,
             'title': 'Simulation', 'text': text, 'section_heading': 'Method'}
            for i, (source, text) in enumerate([
                ('1807.01954v2', 'WGAN shower simulation'),
                ('1807.01954v2', 'WGAN shower simulation fast'),
                ('1705.02355v2', 'CaloGAN model')])]


def test_extract_versions_urls_and_no_names_as_hidden_constraints():
    assert extract_requested_arxiv_ids('arXiv:1705.02355v2 and https://arxiv.org/pdf/2106.05285v3.pdf') == ['1705.02355v2','2106.05285v3']
    assert extract_requested_arxiv_ids('1705.02355, 1705.02355v2, hep-ex/9901001') == ['1705.02355','hep-ex/9901001']
    assert extract_requested_arxiv_ids('What is CaloGAN?') == []
    assert filter_candidate_indices(corpus(), ['1705.02355']) == [2]
    assert filter_candidate_indices(corpus(), ['1705.02355v1']) == []


def test_balancing_preserves_ranking_and_inputs():
    chunks=corpus(); before=copy.deepcopy(chunks)
    selected=select_balanced_results(chunks,['1807.01954v2','1705.02355v2'],2)
    assert [c['chunk_id'] for c in selected] == ['0','2']
    assert selected[1]['citation_label'] == '[C2]'
    assert chunks == before
    with pytest.raises(SourceConstraintError):
        select_balanced_results(chunks,['1807.01954v2','1705.02355v2'],1)


def test_tfidf_mask_keeps_global_scores():
    retriever=TfidfRetrieverV1(corpus())
    all_results=retriever.retrieve('WGAN simulation',top_k=3)
    filtered=retriever.retrieve('WGAN simulation',top_k=3,candidate_indices=[2])
    assert filtered[0].score == next(r.score for r in all_results if r.chunk['chunk_id']=='2')
    assert retriever.retrieve('simulation',candidate_indices=[]) == []


@pytest.mark.parametrize('text,expected',[
    ('According to arXiv:1705.02355v2, what model?', ['1705.02355v2']),
    ('Compare 1705.02355v2 and 1807.01954v2', ['1807.01954v2','1705.02355v2']),
    ('Which models?', ['1807.01954v2','1807.01954v2']),
])
def test_comparison_filters_both_rankers_before_selection(text,expected):
    chunks=corpus()
    # La fonte richiesta ha un punteggio dense basso: deve comunque restare ammessa.
    question={'question':text,'question_type':'method','expected_arxiv_ids':['9999.99999']}
    results,diag=retrieve_for_question(chunks=chunks,vectors=[[1.,0.],[.9,.1],[0.,1.]],
        tfidf=TfidfRetrieverV1(chunks),question=question,query_vector=[1.,0.],
        top_k=2,source_policy='explicit_arxiv')
    assert [c['arxiv_id'] for c in results['minilm']] == expected
    if len(diag['requested_arxiv_ids']) == 1:
        assert [c['arxiv_id'] for c in results['tfidf']] == expected
    if len(diag['requested_arxiv_ids']) == 2:
        assert {c['arxiv_id'] for c in results['tfidf']} == set(expected)


@pytest.mark.parametrize('text,k,status',[
    ('Use 1705.02355v1',5,'requested_source_unavailable'),
    ('Use 1705.02355v2 and 1807.01954v2',1,'source_budget_insufficient'),
])
def test_unavailable_sources_produce_explicit_empty_context(text,k,status):
    question={'question_id':'q','question':text,'question_type':'method'}
    contexts=retrieve_contexts_v1(chunks=corpus(),eval_set=[question],top_k=k,source_policy='explicit_arxiv')
    assert contexts[0]['retrieved_chunks'] == []
    assert contexts[0]['source_constraint']['status'] == status
    assert 'Set answer_status to insufficient_evidence' in build_grounded_qa_prompt_v1(contexts[0])
