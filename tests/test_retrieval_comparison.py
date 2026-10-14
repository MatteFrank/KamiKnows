import pytest
from scripts.compare_retrievers_v1 import aggregate


def test_comparison_aggregation_excludes_abstentions_and_counts_all_source_hits():
    def record(expected,hit,rank,all_sources=False):
        return {'question_id':str(rank),'retrieval_eval':{'has_expected_source':expected,
                'expected_source_hit':hit,'expected_source_hit_rank':rank,
                'required_all_expected_sources':all_sources,
                'reciprocal_rank':1/rank if rank else 0}}
    summary=aggregate([record(True,True,4,True),record(True,False,None,True),
                       record(False,False,None)])
    assert summary['expected_source_questions']==2
    assert summary['expected_source_hit_rate_at_k']==.5
    assert summary['mrr']==pytest.approx(.125)
    assert summary['all_expected_sources_hit']=={'hits':1,'total':2}
