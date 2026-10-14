import hashlib
import json
import pytest
from kamiknows.rag.saved_retrieval import load_saved_contexts
from kamiknows.rag.load_dataset import RagDatasetError


def fixture(tmp_path):
    chunk={'chunk_id':'c','paper_id':'p','arxiv_id':'a','text':'Full original evidence.'}
    question={'question_id':'q','question':'What?','question_type':'method',
              'expected_behavior':'answerable','expected_arxiv_ids':['a']}
    cp=tmp_path/'chunks_v1.jsonl';cp.write_text(json.dumps(chunk)+'\n')
    qp=tmp_path/'questions.jsonl';qp.write_text(json.dumps(question)+'\n')
    rp=tmp_path/'minilm_retrieval_v1.jsonl'
    row={'question_id':'q','question':'What?','retrieved_chunks':[
        {'chunk_id':'c','paper_id':'p','arxiv_id':'a','rank':1,'score':.7,'text_preview':'Full'}]}
    rp.write_text(json.dumps(row)+'\n')
    manifest={'rag_ready_dir':str(tmp_path),'questions_file':str(qp),'top_k':1,
              'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [cp,qp]}}
    (tmp_path/'retrieval_comparison_manifest_v1.json').write_text(json.dumps(manifest))
    return rp,dict(dataset={'paths':{'chunks':str(cp)},'chunks':[chunk]},
                   eval_set=[question],questions_file=qp,top_k=1)


def test_hydrates_full_text_labels_and_source_metadata(tmp_path):
    path,args=fixture(tmp_path);rows,provenance=load_saved_contexts(path,**args)
    assert rows[0]['retrieved_chunks'][0]['text']=='Full original evidence.'
    assert rows[0]['retrieved_chunks'][0]['citation_label']=='[C1]'
    assert rows[0]['retriever']['name']=='minilm_local_v1'
    assert rows[0]['retrieval_eval']['expected_source_hit']
    assert provenance['sha256']


@pytest.mark.parametrize('change',['hash','id','question','rank'])
def test_rejects_mismatched_saved_inputs(tmp_path,change):
    path,args=fixture(tmp_path)
    if change=='hash': args['questions_file'].write_text('{}')
    else:
        r=json.loads(path.read_text())
        if change=='id':r['retrieved_chunks'][0]['chunk_id']='unknown'
        if change=='question':r['question']='Different'
        if change=='rank':r['retrieved_chunks'][0]['rank']=2
        path.write_text(json.dumps(r))
    with pytest.raises(RagDatasetError):load_saved_contexts(path,**args)


@pytest.mark.parametrize('mode',['short','missing','tampered','policy_mismatch'])
def test_constrained_saved_contexts_validate_short_empty_and_tampered_records(tmp_path,mode):
    from kamiknows.rag.source_constraints import plan_source_constraints, SOURCE_POLICY_VERSION
    path,args=fixture(tmp_path)
    chunk=args['dataset']['chunks'][0];chunk['arxiv_id']='1705.02355v2'
    q=args['eval_set'][0];q['question']='According to arXiv:1705.02355v2, what model?'
    if mode=='missing':q['question']='According to arXiv:1705.02355v1, what model?'
    args['top_k']=5
    cp=tmp_path/'chunks_v1.jsonl';cp.write_text(json.dumps(chunk)+'\n')
    args['questions_file'].write_text(json.dumps(q)+'\n')
    _,_,diag=plan_source_constraints([chunk],q['question'],5,'explicit_arxiv')
    row=json.loads(path.read_text());row['question']=q['question'];row['source_constraint']=diag
    row['retrieved_chunks'][0]['arxiv_id']=chunk['arxiv_id']
    if mode=='missing':row['retrieved_chunks']=[]
    if mode=='tampered':row['source_constraint']['candidate_count']=42
    path.write_text(json.dumps(row)+'\n')
    manifest={'rag_ready_dir':str(tmp_path),'questions_file':str(args['questions_file']),
              'top_k':5,'source_policy':'explicit_arxiv','source_policy_version':SOURCE_POLICY_VERSION,
              'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [cp,args['questions_file']]}}
    (tmp_path/'retrieval_comparison_manifest_v1.json').write_text(json.dumps(manifest))
    if mode=='policy_mismatch':args['source_policy']='none'
    if mode in {'tampered','policy_mismatch'}:
        with pytest.raises(RagDatasetError):load_saved_contexts(path,**args)
    else:
        contexts,provenance=load_saved_contexts(path,**args)
        assert len(contexts[0]['retrieved_chunks']) == (0 if mode=='missing' else 1)
        assert provenance['source_policy']=='explicit_arxiv'
