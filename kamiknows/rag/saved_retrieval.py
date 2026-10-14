"""Validate comparison artifacts and hydrate full QA evidence without reranking."""
import hashlib
from pathlib import Path
from kamiknows.rag.load_dataset import read_json, read_jsonl, RagDatasetError


def load_saved_contexts(path, *, dataset, eval_set, questions_file, top_k, source_policy=None):
    from kamiknows.rag.qa_grounded_v1 import build_retrieved_qa_chunk, evaluate_retrieval_for_question
    path=Path(path)
    names={'minilm_retrieval_v1.jsonl':'minilm_local_v1','tfidf_retrieval_v1.jsonl':'tfidf_local_v1'}
    if path.name not in names:
        raise RagDatasetError('expected a retrieval JSONL from compare_retrievers_v1.py')
    manifest_path=path.parent/'retrieval_comparison_manifest_v1.json'
    manifest=read_json(manifest_path)
    from kamiknows.rag.source_constraints import plan_source_constraints, select_balanced_results, SOURCE_POLICY_VERSION
    saved_policy = manifest.get('source_policy', 'none')
    if saved_policy not in {'none', 'explicit_arxiv'} or (source_policy is not None and source_policy != saved_policy):
        raise RagDatasetError('saved retrieval source policy mismatch or unknown policy')
    if saved_policy == 'explicit_arxiv' and manifest.get('source_policy_version') != SOURCE_POLICY_VERSION:
        raise RagDatasetError('unknown saved source policy version')
    if questions_file is None or manifest['top_k']!=top_k:
        raise RagDatasetError('saved retrieval requires matching explicit questions file and top-k')
    chunk_path=Path(dataset['paths']['chunks'])
    for file,original in [(chunk_path,Path(manifest['rag_ready_dir'])/'chunks_v1.jsonl'),
                          (Path(questions_file),Path(manifest['questions_file']))]:
        expected=manifest['sha256'].get(str(original))
        if expected!=hashlib.sha256(file.read_bytes()).hexdigest():
            raise RagDatasetError(f'saved retrieval input hash mismatch: {file}')
    rows=read_jsonl(path)
    lookup={r['question_id']:r for r in rows}
    if len(lookup)!=len(rows) or set(lookup)!={q['question_id'] for q in eval_set}:
        raise RagDatasetError('saved retrieval question IDs must match benchmark exactly')
    chunks={c['chunk_id']:c for c in dataset['chunks']}
    if len(chunks)!=len(dataset['chunks']):raise RagDatasetError('duplicate dataset chunk IDs')
    retriever={'name':names[path.name],'version':'saved_retrieval_v1'}
    contexts=[]
    for question in eval_set:
        row=lookup[question['question_id']]
        if row['question']!=question['question']:
            raise RagDatasetError('saved question text does not match benchmark')
        items=row['retrieved_chunks']
        indices, ids, diagnostics = plan_source_constraints(dataset['chunks'], question['question'], top_k, saved_policy)
        if saved_policy == 'explicit_arxiv' and row.get('source_constraint') != diagnostics:
            raise RagDatasetError('saved source constraint diagnostics do not match question/dataset')
        if len(items)!=diagnostics['expected_result_count'] or [c['rank'] for c in items]!=list(range(1,len(items)+1)):
            raise RagDatasetError('saved ranks/count do not match top-k')
        if len({c['chunk_id'] for c in items})!=len(items):raise RagDatasetError('duplicate retrieved chunk')
        hydrated=[]
        allowed = {dataset['chunks'][i]['chunk_id'] for i in indices}
        for item in items:
            chunk=chunks.get(item['chunk_id'])
            if chunk is None or any(item.get(k)!=chunk.get(k) for k in ['paper_id','arxiv_id']):
                raise RagDatasetError('saved chunk IDs or source identifiers mismatch')
            if item['chunk_id'] not in allowed:
                raise RagDatasetError('saved chunk violates requested source constraint')
            hydrated.append(build_retrieved_qa_chunk(chunk=chunk,rank=item['rank'],score=item['score']))
        if hydrated and ids:
            try:
                select_balanced_results(hydrated, ids, top_k)
            except ValueError as exc:
                raise RagDatasetError('saved results omit a requested source') from exc
        contexts.append({'question_id':question['question_id'],'question':question['question'],
                         'question_type':question['question_type'],'expected_behavior':question['expected_behavior'],
                         'expected_arxiv_ids':question.get('expected_arxiv_ids',[]),
                         'expected_paper_ids':question.get('expected_paper_ids',[]),'top_k':top_k,
                         'retriever':retriever,'retrieved_chunks':hydrated,
                         'source_constraint':diagnostics,
                         'retrieval_eval':evaluate_retrieval_for_question(question=question,retrieved_chunks=hydrated)})
    provenance={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'source_policy':saved_policy,
                'comparison_manifest':manifest,'comparison_manifest_path':str(manifest_path)}
    return contexts,provenance
