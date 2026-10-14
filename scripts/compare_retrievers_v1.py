"""Compare unchanged TF-IDF and local MiniLM with shared source-level evaluation."""
import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kamiknows.rag.embedding_registry import get_embedding_model_config
from kamiknows.rag.embedding_selection import EmbeddingEncoder, build_embedding_text, retrieve_embedding_chunks, compute_embedding_metrics
from kamiknows.rag.load_dataset import read_jsonl
from kamiknows.rag.outputs import write_json, write_jsonl
from kamiknows.rag.qa_grounded_v1 import load_eval_questions_from_file, build_retrieved_qa_chunk
from kamiknows.rag.reranker_v1 import evaluate_retrieval
from kamiknows.rag.retriever_v1 import TfidfRetrieverV1, QUESTION_TYPE_EXPANSIONS
from kamiknows.rag_ready.semantic_dataset import encode_token_windows
from kamiknows.rag.source_constraints import plan_source_constraints, select_balanced_results


def retrieve_for_question(*, chunks, vectors, tfidf, question, query_vector, top_k, source_policy='none'):
    """Filtra prima del ranking; mantiene IDF globale e allineamento chunk/vettori."""
    indices, ids, diagnostics = plan_source_constraints(chunks, question['question'], top_k, source_policy)
    if not indices:
        return {'tfidf': [], 'minilm': []}, diagnostics
    # Ordiniamo tutti i candidati ammessi: ciascuna fonte può trovarsi oltre top-k.
    count = len(indices) if ids else top_k
    lexical = [build_retrieved_qa_chunk(chunk=r.chunk,rank=r.rank,score=r.score)
               for r in tfidf.retrieve(question['question'], top_k=count,
                                      question_type=question['question_type'],
                                      expected_section_keywords=question.get('expected_section_keywords', []),
                                      candidate_indices=indices)]
    dense = retrieve_embedding_chunks(query_vector=query_vector,
                                      chunk_vectors=[vectors[i] for i in indices],
                                      chunks=[chunks[i] for i in indices], top_k=count)
    return {name: select_balanced_results(rows, ids, top_k)
            for name, rows in [('tfidf', lexical), ('minilm', dense)]}, diagnostics


def aggregate(records):
    result=compute_embedding_metrics(records)
    multi=[r for r in records if r['retrieval_eval'].get('required_all_expected_sources')]
    result['all_expected_sources_hit']={'hits':sum(r['retrieval_eval']['expected_source_hit'] for r in multi),'total':len(multi)}
    return result


def run(args):
    if args.output_dir.exists(): raise FileExistsError(f'output exists: {args.output_dir}')
    if args.top_k<1: raise ValueError('top-k must be positive')
    source_policy = getattr(args, 'source_policy', 'none')
    if source_policy not in {'none', 'explicit_arxiv'}:
        raise ValueError('invalid source policy')
    chunks=read_jsonl(args.rag_ready_dir/'chunks_v1.jsonl')
    questions=load_eval_questions_from_file(args.eval_questions_file)
    tfidf=TfidfRetrieverV1(chunks)
    encoder=EmbeddingEncoder(get_embedding_model_config('all-MiniLM-L6-v2'),backend='sentence-transformers')
    encoder.model.to('cpu')
    tokenizer=encoder.model.tokenizer
    encoding=[]
    def encode(texts,kind,identifiers):
        result=[]
        for text,identifier in zip(texts,identifiers):
            ids=tokenizer(text,add_special_tokens=False,truncation=False,verbose=False)['input_ids']
            count=len(ids)+tokenizer.num_special_tokens_to_add(pair=False)
            if count>encoder.model.max_seq_length:
                vector,lengths=encode_token_windows(encoder.model,ids)
            else:
                vector=encoder.encode([text],text_kind=kind)[0]
                lengths=[len(ids)]
            result.append(vector)
            encoding.append({'kind':kind,'id':identifier,'content_tokens':len(ids),'window_lengths':lengths})
        return result
    vectors=encode([build_embedding_text(c) for c in chunks],'passage',[c['chunk_id'] for c in chunks])
    records={'tfidf':[],'minilm':[]}
    for question in questions:
        text=question['question'];qtype=question['question_type'];keywords=question.get('expected_section_keywords',[])
        # Keep natural question wording for MiniLM, append the same query hints TF-IDF uses.
        dense_query=' '.join([text,*QUESTION_TYPE_EXPANSIONS.get(qtype,()),*keywords])
        qvec=encode([dense_query],'query',[question['question_id']])[0]
        retrieved, diagnostics = retrieve_for_question(
            chunks=chunks, vectors=vectors, tfidf=tfidf, question=question,
            query_vector=qvec, top_k=args.top_k, source_policy=source_policy)
        for name,items in retrieved.items():
            evaluation=evaluate_retrieval(question=question,retrieved_chunks=items,top_k=args.top_k)
            rank=evaluation['expected_source_hit_rank']
            evaluation['reciprocal_rank']=1/rank if rank else 0.0
            records[name].append({'question_id':question['question_id'],'question':text,
                                  'query_text':dense_query if name=='minilm' else text,
                                  'retrieved_chunks':items,'retrieval_eval':evaluation,
                                  'source_constraint':diagnostics})
    summary={name:aggregate(rows) for name,rows in records.items()}
    args.output_dir.mkdir(parents=True,exist_ok=False)
    for name,rows in records.items():write_jsonl(args.output_dir/f'{name}_retrieval_v1.jsonl',rows)
    write_jsonl(args.output_dir/'encoding_diagnostics_v1.jsonl',encoding)
    write_json(args.output_dir/'retrieval_comparison_summary_v1.json',summary)
    write_json(args.output_dir/'retrieval_comparison_manifest_v1.json',{
        'rag_ready_dir':str(args.rag_ready_dir),'questions_file':str(args.eval_questions_file),'top_k':args.top_k,
        'chunks':len(chunks),'questions':len(questions),'model':encoder.config,'backend':encoder.backend,
        'device':'cpu','long_input_policy':'exact nonoverlapping token windows; length-weighted normalized average',
        'source_policy':source_policy,
        'source_policy_version':'explicit_arxiv_sources_v1' if source_policy == 'explicit_arxiv' else 'none',
        'query_policy':'TF-IDF native expansion; MiniLM natural question plus same question-type terms and expected section keywords',
        'passage_policy':'MiniLM title + section + text; TF-IDF unchanged title/section weighting',
        'evaluation':'existing reranker evaluate_retrieval + embedding compute_embedding_metrics; required-all ranks use maximum matching rank',
        'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.rag_ready_dir/'chunks_v1.jsonl',args.eval_questions_file]}})
    print(summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rag-ready-dir',type=Path,required=True)
    parser.add_argument('--eval-questions-file',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--top-k',type=int,default=5)
    parser.add_argument('--source-policy',choices=['none','explicit_arxiv'],default='none',
                        help='Constrain candidates to arXiv IDs explicitly present in question text.')
    try: run(parser.parse_args())
    except (OSError,ValueError,RuntimeError) as exc:
        print(f'Retrieval comparison failed: {exc}',file=sys.stderr);sys.exit(1)
