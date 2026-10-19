"""Ripete domande salvate cambiando solo il profilo di prompt del generatore."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kamiknows.rag.load_dataset import read_jsonl, read_json
from kamiknows.rag.outputs import write_json, write_jsonl
from kamiknows.rag.qa_grounded_v1 import (
    annotate_registry_availability, build_grounded_qa_prompt_v1, generate_model_answer_records,
)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--question-ids',nargs='+',required=True)
    parser.add_argument('--prompt-profile',choices=['grounded_json_v1','synthesis_v2'],default='synthesis_v2')
    parser.add_argument('--base-url',default='http://127.0.0.1:11451')
    args=parser.parse_args()
    try:
        config=read_json(args.baseline_dir/'qa_run_config_v1.json')
        registry=read_json(args.baseline_dir/'qa_model_registry_v1.json')
        if len(registry['models'])!=1:raise ValueError('Replay requires one baseline model')
        model=dict(registry['models'][0]);model['prompt_profile']=args.prompt_profile
        model=annotate_registry_availability(registry={'models':[model]},base_url=args.base_url)['models'][0]
        if model['availability_status']!='available':raise RuntimeError(str(model))
        contexts=read_jsonl(args.baseline_dir/'retrieved_contexts_v1.jsonl')
        answers_path=args.baseline_dir/f"rag_answers_{model['model_slug']}_v1.jsonl"
        answers=read_jsonl(answers_path)
        context_map={r['question_id']:r for r in contexts}
        answer_map={r['question_id']:r for r in answers}
        if len(set(args.question_ids))!=len(args.question_ids):raise ValueError('Duplicate requested IDs')
        selected=[context_map[qid] for qid in args.question_ids]
        baseline=[answer_map[qid] for qid in args.question_ids]
        if any(c['question']!=a['question'] for c,a in zip(selected,baseline)):
            raise ValueError('Baseline question/context mismatch')
        out=args.output_dir
        if any((out/name).exists() for name in ['replay_manifest_v1.json','answers_v1.jsonl','baseline_answers_v1.jsonl','contexts_v1.jsonl','prompts']):
            raise FileExistsError('Replay artifacts already exist; use a new output directory')
        out.mkdir(parents=True,exist_ok=True)
        (out/'prompts').mkdir()
        inputs=[args.baseline_dir/'retrieved_contexts_v1.jsonl',answers_path,
                args.baseline_dir/'qa_run_config_v1.json',args.baseline_dir/'qa_model_registry_v1.json']
        prompt_hashes={}
        for context in selected:
            qid=context['question_id'];prompt_hashes[qid]={}
            for profile in ['grounded_json_v1',args.prompt_profile]:
                text=build_grounded_qa_prompt_v1(context,profile)
                (out/'prompts'/f'{qid}_{profile}.txt').write_text(text,encoding='utf-8')
                prompt_hashes[qid][profile]=hashlib.sha256(text.encode()).hexdigest()
        write_json(out/'replay_manifest_v1.json',{
            'baseline_dir':str(args.baseline_dir),'question_ids':args.question_ids,
            'model':model,'prompt_profile':args.prompt_profile,'base_url':args.base_url,
            'timeout_seconds':config.get('timeout_seconds',300),
            'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
            'prompt_sha256':prompt_hashes,'retrieval':'exact saved contexts; no retrieval rerun',
            'answer_status_semantics':'synthesis_v2 answered may be partial; human review required'})
        write_jsonl(out/'contexts_v1.jsonl',selected)
        write_jsonl(out/'baseline_answers_v1.jsonl',baseline)
        known_ids=list({c['chunk_id'] for context in contexts for c in context['retrieved_chunks']})
        with (out/'answers_v1.jsonl').open('x',encoding='utf-8') as f:
            for i,context in enumerate(selected,1):
                print(f"[{i}/{len(selected)}] {context['question_id']}",flush=True)
                record=generate_model_answer_records(model_config=model,retrieved_contexts=[context],
                    known_chunk_ids=known_ids,base_url=args.base_url,
                    timeout_seconds=config.get('timeout_seconds',300))[0]
                f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
                print(record['answer_status'],record['answer'],record['warnings'],record['backend_error'],flush=True)
    except (OSError,ValueError,RuntimeError,KeyError) as exc:
        print(f'Replay failed: {exc}',file=sys.stderr);return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
