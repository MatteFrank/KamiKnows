"""Local MiniLM semantic dataset construction and standalone diagnostic figures."""
from collections import Counter
import math
from pathlib import Path

from kamiknows.fulltext.semantic_chunking import SemanticChunker
from kamiknows.rag.embedding_registry import get_embedding_model_config
from kamiknows.rag.embedding_selection import EmbeddingEncoder
from kamiknows.rag.load_dataset import read_jsonl
from kamiknows.rag.outputs import write_json, write_jsonl
from kamiknows.rag_ready.sentence_dataset import build_sentence_dataset


def encode_token_windows(model, token_ids, *, forward_windows=None):
    """Encode exact token-ID windows, then length-weight and normalize.

    No decode/re-tokenize round trip: all original content tokens are retained
    once. Special tokens are added independently to each window.
    """
    tokenizer = model.tokenizer
    budget = model.max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)
    if budget < 1 or not token_ids:
        raise ValueError('invalid token window budget or empty content')
    windows = [token_ids[i:i+budget] for i in range(0,len(token_ids),budget)]
    if forward_windows is None:
        import torch
        features = build_minilm_window_features(tokenizer, windows)
        features = tokenizer.pad(features, padding=True, return_tensors='pt')
        features = {key:value.to(model.device) for key,value in features.items()}
        model.eval()
        with torch.no_grad():
            vectors = model(features)['sentence_embedding'].detach().cpu().tolist()
    else:
        vectors = forward_windows(windows)
    if len(vectors) != len(windows) or not vectors or not vectors[0]:
        raise ValueError('invalid token-window embeddings')
    dimension = len(vectors[0])
    normalized = []
    for vector in vectors:
        norm = math.sqrt(sum(x*x for x in vector))
        if len(vector) != dimension or not math.isfinite(norm) or norm == 0:
            raise ValueError('invalid token-window vector')
        normalized.append([x/norm for x in vector])
    combined = [sum(v[j]*len(w) for v,w in zip(normalized,windows))/len(token_ids)
                for j in range(dimension)]
    norm = math.sqrt(sum(x*x for x in combined))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError('invalid combined sentence embedding')
    return [x/norm for x in combined], [len(w) for w in windows]


def build_minilm_window_features(tokenizer, windows):
    """Build single-sequence BERT inputs without removed tokenizer methods.

    This builder is specifically for the registered BERT-based MiniLM model.
    Check its actual single-sequence template before constructing exact IDs.
    """
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    if cls_id is None or sep_id is None or tokenizer.num_special_tokens_to_add(pair=False) != 2:
        raise ValueError('expected MiniLM single-sequence BERT CLS/SEP template')
    probe = tokenizer('', add_special_tokens=True, truncation=False)['input_ids']
    if probe != [cls_id, sep_id]:
        raise ValueError('unsupported tokenizer special-token template')
    rows = []
    for window in windows:
        ids = [cls_id, *window, sep_id]
        row = {'input_ids': ids, 'attention_mask': [1]*len(ids)}
        if 'token_type_ids' in tokenizer.model_input_names:
            row['token_type_ids'] = [0]*len(ids)
        rows.append(row)
    return rows


def plot_diagnostics(output_dir, chunker, length_comparison):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots = output_dir/'plots'
    plots.mkdir()
    colors = {'semantic_drop':'tab:red','length_limit':'tab:orange','section_end':'tab:green'}
    files = []

    def save(fig, stem):
        for suffix in ('png','pdf'):
            path = plots/f'{stem}.{suffix}'
            fig.savefig(path, bbox_inches='tight', dpi=160)
            files.append(str(path))
        plt.close(fig)

    for section in chunker.section_summaries:
        key = (section['paper_id'],section['section_index'])
        rows = [r for r in chunker.boundaries if (r['paper_id'],r['section_index'])==key]
        groups = [g for g in chunker.groups if (g['paper_id'],g['section_index'])==key]
        fig,(ax, spans)=plt.subplots(2,1,figsize=(11,5),sharex=True,
                                    gridspec_kw={'height_ratios':[3,1]})
        ax.plot([r['word_position'] for r in rows],[r['similarity'] for r in rows],'.-',label='Cosine similarity')
        if chunker.cut_policy == 'first_drop' and rows:
            drop_axis = ax.twinx()
            drop_axis.plot([r['word_position'] for r in rows], [r['score_drop'] for r in rows],
                           color='teal',alpha=.35,label='Cosine decrease')
            drop_axis.axhline(chunker.min_score_drop,color='teal',linestyle=':',
                              label=f'Min decrease {chunker.min_score_drop:g}')
            drop_axis.set_ylabel('Previous cosine − current cosine',color='teal')
            drop_axis.legend(fontsize=6,loc='lower right')
        if section['threshold'] is not None:
            ax.axhline(section['threshold'],linestyle='--',color='purple',label='Section quantile threshold')
            ax.axhline(section['mean_similarity'],linestyle=':',color='gray',label='Section mean')
        rejected = [r for r in rows if r['semantic_candidate'] and not r['selected']]
        ax.scatter([r['word_position'] for r in rejected],[r['similarity'] for r in rejected],
                   marker='x',color='gray',label='Unselected semantic candidate')
        used = set()
        for index,g in enumerate(groups):
            reason=g['cut_reason']
            ax.axvline(g['end_word'],color=colors[reason],alpha=.7,
                       label=reason.replace('_',' ') if reason not in used else None)
            used.add(reason)
            spans.plot([g['start_word'],g['end_word']],[index%3]*2,lw=5,color=colors[reason],alpha=.65)
            if g['overlap_words']:
                spans.plot([g['start_word'],g['start_word']+g['overlap_words']],[index%3]*2,
                           lw=8,color='black',alpha=.35)
        ax.set(ylabel='Cosine similarity',ylim=(-1.05,1.05),
               title=f"{section['arxiv_id']} — {section['section_heading']} ({chunker.cut_policy})")
        ax.legend(fontsize=7,loc='best')
        spans.set(xlabel='Word position within section',ylabel='Groups',yticks=[],
                  title='Chunk spans; black segments show overlap',xlim=(0,max(1,section['words'])))
        save(fig,f"{section['paper_id']}_section_{section['section_index']:03d}")
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    values=[r['similarity'] for r in chunker.boundaries]
    axes[0].hist(values,bins=25)
    if values:
        axes[0].axvline(sum(values)/len(values),color='red',label='Global boundary mean')
        axes[0].legend(fontsize=8)
    axes[0].set(title='Boundary similarity distribution',xlabel='Cosine similarity',ylabel='Boundaries')
    axes[1].boxplot(list(length_comparison.values()),tick_labels=list(length_comparison))
    axes[1].set(title='Chunk lengths',ylabel='Whitespace words')
    reasons=Counter(g['cut_reason'] for g in chunker.groups)
    axes[2].bar(list(reasons),list(reasons.values()),color=[colors[r] for r in reasons])
    axes[2].set(title='Selected cut reasons',ylabel='Cuts')
    axes[2].tick_params(axis='x',labelrotation=25)
    save(fig,'semantic_overview_v1')
    return files


def build_semantic_dataset(*, reference_dir: Path, output_dir: Path,
                           target_words=400, overlap_words=60, min_words=250,
                           window_sentences=2, threshold_quantile=0.25,
                           sentence_comparison_dir: Path | None = None,
                           exclude_backmatter: bool = False,
                           cut_policy='lowest_similarity', min_score_drop=0.10):
    if output_dir.exists():
        raise FileExistsError(f'output directory already exists: {output_dir}')
    if target_words < 100 or not 0 <= overlap_words < target_words or not 1 <= min_words <= target_words:
        raise ValueError('invalid word limits')
    if window_sentences < 1 or not 0 <= threshold_quantile <= 1:
        raise ValueError('invalid window/quantile')
    if cut_policy not in {'lowest_similarity', 'first_drop'} or not 0 <= min_score_drop <= 2:
        raise ValueError('invalid cut policy/score drop')
    # Fail before output creation if plotting dependencies or comparison files are absent.
    import matplotlib
    comparison={'word':[len(c['text'].split()) for c in read_jsonl(reference_dir/'chunks.jsonl')]}
    if sentence_comparison_dir is not None:
        comparison['sentence']=[len(c['text'].split()) for c in read_jsonl(sentence_comparison_dir/'chunks_v1.jsonl')]
    config=get_embedding_model_config('all-MiniLM-L6-v2')
    encoder=EmbeddingEncoder(config,backend='sentence-transformers')
    encoder.model.to('cpu')
    tokenizer=encoder.model.tokenizer
    token_counts=[]
    long_inputs=[]

    def encode(sentences):
        ids=[tokenizer(s,add_special_tokens=False,truncation=False,verbose=False)['input_ids']
             for s in sentences]
        special=tokenizer.num_special_tokens_to_add(pair=False)
        counts=[len(tokens)+special for tokens in ids]
        offset=len(token_counts)
        token_counts.extend(counts)
        short=[i for i,n in enumerate(counts) if n <= encoder.model.max_seq_length]
        result=[None]*len(sentences)
        if short:
            for i,vector in zip(short,encoder.encode([sentences[i] for i in short],text_kind='passage')):
                result[i]=vector
        for i,n in enumerate(counts):
            if n > encoder.model.max_seq_length:
                result[i],lengths=encode_token_windows(encoder.model,ids[i])
                long_inputs.append({'global_sentence_index':offset+i,'sentence_text':sentences[i],
                                    'content_tokens':len(ids[i]),'tokens_with_special':n,
                                    'window_count':len(lengths),'window_content_tokens':lengths})
        return result

    chunker=SemanticChunker(encode,min_words=min_words,window_sentences=window_sentences,
                           threshold_quantile=threshold_quantile,cut_policy=cut_policy,
                           min_score_drop=min_score_drop)
    manifest=build_sentence_dataset(reference_dir=reference_dir,output_dir=output_dir,
                                   target_words=target_words,overlap_words=overlap_words,
                                   chunk_generator=chunker,strategy='semantic_groups_v1',
                                   exclude_backmatter=exclude_backmatter)
    comparison['semantic']=[g['words'] for g in chunker.groups]
    scores=[r['similarity'] for r in chunker.boundaries]
    offset=0
    for section in chunker.section_summaries:
        for row in long_inputs:
            if offset <= row['global_sentence_index'] < offset+section['sentences']:
                row.update(paper_id=section['paper_id'],section_id=section['section_id'],
                           section_index=section['section_index'],
                           sentence_index=row['global_sentence_index']-offset)
        offset += section['sentences']
    settings={'model':config['model_name'],'backend':encoder.backend,'device':'cpu',
              'model_max_seq_length':encoder.model.max_seq_length,
              'max_sentence_tokens':max(token_counts),'min_words':min_words,
              'long_input_policy':'nonoverlapping exact token-ID windows; content-length-weighted normalized embeddings',
              'long_sentence_count':len(long_inputs),
              'window_sentences':window_sentences,'threshold_quantile':threshold_quantile,
              'threshold_scope':'per section; linear quantile; strict similarity < threshold',
              'cut_policy':cut_policy,'min_score_drop':min_score_drop,
              'drop_definition':'previous boundary cosine minus current; first boundary uses section mean',
              'selection':('first boundary below quantile with sufficient score drop and min length; avoid short tails'
                           if cut_policy == 'first_drop' else
                           'lowest eligible cosine within min/target size; end of section takes priority'),
              'comparison_paths':{'word':str(reference_dir),
                                  'sentence':str(sentence_comparison_dir) if sentence_comparison_dir else None}}
    summary={**settings,'sections':chunker.section_summaries,
             'mean_boundary_similarity':sum(scores)/len(scores) if scores else None,
             'cut_reason_counts':dict(Counter(g['cut_reason'] for g in chunker.groups)),
             'plot_status':'pending'}
    write_jsonl(output_dir/'semantic_boundary_scores_v1.jsonl',chunker.boundaries)
    write_jsonl(output_dir/'semantic_chunk_spans_v1.jsonl',chunker.groups)
    write_jsonl(output_dir/'semantic_long_inputs_v1.jsonl',long_inputs)
    write_json(output_dir/'semantic_chunking_summary_v1.json',summary)
    manifest['chunking'].update(settings)
    manifest['validation'].update({
        'below_min_words_count':sum(g['words'] < min_words for g in chunker.groups),
        'above_max_words_count':sum(g['words'] > target_words for g in chunker.groups),
        'length_exceptions':[g for g in chunker.groups if not min_words <= g['words'] <= target_words]})
    enriched = read_jsonl(output_dir/'chunks_v1.jsonl')
    for chunk in enriched:
        length = len(chunk['text'].split())
        if length < min_words:
            chunk['quality_flags'].append('below_min_words')
        if length > target_words:
            chunk['quality_flags'].append('above_max_words')
    write_jsonl(output_dir/'chunks_v1.jsonl',enriched)
    write_json(output_dir/'rag_manifest_v1.json',manifest)
    try:
        summary['plot_files']=plot_diagnostics(output_dir,chunker,comparison)
        summary['plot_status']='PASS'
    except Exception as exc:
        summary['plot_status']='FAIL'
        summary['plot_error']=str(exc)
        write_json(output_dir/'semantic_chunking_summary_v1.json',summary)
        raise
    write_json(output_dir/'semantic_chunking_summary_v1.json',summary)
    return manifest
