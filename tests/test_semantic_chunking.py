import pytest
from kamiknows.fulltext.semantic_chunking import SemanticChunker, quantile
from kamiknows.rag_ready.semantic_dataset import plot_diagnostics
from kamiknows.rag_ready.semantic_dataset import encode_token_windows


def test_exact_token_windows_cover_all_tokens_and_weight_short_tail():
    from types import SimpleNamespace
    tokenizer=SimpleNamespace(num_special_tokens_to_add=lambda pair:2)
    model=SimpleNamespace(tokenizer=tokenizer,max_seq_length=6)
    tokens=list(range(9))
    seen=[]
    def forward(windows):
        seen.extend(windows)
        return [[1.,0.],[1.,0.],[0.,1.]]
    vector,lengths=encode_token_windows(model,tokens,forward_windows=forward)
    assert seen==[[0,1,2,3],[4,5,6,7],[8]]
    assert [t for w in seen for t in w]==tokens
    assert lengths==[4,4,1]
    assert vector==pytest.approx([8/(65**.5),1/(65**.5)])


def test_token_window_rejects_zero_vectors():
    from types import SimpleNamespace
    model=SimpleNamespace(tokenizer=SimpleNamespace(num_special_tokens_to_add=lambda pair:2),
                          max_seq_length=6)
    with pytest.raises(ValueError,match='vector'):
        encode_token_windows(model,[1,2],forward_windows=lambda windows:[[0.,0.]])


def test_window_forward_adds_special_tokens_and_masks_padding():
    torch=pytest.importorskip('torch')
    class Tokenizer:
        cls_token_id=101
        sep_token_id=102
        model_input_names=['input_ids','attention_mask']
        def __call__(self,text,**kwargs): return {'input_ids':[101,102]}
        def num_special_tokens_to_add(self,pair=False): return 2
        def prepare_for_model(self,ids,**kwargs):
            assert not kwargs['truncation']
            return {'input_ids':[101]+ids+[102],'attention_mask':[1]*(len(ids)+2)}
        def pad(self,rows,**kwargs):
            size=max(len(r['input_ids']) for r in rows)
            return {key:torch.tensor([r[key]+[0]*(size-len(r[key])) for r in rows])
                    for key in ['input_ids','attention_mask']}
    class Model:
        tokenizer=Tokenizer()
        max_seq_length=6
        device='cpu'
        def eval(self): pass
        def __call__(self,features):
            assert features['input_ids'].tolist()==[[101,1,2,3,4,102],[101,5,102,0,0,0]]
            assert features['attention_mask'].tolist()==[[1]*6,[1,1,1,0,0,0]]
            return {'sentence_embedding':torch.tensor([[1.,0.],[0.,1.]])}
    vector,lengths=encode_token_windows(Model(),[1,2,3,4,5])
    assert lengths==[4,1]
    assert vector==pytest.approx([4/(17**.5),1/(17**.5)])


def test_real_cached_minilm_tokenizer_window_inputs():
    transformers=pytest.importorskip('transformers')
    from kamiknows.rag_ready.semantic_dataset import build_minilm_window_features
    try:
        tokenizer=transformers.AutoTokenizer.from_pretrained(
            'sentence-transformers/all-MiniLM-L6-v2',local_files_only=True)
    except OSError:
        pytest.skip('MiniLM tokenizer is not cached locally')
    ids=tokenizer('A shower with energy 3.14 GeV.',add_special_tokens=False)['input_ids']
    rows=build_minilm_window_features(tokenizer,[ids,ids[:2]])
    assert rows[0]['input_ids']==tokenizer('A shower with energy 3.14 GeV.')['input_ids']
    features=tokenizer.pad(rows,padding=True,return_tensors='pt')
    assert features['attention_mask'][1].sum().item()==4
    assert features['input_ids'][0].tolist()==[tokenizer.cls_token_id,*ids,tokenizer.sep_token_id]


def sentence(label):
    return ' '.join([label]*50)+'.'


def test_topic_boundary_overlap_coverage_and_diagnostics():
    def encode(texts):
        return [[1.,0.] if s.startswith('alpha') else [0.,1.] for s in texts]
    chunker=SemanticChunker(encode,min_words=100,window_sentences=1,threshold_quantile=.5)
    sentences=[sentence('alpha')]*3+[sentence('beta')]*4
    chunks=chunker(paper_id='p',arxiv_id='a',title='t',
                   sections=[{'section_id':'s','heading':'H','text':' '.join(sentences)}],
                   target_words=200,overlap_words=50)
    assert chunks[0]['sentence_span']=={'start':0,'end_exclusive':3}
    assert chunks[0]['cut_reason']=='semantic_drop'
    assert chunks[1]['sentence_span']['start']==2
    coverage=[]
    end=0
    for c in chunks:
        span=c['sentence_span']
        coverage.extend(range(max(end,span['start']),span['end_exclusive']))
        end=span['end_exclusive']
        assert len(c['text'].split())<=200
    assert coverage==list(range(7))
    row=next(r for r in chunker.boundaries if r['sentence_boundary']==3)
    assert row['similarity']==0 and row['selected'] and row['decision_reason']=='semantic_drop'


def test_uniform_scores_force_size_cuts_and_sections_stay_separate():
    chunker=SemanticChunker(lambda texts:[[1.,0.]]*len(texts),min_words=100)
    chunks=chunker(paper_id='p',arxiv_id='a',title='t',sections=[
        {'section_id':'one','text':' '.join([sentence('uniform')]*7)},
        {'section_id':'two','text':' '.join(['long']*220)+'.'}],target_words=200,overlap_words=50)
    assert chunks[0]['cut_reason']=='length_limit'
    assert not any(r['semantic_candidate'] for r in chunker.boundaries)
    assert chunks[-1]['section_id']=='two' and chunks[-1]['oversize_sentence']
    assert chunks[-1]['cut_reason']=='section_end'


def test_invalid_vectors_and_quantile():
    assert quantile([0,1],.25)==.25
    chunker=SemanticChunker(lambda texts:[[0,0]]*len(texts))
    with pytest.raises(ValueError,match='zero'):
        chunker(paper_id='p',arxiv_id='a',title='t',sections=[{'text':'First sentence. Second sentence.'}])


def test_first_drop_cuts_earliest_even_if_section_fits(monkeypatch):
    # Both 100 and 150 words are significant drops; the earlier one must win.
    import kamiknows.fulltext.semantic_chunking as module
    scores = iter([1., .6, .2, 1., 1.])
    monkeypatch.setattr(module, 'cosine', lambda a,b: next(scores))
    chunker=SemanticChunker(lambda texts:[[1.,0.]]*len(texts),min_words=100,
                           threshold_quantile=.75,cut_policy='first_drop',min_score_drop=.1)
    chunks=chunker(paper_id='p',arxiv_id='a',title='t',
                   sections=[{'section_id':'s','text':' '.join(sentence(str(i)) for i in range(6))}],
                   target_words=400,overlap_words=0)
    assert chunks[0]['sentence_span']['end_exclusive']==2
    assert chunks[0]['cut_reason']=='semantic_drop'
    assert all(100 <= len(c['text'].split()) <= 400 for c in chunks)
    assert len(chunks)==2


def test_first_drop_rebalances_short_tail_and_rejects_small_decrease(monkeypatch):
    import kamiknows.fulltext.semantic_chunking as module
    scores=iter([1.,1.,1.,.95,.95,.95,.95,.95])
    monkeypatch.setattr(module,'cosine',lambda a,b:next(scores))
    chunker=SemanticChunker(lambda texts:[[1.,0.]]*len(texts),min_words=100,
                           cut_policy='first_drop',min_score_drop=.1,threshold_quantile=.75)
    chunks=chunker(paper_id='p',arxiv_id='a',title='t',
                   sections=[{'section_id':'s','text':' '.join(sentence(str(i)) for i in range(9))}],
                   target_words=400,overlap_words=0)
    assert [len(c['text'].split()) for c in chunks]==[350,100]
    assert not any(r['semantic_candidate'] for r in chunker.boundaries)
    assert ' '.join(c['text'] for c in chunks)==' '.join(sentence(str(i)) for i in range(9))


def test_plots_include_single_sentence_and_pdf_png(tmp_path):
    pytest.importorskip('matplotlib')
    chunker=SemanticChunker(lambda texts:[[1.,0.]]*len(texts),min_words=100)
    chunker(paper_id='p',arxiv_id='a',title='t',sections=[{'section_id':'s','heading':'H','text':'Only one sentence.'}])
    files=plot_diagnostics(tmp_path,chunker,{'word':[100,200],'semantic':[4]})
    assert len(files)==4
    from pathlib import Path
    assert all(Path(p).stat().st_size>0 for p in files)
