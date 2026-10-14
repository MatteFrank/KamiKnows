import json

import pytest

from kamiknows.fulltext.sentence_chunking import generate_sentence_chunks, split_sentences
from kamiknows.rag_ready.sentence_dataset import build_sentence_dataset


def test_abbreviations_decimals_and_normalized_coverage():
    text = 'Dr. Smith uses e.g. 3.14 GeV. Fig. 2 shows showers! Next result.'
    sentences = split_sentences(text)
    assert sentences == ['Dr. Smith uses e.g. 3.14 GeV.', 'Fig. 2 shows showers!', 'Next result.']
    assert ' '.join(sentences) == text


def test_groups_overlap_sections_and_long_sentence():
    sentences = [' '.join([f's{i}'] * 30) + '.' for i in range(7)]
    chunks = generate_sentence_chunks(paper_id='p', arxiv_id='a', title='t',
        sections=[{'section_id': 'one', 'heading': 'First', 'text': ' '.join(sentences)},
                  {'section_id': 'two', 'heading': 'Second', 'text': ' '.join(['long']*130)+'.'}],
        target_words=100, overlap_words=30)
    assert [c['sentence_span'] for c in chunks[:3]] == [
        {'start': 0, 'end_exclusive': 3}, {'start': 2, 'end_exclusive': 5},
        {'start': 4, 'end_exclusive': 7}]
    assert chunks[-1]['oversize_sentence']
    assert chunks[-1]['section_id'] == 'two'
    assert all(len(c['text'].split()) <= 100 for c in chunks[:-1])
    assert len({c['chunk_id'] for c in chunks}) == len(chunks)


@pytest.mark.parametrize('target,overlap', [(99,0), (100,-1), (100,100)])
def test_invalid_parameters(target, overlap):
    with pytest.raises(ValueError):
        generate_sentence_chunks(paper_id='p',arxiv_id='a',title='t',sections=[],
                                 target_words=target, overlap_words=overlap)


def test_builder_preserves_selection_sources_and_refuses_overwrite(tmp_path):
    reference = tmp_path/'reference'
    reference.mkdir()
    paper_dir = tmp_path/'paper'
    paper_dir.mkdir()
    text = ' '.join(['A complete sentence.']*50)
    sections = paper_dir/'sections.json'
    sections.write_text(json.dumps([{'section_id':'s','heading':'Method','text':text}]))
    before = sections.read_bytes()
    paper = {'paper_id':'p','arxiv_id':'a','title':'Title','paper_dir':str(paper_dir),
             'parsing_status':'partial','files':{}}
    for name,rows in [('papers',[paper]),('chunks',[{'chunk_id':'old','text':text,'metadata':{'domain':'test'}}]),
                      ('equations',[]),('eval_questions_v0',[{'question_id':'legacy'}])]:
        (reference/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (reference/'rag_manifest_v0.json').write_text('{}')
    output = tmp_path/'new'
    manifest = build_sentence_dataset(reference_dir=reference,output_dir=output,
                                      target_words=100,overlap_words=30)
    assert manifest['papers_count'] == 1
    assert manifest['validation']['sentence_coverage'] == 'PASS'
    assert sections.read_bytes() == before
    records=[json.loads(l) for l in (output/'chunks_v1.jsonl').read_text().splitlines()]
    assert all(c['chunk_id'].startswith('p::sentence_v1::') for c in records)
    assert all(c['metadata']['parsing_status']=='partial' for c in records)
    assert json.loads((output/'eval_questions_v0.jsonl').read_text()) == {'question_id':'legacy'}
    with pytest.raises(FileExistsError):
        build_sentence_dataset(reference_dir=reference,output_dir=output)
