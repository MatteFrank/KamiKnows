"""Contiguous semantic groups with auditable cosine scores and size constraints."""
import math
from statistics import mean

from kamiknows.fulltext.chunking import count_words
from kamiknows.fulltext.sentence_chunking import split_sentences


def cosine(a, b):
    denominator = math.sqrt(sum(x*x for x in a) * sum(x*x for x in b))
    if not denominator:
        raise ValueError('zero embedding/window vector; cannot compute semantic similarity')
    return max(-1.0, min(1.0, sum(x*y for x,y in zip(a,b))/denominator))


def quantile(values, fraction):
    """Linearly interpolated quantile; no benchmark labels are consulted."""
    values = sorted(values)
    index = (len(values)-1)*fraction
    lo = int(index)
    hi = min(lo+1, len(values)-1)
    return values[lo] + (values[hi]-values[lo])*(index-lo)


def window_mean(vectors):
    return [mean(column) for column in zip(*vectors)]


class SemanticChunker:
    """Encoder injected for tests; production requires real normalized embeddings."""
    def __init__(self, encode, *, min_words=250, window_sentences=2, threshold_quantile=0.25,
                 cut_policy='lowest_similarity', min_score_drop=0.10):
        if min_words < 1 or window_sentences < 1 or not 0 <= threshold_quantile <= 1:
            raise ValueError('invalid semantic parameters')
        if cut_policy not in {'lowest_similarity', 'first_drop'} or not math.isfinite(min_score_drop) or not 0 <= min_score_drop <= 2:
            raise ValueError('invalid cut policy or min_score_drop')
        self.encode = encode
        self.min_words = min_words
        self.window = window_sentences
        self.quantile = threshold_quantile
        self.cut_policy = cut_policy
        self.min_score_drop = min_score_drop
        self.boundaries = []
        self.groups = []
        self.section_summaries = []

    def __call__(self, *, paper_id, arxiv_id, title, sections, target_words=400, overlap_words=60):
        if target_words < 100 or not 0 <= overlap_words < target_words or self.min_words > target_words:
            raise ValueError('require target>=100, 0<=overlap<target and min<=target')
        chunks = []
        for section_index, section in enumerate(sections):
            sentences = split_sentences(str(section.get('text') or ''))
            if not sentences:
                continue
            vectors = self.encode(sentences)
            if len(vectors) != len(sentences) or not vectors or not vectors[0]:
                raise ValueError('encoder returned invalid sentence vectors')
            if any(len(v)!=len(vectors[0]) or not all(math.isfinite(x) for x in v) for v in vectors):
                raise ValueError('encoder returned inconsistent/non-finite vectors')
            lengths = [len(s.split()) for s in sentences]
            positions = [0]
            for length in lengths:
                positions.append(positions[-1]+length)
            scores = {i:cosine(window_mean(vectors[max(0,i-self.window):i]),
                              window_mean(vectors[i:min(len(vectors),i+self.window)]))
                      for i in range(1,len(sentences))}
            threshold = quantile(list(scores.values()),self.quantile) if scores else None
            section_mean = mean(scores.values()) if scores else None
            drops = {i: (scores.get(i-1, section_mean) - score) for i, score in scores.items()}
            eligible = {i: score < threshold and (self.cut_policy == 'lowest_similarity'
                        or drops[i] >= self.min_score_drop) for i, score in scores.items()}
            identity = {'paper_id':paper_id,'arxiv_id':arxiv_id,'section_id':section.get('section_id'),
                        'section_index':section_index,'section_heading':section.get('heading')}
            rows = {i:{**identity,'sentence_boundary':i,'word_position':positions[i],
                       'similarity':score,'threshold':threshold,'score_drop':drops[i],
                       'min_score_drop':self.min_score_drop,'cut_policy':self.cut_policy,
                       'semantic_candidate':eligible[i],
                       'selected':False,'decision_reason':'not_selected'} for i,score in scores.items()}
            start, previous_end = 0, 0
            while start < len(sentences):
                end = start+1
                while end < len(sentences) and positions[end+1]-positions[start] <= target_words:
                    end += 1
                # Discard overlap that prevents including a fresh sentence.
                while end <= previous_end and start < previous_end:
                    start += 1
                    end = start+1
                    while end < len(sentences) and positions[end+1]-positions[start] <= target_words:
                        end += 1
                candidates = [i for i in range(max(start+1,previous_end+1),end+1)
                              if i in scores and eligible[i]
                              and positions[i]-positions[start] >= self.min_words
                              and (self.cut_policy == 'lowest_similarity'
                                   or positions[-1]-positions[i] >= self.min_words)]
                if self.cut_policy == 'first_drop' and candidates:
                    end = candidates[0]
                    reason = 'semantic_drop'
                elif end == len(sentences):
                    reason = 'section_end'
                elif candidates:
                    end = min(candidates,key=lambda i:(scores[i],i))
                    reason = 'semantic_drop'
                else:
                    reason = 'length_limit'
                    if self.cut_policy == 'first_drop' and positions[-1]-positions[end] < self.min_words:
                        # Avoid creating a short terminal tail when a whole-sentence
                        # boundary can satisfy both the minimum and maximum.
                        balanced = [i for i in range(max(start+1,previous_end+1),end+1)
                                    if positions[i]-positions[start] >= self.min_words
                                    and positions[-1]-positions[i] >= self.min_words]
                        if balanced:
                            end = balanced[-1]
                text = ' '.join(sentences[start:end])
                chunk = {'chunk_id':f'{paper_id}::semantic_v1::{len(chunks)+1:04d}',
                         'paper_id':paper_id,'arxiv_id':arxiv_id,'title':title,
                         'section_id':section.get('section_id'),'section_heading':section.get('heading'),
                         'source_type':'section','text':text,'word_count':count_words(text),
                         'sentence_span':{'start':start,'end_exclusive':end},
                         'oversize_sentence':lengths[start]>target_words,'cut_reason':reason}
                chunks.append(chunk)
                self.groups.append({**identity,'chunk_id':chunk['chunk_id'],'start_word':positions[start],
                                    'end_word':positions[end],'words':positions[end]-positions[start],
                                    'overlap_words':max(0,positions[previous_end]-positions[start]),
                                    'cut_reason':reason})
                if end in rows:
                    rows[end].update(selected=True,decision_reason=reason)
                previous_end = end
                if end == len(sentences):
                    break
                start, overlap = end, 0
                while start > chunk['sentence_span']['start']+1 and overlap+lengths[start-1] <= overlap_words:
                    start -= 1
                    overlap += lengths[start]
            for row in rows.values():
                if row['semantic_candidate'] and not row['selected']:
                    row['decision_reason'] = ('candidate_not_selected_by_size_tail_or_earlier_boundary'
                                              if self.cut_policy == 'first_drop' else
                                              'candidate_not_selected_by_size_or_stronger_boundary')
            self.boundaries.extend(rows.values())
            self.section_summaries.append({**identity,'sentences':len(sentences),'words':positions[-1],
                                           'mean_similarity':mean(scores.values()) if scores else None,
                                           'threshold':threshold})
        return chunks
