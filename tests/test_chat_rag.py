import json
from types import SimpleNamespace

from scripts.chat_rag import ChatRetriever, run_chat
from kamiknows.rag import qa_grounded_v1 as qa


def chunks():
    return [{"chunk_id":"c1","arxiv_id":"1807.01954v2","text":"WGAN showers.","section_heading":"Method"},
            {"chunk_id":"c2","arxiv_id":"1705.02355v2","text":"CaloGAN generates showers.","section_heading":"Method"}]


def test_dense_retrieval_uses_full_evidence_and_requested_source():
    # Non carica MiniLM: verifica il ranking reale e il recupero del testo completo.
    retriever=ChatRetriever.__new__(ChatRetriever)
    retriever.chunks=chunks();retriever.name="minilm"
    retriever.vectors=[[1.,0.],[0.,1.]]
    retriever.encode=lambda text:[1.,0.]
    context=retriever.retrieve("According to 1705.02355v2, what model?",5,"explicit_arxiv","q")
    assert len(context["retrieved_chunks"])==1
    assert context["retrieved_chunks"][0]["text"]=="CaloGAN generates showers."
    assert context["retrieved_chunks"][0]["citation_id"]=="C1"


def test_interactive_loop_saves_each_answer_and_survives_backend_error(tmp_path,monkeypatch):
    calls=[]
    def generate(**kwargs):
        calls.append(kwargs["prompt"])
        if len(calls)==2:raise RuntimeError("simulated outage")
        return json.dumps({"answer_status":"answered","answer":"CaloGAN [C1]","citations":["C1"]})
    monkeypatch.setattr(qa,"generate_ollama_json",generate)
    prompts=iter(["", "According to 1705.02355v2, what model?", "What models?", "/exit"])
    displayed=[]
    args=SimpleNamespace(top_k=5,source_policy="explicit_arxiv",show_context=False,
                         base_url="http://fake",timeout_seconds=2)
    run_chat(retriever=ChatRetriever(chunks(),"tfidf"),model_config={"model_name":"fake"},
             args=args,session_dir=tmp_path,input_fn=lambda prompt:next(prompts),
             output_fn=lambda text,**kwargs:displayed.append(text))
    records=[json.loads(line) for line in (tmp_path/"answers.jsonl").read_text().splitlines()]
    contexts=[json.loads(line) for line in (tmp_path/"contexts.jsonl").read_text().splitlines()]
    assert len(records)==len(contexts)==2
    assert records[0]["citation_validation"]["valid"]
    assert records[1]["answer_status"]=="backend_error"
    assert len(contexts[1]["retrieved_chunks"])==2
    assert "WGAN showers." not in calls[0]
    assert "CaloGAN generates showers." in calls[0]
    assert any("simulated outage" in text for text in displayed)
