from kamiknows.rag import qa_grounded_v1 as qa


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


def context():
    return {
        "question_id": "q1",
        "question": "Which model?",
        "retrieved_chunks": [
            {"citation_label": "[C1]", "citation_id": "C1", "chunk_id": "c1", "text": "CaloGAN."}
        ],
    }


def fake_ollama(monkeypatch, data):
    payloads = []

    def post(url, json, timeout):
        payloads.append(json)
        return FakeResponse(data)

    monkeypatch.setattr(qa.requests, "post", post)
    return payloads


def test_truncated_answer_is_flagged_with_generation_metadata(monkeypatch):
    payloads = fake_ollama(monkeypatch, {
        "response": '{"answer_status": "answered", "answer": "Calo',
        "done_reason": "length",
        "eval_count": 512,
        "prompt_eval_count": 3100,
    })
    records = qa.generate_model_answer_records(
        model_config={"model_name": "fake"},
        retrieved_contexts=[context()],
        known_chunk_ids=["c1"],
        base_url="http://fake",
    )
    assert payloads[0]["format"] == "json"
    generation = records[0]["generation"]
    assert generation["done_reason"] == "length"
    assert generation["eval_count"] == 512
    assert generation["prompt_eval_count"] == 3100
    assert "truncated_at_token_limit" in records[0]["warnings"]


def test_default_format_is_unchanged_and_complete_answer_is_not_flagged(monkeypatch):
    payloads = fake_ollama(monkeypatch, {
        "response": '{"answer_status": "answered", "answer": "CaloGAN [C1]", "citations": ["C1"]}',
        "done_reason": "stop",
    })
    records = qa.generate_model_answer_records(
        model_config={"model_name": "fake"},
        retrieved_contexts=[context()],
        known_chunk_ids=["c1"],
        base_url="http://fake",
    )
    assert payloads[0]["format"] == "json"
    assert records[0]["generation"]["json_format"] == "json"
    assert "truncated_at_token_limit" not in records[0]["warnings"]
    assert records[0]["citation_validation"]["valid"]
