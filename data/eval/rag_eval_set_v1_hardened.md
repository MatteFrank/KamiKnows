# RAG Eval Set v1 Hardened

## Summary
- Total questions: 13
- Answerable: 11
- Insufficient evidence: 2 (eval_v0_q008, eval_v1_q013)
- Multi-source questions: 3 (eval_v0_q005, eval_v0_q006, eval_v1_q012)
- Questions needing manual review: 6 (eval_v0_q002, eval_v0_q005, eval_v0_q006, eval_v0_q007, eval_v0_q008, eval_v1_q012)
- Questions needing expected-source review: 6 (eval_v0_q002, eval_v0_q005, eval_v0_q006, eval_v0_q007, eval_v0_q008, eval_v1_q012)

## Question table

| question_id | question_type | expected_behavior | expected sources | required_all_expected_sources | needs_manual_question_review | question |
| --- | --- | --- | --- | --- | --- | --- |
| eval_v0_q001 | method | answerable | 1705.02355v2 | false | false | In arXiv:1705.02355v2, what simulation or modeling method is proposed for fast calorimeter shower simulation? |
| eval_v0_q002 | method | answerable | 1712.10321v1 | false | true | In arXiv:1712.10321v1, which neural or statistical modeling method is used for calorimeter shower generation? |
| eval_v0_q003 | limitation | answerable | 1705.02355v2 | false | false | According to arXiv:1705.02355v2, what limitations or validation caveats are stated for the proposed calorimeter-simulation approach? |
| eval_v0_q004 | limitation | answerable | 1805.00850v2 | false | false | According to arXiv:1805.00850v2, what limitations, open issues, or future-work constraints are stated for its fast calorimeter simulation approach? |
| eval_v0_q005 | comparison | answerable | 1705.02355v2, 1712.10321v1 | true | true | Using evidence from both arXiv:1705.02355v2 and arXiv:1712.10321v1, compare the modeling goals of the two selected FastCalo papers. |
| eval_v0_q006 | comparison | answerable | 1805.00850v2, 1807.01954v2 | true | true | Using evidence from both arXiv:1805.00850v2 and arXiv:1807.01954v2, compare how the two papers discuss validation or performance of fast calorimeter simulation. |
| eval_v0_q007 | source_lookup | answerable | 1705.02355v2, 1712.10321v1, 1805.00850v2, 1807.01954v2, 2106.05285v3 | false | true | Which of the selected FastCalo papers discuss calorimeter simulation as a computational bottleneck, and in which retrieved section or chunk is that evidence found? |
| eval_v0_q008 | unanswerable | insufficient_evidence | 1705.02355v2, 1712.10321v1, 1805.00850v2, 1807.01954v2, 2106.05285v3 | false | true | Do the retrieved chunks provide enough evidence to recommend an exact production-ready detector-simulation replacement based only on these five papers? Answer insufficient evidence unless an exact deployment recommendation is explicitly supported. |
| eval_v1_q009 | fact_lookup | answerable | 1705.02355v2 | false | false | In the CaloGAN paper, what is the name of the GAN-based model proposed for high-fidelity fast simulation of particle showers in electromagnetic calorimeters? |
| eval_v1_q010 | method | answerable | 1705.02355v2 | false | false | According to the CaloGAN paper, what physics-motivated mechanism or loss term is used to encourage generated showers to be consistent with the requested particle energy? |
| eval_v1_q011 | limitation | answerable | 1705.02355v2 | false | false | According to the CaloGAN paper, why are one-dimensional shower-shape statistics not sufficient by themselves to fully validate generated calorimeter showers? |
| eval_v1_q012 | comparison | answerable | 1705.02355v2, 2106.05285v3 | true | true | Using evidence from both the CaloGAN and CaloFlow papers, compare the simulation approach used by CaloGAN with the normalizing-flow approach used by CaloFlow. The answer should cite evidence from both papers if both are retrieved. |
| eval_v1_q013 | no_answer | insufficient_evidence | <none> | false | false | Do the retrieved chunks state an exact ATLAS Run 4 production deployment date and a detector-wide policy for replacing Geant4 calorimeter simulation? Answer insufficient evidence unless both are explicitly stated in the provided context. |

## Questions

### eval_v0_q001 - method

Question:
In arXiv:1705.02355v2, what simulation or modeling method is proposed for fast calorimeter shower simulation?

Expected behavior:
answerable

Expected sources:
1705.02355v2

Required all expected sources:
false

Expected section keywords:
method, model, simulation, approach, GAN, CaloGAN

Evaluation criteria:
A valid answer must cite only the expected source paper and describe the proposed simulation or modeling method using retrieved evidence, not prior knowledge.

Notes:
Hardened from v0: replaces vague 'first selected FastCalo paper' with explicit source constraint.

Review flags:
- needs_manual_question_review: false
- needs_expected_source_review: false

### eval_v0_q002 - method

Question:
In arXiv:1712.10321v1, which neural or statistical modeling method is used for calorimeter shower generation?

Expected behavior:
answerable

Expected sources:
1712.10321v1

Required all expected sources:
false

Expected section keywords:
GAN, generative, model, architecture, training, calorimeter shower generation

Evaluation criteria:
A valid answer must identify the method only if supported by chunks from the constrained source paper.

Notes:
Hardened from v0: removes ambiguous wording 'another selected paper'. Still flagged because this was a retrieval miss in P2T4.

Review flags:
- needs_manual_question_review: true
- needs_expected_source_review: true

### eval_v0_q003 - limitation

Question:
According to arXiv:1705.02355v2, what limitations or validation caveats are stated for the proposed calorimeter-simulation approach?

Expected behavior:
answerable

Expected sources:
1705.02355v2

Required all expected sources:
false

Expected section keywords:
limitation, validation, future, uncertainty, conclusion, caveat

Evaluation criteria:
A valid answer must quote or paraphrase limitations from retrieved text and avoid inventing caveats.

Notes:
Hardened from v0: replaces 'first selected paper' with explicit source constraint.

Review flags:
- needs_manual_question_review: false
- needs_expected_source_review: false

### eval_v0_q004 - limitation

Question:
According to arXiv:1805.00850v2, what limitations, open issues, or future-work constraints are stated for its fast calorimeter simulation approach?

Expected behavior:
answerable

Expected sources:
1805.00850v2

Required all expected sources:
false

Expected section keywords:
limitation, future, outlook, discussion, conclusion, open issues

Evaluation criteria:
A valid answer must be grounded in the expected paper and should not generalize beyond the retrieved source.

Notes:
Hardened from v0: replaces 'later FastCalo paper' with explicit source constraint.

Review flags:
- needs_manual_question_review: false
- needs_expected_source_review: false

### eval_v0_q005 - comparison

Question:
Using evidence from both arXiv:1705.02355v2 and arXiv:1712.10321v1, compare the modeling goals of the two selected FastCalo papers.

Expected behavior:
answerable

Expected sources:
1705.02355v2, 1712.10321v1

Required all expected sources:
true

Expected section keywords:
introduction, method, simulation, calorimeter, modeling goals

Evaluation criteria:
A valid comparison must mention evidence from both expected papers and avoid claims unsupported by retrieved chunks.

Notes:
Hardened from v0: explicit two-source comparison. Still flagged because this was a retrieval miss in P2T4.

Review flags:
- needs_manual_question_review: true
- needs_expected_source_review: true

### eval_v0_q006 - comparison

Question:
Using evidence from both arXiv:1805.00850v2 and arXiv:1807.01954v2, compare how the two papers discuss validation or performance of fast calorimeter simulation.

Expected behavior:
answerable

Expected sources:
1805.00850v2, 1807.01954v2

Required all expected sources:
true

Expected section keywords:
validation, performance, results, comparison

Evaluation criteria:
A valid answer must compare retrieved evidence from both constrained papers, not provide a generic survey.

Notes:
Hardened from v0: explicit two-source comparison; keep expected-source review flag because multi-source retrieval must be checked carefully.

Review flags:
- needs_manual_question_review: true
- needs_expected_source_review: true

### eval_v0_q007 - source_lookup

Question:
Which of the selected FastCalo papers discuss calorimeter simulation as a computational bottleneck, and in which retrieved section or chunk is that evidence found?

Expected behavior:
answerable

Expected sources:
1705.02355v2, 1712.10321v1, 1805.00850v2, 1807.01954v2, 2106.05285v3

Required all expected sources:
false

Expected section keywords:
introduction, simulation, computational, CPU, bottleneck

Evaluation criteria:
A valid answer should identify the source paper or papers and cite chunk or section evidence. Multiple expected IDs are allowed.

Notes:
Hardened from v0 but still flagged: broad source-lookup questions should not be interpreted like single-source fact questions.

Review flags:
- needs_manual_question_review: true
- needs_expected_source_review: true

### eval_v0_q008 - unanswerable

Question:
Do the retrieved chunks provide enough evidence to recommend an exact production-ready detector-simulation replacement based only on these five papers? Answer insufficient evidence unless an exact deployment recommendation is explicitly supported.

Expected behavior:
insufficient_evidence

Expected sources:
1705.02355v2, 1712.10321v1, 1805.00850v2, 1807.01954v2, 2106.05285v3

Required all expected sources:
false

Expected section keywords:
validation, limitations, future, production, deployment, replacement

Evaluation criteria:
A valid answer should state that the dataset is insufficient for an exact deployment recommendation unless retrieved chunks explicitly support it.

Notes:
Hardened from v0: moved broad source list to candidate_context_arxiv_ids because this is an insufficient-evidence question, not a source-hit question.

Review flags:
- needs_manual_question_review: true
- needs_expected_source_review: true

### eval_v1_q009 - fact_lookup

Question:
In the CaloGAN paper, what is the name of the GAN-based model proposed for high-fidelity fast simulation of particle showers in electromagnetic calorimeters?

Expected behavior:
answerable

Expected sources:
1705.02355v2

Required all expected sources:
false

Expected section keywords:
CaloGAN, GAN, fast simulation, particle showers, electromagnetic calorimeters

Evaluation criteria:
A valid answer must identify the proposed named model and cite the expected source.

Notes:
Hardened v1 wording: single-source fact lookup; avoids vague 'named model'.

Review flags:
- needs_manual_question_review: false
- needs_expected_source_review: false

### eval_v1_q010 - method

Question:
According to the CaloGAN paper, what physics-motivated mechanism or loss term is used to encourage generated showers to be consistent with the requested particle energy?

Expected behavior:
answerable

Expected sources:
1705.02355v2

Required all expected sources:
false

Expected section keywords:
physics-specific loss, energy, reconstructed energy, nominal energy, energy conservation

Evaluation criteria:
A valid answer must cite retrieved evidence describing the physics-motivated mechanism or loss term.

Notes:
Hardened v1 wording: asks for the mechanism, not a generic description.

Review flags:
- needs_manual_question_review: false
- needs_expected_source_review: false

### eval_v1_q011 - limitation

Question:
According to the CaloGAN paper, why are one-dimensional shower-shape statistics not sufficient by themselves to fully validate generated calorimeter showers?

Expected behavior:
answerable

Expected sources:
1705.02355v2

Required all expected sources:
false

Expected section keywords:
one-dimensional, shower shape, correlations, higher dimensional, validation

Evaluation criteria:
A valid answer must explain the stated validation limitation using retrieved evidence.

Notes:
Hardened v1 wording: focuses on the stated limitation and expected evidence.

Review flags:
- needs_manual_question_review: false
- needs_expected_source_review: false

### eval_v1_q012 - comparison

Question:
Using evidence from both the CaloGAN and CaloFlow papers, compare the simulation approach used by CaloGAN with the normalizing-flow approach used by CaloFlow. The answer should cite evidence from both papers if both are retrieved.

Expected behavior:
answerable

Expected sources:
1705.02355v2, 2106.05285v3

Required all expected sources:
true

Expected section keywords:
CaloGAN, GAN, CaloFlow, normalizing flows, likelihood, stable training

Evaluation criteria:
A valid answer should compare both approaches and cite evidence from both expected papers when available in retrieved context.

Notes:
Hardened v1 wording: explicitly marks multi-source requirement; review needed if one source is missing from top-k.

Review flags:
- needs_manual_question_review: true
- needs_expected_source_review: true

### eval_v1_q013 - no_answer

Question:
Do the retrieved chunks state an exact ATLAS Run 4 production deployment date and a detector-wide policy for replacing Geant4 calorimeter simulation? Answer insufficient evidence unless both are explicitly stated in the provided context.

Expected behavior:
insufficient_evidence

Expected sources:
<none>

Required all expected sources:
false

Expected section keywords:
ATLAS Run 4, production deployment date, detector-wide policy, Geant4 replacement

Evaluation criteria:
A valid answer should abstain unless the exact deployment date and detector-wide replacement policy are both explicitly supported by retrieved context.

Notes:
Hardened no-answer wording: makes abstention criterion explicit and reduces accidental overclaim.

Review flags:
- needs_manual_question_review: false
- needs_expected_source_review: false
