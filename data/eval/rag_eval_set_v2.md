# RAG Eval Set v2

## Summary
- Total questions: 13
- Answerable: 11
- Insufficient evidence: 2 (eval_v0_q008, eval_v1_q013)
- Multi-source questions: 3 (eval_v0_q005, eval_v0_q006, eval_v1_q012)
- Questions needing manual review: 0
- Questions needing expected-source review: 0

## Question table

| question_id | question_type | expected_behavior | expected sources | required_all_expected_sources | needs_manual_question_review | question |
| --- | --- | --- | --- | --- | --- | --- |
| eval_v0_q001 | method | answerable | 1705.02355v2 | false | false | According to arXiv:1705.02355v2, which generative model is proposed for fast simulation of calorimeter showers? |
| eval_v0_q002 | method | answerable | 1712.10321v1 | false | false | According to arXiv:1712.10321v1, which generative model is used to simulate particle showers, and how are the calorimeter layers represented in its output? |
| eval_v0_q003 | limitation | answerable | 1705.02355v2 | false | false | According to arXiv:1705.02355v2, why is agreement in one-dimensional shower-shape distributions not sufficient to fully validate the generated showers? |
| eval_v0_q004 | limitation | answerable | 1805.00850v2 | false | false | According to arXiv:1805.00850v2, what difficulty with hyperparameter tuning or noise injection do the authors identify as a limitation of their generative model? |
| eval_v0_q005 | comparison | answerable | 1705.02355v2, 1712.10321v1 | true | false | Using arXiv:1705.02355v2 and arXiv:1712.10321v1, explain what both papers aim to simulate and identify one aspect of CaloGAN described in greater detail in the latter paper. Cite both sources. |
| eval_v0_q006 | comparison | answerable | 1805.00850v2, 1807.01954v2 | true | false | Using arXiv:1805.00850v2 and arXiv:1807.01954v2, compare the simulated objects and one validation observable or criterion used in each study. Cite both sources. |
| eval_v0_q007 | source_lookup | answerable | 1807.01954v2 | false | false | According to arXiv:1807.01954v2, how does the computational time for WGAN-generated electromagnetic showers compare with GEANT4 simulation? |
| eval_v0_q008 | unanswerable | insufficient_evidence | <none> | false | false | Based on the five papers in this collection, what exact detector-wide production configuration and deployment procedure should be used to replace Geant4? |
| eval_v1_q009 | fact_lookup | answerable | 1705.02355v2 | false | false | What is the name of the GAN-based fast calorimeter shower simulation model introduced in arXiv:1705.02355v2? |
| eval_v1_q010 | method | answerable | 1705.02355v2 | false | false | According to arXiv:1705.02355v2, what physics-motivated loss component encourages agreement between requested particle energy and reconstructed shower energy? |
| eval_v1_q011 | limitation | answerable | 1705.02355v2 | false | false | According to arXiv:1705.02355v2, which information about the shower distribution is not tested by one-dimensional shower-shape statistics, and what additional validation approach is discussed? |
| eval_v1_q012 | comparison | answerable | 1705.02355v2, 2106.05285v3 | true | false | Using arXiv:1705.02355v2 and arXiv:2106.05285v3, compare the generative modeling approach of CaloGAN with that of CaloFlow. Cite both sources. |
| eval_v1_q013 | no_answer | insufficient_evidence | <none> | false | false | What is the exact ATLAS Run 4 production deployment date and detector-wide policy for replacing Geant4 calorimeter simulation, as specified in the five papers? |

## Questions

### eval_v0_q001 - method

Question:
According to arXiv:1705.02355v2, which generative model is proposed for fast simulation of calorimeter showers?

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
Reviewed wording and source constraints against the five parsed papers.

Review flags:
- needs_manual_question_review: false
- needs_expected_source_review: false

### eval_v0_q002 - method

Question:
According to arXiv:1712.10321v1, which generative model is used to simulate particle showers, and how are the calorimeter layers represented in its output?

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
Reviewed wording and source constraints against the five parsed papers.

Review flags:
- needs_manual_question_review: false
- needs_expected_source_review: false

### eval_v0_q003 - limitation

Question:
According to arXiv:1705.02355v2, why is agreement in one-dimensional shower-shape distributions not sufficient to fully validate the generated showers?

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
Reviewed wording and source constraints against the five parsed papers.

Review flags:
- needs_manual_question_review: false
- needs_expected_source_review: false

### eval_v0_q004 - limitation

Question:
According to arXiv:1805.00850v2, what difficulty with hyperparameter tuning or noise injection do the authors identify as a limitation of their generative model?

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
Reviewed wording and source constraints against the five parsed papers.

Review flags:
- needs_manual_question_review: false
- needs_expected_source_review: false

### eval_v0_q005 - comparison

Question:
Using arXiv:1705.02355v2 and arXiv:1712.10321v1, explain what both papers aim to simulate and identify one aspect of CaloGAN described in greater detail in the latter paper. Cite both sources.

Expected behavior:
answerable

Expected sources:
1705.02355v2, 1712.10321v1

Required all expected sources:
true

Expected section keywords:
CaloGAN, method, model architecture, loss formulation, simulation

Evaluation criteria:
A valid comparison must mention evidence from both expected papers and avoid claims unsupported by retrieved chunks.

Notes:
Reviewed wording and source constraints against the five parsed papers.

Review flags:
- needs_manual_question_review: false
- needs_expected_source_review: false

### eval_v0_q006 - comparison

Question:
Using arXiv:1805.00850v2 and arXiv:1807.01954v2, compare the simulated objects and one validation observable or criterion used in each study. Cite both sources.

Expected behavior:
answerable

Expected sources:
1805.00850v2, 1807.01954v2

Required all expected sources:
true

Expected section keywords:
hadronic jets, electromagnetic showers, results, performance, correlations, validation

Evaluation criteria:
A valid answer must compare retrieved evidence from both constrained papers, not provide a generic survey.

Notes:
Reviewed wording and source constraints against the five parsed papers.

Review flags:
- needs_manual_question_review: false
- needs_expected_source_review: false

### eval_v0_q007 - source_lookup

Question:
According to arXiv:1807.01954v2, how does the computational time for WGAN-generated electromagnetic showers compare with GEANT4 simulation?

Expected behavior:
answerable

Expected sources:
1807.01954v2

Required all expected sources:
false

Expected section keywords:
computational speed-up, computational, GEANT4, WGAN, performance

Evaluation criteria:
A valid answer should report the comparison with GEANT4 based on evidence from arXiv:1807.01954v2.

Notes:
Reviewed wording and source constraints against the five parsed papers.

Review flags:
- needs_manual_question_review: false
- needs_expected_source_review: false

### eval_v0_q008 - unanswerable

Question:
Based on the five papers in this collection, what exact detector-wide production configuration and deployment procedure should be used to replace Geant4?

Expected behavior:
insufficient_evidence

Expected sources:
<none>

Required all expected sources:
false

Expected section keywords:
validation, limitations, future, production, deployment, replacement

Evaluation criteria:
A valid answer must abstain from giving an exact production configuration, date, or policy that is not supported by the papers.

Notes:
Reviewed wording and source constraints against the five parsed papers.

Review flags:
- needs_manual_question_review: false
- needs_expected_source_review: false

### eval_v1_q009 - fact_lookup

Question:
What is the name of the GAN-based fast calorimeter shower simulation model introduced in arXiv:1705.02355v2?

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
Reviewed wording and source constraints against the five parsed papers.

Review flags:
- needs_manual_question_review: false
- needs_expected_source_review: false

### eval_v1_q010 - method

Question:
According to arXiv:1705.02355v2, what physics-motivated loss component encourages agreement between requested particle energy and reconstructed shower energy?

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
Reviewed wording and source constraints against the five parsed papers.

Review flags:
- needs_manual_question_review: false
- needs_expected_source_review: false

### eval_v1_q011 - limitation

Question:
According to arXiv:1705.02355v2, which information about the shower distribution is not tested by one-dimensional shower-shape statistics, and what additional validation approach is discussed?

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
Reviewed wording and source constraints against the five parsed papers.

Review flags:
- needs_manual_question_review: false
- needs_expected_source_review: false

### eval_v1_q012 - comparison

Question:
Using arXiv:1705.02355v2 and arXiv:2106.05285v3, compare the generative modeling approach of CaloGAN with that of CaloFlow. Cite both sources.

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
Reviewed wording and source constraints against the five parsed papers.

Review flags:
- needs_manual_question_review: false
- needs_expected_source_review: false

### eval_v1_q013 - no_answer

Question:
What is the exact ATLAS Run 4 production deployment date and detector-wide policy for replacing Geant4 calorimeter simulation, as specified in the five papers?

Expected behavior:
insufficient_evidence

Expected sources:
<none>

Required all expected sources:
false

Expected section keywords:
ATLAS Run 4, production deployment date, detector-wide policy, Geant4 replacement

Evaluation criteria:
A valid answer must abstain from giving an exact production configuration, date, or policy that is not supported by the papers.

Notes:
Reviewed wording and source constraints against the five parsed papers.

Review flags:
- needs_manual_question_review: false
- needs_expected_source_review: false
