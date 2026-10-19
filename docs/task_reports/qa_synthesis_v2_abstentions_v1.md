# Synthesis v2: fixed-context replay of four abstentions

## Objective and inputs

Compare the proposed synthesis prompt against grounded_json_v1 on
pair_01_implicit, pair_06_explicit, pair_07_explicit and pair_07_implicit. All
four baseline records were insufficient_evidence. Inputs are the exact saved
contexts, model registry, run configuration and answers from
`outputs/qa_grounded_qwen4b_minilm_eval_v3_paired_top5`.

## Changes and commands

- New `rag/prompt_profiles.py`: synthesis_v2 instructions permit supported
  cross-passage synthesis and partial answers, explain comparison behavior,
  require concrete content and nonempty abstention explanations.
- `build_grounded_qa_prompt_v1` accepts an optional profile; the legacy default
  is retained. JSON schema, question and retrieved-context suffix are unchanged.
- Generation reads the profile from model configuration; answer records retain
  the profile name. Chat and ordinary QA defaults are not switched automatically.
- New `scripts/replay_qa_prompt_profile_v1.py` replays selected saved contexts,
  writes input/prompt hashes and both prompts, and saves each answer immediately.
- Native shared Ollama qwen3:4b-instruct. Temperature=0, top_p=0.9 and
  num_predict=512 copied from the baseline; request timeout=300 seconds.
- No retrieval, chunking, corpus or reference-answer changes. Expected answers
  are not loaded by the replay script.

```bash
sbatch scripts/run_qa_synthesis_v2_abstentions_v1.slurm
```

## Validation and outputs

- Full software suite: 259 passed, 2 skipped.
- CLI help, Slurm bash syntax and git diff --check passed.
- Test verifies identical JSON-schema/question/context suffix, retained source
  constraints and legacy default, and rejection of unknown profiles.
- Outputs: `outputs/qa_qwen4b_synthesis_v2_abstentions_v1`, including original
  baseline answers, copied contexts, both prompts, replay manifest and new answers.
- Job 12333 completed: 4 records, zero backend errors; exact context equality with baseline verified.
- New normalized statuses: 2 answered, 2 insufficient; one JSON parse warning.
- One answered record (pair_07_explicit) is actually an incomplete JSON abstention
  normalized by fallback, not a successfully generated comparison.

## Limits and next step

Under synthesis_v2, answered may be partial. A change from insufficient_evidence
to answered alone does not establish correctness or completeness. This selected
four-case sample cannot establish overall quality or retained abstention safety;
negative controls are not part of this initial replay. Reference drafts and
factual entailment remain pending human review. Compare every new claim with
retrieved evidence and inspect truncation/parse warnings under the fixed budget.


## Preliminary content inspection

Only pair_06_explicit changes to a substantive answer: it distinguishes GANs from
normalizing flows and explicit likelihood access. Its extra classifier/training
claims and claim-level citation support still need review. Pair_01_implicit and
pair_07_implicit continue abstaining despite useful retrieved evidence. The jet
comparison in pair_07_explicit also attempts abstention in an incomplete JSON;
its normalized answered status must not be counted as a quality gain.

The same 512-token output budget was retained. The new prompt does not reliably
fix synthesis or evidence recognition, and longer output can still be malformed.
No global default was changed and no broad improvement claim is made. Next,
isolate evidence availability with a reference-context replay before choosing
another prompt/model change, and include negative controls before adoption.

Artifacts: prompt_comparison_v1.md and .csv in the replay directory, with all
four human review rows pending; raw output, old limitations and full context are
retained. Input/prompt hashes and model settings are in replay_manifest_v1.json.
