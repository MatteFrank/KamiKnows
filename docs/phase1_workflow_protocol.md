# KamiKnows — Fase 1 workflow protocol

## Rule

Every Fase 1 task must follow this workflow:

1. Analyze the task before coding.
2. Decide whether Codex is needed.
3. If Codex is needed, use a specific prompt.
4. Codex must produce a final Markdown report.
5. The report must be pasted back into the ChatGPT project chat.
6. The next task starts only after the report is reviewed.
7. Do not accumulate unreviewed changes.

## Required Codex report

Every Codex task that changes code, docs, tests, prompts, schemas, or scripts must produce a report named:

```text
docs/task_reports/YYYY-MM-DD_<short_task_name>.md