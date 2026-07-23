# PromptWatch

A CI/CD-style regression testing system for LLM prompts. It runs a golden dataset of hand-labeled test cases through an email classifier feature, scores the outputs, and flags regressions whenever the prompt or model changes.

## What it does

The core feature under test is a job-search inbox email classifier: given an email's subject and body, it returns a category (`application_ack`, `rejection`, `job_alert`, `newsletter`, `misc`) and a one-sentence summary. Prompts are versioned as YAML files so changes to them can be tested the same way code changes are tested.

## Setup

```bash
python -m venv promptwatch-env
promptwatch-env\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file with:

```
GEMINI_API_KEY=your-key-here
```



