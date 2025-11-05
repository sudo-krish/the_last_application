# The Last Application

> ⚠️ Work in Progress — under active development.

## Overview

One bot to rule all applications, one bot to find them,  
One bot to bring all offers and in employment bind them.

Automates LinkedIn Easy Apply end-to-end: finds jobs, answers application questions using a retrieval-augmented AI over your PDFs and personal notes, and logs everything in a local DuckDB. Designed for reliability, concise answers, and easy extension.

## Features

- Antibot-aware automation
  - Connects to an existing Chromium CDP session for realistic control.
- Smart answering
  - Defaults first: data/defaultanswers.json by label.
  - RAG fallback: FAISS over data/*.pdf with data/myinfo.txt as authoritative context.
  - Strict output rules: numbers only, Yes/No, or concise text when required.
  - Guaranteed fallback: if AI fails or is disabled, uses deterministic answers.
- Persistence
  - DuckDB with jobs, hirers, sessions, applications, form questions and responses.
  - Bulk inserts, indices, stats, CSV import/export.
- Easy Apply flow
  - Inputs, selects, radios/checkboxes supported.
  - Iterates steps, handles inline feedback, submits, and records Q&A.
- Extensible
  - Clear module boundaries for scraping, AI, and storage.

## Quick start

1) Launch Chromium using terminal in the base directory(must be running before the app)
```
chromium-browser --remote-debugging-port=9222 --user-data-dir=data/chromium_user_data --no-first-run --no-default-browser-check --no-sandbox --disable-gpu --disable-dev-shm-usage
```
2) Install Python deps (3.10+ recommended)
```
pip install -r requirements.txt
```

3) Prepare env and data
- Create .env with OPENAI_API_KEY=your_key_here
- Configure config/tla_config.yaml with:
- Populate data/:
  - defaultanswers.json — known label→answer map
  - myinfo.txt — short personal facts (authoritative)
  - *.pdf — resume/portfolio/certs for retrieval
4) Run
    ```
    python main.py
    ```

## Usage notes

- Default → AI → fallback
  - If label found in defaultanswers.json, use it directly.
  - Else, if use_ai is true, call AI RAG over PDFs with myinfo.txt as authoritative context.
  - Else, or if AI returns nothing/invalid, use deterministic fallback (safe option, empty, or rule-based).
- Strict outputs
  - Numbers-only when asked for numeric/years.
  - Exactly Yes/No for binary.
  - Otherwise concise single word/short sentence.
- Storage
  - Jobs and applications, plus each question/response, are persisted in DuckDB under database/.

## Configuration reference

- .env
  - OPENAI_API_KEY: required for current provider (ChatGPT path).
- config/nodriver_config.yaml
  - connection: host, port, userDataDir, headless, noSandbox.
  - search: baseUrl (LinkedIn jobs), query (keyword), geoId (location), easyApply (bool).
  - langchain: model_name, temperature, using_ollama (WIP; keep false for now), openai_api_key (optional).
- data/
  - defaultanswers.json: hard overrides by label.
  - myinfo.txt: authoritative add-on context (short and factual).
  - PDFs: indexed by FAISS for retrieval.

## Architecture

- Scraper: opens LinkedIn search, iterates job cards, extracts details, and stores jobs.
- RAG: FAISS index over PDFs; myinfo.txt injected as authoritative additional_context.
- Chain: retriever → stuff-docs chain → LLM with strict prompt rules; post-processed if needed.
- Form runner: discovers fields, applies defaults/AI/fallback, advances steps, and submits.
- DB: DuckDB schema for jobs, hirers, sessions, applications, questions, responses; CSV IO and stats.

## Checklist before running

- Chromium is launched with:
  - --remote-debugging-port=9222
  - --user-data-dir=data/chromium_user_data
  - --no-first-run --no-default-browser-check
  - --no-sandbox --disable-gpu --disable-dev-shm-usage
- OPENAI_API_KEY set in .env.
- config/nodriver_config.yaml has:
  - connection.host=127.0.0.1, port=9222, userDataDir=data/chromium_user_data
  - search fields (baseUrl, query, geoId, easyApply)
  - langchain.model_name set; using_ollama=false
- data/ contains:
  - defaultanswers.json
  - myinfo.txt
  - at least one PDF for retrieval (resume recommended)
- use_ai flag set as desired:
  - true: use AI with guaranteed fallback
  - false: always use deterministic fallback if not found in defaults
- Database/ directory is writable (created automatically if missing).

## Roadmap

- Ollama support
  - Add parity path for local models; model selection and graceful provider fallback.
- Stronger fallback strategies
  - Per-field deterministic rules, option ranking heuristics, and inline-feedback driven retries.
- Stability hardening
  - Better waits/retries, selector updates, anti-fingerprint refinements.
- Dashboard
  - Visualization for throughput, success rates, and answer quality.

## Disclaimer

Automating third-party websites may violate their terms. Use responsibly and at your own risk.

