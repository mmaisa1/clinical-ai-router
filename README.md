# Clinical AI Inference Router

A confidence-based routing system for cancer type classification from clinical pathology reports. Built on top of a hybrid NLP pipeline developed during my master's capstone at CSUDH.

> **Note:** This is a research prototype built on de-identified public TCGA data. It is not validated for clinical use.

---

## Background

During my master's capstone, I evaluated multiple approaches for classifying TCGA pathology reports into 32 cancer types across 9,523 reports.

One result stood out beyond overall accuracy: the hybrid model's routing score was strongly correlated with whether it was likely to be correct. On high-confidence cases, the hybrid achieved 90.5% accuracy. On low-confidence cases, that dropped to 53.0%.

That finding turned a model output into a routing signal. This project builds on that idea.

---

## Core Idea

Use the fast hybrid pipeline when it is confident, and route uncertain cases to a stronger fallback model.

Every incoming report goes through a two-stage hybrid pipeline:

**Stage 1 — Candidate narrowing**
TF-IDF + LinearSVC scores the full document against all 32 cancer types and returns the top 5 most likely candidates.

**Stage 2 — Chunked NLI reranking**
The full document is split into overlapping 256-token windows. Each window is scored against the 5 candidates using BART-large-MNLI zero-shot classification. The highest-scoring chunk prediction is used as the hybrid prediction.

If the routing score meets the threshold, the hybrid prediction is returned. If it falls below the threshold, the request is planned to escalate to a Qwen2.5-7B local fallback model.

```text
Incoming report
      ↓
Stage 1: TF-IDF + LinearSVC → Top-5 candidates
      ↓
Stage 2: Chunked NLI reranking over full document
      ↓
Score ≥ threshold → return hybrid prediction
Score < threshold → planned Qwen2.5-7B fallback
      ↓
Log prediction and routing details
```

---

## Routing Threshold

The 0.6 threshold was examined on the full 1,905-report test set, where it showed a sharp accuracy split: 90.5% above the threshold versus 53.0% below.

This is a known limitation. A separate validation split would be needed before making stronger generalization claims.

---

## Fallback Model Selection

Offline evaluation compared five fallback candidates on the same 1,905-report test set, focusing on accuracy on low-confidence cases only.

| Model             | Low-conf accuracy | Net gain over hybrid |
| ----------------- | ----------------: | -------------------: |
| Hybrid alone      |             53.0% |                    — |
| Qwen2.5-1.5B      |             78.6% |            +90 cases |
| Qwen2.5-3B        |             85.8% |           +115 cases |
| Qwen2.5-7B        |             91.5% |           +135 cases |
| Claude Haiku      |             95.4% |           +149 cases |
| Groq Llama3.1-8B* |             86.4% |            +25 cases |

*N=300 subset due to Groq free tier token limits.

Qwen2.5-7B was selected as the planned fallback because it crossed 90% accuracy on low-confidence cases at zero API cost. Groq Llama3.1-8B is also being considered as a configurable alternative via `LLM_BACKEND=groq`.

---

## Status

The system is currently under development.

* [x] Offline evaluation complete
* [x] Architecture decided
* [x] Models trained and serialized
* [ ] API endpoints
* [ ] Docker setup
* [ ] PostgreSQL logging
* [ ] Admin metrics endpoint
* [ ] Evaluation endpoint
* [ ] Fallback model integration

---

## Planned Endpoints

**Public:**

```text
POST /classify
POST /classify/batch
GET  /health
```

**Admin:**

```text
GET  /admin/health
GET  /admin/model-info
GET  /admin/metrics
POST /admin/evaluate
POST /admin/threshold
```

Admin endpoints will require an `X-Admin-Key` header.

---

## Planned Stack

* **API:** FastAPI
* **Database:** PostgreSQL
* **Containerization:** Docker
* **Stage 1:** scikit-learn TF-IDF + LinearSVC
* **Stage 2:** BART-large-MNLI using Hugging Face Transformers
* **Fallback:** Qwen2.5-7B-Instruct, planned as a local fallback model
* **Deployment target:** Hugging Face Spaces

---

## Setup

Setup instructions will be updated as implementation progresses.

Planned flow:

```bash
pip install -r requirements.txt
python train.py
docker-compose up
```

---

## Research

The `research/` directory contains the offline evaluation that motivated this project:

* `01_train_and_evaluate.ipynb` — trains models and evaluates fallback candidates
* `02_results_viewer.ipynb` — reads checkpoint and prints comparison tables
* `README.md` — evaluation methodology and full results

---

*Capstone report available on request.*
