# Offline Evaluation

Compares fallback model candidates for the Clinical AI Inference Router
on 1,905 labeled TCGA pathology reports across 32 cancer types.

## Notebooks

**01_train_and_evaluate.ipynb**
Trains TF-IDF + LinearSVC and evaluates all fallback candidates.
Each section is independent. Run only the sections you need.
Results save to Google Drive automatically after each section.

**02_results_viewer.ipynb**
Reads results from Drive and prints comparison tables.
No GPU needed. Run any time to see current results.

## Results

| Model | All cases | Low-confidence | Net gain | Status |
|---|---|---|---|---|
| Hybrid (baseline) | 83.6% | 53.0% | — | baseline |
| Qwen2.5-1.5B | 81.1% | 78.6% | +90 | evaluated |
| Qwen2.5-3B | 89.9% | 85.8% | +115 | evaluated |
| Qwen2.5-7B | 93.7% | 91.5% | +135 | **chosen** |
| Claude Haiku | 97.5% | 95.4% | +149 | evaluated |
| Groq Llama3.1-8B* | 86.7% | 86.4% | +25 | evaluated |

*N=300 subset — full evaluation limited by Groq free tier (500K tokens/day)

## Decision

Qwen2.5-7B selected as default fallback. Matches Claude Haiku on
low-confidence cases at zero cost with no external API dependency.

Groq Llama3.1-8B available as alternative via `LLM_BACKEND=groq`
for RAM-constrained deployments.

## Reproducing Results

1. Open `01_train_and_evaluate.ipynb` in Google Colab
2. Set `ANTHROPIC_API_KEY` and `GROQ_API_KEY` in Colab Secrets
3. Run Section 0, then any section you want to reproduce
4. Results save automatically to your Google Drive