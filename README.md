# Telugu Complaint Summariser on Amazon Bedrock

Summarise Telugu citizen complaints **in Telugu** with **Qwen3-32B on Amazon Bedrock**, without deploying any GPU servers.

For each complaint you get:
- a 1–2 sentence Telugu summary
- the category
- the requested action
- an urgency flag
- a verbatim quote from the complaint, so a reviewer can check the summary

For the whole set you get:
- counts per category, computed in code rather than by the model
- a Telugu group summary for each category, citing complaint IDs

```
Telugu complaints (.txt .csv .json .jsonl .xlsx .docx)
        │
        ▼
bedrock_summarize.py ──► Amazon Bedrock (ap-south-1, Mumbai) ── Qwen3-32B
        │                    one call per complaint (parallel), one per category
        ▼
output/report.html   output/report.json   output/complaints.jsonl
```

## Why this setup

| Decision | Reason |
|---|---|
| **Qwen3-32B** | Best Telugu → Telugu summaries among the models we tested on 100 Telugu complaints: Qwen3-32B in full precision, FP8 and INT4, Qwen3-4B, Qwen3.5-0.8B, Gemma 3 12B, Gemma 4 31B, Sarvam and IndicBART. Apache 2.0, so commercial use is allowed. |
| **Bedrock** | Managed and pay per token: no GPUs to deploy for the demo. Qwen3-32B is offered **in-region in Mumbai (ap-south-1)**. |
| **Direct Telugu → Telugu** | Translating via English added translation errors and invented emphasis. |
| **No RAG or knowledge graph** | See [below](#why-not-rag-or-a-knowledge-graph). |
| **Same model offline later** | For an air-gapped client, the same Qwen3-32B runs self-hosted (FP8, one 48 GB GPU, vLLM) with the same prompts. |

## Model facts (from the AWS model card)

- Model ID: `qwen.qwen3-32b-v1:0` (endpoint `bedrock-runtime`)
- Regions: Mumbai `ap-south-1` is supported **in-region**. Cross-region inference is not supported for this model, so requests stay in the chosen region.
- Context window: 32K tokens. Maximum output: 8K tokens.
- APIs: Converse, Invoke, and Chat Completions (OpenAI-compatible).

## Steps

### 1. AWS account and region
Use an AWS account where you can create IAM users or roles. In the console, switch the region to **Asia Pacific (Mumbai) ap-south-1**.

### 2. Model access
Open **Amazon Bedrock → Model catalog → Qwen3 32B** in ap-south-1. If the console shows **Request access** or **Enable**, accept the terms; serverless models are otherwise enabled on first use.

### 3. Credentials (choose one)

**Option A, quickest for a demo: Bedrock API key**
Bedrock console → **API keys** → *Generate long-term API key*, then:
```bash
export AWS_BEARER_TOKEN_BEDROCK="<your Bedrock API key>"
```

**Option B, recommended: an IAM user or role with least privilege**
Attach [`iam-policy.json`](iam-policy.json), which only allows invoking Qwen3-32B in ap-south-1. Then:
```bash
aws configure          # access key, secret, region ap-south-1
# or, on EC2: attach an instance role with this policy, and no keys are needed
```

### 4. Install
Python 3.10 or newer:
```bash
git clone <this repo> && cd <this repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 5. Smoke test (one call)
```bash
aws bedrock-runtime converse --region ap-south-1 --model-id qwen.qwen3-32b-v1:0 \
  --messages '[{"role":"user","content":[{"text":"ఒక వాక్యంలో నమస్కారం చెప్పండి. /no_think"}]}]'
```
A Telugu reply means access works.

### 6. Check your input without calling the model
```bash
python bedrock_summarize.py examples/ --dry-run
```

### 7. Run
```bash
python bedrock_summarize.py examples/ --out output/ --region ap-south-1 --workers 8
open output/report.html        # macOS; use xdg-open on Linux
```

Input formats (files or folders, mixed):

| Format | One complaint per | Text field |
|---|---|---|
| `.txt`, `.md`, `.docx` | file | whole file |
| `.csv`, `.xlsx` | row | column `text` (optional `id`) |
| `.json` | object in a list, or `{"complaints": [...]}` | field `text` (optional `id`) |
| `.jsonl` | line | field `text` (optional `id`) |

PDFs and scanned images are out of scope for this version. Convert them to text first.

Outputs in `--out`:
- `report.html`: category counts, Telugu group summaries, and every complaint's summary, urgency flag and quote.
- `report.json`: counts, flagged IDs, failures, token usage and run time.
- `complaints.jsonl`: one line per complaint with all fields.

Useful flags:
- `--workers`: number of parallel requests. Lower it if you hit throttling.
- `--max-tokens`: default 3000. Telugu uses many tokens, and lower values can cut summaries short.
- `--model-id`: use a different Bedrock model.

## Cost and speed
Pay per input and output token; see [Bedrock pricing](https://aws.amazon.com/bedrock/pricing/) for current rates in ap-south-1. Each complaint is one call, plus one call per category. `report.json` records the exact token counts, so you can multiply by the current price. Increase `--workers` for speed, within your account's Bedrock quotas.

## Data and safety notes
- Bedrock is a managed cloud service. Send only **synthetic or approved data** until the deployment is moved offline. Requests stay in the chosen region.
- The urgency flag is a prompt for human review, not a decision.
- Each quote is checked against the source text. Quotes that don't match exactly are marked "not verbatim — check".
- Counts and percentages are computed by the script, never by the model.

## Moving offline (air-gapped client)
The same model runs on the client's own GPU: **Qwen3-32B-FP8** (about 32 GB, one 48 GB+ GPU such as an L40S, A100 or H100) served by vLLM's OpenAI-compatible server. This script can point to it with small changes, or use the full offline app with `--backend vllm`. No internet is needed at runtime.

## Why not RAG or a knowledge graph
- The task is **summarisation**, not search or question answering. Each complaint fits in a single model prompt, so there is nothing to retrieve.
- Group summaries are built from all per-complaint summaries, and counts are computed exactly in code. Retrieval would only risk dropping complaints.
- RAG and knowledge graphs need Telugu embedding or entity-extraction models plus a vector or graph database. That is more to build, test and later ship into an air-gapped server, with no gain in summary quality.
- Every summary already cites complaint IDs and a verbatim quote, which is simpler to audit.
- They become useful later for question answering across thousands of complaints, tracking villages or departments over time, or detecting duplicate complaints.

## Troubleshooting

| Error | Fix |
|---|---|
| `AccessDeniedException` | Enable Qwen3 32B in the Bedrock console (step 2), and check that the IAM policy region is `ap-south-1`. |
| `ValidationException: model identifier is invalid` | Wrong region. Use `--region ap-south-1` and model ID `qwen.qwen3-32b-v1:0`. |
| `ThrottlingException` | Lower `--workers`. The script already retries with backoff. |
| Summary cut off, or "no valid JSON" | Raise `--max-tokens`, up to 8000. |
| Telugu shows as boxes in the HTML | Install a Telugu font such as Noto Sans Telugu. |

## Repository contents

| File | Purpose |
|---|---|
| `bedrock_summarize.py` | The summariser (boto3 Converse API) |
| `iam-policy.json` | Least-privilege IAM policy (Qwen3-32B in ap-south-1 only) |
| `requirements.txt` | `boto3`, plus `openpyxl` for `.xlsx` |
| `examples/` | Two synthetic Telugu complaints, as `.txt` and `.csv` |
