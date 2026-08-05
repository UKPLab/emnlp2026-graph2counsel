# Therapy-Eval

Therapy-Eval is a multi-turn counseling evaluation framework. A locally-hosted Llama-3-8B-Instruct model role-plays a therapy client seeded from a
character card (100 cards sampled from Eeyore), holds a multi-turn conversation with a counselor model under test, and the resulting session is then scored by GPT-4o as an LLM judge on five criteria: **specificity, authenticity, competence, flow, and safety**. It's used in the paper to demonstrate the effectiveness of our fine-tuned model in realistic therapeutic conversations, comparing it against counselor models fine-tuned on three baseline datasets (SQPsychConv, MAGNeT, CACTUS).

## Workflow

1. **Generate sessions** — role-play a client against each counselor checkpoint
   (`generate-sessions.py`).
2. **Score sessions** — judge each generated session on the five criteria with GPT-4o
   (`therapy-eval-gpt4o.py`).

---

### `data/eeyore.json`
100 client profiles (`{"profile": "..."}`) used to seed the client role-play, sampled
from the Eeyore dataset (Liu et al., 2025).

### `prompts/`
Contains the GPT-4o judge prompt for each of the five Therapy-Eval criteria
(`specificity.txt`, `authenticity.txt`, `competence.txt`, `flow.txt`, `safety.txt`).

---

### `generate-sessions.py`
Role-plays a client (via a locally-hosted Llama-3-8B-Instruct-style model) against each of four counselor checkpoints — the Graph2Counsel-fine-tuned model and three baselines (SQPsychConv, MAGNeT, CACTUS) — producing one multi-turn session per client profile per counselor. Sessions end early if the client signals the conversation feels complete, and otherwise run for up to 20 client/counselor exchanges (40 utterances total).

| Argument | Description |
|---|---|
| `-i` | Path to the client profiles JSON file (e.g. `data/eeyore.json`) |
| `-o` | Directory to save the generated sessions |
| `--client-model-path` | Local path or HF repo ID of the client-simulator model |
| `--graph2counsel-model-path` | Local path or HF repo ID of the Graph2Counsel-fine-tuned counselor checkpoint |
| `--sqpsychconv-model-path` | Local path or HF repo ID of the SQPsychConv-fine-tuned counselor checkpoint |
| `--magnet-model-path` | Local path or HF repo ID of the MAGNeT-fine-tuned counselor checkpoint |
| `--cactus-model-path` | Local path or HF repo ID of the CACTUS-fine-tuned counselor checkpoint |

```bash
python generate-sessions.py \
  -i data/eeyore.json \
  -o results/ \
  --client-model-path meta-llama/Meta-Llama-3-8B-Instruct \
  --graph2counsel-model-path <path-or-hf-repo-id> \
  --sqpsychconv-model-path <path-or-hf-repo-id> \
  --magnet-model-path <path-or-hf-repo-id> \
  --cactus-model-path <path-or-hf-repo-id>
```

Sessions are written to `<output_dir>/therapy_final/<method>.json`, where `<method>` is one of `graph2counsel`, `sqpsychconv`, `magnet`, `cactus`.

### `therapy-eval-gpt4o.py`
Scores each of the five Therapy-Eval criteria for every generated session using GPT-4o.

| Argument | Description |
|---|---|
| `-i` | Path to the sessions JSON file to be rated (e.g. one of the files produced above) |
| `-o` | Directory to save the results |
| `-oai_key` | OpenAI API key |

```bash
python therapy-eval-gpt4o.py -i results/therapy_final/graph2counsel.json -o scores/ -oai_key <openai_api_key>
```

Results are written to `<output_dir>/<input_stem>_results.json`, with a
`{"score": ..., "justification": ...}` entry per criterion per session.

If this folder is merged into the main Graph2Counsel repo, it can instead run inside the
repo's existing `inf_env` (see `requirements_inf_eval.txt`), which already covers these
dependencies.
