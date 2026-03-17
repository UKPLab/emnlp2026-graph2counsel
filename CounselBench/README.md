## CounselBench

Download the data files from [CounselBench-Eval](https://huggingface.co/datasets/izi-ano/CounselBench-Eval) and [CounselBench-Adv](https://huggingface.co/datasets/izi-ano/CounselBench-Adv) before running any scripts.

Start a vLLM server for the fine-tuned model using:
```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --dtype float16 \
  --host 0.0.0.0 \
  --port $PORT
```

---

### `prompts/`
Contains prompts for generating responses to CounselBench queries.

---

### `counselbench_eval.py`
Generates responses to queries in CounselBench-Eval.

| Argument | Description |
|---|---|
| `-m` | Path to the fine-tuned model |
| `-o` | Path to the output file to save responses |
```bash
python counselbench_eval.py -m path_to_model -o output_file
```

---

### `counselbench_adv.py`
Generates responses to queries in CounselBench-Adv.

| Argument | Description |
|---|---|
| `-m` | Path to the fine-tuned model |
| `-o` | Path to the output file to save responses |
```bash
python counselbench_adv.py -m path_to_model -o output_file
```

---

### `evaluation/`
Contains scripts for evaluating generated responses for both CounselBench-Eval and CounselBench-Adv.

#### `evaluation/eval/`
Evaluation for CounselBench-Eval responses.
- `prompts/` — Contains the evaluation prompt.
- `results.ipynb` — Scores and aggregates responses. Update the following before running:
  - Path to the responses directory
  - OpenAI API key
  - Path to save output files

#### `evaluation/adv/`
Evaluation for CounselBench-Adv responses.
- `prompts/` — Contains the evaluation prompt.
- `issues_examples.json` — Few-shot examples for each failure mode.
- `results.ipynb` — Scores and aggregates responses. Update the following before running:
  - Path to the responses directory
  - OpenAI API key
  - Path to save output files
