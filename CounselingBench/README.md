## CounselingBench

Download the data files from [CounselingBench](https://huggingface.co/datasets/ckmjx/CounselingBench) before running any scripts.

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
Contains prompts for generating responses to CounselingBench questions across three prompting strategies: Zero-Shot (ZS), Few-Shot (FS), and Few-Shot Chain-of-Thought (FS-CoT).

---

### `counselingbench_eval_zs.py`
Generates responses to questions using zero-shot prompting.

| Argument | Description |
|---|---|
| `-m` | Path to the fine-tuned model weights |
| `-o` | Path to the output file |
```bash
python counselingbench_eval_zs.py -m path_to_model -o output_file
```

---

### `counselingbench_eval_fs.py`
Generates responses to questions using few-shot prompting.

| Argument | Description |
|---|---|
| `-m` | Path to the fine-tuned model weights |
| `-o` | Path to the output file |
```bash
python counselingbench_eval_fs.py -m path_to_model -o output_file
```

---

### `counselingbench_eval_fs_cot.py`
Generates responses to questions using few-shot chain-of-thought prompting.

| Argument | Description |
|---|---|
| `-m` | Path to the fine-tuned model weights |
| `-o_labels` | Path to the output file for predicted labels |
| `-o_reasons` | Path to the output file for reasoning chains |
```bash
python counselingbench_eval_fs_cot.py -m path_to_model -o_labels output_file_labels -o_reasons output_file_reasoning_chains
```

---

### `results.ipynb`
Aggregates results across all prompting methods. Update the paths to the output label file and output reasoning file before running. Returns:
- Classification metrics (F1, Accuracy, etc.) for predicted labels across ZS, FS, and FS-CoT
- Reference-based metrics for FS-CoT reasoning chains

For ROSCOE evaluation of reasoning chains, follow the instructions in the [ROSCOE repository](https://github.com/facebookresearch/ParlAI/tree/main/projects/roscoe).
