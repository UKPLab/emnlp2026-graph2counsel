## Faithfulness Evaluation

Faithfulness measures how well generated sessions reflect their inputs:
- **CPG Faithfulness:** The proportion of CPG nodes reflected in at least one utterance in the generated session.
- **Profile Faithfulness:** Binary score — `1` if no utterance contradicts the profile input, `0` otherwise.

### `prompts/`
Contains prompts for evaluating CPG and Profile faithfulness.

---

### `graph.py`
Assesses CPG faithfulness.

| Argument | Description |
|---|---|
| `-i` | Path to the input directory for the generated sessions |
| `-i_graph` | Path to the input directory for the CPGs |
| `-o` | Path to the output directory for saving results |
| `-oai_key` | OpenAI API key |
```bash
python graph.py -i input_directory_session -i_graph input_directory_graph -o output_directory -oai_key openai_api_key
```

---

### `profile.py`
Assesses profile faithfulness.

| Argument | Description |
|---|---|
| `-i` | Path to the input directory for the generated sessions |
| `-i_profile` | Path to the input directory for the profiles |
| `-o` | Path to the output directory for saving results |
| `-oai_key` | OpenAI API key |
```bash
python profile.py -i input_directory_session -i_profile input_directory_profile -o output_directory -oai_key openai_api_key
```

---

### `results.ipynb`
Aggregates CPG and profile faithfulness results. Update the following paths in the notebook before running:
- Output directory for CPG faithfulness results
- Output directory for profile faithfulness results
