## Client Profile Generation

### `client_profile.py`
Generates a single client profile for each CPG.

| Argument | Description |
|---|---|
| `-i` | Path to the directory containing the CPG representations |
| `-o` | Path to the output directory to save the generated client profiles |
| `-oai_key` | OpenAI API key |
```bash
python client_profile.py -i input_directory -o output_directory -oai_key openai_api_key
```

### `client_profile_expansion.py`
Generates 10 diverse client profiles from each CPG.

| Argument | Description |
|---|---|
| `-i` | Path to the directory containing the CPG representations |
| `-o` | Path to the output directory to save the generated client profiles |
| `-oai_key` | OpenAI API key |
```bash
python client_profile_expansion.py -i input_directory -o output_directory -oai_key openai_api_key
```
