## Configurations

All configurations share the following common arguments:

| Argument | Applies To | Description |
|---|---|---|
| `-i_graph` | CPG and CPG+Profile variants | Path to the CPG representation directory |
| `-i_profile` | Profile and CPG+Profile variants | Path to the client profile directory |
| `-o` | All | Path to the output directory to save generated sessions |
| `-oai_key` | All | OpenAI API key |

---

### Base Prompting

| Variant | Script |
|---|---|
| Base (CPG) | `base_cpg.py` |
| Base (Profile) | `base_profile.py` |
| Base (CPG+Profile) | `base_cpg_profile.py` |
```bash
python base_cpg.py -i_graph cpg_directory -o output_directory -oai_key openai_api_key
python base_profile.py -i_profile profile_directory -o output_directory -oai_key openai_api_key
python base_cpg_profile.py -i_graph cpg_directory -i_profile profile_directory -o output_directory -oai_key openai_api_key
```

---

### GC Prompting

| Variant | Script |
|---|---|
| GC (CPG) | `gc_cpg.py` |
| GC (Profile) | `gc_profile.py` |
| GC (CPG+Profile) | `gc_cpg_profile.py` |
```bash
python gc_cpg.py -i_graph cpg_directory -o output_directory -oai_key openai_api_key
python gc_profile.py -i_profile profile_directory -o output_directory -oai_key openai_api_key
python gc_cpg_profile.py -i_graph cpg_directory -i_profile profile_directory -o output_directory -oai_key openai_api_key
```

---

### GC+CoT Prompting

| Variant | Script |
|---|---|
| GC+CoT (CPG) | `CoT_gc_cpg.py` |
| GC+CoT (Profile) | `CoT_gc_profile.py` |
| GC+CoT (CPG+Profile) | `CoT_gc_cpg_profile.py` |
```bash
python CoT_gc_cpg.py -i_graph cpg_directory -o output_directory -oai_key openai_api_key
python CoT_gc_profile.py -i_profile profile_directory -o output_directory -oai_key openai_api_key
python CoT_gc_cpg_profile.py -i_graph cpg_directory -i_profile profile_directory -o output_directory -oai_key openai_api_key
```

---

### GC+MA Prompting

In addition to the common arguments, GC+MA variants take:

| Argument | Description |
|---|---|
| `-t` | Number of feedback turns |

| Variant | Script |
|---|---|
| GC+MA (CPG) | `ma_gc_cpg.py` |
| GC+MA (Profile) | `ma_gc_profile.py` |
| GC+MA (CPG+Profile) | `ma_gc_cpg_profile.py` |
```bash
python ma_gc_cpg.py -i_graph cpg_directory -t feedback_turns -o output_directory -oai_key openai_api_key
python ma_gc_profile.py -i_profile profile_directory -t feedback_turns -o output_directory -oai_key openai_api_key
python ma_gc_cpg_profile.py -i_graph cpg_directory -i_profile profile_directory -t feedback_turns -o output_directory -oai_key openai_api_key
```

---

### Expansion of Optimal Configuration — GC (CPG+Profile)

Script: `gc_cpg_profile_expanded.py`
```bash
python gc_cpg_profile_expanded.py -i_graph cpg_directory -i_profile profile_directory -o output_directory -oai_key openai_api_key
```
