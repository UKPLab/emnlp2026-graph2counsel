<p  align="center">
  <img src='img/Graph2Counsel_Diagram.png' width='600'>
</p>

# Graph2Counsel: Clinically Grounded Synthetic Counseling Dialogue Generation from Client Psychological Graphs
[![Arxiv](https://img.shields.io/badge/Arxiv-YYMM.NNNNN-red?style=flat-square&logo=arxiv&logoColor=white)](https://put-here-your-paper.com)
[![License](https://img.shields.io/github/license/UKPLab/graph2counsel)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/badge/Python-3.9-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://github.com/UKPLab/graph2counsel/actions/workflows/main.yml/badge.svg)](https://github.com/UKPLab/graph2counsel/actions/workflows/main.yml)

> **Abstract:** Rising demand for mental health support has increased interest in using Large Language Models (LLMs) for counseling. However, adapting LLMs to this high-risk safety-critical domain is hindered by the scarcity of real-world counseling data due to privacy constraints. Synthetic datasets provide a promising alternative, but existing approaches often rely on unstructured or semi-structured text inputs and overlook structural dependencies between a client's cognitive, emotional, and behavioral states, often producing psychologically inconsistent interactions and reducing data realism and quality. We introduce Graph2Counsel, a framework for generating synthetic counseling sessions grounded in Client Psychological Graphs (CPGs) that encode relationships among clients’ thoughts, emotions, and behaviors. Graph2Counsel employs a structured prompting pipeline guided by counselor strategies and CPG, and explores prompting strategies including CoT and Multi-Agent Feedback. Graph2Counsel produces 760 sessions from 76 CPGs across diverse client profiles. In expert evaluation, our dataset outperforms prior datasets on specificity, counselor competence, authenticity, conversational flow, and safety, with substantial inter-annotator agreement (Krippendorff's $\alpha$ = 0.70). Fine-tuning an open-source model on this dataset improves performance on CounselingBench and CounselBench, showing downstream utility.

Contact person: [Aishik Mandal](mailto:aishik.mandal@tu-darmstadt.de) 

[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/
)

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

## Creating the environment

To create the environment for Inference and Evaluation (i.e. run scripts in folder src/ and evaluation/) use:

```
python3 -m venv inf_env
source inf_env/bin/activate
pip install -r requirements_inf_eval.txt
```

To create the environment for Qlora Fine-tuning (i.e. run scripts in folder qlora/) use:

```
python3 -m venv qlora_env
source qlora_env/bin/activate
pip install -r requirements_qlora.txt
```

## Repository Structure
```
graph2counsel/
│
├── prompts_system_user/                  # System and user prompts used in configurations and client_profile_generation
│
├── Graph_Dicts.ipynb                     # Build KG triples from CPGs
│
├── configurations/                       # Generation configuration experiments
│   └── gc_cpg_profile_expanded.py        # Final dataset generation (GC optimal config)
│
├── client_profile_generation/            # Scripts for generating client profiles
│
├── LLM_judge_evaluation/                 # CTRS and WAI evaluation with LLM-as-a-judge
│
├── faithfulness_evaluation/              # Faithfulness of sessions to CPG inputs
│
├── evaluation/                           # General evaluation utilities
│
├── qlora_prompts/                        # Prompts used for QLoRA fine-tuning
│
├── create_qlora_fine-tuning_datasets.ipynb  # Prepare datasets for QLoRA
│
│── graph2counsel_qlora.py                # Fine-tune on Graph2Counsel data
│── sqpconv_qlora.py                      # Fine-tune on SQPsychConv data
│── magnet_qlora.py                       # Fine-tune on MAGneT data
│
├── CounselBench/                         # Downstream evaluation on CounselBench
├── CounselingBench/                      # Downstream evaluation on CounselingBench
│
├── requirements_inf_eval.txt             # Dependencies for inference & evaluation
└── requirements_qlora.txt                # Dependencies for QLoRA fine-tuning
```

## Downloading Data

| Dataset | Link |
|---|---|
| **Graph2Counsel** (ours) | Coming soon |
| **SQPsychConv** (We use Gemma split) | [HuggingFace](https://huggingface.co/collections/AIMH/sqpsychconv) |
| **CACTUS** | [HuggingFace](https://huggingface.co/datasets/LangAGI-Lab/cactus) |
| **MAGneT** | [GitHub](https://github.com/UKPLab/arxiv2025-MAGneT) |
| **CounselBench-Eval** | [HuggingFace](https://huggingface.co/datasets/izi-ano/CounselBench-Eval) |
| **CounselBench-Adv** | [HuggingFace](https://huggingface.co/datasets/izi-ano/CounselBench-Adv) |
| **CounselingBench** | [HuggingFace](https://huggingface.co/datasets/ckmjx/CounselingBench) |

Place CounselBench data in the `CounselBench/` directory and CounselingBench data in `CounselingBench/`.

## Reproducing Experiments

Follow these steps in order after setting up your environment and downloading data.

**1. Build CPG Representations**

Run `Graph_Dicts.ipynb` to create `(process_i, relation, process_j)` KG triples from the CPGs. Outputs are saved to `CPG_processed/`.

**2. Generate Client Profiles**

See `client_profile_generation/` for instructions and scripts.

**3. Run Configuration Experiments**

Use files in `configurations/` to experiment with different generation strategies. See the folder README for details. We found the **GC (CPG + Profile)** configuration to be optimal.

**4. Evaluate Generated Sessions**

Use files in `LLM_judge_evaluation/` to score sessions on CTRS and WAI metrics using LLM-as-a-judge. See the folder README for details.

**5. Generate the Final Dataset**

Using the optimal configuration:
```bash
python configurations/gc_cpg_profile_expanded.py
```

**6. Evaluate Faithfulness**

Use files in `faithfulness_evaluation/` to assess how faithfully generated sessions reflect their CPG and profile inputs.

**7. Prepare Fine-tuning Datasets**

Run `create_qlora_fine-tuning_datasets.ipynb`, updating dataset paths to match your download locations.

**8. Run QLoRA Fine-tuning**

Update the base model path and output model path in the relevant script (we used `Llama3-8B-Instruct`), then run:
```bash
deepspeed --num_gpus=num_gpus qlora/graph2counsel_qlora.py
# or for baselines:
deepspeed --num_gpus=num_gpus qlora/sqpconv_qlora.py
deepspeed --num_gpus=num_gpus qlora/magnet_qlora.py
```

**9. Downstream Evaluation**

Use scripts in `CounselBench/` and `CounselingBench/` for benchmark evaluation. See each folder for details.

## Cite

Please use the following citation:

```
@InProceedings{smith:20xx:CONFERENCE_TITLE,
  author    = {Smith, John},
  title     = {My Paper Title},
  booktitle = {Proceedings of the 20XX Conference on XXXX},
  month     = mmm,
  year      = {20xx},
  address   = {Gotham City, USA},
  publisher = {Association for XXX},
  pages     = {XXXX--XXXX},
  url       = {http://xxxx.xxx}
}
```

## Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 
