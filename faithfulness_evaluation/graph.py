import time
import os
import json
import csv
import argparse
import pandas as pd
import re
import openai
from tqdm import tqdm
import numpy as np
import codecs
import tiktoken
from langchain.prompts import PromptTemplate
from pathlib import Path
import ast

def load_prompt(file_name):
    base_dir = "prompts/"
    file_path = base_dir + file_name
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def get_utterances(prompt, retries=5):
    for attempt in range(retries):
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = response.choices[0].message.content

        try:
            return ast.literal_eval(content)
        except (ValueError, SyntaxError):
            if attempt == retries - 1:
                break
            # regenerate on error
            continue

def save_as_json(dictionary, filename):
    """Save dictionary to a JSON file"""
    with open(filename, 'w') as file:
        json.dump(dictionary, file, indent=4)
    print(f"Dictionary saved to {filename}")

df = pd.read_csv("../therapy_strategies_summary.csv", sep=',')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Faithfulness on CPG")
    parser.add_argument("-i","--input_dir", type=str, default=".",
                        help="Directory to read the dialogues")
    parser.add_argument("-i_graph","--input_dir_graph", type=str, default=".",
                        help="Directory to read the graphs")
    parser.add_argument("-o","--output_dir", type=str, default=".",
                        help="Directory to to save the results.")
    parser.add_argument("-oai_key","--openai_key", type=str, default=".",
                        help="OpenAI API key for GPT-4o")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    openai.api_key = args.openai_key
    for filename in os.listdir(args.input_dir):
        if filename.endswith('.json') and filename not in os.listdir(args.output_dir):
            graph_results = {}
            process_count = 0
            process_utt_count = 0
            with open(os.path.join(args.input_dir_graph,'_'.join(filename.split('_')[:3]) + ".json")) as f:
                graph_dict = json.load(f)
            if graph_dict['nodes']:
                with open(os.path.join(args.input_dir,filename)) as f:
                    dialogues = json.load(f)
                counselor_list = []
                client_list = []

                for dialogue in dialogues['History']:
                    if dialogue['speaker'] == 'counselor':
                        counselor_list.append(dialogue['message'])
                    else:
                        client_list.append(dialogue['message'])
                prompt_text = load_prompt("graph.txt")
                prompt_template = PromptTemplate(input_variables=["psychological_process","client_utt_list"],
                                        template=prompt_text)
                for node in graph_dict['nodes']:
                    prompt = prompt_template.format(psychological_process=node['type'],client_utt_list=client_list)
                    utterances = get_utterances(prompt)
                    graph_results[node['type']] = utterances

                for key in graph_results.keys():
                    if key!='score':
                        process_count = process_count+1
                        if graph_results[key]:
                            process_utt_count = process_utt_count + 1
                graph_results['score'] = process_utt_count/process_count
                save_as_json(graph_results, os.path.join(args.output_dir, filename))