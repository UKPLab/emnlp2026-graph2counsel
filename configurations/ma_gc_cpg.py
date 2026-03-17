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
import chardet

def load_prompt(file_name):
    base_dir = "../prompts_system_user/"
    file_path = base_dir + file_name
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def save_as_json(dictionary, filename):
    """Save dictionary to a JSON file"""
    with open(filename, 'w') as file:
        json.dump(dictionary, file, indent=4)
    print(f"Dictionary saved to {filename}")

def generate_history(history):
    history_text = '\n'.join(
            [
                f"{message['speaker'].capitalize()}: {message['message']}"
                for message in history
            ]
    )
    return history_text

def fix_encoding_txt(txt):
    """Detect encoding and safely re-decode as UTF-8."""
    raw = txt.encode("utf-8")

    # Detect probable encoding
    detected = chardet.detect(raw)
    enc = detected["encoding"] or "utf-8"

    try:
        text = raw.decode(enc)
    except Exception:
        # fallback if detection fails
        text = raw.decode("utf-8", errors="replace")

    # Fix double-decoding like Ã© → é by re-encoding and decoding properly
    try:
        text_fixed = text.encode("latin1").decode("utf-8")
    except Exception:
        text_fixed = text

    # Remove leading spaces but preserve blank lines
    lines = [line.lstrip() for line in text_fixed.splitlines()]
    text_fixed = "\n".join(lines)

    return text_fixed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate sessions from Graph")
    parser.add_argument("-i","--input_dir", type=str, default=".",
                        help="Directory to read the graphs")
    parser.add_argument("-o","--output_dir", type=str, default=".",
                        help="Directory to to save the results.")
    parser.add_argument("-t","--turns", type=int, default=1,
                        help="Maximum number of feedback turns")
    parser.add_argument("-oai_key","--openai_key", type=str, default=".",
                        help="OpenAI API key for GPT-4o")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    openai.api_key = args.openai_key

    df = pd.read_csv("../therapy_strategies_summary.csv", sep=',')
    
    for filename in os.listdir(args.input_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(args.input_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                graph_dict = json.load(f)

            df_mod = df[df["Session"]=='_'.join(filename.split('_')[:2])]
            counselor_strategy_list = df_mod["Strategy"].tolist()

            prompt_system_initial = load_prompt("ma_gc_cpg_system_initial.txt")

            prompt_text_user_initial = load_prompt("ma_gc_cpg_user_initial.txt")
            prompt_template_user_initial = PromptTemplate(
            input_variables=["Edges","counselor_strats"],
            template=prompt_text_user_initial)
            
            prompt_user_initial = prompt_template_user_initial.format(
            Edges = graph_dict["edges"],
            counselor_strats = counselor_strategy_list
            )

            messages_initial = [{'role': 'system', 'content': prompt_system_initial},{'role': 'user', 'content': prompt_user_initial}]

            prompt_system_feedback = load_prompt("ma_gc_cpg_system_feedback.txt")

            prompt_text_user_feedback = load_prompt("ma_gc_cpg_user_feedback.txt")
            prompt_template_user_feedback = PromptTemplate(
            input_variables=["Edges","counselor_strats","dialogue"],
            template=prompt_text_user_feedback)

            prompt_system_after_feedback = load_prompt("ma_gc_cpg_system_after_feedback.txt")

            prompt_text_user_after_feedback = load_prompt("ma_gc_cpg_user_after_feedback.txt")
            prompt_template_user_after_feedback = PromptTemplate(
            input_variables=["Edges","counselor_strats","dialogue","feedback"],
            template=prompt_text_user_after_feedback)

            for attempt in range(3):

                response = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages_initial,
                temperature=0.7
                )
    
                output = response.choices[0].message.content
                cleaned = re.sub(r"^```json\n|```$", "", output.strip())
                cleaned = re.sub(r'[\x00-\x1f]+', '', cleaned)
                cleaned = cleaned.encode('utf-8').decode('unicode_escape')

                try:
                    dialogue = json.loads(cleaned)
                    break
                except json.JSONDecodeError:
                    if attempt == 2:
                        raise

            for turns in range(args.turns):
                clean_dialogue = []
                for i in range(len(dialogue)):
                    clean_dialogue.append({"speaker":dialogue[i]["speaker"],"message":fix_encoding_txt(dialogue[i]["message"])})
                dialogue_text = generate_history(clean_dialogue)

                prompt_user_feedback = prompt_template_user_feedback.format(
                Edges = graph_dict["edges"],
                counselor_strats = counselor_strategy_list,
                dialogue = dialogue_text
                )
    
                messages_feedback = [{'role': 'system', 'content': prompt_system_feedback},{'role': 'user', 'content': prompt_user_feedback}]

                feedback_i = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages_feedback,
                temperature=0.7
                )

                prompt_user_after_feedback = prompt_template_user_after_feedback.format(
                Edges = graph_dict["edges"],
                counselor_strats = counselor_strategy_list,
                dialogue = dialogue_text,
                feedback = feedback_i.choices[0].message.content
                )
    
                messages_after_feedback = [{'role': 'system', 'content': prompt_system_after_feedback},{'role': 'user', 'content': prompt_user_after_feedback}]

                for attempt_feedback in range(3):
                
                    response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=messages_after_feedback,
                    temperature=0.7
                    )
                    
                    output = response.choices[0].message.content
                    cleaned = re.sub(r"^```json\n|```$", "", output.strip())
                    cleaned = re.sub(r'[\x00-\x1f]+', '', cleaned)
                    cleaned = cleaned.encode('utf-8').decode('unicode_escape')
                    try:
                        dialogue = json.loads(cleaned)
                        break
                    except json.JSONDecodeError:
                        if attempt_feedback == 2:
                            raise
            
            dialogue_dict = {"History":dialogue}
            d_cleaned = {"History":[]}

            for i in range(len(dialogue_dict["History"])):
                d_cleaned["History"].append({"speaker":dialogue_dict["History"][i]["speaker"],"message":fix_encoding_txt(dialogue_dict["History"][i]["message"])})
                
            save_as_json(d_cleaned, os.path.join(args.output_dir, filename))