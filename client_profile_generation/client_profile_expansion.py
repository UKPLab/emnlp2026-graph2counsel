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

def fix_encoding(raw):
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

def load_prompt(file_name):
    base_dir = "../prompts_system_user/"
    file_path = base_dir + file_name
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def save_as_txt(text, filename):
    """Save dictionary to a JSON file"""
    with open(filename, "w", encoding="utf-8") as file:
        file.write(text)
    print(f"Text saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate profiles from Graph")
    parser.add_argument("-i","--input_dir", type=str, default=".",
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
        if filename.endswith('.json'):
            file_path = os.path.join(args.input_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                graph_dict = json.load(f)

            prompt_system = load_prompt("client_profile_expansion_system.txt")

            prompt_text_user = load_prompt("client_profile_expansion_user.txt")
            prompt_template_user = PromptTemplate(
            input_variables=["Edges"],
            template=prompt_text_user)
            
            prompt_user = prompt_template_user.format(
            Edges = graph_dict["edges"]
            )

            messages = [{'role': 'system', 'content': prompt_system},{'role': 'user', 'content': prompt_user}]

            for attempt in range(3):
                response = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7
                )
    
                output = response.choices[0].message.content
                cleaned = re.sub(r"^```json\n|```$", "", output.strip())
                cleaned = re.sub(r'[\x00-\x1f]+', '', cleaned)
                cleaned = cleaned.encode('utf-8').decode('unicode_escape')

                try:
                    profiles = json.loads(cleaned)
                    break
                except json.JSONDecodeError:
                    if attempt == 2:
                        raise

            for i in range(len(profiles)):
                filename_txt = Path(filename).stem + "_" + str(i) + ".txt"
                text_fixed = fix_encoding(profiles[i]['profile'].encode('utf-8'))
                save_as_txt(text_fixed, os.path.join(args.output_dir, filename_txt))