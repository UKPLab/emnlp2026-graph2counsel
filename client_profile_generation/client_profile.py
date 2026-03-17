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
            
            ex1 = "Name: Miguel Alvarez Age: 39 Gender: Male Occupation: Truck Driver Education: Vocational School Marital Status: Married Family Details: Wife and two teenage children 2. Presenting Problem I constantly feel on edge and irritable, especially during long drives. The problem started three years ago after a near-fatal accident on the highway. At first, the stress level was extreme—I couldn’t even drive for weeks. Over time, I pushed myself back into work, but the tension never fully went away. Now I experience it daily, particularly on high-traffic routes. I’ve tried deep breathing, music, and coffee breaks, but nothing really works. 3. Reason for Seeking Counseling My wife urged me to seek counseling because the irritability is affecting my relationship with her and the kids. 4. Past History (including medical history) I’ve had mild anxiety during stressful periods before but nothing this severe. Never had counseling before. No major illnesses. 5. Academic/occupational functioning level: My job performance has been shaky, sometimes missing deadlines due to anxiety. Interpersonal relationships: Strained at home. Daily life: Sleep is poor, appetite is inconsistent. 6. Social Support System My wife is supportive, though frustrated, and I occasionally talk to a close coworker."
            ex2 = "Name: Aisha Rahman Age: 22 Gender: Female Occupation: University Student Education: Currently pursuing Bachelor's Degree Marital Status: Single Family Details: Lives with parents and younger brother 2. Presenting Problem I experience overwhelming pressure before exams, leading to panic attacks and difficulty concentrating. It started two years ago after failing one midterm. At that time, I felt devastated and ashamed. Since then, the anxiety worsens every exam season. I now dread studying, often procrastinate, and then spiral into self-criticism. I’ve tried meditation apps and cutting caffeine, but the anxiety still hits hard. 3. Reason for Seeking Counseling My most recent exam panic led me to faint in the library, and I realized I can’t keep going like this. 4. Past History (including medical history) I struggled with performance anxiety in high school too, but it was less intense. Never received counseling before. No physical illnesses. 5. Academic/occupational functioning level: Grades are average, slipping compared to earlier years. Interpersonal relationships: I avoid group study out of embarrassment. Daily life: Erratic sleep, sometimes skipping meals before exams. 6. Social Support System I talk to my mother sometimes, but I usually hide my stress to avoid worrying my family."
            
            prompt_text = load_prompt(f"client-profile.txt")
            
            prompt_template = PromptTemplate(
            input_variables=["Edges","example_output_1","example_output_2"],
            template=prompt_text)
            
            prompt = prompt_template.format(
            Edges = graph_dict['edges'],
            example_output_1 = ex1,
            example_output_2 = ex2
            )

            messages = [{'role': 'user', 'content': prompt}]

            response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7
            )

            output = response.choices[0].message.content
            filename_txt = Path(filename).stem + ".txt"
            save_as_txt(output, os.path.join(args.output_dir, filename_txt))