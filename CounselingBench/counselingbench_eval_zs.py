import pandas as pd
import torch
from langchain.prompts import PromptTemplate
import openai
import re
import json
import argparse

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"  # Any value works
)

def generate(prompt,model_path) -> str:
    response = client.completions.create(
    model=model_path,
    prompt=prompt,
    temperature=0,
    top_p=0.9,
    max_tokens=512
    )
    return response.choices[0].text

def load_prompt(file_name):
    base_dir = "prompts/"
    file_path = base_dir + file_name
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get zero-shot responses in CounselingBench.")
    parser.add_argument("-m","--model_path", type=str, default=".",
                        help="Path to fine-tuned model weights.")
    parser.add_argument("-o","--output_file", type=str, default=".",
                        help="File to save the session results.")

    args = parser.parse_args()
    
    df = pd.read_csv("counselingBench.csv")
    df_filled = df.fillna("")
    prompt_text = load_prompt(f"zs.txt")
    prompt_template = PromptTemplate(input_variables=["patient_demographics","presenting_problem","mental_status_examination","other_contexts","question","candidate_answers"],
            template=prompt_text)
    predicted_labels = []

    for index, row in df_filled.iterrows():
        prompt = prompt_template.format(patient_demographics=row["Patient Demographic"],
                                   presenting_problem=row["Presenting Problem"],
                                   mental_status_examination=row["Mental Status Exam"],
                                   other_contexts=row["Other Contexts"],
                                   question=row["Question"],
                                   candidate_answers=row["Potential Answers"])
        response = generate(prompt,args.model_path)
        match = re.match(r"^\s*\(?\s*([A-Da-d])\s*\)?\s*[:\)]?", response)
        if match:
            response_label = match.group(1).upper()
        predicted_labels.append(response_label)
    
    with open(args.output_file, "w") as f:
        json.dump(predicted_labels, f)