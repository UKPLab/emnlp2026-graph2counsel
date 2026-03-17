import json
import torch
from langchain.prompts import PromptTemplate
import openai
import re
import argparse

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"  # Any value works
)

def generate(prompt,model_path) -> str:
    response = client.completions.create(
    model=model_path,
    prompt=prompt,
    temperature=0.7,
    top_p=1.0,
    max_tokens=1024
    )
    return response.choices[0].text

def load_prompt(file_name):
    base_dir = "prompts/"
    file_path = base_dir + file_name
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get responses for CounselBench-Adv.")
    parser.add_argument("-m","--model_path", type=str, default=".",
                        help="Path to fine-tuned model weights")
    parser.add_argument("-o","--output_dir", type=str, default=".",
                        help="Directory to save the session results.")

    args = parser.parse_args()

    with open('counselbench_adv_questions.json') as f:
        data = json.load(f)

    prompt_text = load_prompt(f"gen.txt")
    prompt_template = PromptTemplate(
            input_variables=["user_query"],
            template=prompt_text)

    output_all = []
    for i in range(len(data)):
        prompt = prompt_template.format(user_query=data[i]["question"])
        response = generate(prompt,args.model_path)
        output_all.append({"issue":data[i]["category"], "question":data[i]["question"], "response": response})
    with open(args.output_dir, "w") as f:
        json.dump(output_all, f)