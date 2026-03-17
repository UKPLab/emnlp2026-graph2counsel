from langchain.prompts import PromptTemplate
import os
import json
import argparse
import openai
import tiktoken
from pathlib import Path

def generate_history(history):
    history_text = '\n'.join(
            [
                f"{message['speaker'].capitalize()}: {message['message']}"
                for message in history
            ]
    )

    return history_text

def load_prompt(file_name):
    base_dir = "prompts/"
    file_path = base_dir + file_name
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def save_as_json(dictionary, filename):
    """Save dictionary to a JSON file"""
    with open(filename, 'w') as file:
        json.dump(dictionary, file, indent=4)
    print(f"Dictionary saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate CTRS results")
    parser.add_argument("-i","--input_dir", type=str, default=".",
                        help="Directory to read the sessions")
    parser.add_argument("-o","--output_dir", type=str, default=".",
                        help="Directory to to save the results.")
    parser.add_argument("-m_iter","--max_iter", type=int, default=3,
                        help="Maximum number of evaluation iterations.")
    parser.add_argument("-oai_key","--openai_key", type=str, default=".",
                        help="OpenAI API key for GPT-4o")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    openai.api_key = args.openai_key

    ctrs_list = ["general_1_understanding", "general_2_interpersonal_effectiveness", "general_3_collaboration", "CBT_1_guided_discovery", "CBT_2_focus", "CBT_3_strategy"]
    
    for filename in os.listdir(args.input_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(args.input_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            score_dict = {}
            for i in range(len(ctrs_list)):
                score = 0
                prompt_text = load_prompt(ctrs_list[i] + ".txt")
                prompt_template = PromptTemplate(
                                    input_variables=["conversation"],
                                    template=prompt_text)
                prompt = prompt_template.format(conversation = generate_history(json_data["History"]))
                messages = [{'role': 'user', 'content': prompt}]
                response = openai.chat.completions.create(
                            model="gpt-4o",
                            messages=messages,
                            temperature=0,
                            n=args.max_iter
                            )
                for j in range(args.max_iter):
                    txt_response = response.choices[j].message.content
                    score = score + int(txt_response.split(',')[0])
                avg_score = score/args.max_iter
                score_dict[ctrs_list[i]] = avg_score
            save_as_json(score_dict, os.path.join(args.output_dir, filename))