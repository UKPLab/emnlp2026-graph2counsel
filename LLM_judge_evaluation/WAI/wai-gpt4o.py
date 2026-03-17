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
        description="Evaluate WAI results")
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

    for filename in os.listdir(args.input_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(args.input_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            score_dict = {}
            for i in range(12):
                score = 0
                prompt_text = load_prompt("wai" + str(i+1) + ".txt")
                prompt_template = PromptTemplate(
                                    input_variables=["conversation"],
                                    template=prompt_text)
                prompt = prompt_template.format(conversation = generate_history(json_data["History"]))
                messages = [{'role': 'user', 'content': prompt}]
                for attempt in range(5):  # retry up to 5 times
                    print(1)
                    try:
                        print(2)
                        response = openai.chat.completions.create(
                                    model="gpt-4o",
                                    messages=messages,
                                    temperature=0,
                                    n=args.max_iter
                                    )
                        break
                    except openai.RateLimitError as e:
                        print(3)
                        wait_time = (2 ** attempt) + (0.1 * attempt)
                        print(f"Rate limit hit, retrying in {wait_time:.2f}s...")
                        time.sleep(wait_time)
                for j in range(args.max_iter):
                    txt_response = response.choices[j].message.content
                    score = score + int(txt_response.split(',')[0])
                avg_score = score/args.max_iter
                score_dict["wai" + str(i+1)] = avg_score
            save_as_json(score_dict, os.path.join(args.output_dir, filename))