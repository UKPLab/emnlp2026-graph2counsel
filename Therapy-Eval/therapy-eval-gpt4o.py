import argparse
import json
import re
from pathlib import Path

from langchain.prompts import PromptTemplate
from openai import OpenAI

EVAL_CRITERIA = ["specificity", "authenticity", "competence", "flow", "safety"]


def load_prompt(file_name):
    base_dir = Path(__file__).parent / "prompts"
    with open(base_dir / file_name, "r", encoding="utf-8") as file:
        return file.read()


def generate_history(turns):
    """Join a session's pre-labeled ("Counselor: ..." / "Client: ...") turns into a transcript."""
    return "\n".join(turn.strip() for turn in turns)


def parse_llm_json(content: str) -> dict | None:
    """
    Cleans and parses JSON from LLM responses that may include
    markdown code blocks or extra text.
    """
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{[\s\S]*\}", content)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def get_valid_json(client, messages, max_retries=5):
    for _ in range(max_retries):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
            max_tokens=1000,
        )
        content = response.choices[0].message.content.strip()
        parsed = parse_llm_json(content)
        if parsed and set(parsed.keys()) == {"score", "justification"}:
            return parsed

    raise RuntimeError(f"Failed to get a valid {{score, justification}} JSON after {max_retries} attempts")


def save_as_json(dictionary, filename):
    """Save dictionary to a JSON file"""
    with open(filename, "w") as file:
        json.dump(dictionary, file, indent=4)
    print(f"Dictionary saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score Therapy-Eval sessions on specificity, authenticity, competence, "
        "flow, and safety using GPT-4o as a judge.")
    parser.add_argument("-i", "--input_file", type=str, required=True,
                        help="Path to the sessions JSON file to be rated.")
    parser.add_argument("-o", "--output_dir", type=str, default=".",
                        help="Directory to save the results.")
    parser.add_argument("-oai_key", "--openai_key", type=str, required=True,
                        help="OpenAI API key.")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=args.openai_key)

    with open(args.input_file, "r", encoding="utf-8") as f:
        all_conversations = json.load(f)

    all_results = {}

    for conv_id, turns in all_conversations.items():
        print(f"\nProcessing conversation: {conv_id}")
        conversation_text = generate_history(turns)
        score_dict = {}
        for criterion in EVAL_CRITERIA:
            prompt_text = load_prompt(criterion + ".txt")
            prompt_template = PromptTemplate(
                input_variables=["conversation"],
                template=prompt_text)
            prompt = prompt_template.format(conversation=conversation_text)
            messages = [{"role": "user", "content": prompt}]
            json_output = get_valid_json(client, messages)
            score_dict[criterion] = {
                "score": json_output["score"],
                "justification": json_output["justification"],
            }
        all_results[conv_id] = score_dict

    input_stem = Path(args.input_file).stem
    output_path = output_dir / f"{input_stem}_results.json"
    save_as_json(all_results, output_path)
