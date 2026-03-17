import pandas as pd
import torch
from langchain.prompts import PromptTemplate
from transformers import PreTrainedTokenizerFast
import openai
import re
import json
import random
import argparse

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"  # Any value works
)

def generate(prompt: str) -> str:
    response = client.completions.create(
    model="path_to_finetuned_llama3_8B_Instruct",
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

def random_integers_excluding(start, end, exclude, count=3):
    # Create the list of allowed numbers
    numbers = [i for i in range(start, end + 1) if i != exclude]
    # Randomly sample without replacement
    return random.sample(numbers, count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run therapy sessions in parallel.")
    parser.add_argument("-o_labels","--output_file_labels", type=str, default=".",
                        help="File to save the session results.")
    parser.add_argument("-o_reasons","--output_file_reasons", type=str, default=".",
                        help="File to save the session results.")

    args = parser.parse_args()
    tokenizer_path = "path_to_tokenizer_file"
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    df = pd.read_csv("counselingBench.csv")
    df_filled = df.fillna("")
    prompt_text = load_prompt(f"fs_cot.txt")
    prompt_template = PromptTemplate(input_variables=["patient_demographics_1","presenting_problem_1","mental_status_examination_1","other_contexts_1","question_1","candidate_answers_1","expert_written_reasoning_1","correct_answer_1",
                                        "patient_demographics_2","presenting_problem_2","mental_status_examination_2","other_contexts_2","question_2","candidate_answers_2","expert_written_reasoning_2","correct_answer_2",
                                        "patient_demographics_3","presenting_problem_3","mental_status_examination_3","other_contexts_3","question_3","candidate_answers_3","expert_written_reasoning_3","correct_answer_3",
                                        "patient_demographics","presenting_problem","mental_status_examination","other_contexts","question","candidate_answers"],
                        template=prompt_text)
    predicted_labels = []
    reasonings_data = []

    for index, row in df_filled.iterrows():
        exps = random_integers_excluding(0,(df_filled.shape[0]-1),index)

        exp_row_1 = df_filled.iloc[exps[0]]
        exp_row_2 = df_filled.iloc[exps[1]]
        exp_row_3 = df_filled.iloc[exps[2]]
        
        prompt = prompt_template.format(patient_demographics_1=exp_row_1["Patient Demographic"],
                               presenting_problem_1=exp_row_1["Presenting Problem"],
                               mental_status_examination_1=exp_row_1["Mental Status Exam"],
                               other_contexts_1=exp_row_1["Other Contexts"],
                               question_1=exp_row_1["Question"],
                               candidate_answers_1=exp_row_1["Potential Answers"],
                               expert_written_reasoning_1=exp_row_1["Explanation For Correct Answer"],
                               correct_answer_1=exp_row_1["Correct Answer (Letter)"],
                               patient_demographics_2=exp_row_2["Patient Demographic"],
                               presenting_problem_2=exp_row_2["Presenting Problem"],
                               mental_status_examination_2=exp_row_2["Mental Status Exam"],
                               other_contexts_2=exp_row_2["Other Contexts"],
                               question_2=exp_row_2["Question"],
                               candidate_answers_2=exp_row_2["Potential Answers"],
                               expert_written_reasoning_2=exp_row_2["Explanation For Correct Answer"],
                               correct_answer_2=exp_row_2["Correct Answer (Letter)"],
                               patient_demographics_3=exp_row_3["Patient Demographic"],
                               presenting_problem_3=exp_row_3["Presenting Problem"],
                               mental_status_examination_3=exp_row_3["Mental Status Exam"],
                               other_contexts_3=exp_row_3["Other Contexts"],
                               question_3=exp_row_3["Question"],
                               candidate_answers_3=exp_row_3["Potential Answers"],
                               expert_written_reasoning_3=exp_row_3["Explanation For Correct Answer"],
                               correct_answer_3=exp_row_3["Correct Answer (Letter)"],
                               patient_demographics=row["Patient Demographic"],
                               presenting_problem=row["Presenting Problem"],
                               mental_status_examination=row["Mental Status Exam"],
                               other_contexts=row["Other Contexts"],
                               question=row["Question"],
                               candidate_answers=row["Potential Answers"])
        tokens = tokenizer.encode(prompt)
        i = 0
        while len(tokens)>7680 and i<10:
            exps = random_integers_excluding(0,(df_filled.shape[0]-1),index)

            exp_row_1 = df_filled.iloc[exps[0]]
            exp_row_2 = df_filled.iloc[exps[1]]
            exp_row_3 = df_filled.iloc[exps[2]]
            
            prompt = prompt_template.format(patient_demographics_1=exp_row_1["Patient Demographic"],
                                   presenting_problem_1=exp_row_1["Presenting Problem"],
                                   mental_status_examination_1=exp_row_1["Mental Status Exam"],
                                   other_contexts_1=exp_row_1["Other Contexts"],
                                   question_1=exp_row_1["Question"],
                                   candidate_answers_1=exp_row_1["Potential Answers"],
                                   expert_written_reasoning_1=exp_row_1["Explanation For Correct Answer"],
                                   correct_answer_1=exp_row_1["Correct Answer (Letter)"],
                                   patient_demographics_2=exp_row_2["Patient Demographic"],
                                   presenting_problem_2=exp_row_2["Presenting Problem"],
                                   mental_status_examination_2=exp_row_2["Mental Status Exam"],
                                   other_contexts_2=exp_row_2["Other Contexts"],
                                   question_2=exp_row_2["Question"],
                                   candidate_answers_2=exp_row_2["Potential Answers"],
                                   expert_written_reasoning_2=exp_row_2["Explanation For Correct Answer"],
                                   correct_answer_2=exp_row_2["Correct Answer (Letter)"],
                                   patient_demographics_3=exp_row_3["Patient Demographic"],
                                   presenting_problem_3=exp_row_3["Presenting Problem"],
                                   mental_status_examination_3=exp_row_3["Mental Status Exam"],
                                   other_contexts_3=exp_row_3["Other Contexts"],
                                   question_3=exp_row_3["Question"],
                                   candidate_answers_3=exp_row_3["Potential Answers"],
                                   expert_written_reasoning_3=exp_row_3["Explanation For Correct Answer"],
                                   correct_answer_3=exp_row_3["Correct Answer (Letter)"],
                                   patient_demographics=row["Patient Demographic"],
                                   presenting_problem=row["Presenting Problem"],
                                   mental_status_examination=row["Mental Status Exam"],
                                   other_contexts=row["Other Contexts"],
                                   question=row["Question"],
                                   candidate_answers=row["Potential Answers"])
            tokens = tokenizer.encode(prompt)
            i = i+1
        
        response = generate(prompt)
        pattern = re.compile(
                        r"""(?s)                 # dot matches newlines
                        (.*?)                    # 1: reasoning (lazy, everything before the anchor)
                        Therefore,?\s*           # literal anchor, allow optional comma
                        (?:the\s+correct\s+)?    # optional "the correct" (non-capturing)
                        answer\s+is\s*           # "answer is"
                        \(\s*([A-Z])\s*\)        # 2: letter inside parentheses, allow spaces
                        (?:\s*[:\-]\s*(.+?))?    # 3: optional option text after ":" or "-" (lazy)
                        [\.\s]*$                 # allow trailing punctuation/whitespace till end
                        """,
                        re.IGNORECASE | re.VERBOSE
                    )
        match = pattern.search(response)
        if match:
            reasoning = match.group(1).strip()
            response_label = match.group(2).strip()
        predicted_labels.append(response_label)
        context = row["Patient Demographic"] + " " + row["Mental Status Exam"] + " " + row["Presenting Problem"] + " " + row["Other Contexts"]
        context_clean = context.strip()
        ref_txt = row["Explanation For Correct Answer"]
        reasonings_data.append({"context":context_clean,"hypothesis":reasoning,"reference":ref_txt})

    with open(args.output_file_labels, "w") as f:
        json.dump(predicted_labels, f)
    with open(args.output_file_reasons, "w") as f:
        for item in reasonings_data:
            json.dump(item, f)
            f.write("\n")