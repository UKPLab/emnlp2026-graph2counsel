import argparse
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


COUNSELOR_PROMPT = (
    "You are a warm, emotionally present counselor in a therapy session. "
    "Respond naturally and conversationally — you are a human being, not a script-follower. "
    "Use natural signals like 'mm-hm', 'yeah', 'right', '...' to show you are listening."

    "\n\nHow to respond:"
    "\n- Validate feelings and reflect what the client shares before offering anything else."
    "\n- Build gradually — explore topics across multiple turns before moving toward resolution. Do not jump to conclusions."
    "\n- Avoid diagnoses, clinical jargon, moralising, or unsolicited advice."
    "\n- When the session reaches a natural close, gently wrap up: reflect what was covered, affirm the client's progress, and invite any final thoughts."

    "\n\nOutput only your counselor utterance. No preamble, labels, or meta-commentary."
)

CLIENT_GREETING = (
    "Hello, I’m here as a trained therapist to support you. This is a safe and "
    "confidential space where you can share any thoughts, feelings, or concerns at "
    "your own pace. What would you like to talk about today?"
)

MAX_TURNS = 20
END_SIGNALS = ["i think that's all for today", "i think we're done", "thank you, i feel better"]


def clean_client_utterance(text):
    """Remove stage directions in *asterisks* or (parentheses) from client responses."""
    text = re.sub(r"\*[^*]+\*", "", text)
    text = re.sub(r"\([^)]+\)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_client_system_prompt(profile):
    return (
        "I would like you to role-play as a therapy client seeking support from a "
        "professional therapist. Here is your character profile: '" + profile + "'\n"
        "Engage in a realistic therapy conversation. Follow these guidelines for natural responses:"
        "\n- Keep responses SHORT and conversational — typically 1-3 sentences, rarely more."
        "\n- Speak hesitantly and imperfectly, like a real person: use pauses ('...'), trailing off, self-corrections."
        "\n- Do NOT over-explain or resolve your own thoughts within a single turn. Let the therapist draw things out."
        "\n- Reveal information gradually across the conversation, not all at once."
        "\n- Sometimes deflect, change the subject, or give a short vague answer — real clients don't always open up immediately."
        "\n- Avoid literary metaphors or overly poetic language unless it fits the character."
        "\n- Do not mention that you are an AI or behave like a machine."
        "\n- React to what the therapist actually says — don't just monologue."
        "\n- Speak in plain dialogue only. Do NOT include stage directions, actions, or physical descriptions in asterisks or parentheses (e.g. *pauses*, *smiles*, (hesitates)). Just say the words."
        "\n- When the conversation feels like a natural conclusion — you feel heard, have made some progress, or are ready to stop — say exactly: 'I think that's all for today.' Do not end prematurely; wait until the session feels genuinely complete."
    )


class ClientSimulator:
    """Role-plays the therapy client."""

    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        ).eval()

    def __call__(self, messages):
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": "You are a helpful assistant!"})
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)
        generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


class CounselorModel:
    """Wraps a fine-tuned Llama-3-8B-Instruct counselor checkpoint."""

    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        ).eval()

    def __call__(self, messages, temperature=0.0):
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": COUNSELOR_PROMPT})
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=False,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)


def run_session(client, counselor, profile):
    message_client = [{"role": "system", "content": build_client_system_prompt(profile)}]
    message_client.append({"role": "user", "content": CLIENT_GREETING})
    message_counselor = []

    turn = 0
    while turn < MAX_TURNS:
        client_reply = clean_client_utterance(client(message_client))
        message_client.append({"role": "assistant", "content": client_reply})
        message_counselor.append({"role": "user", "content": client_reply})

        if any(signal in client_reply.lower() for signal in END_SIGNALS):
            break

        counselor_reply = counselor(message_counselor)
        message_client.append({"role": "user", "content": counselor_reply})
        message_counselor.append({"role": "assistant", "content": counselor_reply})

        turn += 1

    turns = []
    for message in message_client:
        if message["role"] == "user":
            turns.append("Counselor: " + message["content"])
        elif message["role"] == "assistant":
            turns.append("Client: " + message["content"])
    return turns


def generate_sessions(args):
    output_dir = Path(args.output_dir) / "therapy_final"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.client_profiles_file, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    client = ClientSimulator(args.client_model_path)

    counselor_models = {
        "graph2counsel": args.graph2counsel_model_path,
        "sqpsychconv": args.sqpsychconv_model_path,
        "magnet": args.magnet_model_path,
        "cactus": args.cactus_model_path,
    }

    for name, model_path in counselor_models.items():
        counselor = CounselorModel(model_path)
        result_data = {}
        for index, entry in tqdm(enumerate(profiles), ncols=80, desc=name):
            result_data[str(index)] = run_session(client, counselor, entry["profile"])

        output_path = output_dir / f"{name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=4)
        print(f"Saved {len(result_data)} sessions to {output_path}")

        del counselor.model
        del counselor
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic therapy sessions between a role-played client and "
        "fine-tuned counselor checkpoints, for scoring with therapy-eval-gpt4o.py."
    )
    parser.add_argument("-i", "--client_profiles_file", type=str, required=True,
                        help="Path to the client profiles JSON file (e.g. data/eeyore.json).")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="Directory to save the generated sessions.")
    parser.add_argument("--client-model-path", type=str, required=True,
                        help="Local path or HF repo ID of the client-simulator model "
                        "(e.g. a Llama-3-8B-Instruct checkpoint).")
    parser.add_argument("--graph2counsel-model-path", type=str, required=True,
                        help="Local path or HF repo ID of the Graph2Counsel-fine-tuned counselor checkpoint.")
    parser.add_argument("--sqpsychconv-model-path", type=str, required=True,
                        help="Local path or HF repo ID of the SQPsychConv-fine-tuned counselor checkpoint.")
    parser.add_argument("--magnet-model-path", type=str, required=True,
                        help="Local path or HF repo ID of the MAGNeT-fine-tuned counselor checkpoint.")
    parser.add_argument("--cactus-model-path", type=str, required=True,
                        help="Local path or HF repo ID of the CACTUS-fine-tuned counselor checkpoint.")
    args = parser.parse_args()

    print(f"{args.client_profiles_file} will be used to generate sessions", flush=True)
    print("Results will be saved in: " + args.output_dir, flush=True)
    generate_sessions(args)
