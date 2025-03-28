import jsonlines
from tqdm import tqdm
import pickle
import os
# from openai import OpenAI

# openai_api_key = "EMPTY"
# openai_api_base = "http://localhost:8000/v1"
# client = OpenAI(
#     api_key=openai_api_key,
#     base_url=openai_api_base,
# )

from vllm import LLM, SamplingParams
import argparse
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
parser.add_argument("--revision", type=str, default="")
parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
parser.add_argument("--nice_model_name", type=str, default="qwen3B")
parser.add_argument("--num_completions", type=int, default=1)
parser.add_argument("--max_tokens", type=int, default=2048)
parser.add_argument("--data_dir", type=str, default="rl_data")
parser.add_argument("--output_file", type=str, default="test_synopsis_to_possible_1n_2048max_tokens_qwen3b_completions.pkl")
args = parser.parse_args()

MODEL_NAME = args.model_name
nice_model_name = args.nice_model_name

llm = LLM(model=MODEL_NAME, max_seq_len_to_capture=15000, tensor_parallel_size=4, max_model_len=15000, revision=args.revision)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name,
                                          padding_side="left", trust_remote_code=True)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=args.max_tokens, best_of=1)

split = args.split

with jsonlines.open(f"{args.data_dir}/{split}.jsonl", "r") as reader:
    prompts = list(reader)

NUM_COMPLETIONS = args.num_completions
prompt_key = "prompt"
# if you want to load intermediate results, you should load here
synopsis_to_possible_completions = {}

for prompt in tqdm(prompts):
    messages = prompt[prompt_key]
    
    next_chapter_synopsis = messages[-1]['content'].split("### Next Chapter Synopsis: ###")[1].split("###")[0].strip()

    next_chapter_synopsis = tokenizer.decode(tokenizer.encode(next_chapter_synopsis))

    if next_chapter_synopsis in synopsis_to_possible_completions:
        continue

    completions = []
    for _ in range(NUM_COMPLETIONS):
        outputs = llm.chat(messages,
                        sampling_params=sampling_params,
                        use_tqdm=False)
        generated_text = outputs[0].outputs[0].text
        completions.append(generated_text)


    synopsis_to_possible_completions[next_chapter_synopsis] = completions

    print(f"writing {len(synopsis_to_possible_completions)}...")
    with open(args.output_file, "wb") as f:
        pickle.dump(synopsis_to_possible_completions, f)