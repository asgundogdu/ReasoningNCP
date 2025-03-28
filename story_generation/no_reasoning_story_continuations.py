import jsonlines
from prompt_utils import generate_next_chapter_messages
from transformers import AutoTokenizer
from tqdm import tqdm

import pickle
from vllm import LLM, SamplingParams

import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct-1M")
parser.add_argument("--revision", type=str, default=None)
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--output_file", type=str, default="FINAL_test_predbaselinechapters_sftqwen3b_completions.pkl")
parser.add_argument("--prompt_to_datapoint_with_baseline_ppl_file", type=str, default="prompt_to_datapoint_with_baseline_ppl_qwen3B.pkl")
parser.add_argument("--data_dir", type=str, default="rl_data")
parser.add_argument("--use_system_role", type=bool, default=True)
args = parser.parse_args()


MODEL_NAME = args.model_name
split = args.split
prompt_key = "prompt"

llm = LLM(model=MODEL_NAME, max_seq_len_to_capture=25000, tensor_parallel_size=2, max_model_len=25000,revision=args.revision)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left", trust_remote_code=True)

save_fname = args.output_file
    
with open(args.prompt_to_datapoint_with_baseline_ppl_file, "rb") as f:
    prompt_to_datapoint_with_baseline_ppl = pickle.load(f)

with jsonlines.open(f"{args.data_dir}/{split}.jsonl", "r") as reader:
    prompts = list(reader)

synopsis_to_predicted_chapters = {}

def roundup(x):
    return int(math.ceil(x / 100.0)) * 100


for prompt in tqdm(prompts):
    messages = prompt[prompt_key]
    next_chapter_synopsis = messages[-1]['content'].split("### Next Chapter Synopsis: ###")[1].split("###")[0].strip()
    
    if next_chapter_synopsis in synopsis_to_predicted_chapters:
        continue
    
    datapoint_with_baseline_ppl = prompt_to_datapoint_with_baseline_ppl[next_chapter_synopsis]
    print(f"story_id: {datapoint_with_baseline_ppl['story_id']}; chapter_index: {datapoint_with_baseline_ppl['chapter_index']}")
    
    baseline_ppl = datapoint_with_baseline_ppl["baseline_ppl"]
    baseline_ppl = baseline_ppl.detach().cpu().item()
    
    num_tokens_final_chapter = len(tokenizer.encode(datapoint_with_baseline_ppl['next_chapter']))
    num_tokens_header = len(tokenizer.encode(datapoint_with_baseline_ppl['next_chapter_header']))
    
    baseline_num_tokens = num_tokens_final_chapter + num_tokens_header
    
    num_can_gen = int(baseline_num_tokens * 1.5)
    min_tokens = int(baseline_num_tokens * 0.5)
    
    # fake ppl with possible completions
    predicted_chapters = []
        
    next_chapter_messages = generate_next_chapter_messages(
        datapoint_with_baseline_ppl,
        [],
        USE_SYSTEM_ROLE=args.use_system_role,
        for_actual_generation=True,
        approximate_chapter_length=roundup(int(baseline_num_tokens * 0.5 * 0.75)),
    )
    
    actual_gen_ncp_messages = next_chapter_messages[:-1]
    
    # generate between 50% and 150% of the true final chapter
    num_can_gen = num_tokens_final_chapter * 1.5
    min_tokens = num_tokens_final_chapter * 0.5
    
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, min_tokens=min_tokens,
                                        max_tokens=num_can_gen, best_of=1, repetition_penalty=1.05)
    
    outputs = llm.chat(actual_gen_ncp_messages,
                    sampling_params=sampling_params,
                    use_tqdm=False)
    
    generated_text = outputs[0].outputs[0].text
    predicted_chapters.append(generated_text)
    
    synopsis_to_predicted_chapters[next_chapter_synopsis] = predicted_chapters
    
    print(f"writing {len(synopsis_to_predicted_chapters)}...")
    with open(save_fname, "wb") as f:
        pickle.dump(synopsis_to_predicted_chapters, f)