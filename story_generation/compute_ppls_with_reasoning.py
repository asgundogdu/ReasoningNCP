import jsonlines
from prompt_utils import generate_next_chapter_messages, generate_reasoning_from_story_messages
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
parser.add_argument("--prompt_to_datapoint_with_baseline_ppl_file", type=str, default="./prompt_to_datapoint_with_baseline_ppl_qwen3B.pkl")
parser.add_argument("--completions_file", type=str, default="./test_synopsis_to_possible_1n_2048max_tokens_trained_qwen3b_completions.pkl")
parser.add_argument("--output_file", type=str, default="./syn_to_completion_to_ppl_trained_qwen3b.pkl")
args = parser.parse_args()


with open(args.completions_file, "rb") as f:
    syn_to_completions = pickle.load(f)

with open(args.prompt_to_datapoint_with_baseline_ppl_file, "rb") as f:
    syn_to_datapoint_with_baseline_ppl = pickle.load(f)


MODEL_NAME = args.model_name

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left", trust_remote_code=True)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

def find_index_of_last_system_message(
    input_ids, special_token, offset_after_token=4, end_offset=4
):
    # find the index of the last system message in the input_ids
    # offset is to avoid encodign the special tokens from tokenization
    # offset of 4 is used for llama models, end_offset is to make sure we aren't looking in the ending special tokens
    # for i in range(len(input_ids[0]) - end_offset - 1, 0, -1):
    # if input_ids[0][i] == special_token:
    for i in range(len(input_ids) - end_offset - 1, 0, -1):
        if input_ids[i] == special_token:
            return i + offset_after_token
    print("DIDNT FIND IT, RETURNING -1")
    return -1

def get_perplexity(messages):
    tokenized = tokenizer.apply_chat_template(
        messages, tokenize=True, return_tensors="pt"
    )
    tokenized = tokenized.to(model.device)
    start_of_system_message = find_index_of_last_system_message(
        tokenized[0], tokenizer.eos_token_id, offset_after_token=5
    )
    labels = tokenized.clone()
    labels[:, :start_of_system_message] = -100
    
    with torch.no_grad():
        outputs = model(input_ids=tokenized, labels=labels)
    logits = outputs.logits
    loss = outputs.loss
    perplexity = torch.exp(loss)
    return perplexity

overlapping_syns = set(syn_to_completions.keys()) & set(syn_to_datapoint_with_baseline_ppl.keys())

syn_to_completion_to_ppl = {}
for syn in tqdm(overlapping_syns, total=len(overlapping_syns)):
    syn_to_completion_to_ppl[syn] = {}
    datapoint = syn_to_datapoint_with_baseline_ppl[syn]
    baseline_ppl = datapoint["baseline_ppl"]
    completions = syn_to_completions[syn]
    for completion in completions:
        model_response = completion
        model_response = model_response.split("In summary:")[-1].strip()
        model_response = model_response.split("In summary,")[-1].strip()
        model_response = model_response.split("Detailed Plan:")[-1].strip()

        next_chapter_messages = generate_next_chapter_messages(
            datapoint,
            [[model_response, ""]],
        )

        ppl = get_perplexity(next_chapter_messages)
        percent_improvement = (baseline_ppl - ppl) / baseline_ppl * 100
        syn_to_completion_to_ppl[syn][completion] = {"baseline_ppl": baseline_ppl, "with_reasoning_ppl": ppl, "percent_improvement": percent_improvement}


# write prompt to baseline ppl to a pickle file
with open(args.output_file, "wb") as f:
    pickle.dump(syn_to_completion_to_ppl, f)