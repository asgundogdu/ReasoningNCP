import jsonlines
from prompt_utils import generate_next_chapter_messages, generate_reasoning_from_story_messages
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="story_dataset")
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct-1M")
parser.add_argument("--nice_model_name", type=str, default="qwen7B")
args = parser.parse_args()

all_data = []
for split in ["train", "test", "val"]:
    data_fname = args.dataset_name + "_" + split + ".jsonl"
    with jsonlines.open(data_fname) as reader:
        all_data.extend(list(reader))

# for loading the model
MODEL_NAME = args.model_name
# for saving the ppls for this model
nice_model_name = args.nice_model_name

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left", trust_remote_code=True)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

def find_index_of_last_system_message(
    input_ids, special_token, offset_after_token=4, end_offset=10
):
    # find the index of the last system message in the input_ids
    # offset is to avoid encoding the special tokens from tokenization
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
    print(f"labels.shape: {labels.shape}; start_of_system_message: {start_of_system_message}")
    with torch.no_grad():
        outputs = model(input_ids=tokenized, labels=labels)
    logits = outputs.logits
    loss = outputs.loss
    perplexity = torch.exp(loss)
    return perplexity


prompt_to_datapoint_with_baseline_ppl = {}

for datapoint in tqdm(all_data):
    print(f"story_id: {datapoint['story_id']}; chapter_index: {datapoint['chapter_index']}")
    next_chapter_messages = generate_next_chapter_messages(
        datapoint,
        [],
        USE_SYSTEM_ROLE=True,
    )
    
    reasoning_messages = generate_reasoning_from_story_messages(datapoint, [])
    
    reasoning_prompt = tokenizer.apply_chat_template(
        reasoning_messages, tokenize=False
    )
    
    next_chapter_synopsis = reasoning_prompt.split("### Next Chapter Synopsis: ###")[1].split("###")[0].strip()
    
    ppl = get_perplexity(next_chapter_messages)
    datapoint["baseline_ppl"] = ppl
    prompt_to_datapoint_with_baseline_ppl[next_chapter_synopsis] = datapoint
    
import pickle

# write prompt to baseline ppl to a pickle file
with open(f"prompt_to_datapoint_with_baseline_ppl_{nice_model_name}.pkl", "wb") as f:
    pickle.dump(prompt_to_datapoint_with_baseline_ppl, f)