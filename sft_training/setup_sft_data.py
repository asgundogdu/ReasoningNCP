import jsonlines
from prompt_utils import generate_next_chapter_messages
import numpy as np
from transformers import AutoTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
parser.add_argument("--dataset_name", type=str, default="FINAL_long_story_noprompt_dataset")
parser.add_argument("--data_dir", type=str, default="sft_data/")
args = parser.parse_args()

data_dir = args.data_dir

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

def get_initial_dataset(split):
    initial_dataset_fname = args.dataset_name + "_" + split + ".jsonl"
    with jsonlines.open(initial_dataset_fname, "r") as reader:
        initial_dataset = list(reader)

    prompts = []
    num_tokens = []
    for datapoint in initial_dataset:
        messages = generate_next_chapter_messages(datapoint, [])
        
        num_tokens.append(tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt").shape[-1])
        prompts.append(
            {'messages': messages}
        )
    return prompts, num_tokens

train_prompts, train_num_tokens = get_initial_dataset("train")
test_prompts, test_num_tokens = get_initial_dataset("test")
val_prompts, val_num_tokens = get_initial_dataset("val")

print(f"Train num_tokens: max {np.max(train_num_tokens)}, min {np.min(train_num_tokens)}, mean {np.mean(train_num_tokens)}, std {np.std(train_num_tokens)}")
print(f"Test num_tokens: max {np.max(test_num_tokens)}, min {np.min(test_num_tokens)}, mean {np.mean(test_num_tokens)}, std {np.std(test_num_tokens)}")
print(f"Val num_tokens: max {np.max(val_num_tokens)}, min {np.min(val_num_tokens)}, mean {np.mean(val_num_tokens)}, std {np.std(val_num_tokens)}")

train_fname = data_dir + "train.jsonl"
test_fname = data_dir + "test.jsonl"
val_fname = data_dir + "val.jsonl"

for fname, prompts in zip([train_fname, test_fname, val_fname], [train_prompts, test_prompts, val_prompts]):
    # shuffle prompts
    # random.shuffle(prompts)
    with jsonlines.open(fname, "w") as writer:
        writer.write_all(prompts)
    print(f"Wrote {len(prompts)} prompts to {fname}")
