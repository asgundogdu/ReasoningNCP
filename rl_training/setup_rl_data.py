import jsonlines
from prompt_utils import generate_reasoning_from_story_messages
import argparse
# for estimating tokens
from transformers import AutoTokenizer
# for getting stats about the dataset
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
parser.add_argument("--data_dir", type=str, default="rl_data/")
parser.add_argument("--dataset_name", type=str, default="story_dataset")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

# setup dataset

LOW_CHAPTER_WORD_LIMIT = 200
HIGH_CHAPTER_WORD_LIMIT = 5000
MESSAGE_WORD_LIMIT = 10000
MINIMUM_CHAPTER_INDEX = 2

FILTER = True
# FILTER = False

def get_num_message_words(messages):
    combined = " ".join([m['content'] for m in messages])
    return len(combined.split(" "))

def get_initial_dataset(split):
    initial_dataset_fname = args.dataset_name + "_" + split + ".jsonl"
    with jsonlines.open(initial_dataset_fname, "r") as reader:
        initial_dataset = list(reader)

    num_tokens = []
    
    prompts = []
    for datapoint in initial_dataset:
        messages = generate_reasoning_from_story_messages(datapoint, [], USE_SYSTEM_ROLE=True)
        chapter_index = datapoint['chapter_index']
        if FILTER:
            next_chapter = datapoint['next_chapter']
            story_text = datapoint['story_text']
            next_chapter_words = len(next_chapter.split(" "))
            if next_chapter_words <= LOW_CHAPTER_WORD_LIMIT:
                print(f"Skipping chapter {chapter_index} because it has too few words ({next_chapter_words})")
                continue
            if next_chapter_words >= HIGH_CHAPTER_WORD_LIMIT or len(story_text.split(" ")) >= HIGH_CHAPTER_WORD_LIMIT:
                print(f"Skipping chapter {chapter_index} because it has too many words ({next_chapter_words})")
                continue
            num_message_words = get_num_message_words(messages)
            if num_message_words >= MESSAGE_WORD_LIMIT:
                print(f"Skipping chapter {chapter_index} because it has {num_message_words} words")
                continue
            if chapter_index < MINIMUM_CHAPTER_INDEX:
                print(f"Skipping chapter {chapter_index} because it is before the minimum chapter index")
                continue
        num_tokens.append(tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt").shape[-1])
        prompts.append(
            {'prompt': messages}
        )
    return prompts, num_tokens


train_prompts, train_num_tokens = get_initial_dataset("train")
test_prompts, test_num_tokens = get_initial_dataset("test")
val_prompts, val_num_tokens = get_initial_dataset("val")

# get max, min, mean, std of num_tokens
print(f"Train num_tokens: max {np.max(train_num_tokens)}, min {np.min(train_num_tokens)}, mean {np.mean(train_num_tokens)}, std {np.std(train_num_tokens)}")
print(f"Test num_tokens: max {np.max(test_num_tokens)}, min {np.min(test_num_tokens)}, mean {np.mean(test_num_tokens)}, std {np.std(test_num_tokens)}")
print(f"Val num_tokens: max {np.max(val_num_tokens)}, min {np.min(val_num_tokens)}, mean {np.mean(val_num_tokens)}, std {np.std(val_num_tokens)}")

data_dir = args.data_dir

train_fname = data_dir + "train.jsonl"
test_fname = data_dir + "test.jsonl"
val_fname = data_dir + "val.jsonl"

# for fname, prompts in zip([test_fname], [test_prompts]):
for fname, prompts in zip([train_fname, test_fname, val_fname], [train_prompts, test_prompts, val_prompts]):
    print(f"Writing {fname} with {len(prompts)} prompts")
    with jsonlines.open(fname, "w") as writer:
        writer.write_all(prompts)
