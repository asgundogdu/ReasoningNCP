import jsonlines
import pickle
import spacy
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Run evaluation on a specific quarter of the dataset.")

parser.add_argument('--quarter', type=int, choices=[1, 2, 3, 4], help="Quarter of the dataset to process (1, 2, 3, or 4)", default=None)
parser.add_argument('--split', type=str, choices=["FINAL_train", "FINAL_val", "FINAL_test"], help="Split of the dataset to process", default="FINAL_val")
args = parser.parse_args()
QUARTER = args.quarter
split = args.split

nlp = spacy.load("en_core_web_md")

ORIGINAL_STORIES_INPUT_FNAME = f"{split}_long_story_to_3chars+chaps.pkl"

with open(ORIGINAL_STORIES_INPUT_FNAME, "rb") as f:
    STORIES_TO_BASE_CHARACTER_SHEETS_ON = pickle.load(f)
    
# will be (story_name -> (list_of_3_characters, list_of_chapters))

quarter_str = f"quarter{QUARTER}." if QUARTER is not None else ""
# fname = "temp.longstorychapters.output.simplified.vllm.llama70B.40maxtok.jsonl"
fname = f"temp.{quarter_str}{split}_longstorychapters.output.simplified.vllm.llama70B.40maxtok.jsonl"

with jsonlines.open(fname, "r") as reader:
    all_data = list(reader)

print(f"Loaded {len(all_data)} examples")

# d = ['snippet_role', 'prompt', 'character', 'snippet', 'question', 'question_set', 
#      'story_id', 'response', 'sentence_of_interest', 'sentence_from_response_to_split', 
#   'old_response', 'split_sentences_raw', 'split_sentences_list']

story_name_to_list_of_chapters = {story_name: [h+c for h, c in STORIES_TO_BASE_CHARACTER_SHEETS_ON[story_name][1]] for story_name in STORIES_TO_BASE_CHARACTER_SHEETS_ON}

find_matching_snippet = lambda snippet, story_name: story_name_to_list_of_chapters[story_name].index(snippet)

# trying just replacing snippet with index of snippet in list of snippets

new_data = []
for datapoint in tqdm(all_data):
    snippet_index = find_matching_snippet(datapoint['snippet'], datapoint['story_id'])
    # originally generated response (when making character sheets)
    old_response = datapoint['old_response']
    doc = nlp(old_response)
    sentences = [sent.text for sent in doc.sents]
    # get index of sentence_of_interest in sentences
    sentence_of_interest_index = sentences.index(datapoint['sentence_of_interest'])
    # this is the index pre-simplification
    for simplified_sentence_index, simplified_sentence in enumerate(datapoint['split_sentences_list']):
        indices = (sentence_of_interest_index, simplified_sentence_index)
    
        new_datapoint = {
            'character': datapoint['character'],
            'question': datapoint['question'],
            'question_set': datapoint['question_set'],
            'story_id': datapoint['story_id'],
            'sentence_to_verify': simplified_sentence,
            'index_from_original_response': sentence_of_interest_index,
            'index_from_simplified_response': simplified_sentence_index,
        }
        
        new_datapoint['snippet'] = snippet_index
        new_data.append(new_datapoint)

prefix = fname.split(".jsonl")[0]
print(f"Writing {len(new_data)} examples to {prefix}.retrofit.jsonl")
with jsonlines.open(f"{prefix}.retrofit.jsonl", "w") as writer:
    writer.write_all(new_data)
