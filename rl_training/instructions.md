# Training RL-Training models

## Setup

Setup your environment using the `requirements.txt` file and install our custom OpenRLHF fork. We used the `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel` docker image as a base.

```bash
pip install -r requirements.txt
cd openrlhf
pip install -e .
cd ..
```

Setup your dataset (use `setup_data/` to recreate our dataset) - we assume you have a jsonl format with the following fields for each element:

- `story_text`: the previous text of the story (we use the previous chapter)
- `next_chapter`: the next chapter of the story
- `chapter_index`: the current chapter index i (the next chapter is chapter i+1)
- `prior_plot_summary`: a summary of the prior plot
- `high_level_plot_summary`: a high-level summary of the plot
- `character_sheets`: a dictionary of character sheets (e.g. `{character_name1: character_sheet1, character_name2: character_sheet2}`)
- `next_chapter_synopsis`: a synopsis of the next chapter
- `last_n_chapters`: how many previous chapters are included in story text (for correct instruction formatting)
- `next_chapter_header`: the header of the next chapter (e.g. "Chapter 10: 1979: The Next Chapter", important so the model knows the chapter number and any other information the header might contains)

Many of these fields (except story_text, next_chapter, and next_chapter_header) are optional, and can be controlled by the `include_*` flags in functions in `prompt_utils.py`. The next chapter synopsis is technically optional for testing story generation, but is necessary for training (more details below).

We also assume your model's tokenizer has a chat template that supports the system role, but you can use the `USE_SYSTEM_ROLE` flag in `prompt_utils.py` to control this behaviour.

## Creating Training Data

### Creating Prompt Dataset

In order to construct the prompt training data, we use the `setup_rl_data.py` script.

```bash
python setup_rl_data.py --tokenizer_name $TOKENIZER_NAME --data_dir $DATA_DIR --dataset_name $DATASET_NAME
```

`$TOKENIZER_NAME` is the name of the tokenizer you are using.

`$DATA_DIR` is the directory where you want to save the training data. (default is rl_data/)

`$DATASET_NAME` is the name of the dataset you want to use. (default is story_dataset), you should have already split your dataset into train, test, and val sets and saved them as `story_dataset_train.jsonl`, `story_dataset_test.jsonl`, and `story_dataset_val.jsonl`.

### Computing Baseline Perplexity

Instead of computing the baseline perplexity each time during training, we can precompute the perplexity for each datapoint and save it to a file.

```bash
python compute_baseline_perplexity.py --model_name $MODEL_NAME --nice_model_name $NICE_MODEL_NAME
```

`$MODEL_NAME` is the name of the model you want to use, e.g. `Qwen/Qwen2.5-7B-Instruct-1M`.

`$NICE_MODEL_NAME` is the name of the model you want to use for saving the perplexity values, e.g. `qwen7B`. We use this so we can test a variety of models with different naming conventions.

This script saves a pickle file (`prompt_to_datapoint_with_baseline_ppl_{$NICE_MODEL_NAME}.pkl`) which stores a dictionary going from the next chapter synopsis to the baseline perplexity value and underlying datapoint. We use the next chapter synopsis because we want some unique identifier for the datapoint that is extractable from the prompts saved in our RL dataset. If RL libraries make it easier to keep track of the datapoint, we could just include the baseline perplexity in the datapoint.

## Training - Next-Chapter Prediction with Verifiable Rewards via Completion Likelihood

Modify the `train_7b.sh` or `train_3b.sh` scripts based on your needs. You will need to change the directories (e.g. `$WORKING_DIR`) to point to the correct data and working directories, and likely the gpu configuration. Explanations for the arguments are in `train_ncp.py` and OpenRLHF's documentation.

You will also need to change the file pointed to in `ray_utils.py` to point to the `prompt_to_datapoint_with_baseline_ppl_{$NICE_MODEL_NAME}.pkl` file you created in the previous step. This file is used to get the baseline perplexity and next-chapter given the prompt. Hopefully this will be made easier in future RL libraries.

Run the training script.

```bash
./train_7b.sh # or ./train_3b.sh
```

After training you will have a checkpoint and `zero_to_fp32.py` script in the `checkpoint` directory. We recommend converting this model using

```bash
python zero_to_fp32.py ./ $OUTPUT_DIR/pytorch_model.bin -t $CHECKPOINT_NAME
```

to convert the model to fp32, and then pushing this to hub for easier use later.


## Detailed Instructions for Training with VR-CLI Rewards

We use a modified version of [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) (0.6.1.post1) to train our models using GRPO. We provide the modified version in the `openrlhf` directory, however as RL libraries update we encourage you to make the following modifications to support VR-CLI rewards for your own task:

Create a custom reward function that takes in a `generator` model (in our case, the reference model) and the prompt+generated response. Ideally this would also have access to the underlying datapoint, otherwise this method needs to load the corresponding baseline perplexity based on information in the prompt.
- This reward function should compute the perplexity of the gold completion (e.g. the next chapter) using the generator model and the policy's response.
- The _Improvement_ over the baseline perplexity is then computed as `100 * [(baseline_ppl - policy_ppl) / baseline_ppl]`.
- Your reward is then some function of this improvement, e.g. `min(improvement, 0)` or `0 if improvement < 0 else 1`.


### Changes to OpenRLHF

Specifically we made the following modifications for our next-chapter prediction task in OpenRLHF:

1) Created a custom `StoryBasedRemoteExperienceMaker` class that computes rewards using the baseline perplexity and perplexity of the model's response using the reference model.
2) Added a `remote_experience_maker_class` argument to the `ActorModelRayActor` class that allows us to pass in our custom `StoryBasedRemoteExperienceMaker` class.
3) Modified the way eval-dataloaders were handled and actors are evaluated in the `PPOActor` and `PPOTrainer` classes.
4) Added custom evaluation metrics to the `PPOActor` class that compute the reward, raw reward, and percent improvements.
5) Create a custom `CustomStoryBasedActorModelRayActor` class that inherits from `BasePPORole` and uses our custom `StoryBasedRemoteExperienceMaker` class and takes advantage of the new evaluation metrics.
