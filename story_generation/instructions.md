# Story Generation

This section requires you have already built the dataset as described in the `setup_data` section, and are either using a pretrained model or have already trained a model (`rl_training` or `sft_training`).

For convenience we separate the reasoning and story continuation generation into seperate parts. For reasoning you first generate reasoning for each datapoint, then pass that to the story generation script. For non-reasoning story generation, you can call the story generation script directly.

## Generate Reasoning

To generate reasoning, run:

```bash
python generate_reasoning.py --split train \
--model_name Qwen/Qwen2.5-3B-Instruct \
--tokenizer_name Qwen/Qwen2.5-3B-Instruct \
--nice_model_name qwen3B \
--num_completions 1 \
--max_tokens 2048 \
--data_dir rl_data \
--output_file test_synopsis_to_possible_1n_2048max_tokens_qwen3b_completions.pkl
```

By default this will output a file called `test_synopsis_to_possible_1n_2048max_tokens_qwen3b_completions.pkl` in the current directory.

## Generate Story Continuations

### Non-reasoning Story Continuations

To generate story continuations, run:

```bash
python no_reasoning_story_continuations.py --split test --model_name Qwen/Qwen2.5-7B-Instruct-1M --output_file FINAL_test_predbaselinechapters_sftqwen3b_completions.pkl --prompt_to_datapoint_with_baseline_ppl_file prompt_to_datapoint_with_baseline_ppl_qwen3B.pkl --data_dir rl_data --use_system_role
```

### Reasoning Story Continuations

To generate story continuations, run:

```bash
python with_reasoning_story_continuations.py --split test --model_name Qwen/Qwen2.5-7B-Instruct-1M --output_file FINAL_test_predchapters_qwen3b_completions.pkl --reasoning_file test_synopsis_to_possible_1n_2048max_tokens_qwen3b_completions.pkl --data_dir rl_data
```

## Get Percent-Improvements over Baseline

To evaluate your reasoning model via percent-improvements over the baseline, run the following command with your own model and files:

```bash
python compute_ppls_with_reasoning.py --model_name Qwen/Qwen2.5-3B-Instruct --prompt_to_datapoint_with_baseline_ppl_file prompt_to_datapoint_with_baseline_ppl_qwen3B.pkl --completions_file test_synopsis_to_possible_1n_2048max_tokens_trained_qwen3b_completions.pkl --output_file syn_to_completion_to_ppl_trained_qwen3b.pkl
```






