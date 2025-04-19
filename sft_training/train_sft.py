"""
# SFT training (based on Open-R1)
python train_sft.py

accelerate launch --config_file deepspeed_zero2.yaml \
    train_sft.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --dataset_name long_story_sft \
    --learning_rate 2.0e-5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --logging_steps 4 \
    --eval_strategy steps \
    --eval_steps 4 \
    --output_dir Qwen2.5-3B-SFT \
    --bf16 \
    --push_to_hub \
    --load_best_model_at_end \
    --metric_for_best_model loss \
    --report_to wandb \
    --eval_on_start \
    --max_seq_length 20000
    
accelerate launch --config_file deepspeed_zero2.yaml \
    train_sft.py \
    --config 3b_config.yaml

accelerate launch --config_file deepspeed_zero3.yaml \
    train_sft.py \
    --config 3b_config.yaml
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import jsonlines
import trl

def init_wandb_training(training_args):
    """
    Helper function for setting up Weights & Biases logging tools.
    """
    if training_args.wandb_entity is not None:
        os.environ["WANDB_ENTITY"] = training_args.wandb_entity
    if training_args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project
    if training_args.wandb_run_id is not None:
        os.environ["WANDB_RUN_ID"] = training_args.wandb_run_id
    if training_args.wandb_resume is not None:
        os.environ["WANDB_RESUME"] = training_args.wandb_resume    

@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    wandb_run_id: Optional[str] = field(
        default=None,
        metadata={"help": ("The run ID to store runs under.")},
    )
    wandb_resume: Optional[str] = field(
        default=None,
        metadata={"help": ("Should be none, never, must or allow")},
    )

import logging
import os
import sys

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint

from callbacks import get_callbacks
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


logger = logging.getLogger(__name__)


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    ################
    # Load datasets
    ################
      
    train_dataset = load_dataset("json", data_files="sft_data/train.jsonl")
    val_dataset = load_dataset("json", data_files="sft_data/val.jsonl")

    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        # device_map=get_kbit_device_map() if quantization_config is not None else None,
        # quantization_config=quantization_config,
        device_map = None,
    )
    training_args.model_init_kwargs = model_kwargs

    ############################
    # Initialize the SFT Trainer
    ############################
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset['train'],
        eval_dataset=val_dataset['train'],
        processing_class=tokenizer,
        # peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset['train'])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": "ncp_long_story",
        "tags": ["ncp_long_story"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(val_dataset['train'])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
