import pickle
from prompt_utils import generate_next_chapter_messages
import time
import torch
import ray

from openrlhf.trainer.ppo_utils.experience_maker import (
    RemoteExperienceMaker,
    Samples,
    Experience,
)
import itertools
import math
import os
import socket
from typing import Callable, Dict, List

import deepspeed
import ray
import torch
import torch.distributed
from transformers.trainer import get_scheduler

from openrlhf.trainer.ray.launcher import BasePPORole
from openrlhf.models.utils import (
    compute_approx_kl,
    masked_mean,
    unpacking_samples,
)
from openrlhf.trainer.ray.ppo_actor import ActorPPOTrainer

from openrlhf.datasets import PromptDataset, SFTDataset
from openrlhf.models import Actor
from openrlhf.models.ring_attn_utils import pad_sequences, unpad_sequences
from openrlhf.trainer.ppo_utils import Experience, RemoteExperienceMaker
from openrlhf.utils import blending_datasets, get_tokenizer
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.deepspeed.deepspeed_utils import offload_deepspeed_states
from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

with open("/mnt/disk/nrl_ncp/prompt_to_datapoint_with_baseline_ppl_qwen3B.pkl", "rb") as f:
    prompt_to_datapoint_with_baseline_ppl = pickle.load(f)

def find_index_of_last_system_message(
    input_ids, special_token, offset_after_token=4, end_offset=10
):
    # find the index of the last system message in the input_ids
    # offset is to avoid encoding the special tokens from tokenization
    for i in range(len(input_ids) - end_offset - 1, 0, -1):
        if input_ids[i] == special_token:
            return i + offset_after_token
    return -1

def get_next_chapter_messages_from_sequence(sequence, tokenizer):
    decoded_sequence = tokenizer.decode(sequence)
    start_of_system_message = find_index_of_last_system_message(
        sequence, tokenizer.eos_token_id, offset_after_token=5
    )
    original_model_response = tokenizer.decode(
        sequence[start_of_system_message:], skip_special_tokens=True
    ).strip()
    model_response = original_model_response.split("In summary:")[-1].strip()
    model_response = model_response.split("In summary,")[-1].strip()
    model_response = model_response.split("Detailed Plan:")[-1].strip()
    
    split_term = "### Next Chapter Synopsis: ###"
        
    next_chapter_synopsis = (
        decoded_sequence.split(split_term)[1]
        .split("###")[0]
        .strip()
    )

    datapoint = prompt_to_datapoint_with_baseline_ppl[next_chapter_synopsis]
    baseline_ppl = datapoint["baseline_ppl"].detach().to("cpu").item()
    
    next_chapter_messages = generate_next_chapter_messages(
        datapoint,
        [[model_response, ""]],
        
    )

    next_chapter_tokens = tokenizer.apply_chat_template(
        next_chapter_messages, tokenize=True, return_tensors="pt"
    )
    
    start_of_system_message = find_index_of_last_system_message(
        next_chapter_tokens[0],
        tokenizer.eos_token_id,
        offset_after_token=5,
    )
    labels = next_chapter_tokens.clone()
    labels[:, :start_of_system_message] = -100

    return next_chapter_tokens, labels, baseline_ppl, model_response, original_model_response

class StoryBasedRemoteExperienceMaker(RemoteExperienceMaker):

    @torch.no_grad()
    def get_r_refs(
        self, sequences_cpu, attention_mask_cpu, packed_seq_lens, num_actions
    ):
        epsilon = 1e-10
        rewards = []
        pct_improvements = []
        for sequence in sequences_cpu:
            next_chapter_tokens, labels, baseline_ppl, model_response, original_model_response = get_next_chapter_messages_from_sequence(sequence, self.tokenizer)

            attention_mask = next_chapter_tokens.ne(self.tokenizer.pad_token_id)
            num_actions = next_chapter_tokens.size(-1)
            
            loss = self.initial_model.forward.remote(
                next_chapter_tokens,
                attention_mask=attention_mask,
                num_actions=num_actions,
                labels=labels,
                return_loss=True,
                return_output=False,
            )
            loss = ray.get([loss])
            loss = loss[0]
            
            perplexity = torch.exp(loss).item()

            percent_improvement = (baseline_ppl - perplexity) / baseline_ppl * 100
            # e.g. (10 - 9) / 10 * 100 = 10%
            # you get 0.5 for any (at least kind of meaningful) improve, and 1.0 for an improvement over 5%
            
            reward = 0.5 if percent_improvement >= 0.05 else 0
            reward = 0.9 if percent_improvement >= 1 else reward
            reward = 1.0 if percent_improvement >= 2 else reward
            
            reward = max(reward, 0)

            reward += epsilon  # avoid log(0)
            rewards.append(torch.tensor([reward]))
            pct_improvements.append(torch.tensor([percent_improvement]))

        return rewards, pct_improvements
    
    @torch.no_grad()
    def make_experience(self, samples: Samples) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        args = self.strategy.args
        self.actor.eval()
        device = torch.cuda.current_device()

        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions
        packed_seq_lens = samples.packed_seq_lens

        start = time.time()
        sequences_cpu, attention_mask_cpu = (
            sequences.to("cpu"),
            attention_mask.to("cpu"),
        )

        # init log probs
        if self.initial_model is not None:
            base_action_log_probs_ref = self.initial_model.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, logps_allgather=True, packed_seq_lens=packed_seq_lens
            )

            if args.colocate_actor_ref or args.colocate_all_models:
                ray.get([base_action_log_probs_ref])
                ray.get([self.initial_model.empty_cache.remote()])
        else:
            base_action_log_probs_ref = ray.put(None)

        # values
        if self.critic is not None:
            value_ref = self.critic.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
            )
            # avoid CUDA OOM when colocate models
            if args.colocate_critic_reward or args.colocate_all_models:
                ray.get([value_ref])
                ray.get([self.critic.empty_cache.remote()])
        else:
            value_ref = ray.put(None)

        # rewards
        r_refs, pct_improvements = self.get_r_refs(
            sequences_cpu, attention_mask_cpu, packed_seq_lens, num_actions
        )
        
        if args.colocate_actor_ref or args.colocate_all_models:
            ray.get([self.initial_model.empty_cache.remote()])

        # log probs
        action_log_probs = self.actor(
            sequences,
            num_actions,
            attention_mask,
            ring_attn_group=self.strategy.ring_attn_group,
            logps_allgather=True,
            packed_seq_lens=packed_seq_lens,
        )
        actor_value_rm_time = time.time() - start

        # wait initial/critic/reward model done
        start = time.time()
        ref_values = ray.get([base_action_log_probs_ref, value_ref])
        wait_time = time.time() - start

        base_action_log_probs, value, rewards = ref_values[0], ref_values[1], r_refs
        if base_action_log_probs is not None:
            base_action_log_probs = base_action_log_probs.to(device)
        if value is not None:
            value = value.to(device)
        rewards = [r.to(device) for r in rewards]
        r = self.reward_fn(rewards) if len(rewards) > 0 else rewards[0]
        pct_improvements = torch.stack(pct_improvements)

        # avoid CUDA OOM when colocate models
        if args.colocate_critic_reward and not self.remote_rm_url:
            ray.get([self.reward_model[0].empty_cache.remote()])

        if args.colocate_actor_ref or args.colocate_all_models:
            torch.cuda.empty_cache()

        if (self.initial_model is not None) and (not args.use_kl_loss):
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=action_mask,
                kl_estimator=self.strategy.args.kl_estimator,
            )
        else:
            kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=action_log_probs.device)

        if not self.packing_samples:
            kl_mean = masked_mean(kl, action_mask, dim=-1)
        else:
            if self.strategy.ring_attn_group is not None:
                assert samples.pad_len is not None
                sequences, attention_mask, num_actions, packed_seq_lens, _, _, kl = unpad_sequences(
                    pad_len=samples.pad_len,
                    sequences=sequences,
                    attention_mask=attention_mask,
                    num_actions=num_actions,
                    packed_seq_lens=packed_seq_lens,
                    ring_attn_group=self.strategy.ring_attn_group,
                    action_log_probs=action_log_probs,
                    values=value,
                    kl=kl,
                )
            # convert tensor into list of tensors so that it's easier to manipulate
            # within dataset.
            sequences = unpacking_samples(sequences, packed_seq_lens)
            attention_mask = None
            action_log_probs = unpacking_samples(action_log_probs, num_actions)
            if value is not None:
                value = unpacking_samples(value, num_actions)
            if base_action_log_probs is not None:
                base_action_log_probs = unpacking_samples(base_action_log_probs, num_actions)

            kl = unpacking_samples(kl, num_actions)
            kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=device)

        if not args.use_kl_loss:
            base_action_log_probs = None

        info = {
            "kl": kl_mean,
            "reward": r.detach().cpu(),
            "raw_reward": r.detach().cpu(),
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
            "pct_improvements": pct_improvements,
        }

        if self.strategy.args.perf:
            self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
            self.perf_stats["wait_time"] += wait_time

        experience = Experience(
            sequences,
            action_log_probs,
            base_action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        )

        self.actor.train()  # reset model state
        return experience

@ray.remote(num_gpus=1)
class CustomStoryBasedActorModelRayActor(BasePPORole):
    def __init__(self, *args, remote_experience_maker_class=StoryBasedRemoteExperienceMaker, **kwargs):
        super().__init__(*args, **kwargs)
        self.remote_experience_maker_class = remote_experience_maker_class

    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        args = strategy.args

        if getattr(args, "vllm_num_engines", 0) > 0:
            # To prevent hanging during NCCL synchronization of weights between DeepSpeed and vLLM.
            # see https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
            if getattr(args, "vllm_sync_backend", "nccl") == "nccl":
                os.environ["NCCL_CUMEM_ENABLE"] = "0"

        self._setup_distributed(strategy)

        actor = Actor(
            pretrain,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            lora_rank=strategy.args.lora_rank,
            lora_alpha=strategy.args.lora_alpha,
            target_modules=strategy.args.target_modules,
            lora_dropout=strategy.args.lora_dropout,
            ds_config=strategy.get_ds_train_config(is_actor=True),
            packing_samples=strategy.args.packing_samples,
            temperature=strategy.args.temperature,
        )
        strategy.print(actor)

        # configure tokenizer
        self.tokenizer = get_tokenizer(
            pretrain, actor.model, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer
        )

        if args.enable_ema:
            ema_model = Actor(
                pretrain,
                use_flash_attention_2=strategy.args.flash_attn,
                bf16=strategy.args.bf16,
                load_in_4bit=strategy.args.load_in_4bit,
                ds_config=strategy.get_ds_eval_config(offload=True),
                packing_samples=strategy.args.packing_samples,
            )
        else:
            ema_model = None

        # configure optimizer
        actor_optim = strategy.create_optimizer(
            actor, lr=args.actor_learning_rate, betas=strategy.args.adam_betas, weight_decay=args.l2
        )

        # prepare_datasets
        self.prepare_datasets()

        # configure scheduler
        self.num_update_steps_per_episodes = (
            len(self.prompts_dataset) * args.n_samples_per_prompt // args.train_batch_size * args.max_epochs
        )
        max_steps = math.ceil(args.num_episodes * self.num_update_steps_per_episodes)
        self._max_steps = max_steps

        actor_scheduler = get_scheduler(
            "cosine_with_min_lr",
            actor_optim,
            num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.1},
        )

        if args.gradient_checkpointing:
            actor.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
            )

        # prepare models/optimizers...
        self.actor, self.actor_optim, self.actor_scheduler = strategy.prepare(
            (actor, actor_optim, actor_scheduler),
            is_rlhf=True,
        )

        if ema_model:
            ema_model._offload = True
            self.ema_model = strategy.prepare(ema_model, is_rlhf=True)
        else:
            self.ema_model = None

        # load checkpoint
        self.consumed_samples = 0
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path):
            _, states = strategy.load_ckpt(self.actor.model, ckpt_path)
            self.consumed_samples = states["consumed_samples"]
            strategy.print(f"Loaded the checkpoint: {ckpt_path}, consumed_samples: {self.consumed_samples}")

        # initial offload
        if strategy.args.deepspeed_enable_sleep:
            offload_deepspeed_states(self.actor.model)

    def prepare_datasets(self):
        strategy = self.strategy
        args = self.strategy.args

        # prepare datasets
        prompts_data, eval_prompts_data = blending_datasets(
            args.prompt_data,
            args.prompt_data_probs,
            strategy,
            args.seed,
            max_count=args.max_samples,
            return_eval=True,
            train_split=args.prompt_split,
        )
        prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
        eval_prompts_data = eval_prompts_data.select(range(min(args.max_samples, len(eval_prompts_data))))
        self.prompts_dataset = PromptDataset(
            prompts_data, self.tokenizer, strategy, input_template=args.input_template
        )
        self.eval_prompts_dataset = PromptDataset(
            eval_prompts_data, self.tokenizer, strategy, input_template=args.input_template
        )
        self.prompts_dataloader = strategy.setup_dataloader(
            self.prompts_dataset,
            args.rollout_batch_size // (strategy.world_size // strategy.ring_attn_size),
            True,
            True,
        )
        self.eval_prompts_dataloader = strategy.setup_dataloader(
            self.eval_prompts_dataset, args.rollout_batch_size // strategy.world_size, True, False
        )
        

        if args.pretrain_data:
            pretrain_data = blending_datasets(
                args.pretrain_data,
                args.pretrain_data_probs,
                strategy,
                args.seed,
                return_eval=False,
                train_split=args.pretrain_split,
                eval_split=args.eval_split,
            )
            pretrain_max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
            pretrain_dataset = SFTDataset(
                pretrain_data.select(
                    range(
                        min(
                            len(pretrain_data), args.max_epochs * len(self.prompts_dataset) * args.n_samples_per_prompt
                        )
                    )
                ),
                self.tokenizer,
                pretrain_max_len,
                strategy,
                pretrain_mode=True,
            )
            self.pretrain_dataloader = itertools.cycle(
                iter(
                    strategy.setup_dataloader(
                        pretrain_dataset,
                        args.micro_train_batch_size,
                        True,
                        True,
                        pretrain_dataset.collate_fn,
                    )
                )
            )
        else:
            self.pretrain_dataloader = None

    def max_steps(self):
        """Return the maximum number of steps."""
        return self._max_steps

    def fit(
        self,
        critic_model: ray.actor.ActorHandle,
        initial_model: ray.actor.ActorHandle,
        reward_model: List[ray.actor.ActorHandle],
        remote_rm_url: List[str] = None,
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
        vllm_engines: List[ray.actor.ActorHandle] = None,
        critic_train_remote: bool = False,
    ):
        """Train actor model with prompt datasets."""
        strategy = self.strategy
        args = self.strategy.args

        # configure Trainer
        trainer = ActorPPOTrainer(
            strategy,
            self.actor,
            critic_model,
            reward_model,
            initial_model,
            ema_model=self.ema_model,
            actor_optim=None,
            critic_optim=None,
            actor_scheduler=self.actor_scheduler,
            critic_scheduler=None,
            remote_rm_url=remote_rm_url,
            reward_fn=reward_fn,
            vllm_engines=vllm_engines,
            max_epochs=args.max_epochs,
            micro_train_batch_size=args.micro_train_batch_size,
            micro_rollout_batch_size=args.micro_rollout_batch_size,
            gradient_checkpointing=args.gradient_checkpointing,
            critic_train_remote=critic_train_remote,
            remote_experience_maker_class=self.remote_experience_maker_class,
            tokenizer=self.tokenizer,
            prompt_max_len=args.prompt_max_len,
            value_clip=args.value_clip,
            eps_clip=args.eps_clip,
            gamma=args.gamma,
            lambd=args.lambd,
            init_kl_coef=args.init_kl_coef,
            kl_target=args.kl_target,
            ema_beta=0.992,
            ptx_coef=args.ptx_coef,
            max_norm=args.max_norm,
            # for GPT generation
            do_sample=True,
            max_new_tokens=args.generate_max_len,
            max_length=args.max_len,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            save_hf_ckpt=args.save_hf_ckpt,
            disable_ds_ckpt=args.disable_ds_ckpt,
        )

        # broadcast checkpoint
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path) and not vllm_engines is None:
            # vLLM wakeup when vllm_enable_sleep
            if self.strategy.args.vllm_enable_sleep:
                batch_vllm_engine_call(self.vllm_engines, "wake_up")
            torch.distributed.barrier()
            torch.cuda.synchronize()

            trainer._broadcast_to_vllm()

            # vLLM offload when vllm_enable_sleep
            if self.strategy.args.vllm_enable_sleep:
                batch_vllm_engine_call(self.vllm_engines, "sleep")
                torch.distributed.barrier()
                torch.cuda.synchronize()

        trainer.fit(
            args,
            self.prompts_dataloader,
            self.pretrain_dataloader,
            self.consumed_samples,
            self.num_update_steps_per_episodes,
            eval_prompts_dataloader=self.eval_prompts_dataloader,
        )

    def save_model(self):
        args = self.strategy.args

        # save model checkpoint after fitting on only rank0
        self.strategy.save_model(
            self.ema_model if args.enable_ema else self.actor,
            self.tokenizer,
            args.save_path,
        )
