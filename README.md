# Learning to Reason for Long-Form Story Generation

Official repo for Learning to Reason for Long-Form Story Generation.

This repo contains three four parts:

1. `setup_data`: Compile a Next-Chapter Prediction dataset, used for training and story-generation
2. `rl_training`: Train a model using our VR-CLI reward paradigm, using either the NCP task or another task of your choosing
3. `sft_training`: Train a model using supervised finetuning on the NCP task
4. `story_generation`: Generate reasoning and story continuations using either a pretrained model, or a model you have trained yourself
5. `evaluation`: Replicate our evaluations of the story generation models using human annotations and automated metrics

Consult the `instructions.md` files in each directory for more details.
