import argparse
from pathlib import Path
import math

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default=Path("configs/default.yaml"), required=True, type=Path, help="Path to the config file containing the model, training, logging, data, and checkpoint parameters. (default: configs/default.yaml)")

    ## These parameters can be set in the config file or passed in as command line arguments. Command line arguments will override config file parameters if there is a conflict.
    # Data parameters
    parser.add_argument("--train_path", type=Path, help="Train dataset name.")
    parser.add_argument("--valid_path", type=Path, help="Path to the validation dataset.")
    parser.add_argument("--num_shards", type=int, help="Number of data shards (per dataset type). Should be at least the number of GPUs.")
    parser.add_argument("--dataset_type", type=str, help="The type of dataset to train on (causal or masked). This will be set automatically in distributed training based on the hybrid training parameters, but can be set manually for non-distributed training.")
    parser.add_argument("--num_train_tokens", type=int, help="Total number of tokens to train on. This is used to calculate the number of training steps and for learning rate scheduling.")
    ## Masking parameters
    parser.add_argument("--mask_p_max", type=float, help="Masking asking probability.")
    parser.add_argument("--mask_p_min", type=float, help="Minimum masking probability.")
    parser.add_argument("--mask_random_p", type=float, help="Probability of replacing the masked token with a random token.")
    parser.add_argument("--mask_keep_p", type=float, help="Probability of keeping the masked token.")
    ## Tokenizer parameters
    parser.add_argument("--tokenizer_path", type=Path, help="Path to the tokenizer.")
    parser.add_argument('--n_special_tokens', type=int, help="Number of special tokens.")

    # Training parameters
    ## Seeding and reproducibility parameters
    parser.add_argument('--seed', type=int, help="random seed for initialization")
    ## Checkpoint loading parameters
    parser.add_argument("--load_checkpoint", default=False, action="store_true", help="Whether to load a checkpoint and resume training.")
    parser.add_argument("--checkpoint_foldername", type=Path, help="The checkpoint filename to resume training.")
    ## Loss parameters
    parser.add_argument("--hybrid_numerator", type=int, help="The numerator of the hybrid ratio.")
    parser.add_argument("--hybrid_denominator", type=int, help="The denominator of the hybrid ratio (the number of GPUs should be divisible by this number).")
    parser.add_argument('--z_loss_weight', type=float, help="Weight for the z loss.")
    ## Training duration parameters
    parser.add_argument("--max_steps", type=int, help="Maximum number of training steps.")
    parser.add_argument("--number_of_tokens", type=int, help="Total number of tokens to train on.")
    ## Batch Size
    parser.add_argument("--local_batch_size", type=int, help="Batch size for training per GPU.")
    parser.add_argument("--global_batch_size", type=int, help="Total batch size for training per GPUs and per grad accumulation step.")
    ## Learning Rate
    parser.add_argument("--adam_learning_rate", type=float, help="The initial learning rate for AdamW.")
    parser.add_argument("--muon_learning_rate", type=float, help="The initial learning rate for Muon and Normuon.")
    parser.add_argument("--scheduler", type=str, help="Which learning rate scheduler to use.", choices=["trapezoid", "cosine", "flat"])
    parser.add_argument("--warmup_proportion", type=float, help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument("--cooldown_proportion", type=float, help="Proportion of training to perform linear learning rate cooldown for. E.g., 0.1 = 10%% of training.")
    ## Weight Decay
    parser.add_argument("--adam_weight_decay", type=float, help="Weight decay if we apply some.")
    parser.add_argument("--muon_weight_decay", type=float, help="Weight decay if we apply some.")
    ## Gradient Clipping
    parser.add_argument("--max_gradient", type=float, help="Max value for gradient clipping.")
    ## Optimizer parameters
    parser.add_argument("--optimizer", type=str, choices=["muon", "normuon", "adamw"])
    parser.add_argument("--adam_eps", type=float, help="Adam epsilon.")
    parser.add_argument("--adam_beta1", type=float, help="Adam beta1.")
    parser.add_argument("--adam_beta2", type=float, help="Adam beta2.")
    parser.add_argument("--momentum", type=float, help="Momentum for Muon and Normuon optimizers.")
    parser.add_argument("--muon_beta2", type=float, help="Muon beta2.")

    # Logging parameters
    parser.add_argument("--run_name", type=str, help="Name of the run.")
    parser.add_argument("--wandb_project", type=str, help="Name of the WandB project to log into.")
    parser.add_argument("--wandb_entity", type=str, help="The entity to log to on WandB (typically your wandb username).")
    parser.add_argument("--validate_every", type=int, help="Run validation after every X training steps.")
    parser.add_argument("--validation_steps", type=int, help="Number of validation steps.")
    parser.add_argument("--experiment", default="dataset_size", type=str)

    # Checkpointing parameters
    parser.add_argument("--output_dir", type=Path, help="The output directory where the model checkpoints will be written.")
    parser.add_argument('--save_every', type=int, help="save every X steps")
    parser.add_argument("--checkpoint_style", type=str, help="The style of checkpointing", choices=["linear", "exp"])
    parser.add_argument("--first_checkpoint", type=float, help="Represents the number of tokens/steps at which to save the first checkpoint.")
    parser.add_argument('--checkpoint_every', type=int, help="create a model chekpoint every X tokens/steps after the initial checkpoint.")
    parser.add_argument("--checkpoint_mult", type=float, help="Checkpoint every power of X steps/tokens (times a initial checkpoint).")
    parser.add_argument("--checkpoint_on", type=str, help="What to checkpoint on.")
    parser.add_argument("--checkpoint_init", action="store_true", help="Whether to save the initial untrained model.")
    parser.add_argument("--checkpoint_before_cooldown", action="store_true", help="Whether to checkpoint the model before starting cooldown.")

    # Model parameters
    ## Model architecture parameters
    parser.add_argument("--model_name", type=str, help="The name of the model to train.")
    parser.add_argument("--attention_pre_norm_affine", action="store_true", help="Whether to apply an affine transformation before the attention layer in each block.")
    parser.add_argument("--classifier_pre_norm_affine", action="store_true", help="Whether to apply an affine transformation before the classifier layer.")
    parser.add_argument("--feed_forward_pre_norm_affine", action="store_true", help="Whether to apply an affine transformation before the feed forward layer in each block.")
    parser.add_argument("--norm_eps", type=float, help="Epsilon value for layer normalization.")
    parser.add_argument("--d_h", type=int, help="Head dimension.")
    parser.add_argument("--hidden_size", type=int, help="Hidden size of the model.")
    parser.add_argument("--num_attention_heads", type=int, help="Number of attention heads.")
    parser.add_argument("--intermediate_size", type=int, help="Intermediate size of the model.")
    parser.add_argument("--num_layers", type=int, help="Number of layers in the model.")
    parser.add_argument("--vocab_size", type=int, help="Vocabulary size of the model.")
    parser.add_argument("--rope_theta", type=float, help="Base period of the RoPE positional embeddings.")
    parser.add_argument("--tie_weights", action="store_true", help="Whether to tie the input and output token embeddings.")
    parser.add_argument("--max_sequence_length", type=int, help="Maximum sequence length for training.")
    ## Model optimization parameters
    parser.add_argument("--compile_model", action="store_true", help="Whether to compile the model with torch.compile for faster training.")


    args = parser.parse_args()

    return args