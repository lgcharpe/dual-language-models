import sys
import argparse
from tqdm import tqdm
import json
import math
from pathlib import Path

from tokenizers import Tokenizer
import torch
import torch.nn as nn
import torch._dynamo

import wandb

from dual_language_models.model.model_single_gpu import Model
from dual_language_models.optimizers.muon import SingleDeviceMuon as Muon
from dual_language_models.utils import trapezoid_schedule, MaskScheduler, is_main_process, seed_everything, cosine_schedule_with_warmup, cosine_schedule_with_warmup_cooldown, flat_with_warmup_schedule, trapezoid_schedule_sqrt
from dual_language_models.pretraining.dataset_single_gpu import ValidationCausalDataset, ValidationMaskedDataset, TrainDataset
from dual_language_models.metrics import TrainingMetrics
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.guard_nn_modules = False  # Prevent recompilation when self.mask changes


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", default="data/train", required=False, type=Path, help="Train dataset name.")
    parser.add_argument("--valid_path", default="data/valid", type=Path, help="Path to the validation dataset.")
    parser.add_argument("--name", default="Single_GPU_test", type=str, help="Name of the run.")
    parser.add_argument("--wandb_project", default="single_gpu_dual_lm", type=str, help="Name of the WandB project to log into.")
    parser.add_argument("--wandb_entity", default="lgcharpe", type=str, help="The entity to log to on WandB (typically your wandb username).")
    parser.add_argument("--config_file", default="configs/test.json", type=Path, help="The BERT model config")
    parser.add_argument("--tokenizer_path", default="tokenizers/tokenizer_australis.json", type=Path, help="Path to the tokenizer.")
    parser.add_argument("--output_dir", default="checkpoints", type=Path, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--checkpoint_foldername", default=None, type=Path, help="The checkpoint filename to resume training.")
    parser.add_argument("--causal_ratio", default=0.5, type=float, help="The ratio of causal tokens in the hybrid model.")
    parser.add_argument("--max_seq_length", default=512, type=int, help="Sequence length for training.")
    parser.add_argument("--local_batch_size", default=16, type=int, help="Batch size for training per GPU.")
    parser.add_argument("--global_batch_size", default=128, type=int, help="Total batch size for training per GPUs and per grad accumulation step.")
    parser.add_argument("--learning_rate", default=1e-2, type=float, help="The initial learning rate for AdamW.")
    parser.add_argument("--adam_learning_rate", default=5e-3, type=float, help="The learning rate for Adam optimizer.")
    parser.add_argument("--number_of_tokens", default=1_048_576_000, type=int, help="Total number of tokens to train on.")
    parser.add_argument("--max_steps", default=16000, type=int)
    parser.add_argument("--validate_every", default=1000, type=int, help="Run validation after every X training steps.")
    parser.add_argument("--validation_steps", default=1, type=int, help="Number of validation steps.")
    parser.add_argument("--scheduler", default="flat", type=str, help="Which learning rate scheduler to use.", choices=["trapezoid", "cosine", "flat"])
    parser.add_argument("--warmup_proportion", default=0.0, type=float, help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument("--cooldown_proportion", default=0.25, type=float, help="Proportion of training to perform linear learning rate cooldown for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--save_every', type=int, default=2, help="save every X steps")
    parser.add_argument("--checkpoint_style", default="exp", type=str, help="The style of checkpointing", choices=["linear", "exp"])
    parser.add_argument("--first_checkpoint", default=500, type=float, help="Represents the number of tokens/steps at which to save the first checkpoint.")
    parser.add_argument('--checkpoint_every', type=int, default=1e8, help="create a model chekpoint every X tokens/steps after the initial checkpoint.")
    parser.add_argument("--checkpoint_mult", default=math.sqrt(2), type=float, help="Checkpoint every power of X steps/tokens (times a initial checkpoint).")
    parser.add_argument("--checkpoint_on", default="steps", type=str, help="What to checkpoint on.")
    parser.add_argument("--checkpoint_init", default=True, action="store_true", help="Whether to save the initial untrained model.")
    parser.add_argument("--checkpoint_before_cooldown", default=True, action="store_true", help="Whether to checkpoint the model before starting cooldown.")
    parser.add_argument("--mask_p_max", default=0.3, type=float, help="Masking asking probability.")
    parser.add_argument("--mask_p_min", default=0.1, type=float, help="Minimum masking probability.")
    parser.add_argument("--mask_random_p", default=0.1, type=float, help="Probability of replacing the masked token with a random token.")
    parser.add_argument("--mask_keep_p", default=0.1, type=float, help="Probability of keeping the masked token.")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--optimizer_eps", default=1e-8, type=float, help="Optimizer epsilon.")
    parser.add_argument("--optimizer_beta1", default=0.9, type=float, help="Optimizer beta1.")
    parser.add_argument("--optimizer_beta2", default=0.95, type=float, help="Optimizer beta2.")
    parser.add_argument("--max_gradient", default=1e9, type=float, help="Max value for gradient clipping.")
    parser.add_argument('--n_special_tokens', default=16, type=int, help="Number of special tokens.")
    parser.add_argument('--z_loss_weight', default=0.0001, type=float, help="Weight for the z loss.")
    parser.add_argument("--experiment", default="long_test", type=str)
    parser.add_argument("--optimizer", default="muon", type=str, choices=["muon", "adamw"])
    parser.add_argument("--coeffs", default="jordan", type=str, help="Which set of coefficients to use for the Muon optimizer's polynomial approximation.", choices=["jordan", "polar_default", "polar_five_iter"])
    parser.add_argument("--kimi_adjust_lr", action="store_true", help="Whether to adjust the learning rate according to the KIMI paper's suggestion (dividing by sqrt(t) after each step).")
    parser.add_argument("--muonh", action="store_true", help="Whether to use the MuonH optimizer, which applies the Muon update to a subset of parameters and AdamW to the rest.")
    parser.add_argument("--normuon", action="store_true", help="Whether to use the NorMuon optimizer, which is a variant of Muon that normalizes the update to have the same norm as the original gradient.")
    parser.add_argument("--polar_express", action="store_true", help="Whether to use the Polar Express approximation in the Muon optimizer.")
    parser.add_argument("--ns_steps", default=5, type=int, help="Number of Newton-Schulz steps to perform in the Muon optimizer.")
    parser.add_argument("--untie", default=True, action="store_true")
    parser.add_argument("--momentum", default=0.95, type=float)
    parser.add_argument("--n_repetitions", default=64, type=int, help="Number of times to repeat the dataset.")
    parser.add_argument("--num_shards", default=1, type=int, help="Number of data shards (per dataset type). Should be at least the number of GPUs.")
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode (runs on a single GPU with a small subset of the data).")
    parser.add_argument("--log_mode_switch", action="store_true", help="Log per-rank objective switch timing information.")
    parser.add_argument("--projection_adamh", action="store_true", help="Whether to optimize the projection layer (if it exists) with AdamW instead of Muon.")
    parser.add_argument("--classifier_adamh", action="store_true", help="Whether to optimize the classifier embedding layer (if it exists) with AdamW instead of Muon.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_path = (args.output_dir / args.name)
    args.output_path.mkdir(parents=True, exist_ok=True)
    args.steps_in_epoch = int(args.max_steps / args.n_repetitions)

    args.tokens_per_step = args.global_batch_size * args.max_seq_length

    return args


def setup_training(args, tokenizer):
    assert torch.cuda.is_available()
    args.n_gpu = torch.cuda.device_count()
    args.tokens_per_batch = args.global_batch_size * args.max_seq_length
    if args.max_steps is None:
        args.max_steps = (args.number_of_tokens // args.tokens_per_batch) + 1
    else:
        args.number_of_tokens = args.max_steps * args.tokens_per_batch
    if args.checkpoint_on == "steps":
        args.next_checkpoint = args.first_checkpoint
    elif args.checkpoint_on == "tokens":
        args.next_checkpoint = math.ceil(args.first_checkpoint / args.tokens_per_batch)
    if args.checkpoint_before_cooldown:
        args.cooldown_checkpoint = args.max_steps - int(args.cooldown_proportion * args.max_steps)
    else:
        args.cooldown_checkpoint = -1

    args.accumulate_steps = max(1, (args.global_batch_size) // args.local_batch_size)

    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    seed_everything(args.seed)

    args.shard_ranks = [0]

    args.vocab_size = tokenizer.get_vocab_size()

    wandb.init(
        name=args.name,
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.experiment
    )
    wandb.config.update(args)
    wandb.save(sys.argv[0], policy="now")


def load_config(args):
    with args.config_file.open("r") as f:
        config = json.load(f)
    for k, v in config.items():
        setattr(args, k, v)
    return args


def prepare_model_and_optimizer(args):
    print(f"Starting to load the config from {args.config_file}", flush=True)
    args = load_config(args)
    print("Starting to load the model", flush=True)
    model = Model(args)
    print("Model loaded", flush=True)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.config.update(args)
    wandb.config.update({"n_params": n_params})
    print(model)
    print(f"NUMBER OF PARAMETERS: {n_params}\n", flush=True)

    for p in model.parameters():
        if not p.data.is_contiguous():
            print("Warning: non-contiguous parameter data", flush=True)
            p.data = p.data.contiguous()

    model.to(args.device)
    model.create_mask(args.device, args.local_batch_size)

    model = torch.compile(model)

    ddp_model = model
    print(f"Model initialized on device {args.device}", flush=True)

    adamh_params = []
    matrix_params = [(n, p) for n, p in model.encoder.named_parameters() if p.ndim == 2]
    if hasattr(model.classifier, "projection"):
        if args.projection_adamh:
            adamh_params.append(("classifier.projection.weight", model.classifier.projection.weight))
        else:
            matrix_params.append(("classifier.projection.weight", model.classifier.projection.weight))
    other_params = [(n, p) for n, p in model.named_parameters() if p.ndim != 2]
    other_params.append(("embedding.word_embedding.weight", model.embedding.word_embedding.weight))
    if args.untie:
        if args.classifier_adamh:
            adamh_params.append(("classifier.emb2vocab.weight", model.classifier.emb2vocab.weight))
        else:
            other_params.append(("classifier.emb2vocab.weight", model.classifier.emb2vocab.weight))

    muon_parameters = [p for _, p in matrix_params]
    adamw_parameters = [p for _, p in other_params]
    adamh_parameters = [p for _, p in adamh_params]

    param_groups = [
        {"params": muon_parameters, "use_muon": True, "lr": args.learning_rate, "weight_decay": args.weight_decay, "momentum": args.momentum, "beta2": 0.95, "use_adamh": False},
        {"params": adamw_parameters, "use_muon": False, "lr": args.adam_learning_rate, "weight_decay": 0.0, "eps": args.optimizer_eps, "betas": (args.optimizer_beta1, args.optimizer_beta2), "use_adamh": False},
    ]

    if adamh_params:
        param_groups.append(
            {"params": adamh_parameters, "use_muon": False, "lr": args.learning_rate, "weight_decay": 0.0, "eps": args.optimizer_eps, "betas": (args.optimizer_beta1, args.optimizer_beta2), "use_adamh": True}
        )

    print(f"Parameters with {args.optimizer} Optimizer:")
    for n, _ in matrix_params:
        print(n)
    print("\nParameters with AdamW Optimizer:")
    for n, _ in other_params:
        print(n)
    print(flush=True)

    if args.optimizer == "muon":
        optimizer = Muon(
            param_groups,
            ns_steps=args.ns_steps,
            coeffs=args.coeffs,
            kimi_adjust_lr=args.kimi_adjust_lr,
            normuon=args.normuon,
            polar_express=args.polar_express,
            hyperball=args.muonh,
        )
    

    if args.scheduler == "trapezoid":
        lr_scheduler = trapezoid_schedule(
            optimizer,
            int(args.max_steps * args.warmup_proportion),
            int(args.max_steps * args.cooldown_proportion),
            args.max_steps
        )
    elif args.scheduler == "trapezoid_sqrt":
        lr_scheduler = trapezoid_schedule_sqrt(
            optimizer,
            int(args.max_steps * args.warmup_proportion),
            int(args.max_steps * args.cooldown_proportion),
            args.max_steps
        )
    elif args.scheduler == "cosine":
        lr_scheduler = cosine_schedule_with_warmup(
            optimizer,
            int(args.max_steps * args.warmup_proportion),
            args.max_steps,
            min_factor=0.1
        )
    elif args.scheduler == "cosine_cooldown":
        lr_scheduler = cosine_schedule_with_warmup_cooldown(
            optimizer,
            int(args.max_steps * args.warmup_proportion),
            int(args.max_steps * args.cooldown_proportion),
            args.max_steps,
            min_factor=0.1
        )
    elif args.scheduler == "flat":
        lr_scheduler = flat_with_warmup_schedule(
            optimizer,
            int(args.max_steps * args.warmup_proportion),
            args.max_steps
        )

    mask_scheduler = MaskScheduler(
        args.mask_p_min,
        args.mask_p_max,
        int(args.max_steps * args.warmup_proportion),
        int(args.max_steps * args.cooldown_proportion),
        args.max_steps
    )
    args.mask_p = args.mask_p_max

    global_step = 0
    if args.checkpoint_foldername is not None:
        path_to_checkpoint = args.checkpoint_foldername / "state_dict.bin"
        state_dict = torch.load(path_to_checkpoint, map_location=args.device)
        # Support both old format (raw state dict) and new format (dict with keys)
        if "model" in state_dict:
            model_state = state_dict["model"]
            optimizer.load_state_dict(state_dict["optimizer"])
            lr_scheduler.load_state_dict(state_dict["lr_schedulers"])
            mask_scheduler.load_state_dict(state_dict["mask_scheduler"])
            global_step = state_dict["global_step"]
        else:
            print("Warning: checkpoint is in legacy format — optimizer/scheduler state not restored.", flush=True)
            model_state = state_dict
        # Align "_orig_mod." prefix between checkpoint and model (torch.compile adds this prefix)
        ckpt_has_prefix = any(k.startswith("_orig_mod.") for k in model_state)
        model_has_prefix = any(k.startswith("_orig_mod.") for k in model.state_dict())
        if ckpt_has_prefix and not model_has_prefix:
            model_state = {k.removeprefix("_orig_mod."): v for k, v in model_state.items()}
        elif not ckpt_has_prefix and model_has_prefix:
            model_state = {"_orig_mod." + k: v for k, v in model_state.items()}
        model.load_state_dict(model_state)

    return model, ddp_model, optimizer, lr_scheduler, mask_scheduler, global_step


@torch.no_grad()
def get_batch(args, dataset):
    batch = dataset.next(args.local_batch_size)
    input_ids, target_ids, doc_ids, mask_p, causal_mask = [t.cuda(non_blocking=True) for t in batch]
    input_ids, target_ids, mask_p = input_ids.t(), target_ids.t(), mask_p.t()

    return input_ids, target_ids, doc_ids, mask_p, causal_mask


def calculate_num_causal_tokens(target_ids):
    num_causal_sequences = int(args.causal_ratio * target_ids.size(1))
    causal_labels = target_ids[:, :num_causal_sequences].flatten()
    num_causal_tokens = (causal_labels != -100).sum()

    return num_causal_tokens, num_causal_sequences


def _do_checkpoint(global_step, args):
    if global_step >= args.next_checkpoint:
        if args.checkpoint_style == "exp":
            args.next_checkpoint *= args.checkpoint_mult
        elif args.checkpoint_style == "linear":
            args.next_checkpoint += args.checkpoint_every
        return True
    elif global_step == args.cooldown_checkpoint:
        return True
    return False


def training_loop(model, ddp_model, train_dataset, valid_diffusion_dataset, valid_causal_dataset, optimizer, lr_scheduler, mask_scheduler, global_step, args):
    if args.checkpoint_init and global_step == 0:
        save_checkpoint(model, optimizer, lr_scheduler, mask_scheduler, global_step, 0, train_dataset, args)

    model = model.train()
    model.zero_grad(set_to_none=True)

    training_metrics = TrainingMetrics(
        num_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        seq_len=args.max_seq_length,
        world_size=1,
        device=args.device,
        peak_flops_per_sec=989e12  # Set this if you have the peak FLOPS of your device
    )

    # initialize the dataloader and the metrics
    total_loss, total_accuracy, total_z_loss, total_mask_p, total_grad_norm = 0.0, 0.0, 0.0, 0.0, 0.0
    total_clm_loss, total_clm_accuracy, total_mlm_loss, total_mlm_accuracy = 0.0, 0.0, 0.0, 0.0
    tokens_trained = global_step * args.tokens_per_step

    # calculate the number of forward passes to perform
    num_steps = int(args.max_steps * args.accumulate_steps)

    # Initialize the progress bar
    progress_bar = tqdm(total=args.max_steps, initial=global_step, disable=not is_main_process(), desc="Train iteration")

    # Start the timer
    training_metrics.step_start()

    # iterate over the steps
    for local_step in range(num_steps):

        next_batch = get_batch(args, train_dataset)

        input_ids, target_ids, doc_ids, mask_p, causal_mask = next_batch
        num_causal_tokens, num_causal_sequences = calculate_num_causal_tokens(target_ids)

        # forward pass, do a more detailed check of the model every 100 steps
        # with ModelLogger(enable=global_step % 100 == 0, module=model):
        output = ddp_model(input_ids, doc_ids, causal_mask, target_ids)

        loss, accuracy, z_loss, _ = output.loss, output.accuracy, output.z_loss, output.num_tokens

        causal_loss = (loss.detach()[:num_causal_tokens] / input_ids[:, :num_causal_sequences].numel()).sum()
        diffusion_loss = torch.zeros_like(causal_loss)
        if args.causal_ratio < 1.0:
            diffusion_weight = 1.0 / mask_p[target_ids[:, num_causal_sequences:] != -100]
            diffusion_loss = ((loss.detach()[num_causal_tokens:] * diffusion_weight) / input_ids[:, num_causal_sequences:].numel()).sum()

        causal_accuracy = accuracy[:num_causal_tokens].mean()
        diffusion_accuracy = torch.zeros_like(causal_accuracy)
        if args.causal_ratio < 1.0:
            diffusion_accuracy = accuracy[num_causal_tokens:].mean()
        accuracy = accuracy.mean()

        if mask_p.dim() == 2:
            weight = torch.cat([torch.ones(num_causal_tokens, device=loss.device), diffusion_weight.flatten()])
        else:
            weight = 1.0
        
        loss = (loss / input_ids.numel() * weight).sum() / args.accumulate_steps
        z_loss = (z_loss / input_ids.numel() * weight).sum() / args.accumulate_steps

        # backward pass through both losses
        (loss + args.z_loss_weight * z_loss).backward()

        # add the tracked metrics (for gradient accumulation)
        with torch.no_grad():
            total_loss += loss.detach()
            total_clm_loss += causal_loss / args.accumulate_steps
            total_mlm_loss += diffusion_loss / args.accumulate_steps
            total_accuracy += accuracy / args.accumulate_steps
            total_clm_accuracy += causal_accuracy / args.accumulate_steps
            total_mlm_accuracy += diffusion_accuracy / args.accumulate_steps
            total_z_loss += z_loss.detach()
            total_mask_p += mask_p / args.accumulate_steps

        # gradient accumulation -- if we have accumulated enough gradients, we can perform the optimizer step; otherwise, we just continue and backpropagate through the next batch
        if (local_step + 1) % args.accumulate_steps != 0:
            continue

        tokens_trained += args.tokens_per_step

        # clip the gradients
        total_grad_norm += nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient) / args.accumulate_steps

        # optimizer step
        optimizer.step()
        lr_scheduler.step()
        args.mask_p = mask_scheduler.step()
        step_timings = training_metrics.step_end(args.global_batch_size)

        with torch.no_grad():

            # log the metrics
            wandb.log(
                {
                    "train/loss": total_loss,
                    "train/z_loss": total_z_loss,
                    "train/perplexity": math.exp(total_loss),
                    "train/accuracy": total_accuracy * 100.0,
                    "train/mlm_loss": total_mlm_loss,
                    "train/mlm_accuracy": total_mlm_accuracy * 100.0,
                    "train/clm_loss": total_clm_loss,
                    "train/clm_accuracy": total_clm_accuracy * 100.0,
                    "stats/learning_rate_adamw": optimizer.param_groups[0]['lr'],
                    "stats/learning_rate_muon": optimizer.param_groups[0]['lr'],
                    "stats/grad_norm": total_grad_norm,
                    "stats/global_batch_size": args.global_batch_size * args.max_seq_length,
                    "stats/local_batch_size": args.local_batch_size * args.max_seq_length,  # Is this correct?
                    "stats/accumulate_steps": args.accumulate_steps,
                    "stats/mask_p": total_mask_p,
                    "global_step": global_step,
                    "tokens_trained": tokens_trained,
                    "time/step_time": step_timings.step_time_ms,
                    "time/samples_per_sec": step_timings.samples_per_sec,
                    "time/tokens_per_sec": step_timings.tokens_per_sec,
                    "time/flops_per_step": step_timings.flops_per_step,
                    "memory/allocated_mb": step_timings.memory_allocated_mb,
                    "memory/reserved_mb": step_timings.memory_reserved_mb,
                    "time/mfu": step_timings.mfu,
                },
                step=global_step
            )
            wandb.log(training_metrics.get_smoothed(), step=global_step)

        # zero the accumulated gradients and the metrics
        model.zero_grad(set_to_none=True)
        total_loss, total_accuracy, total_z_loss, total_mask_p, total_grad_norm = 0.0, 0.0, 0.0, 0.0, 0.0
        total_clm_loss, total_clm_accuracy, total_mlm_loss, total_mlm_accuracy = 0.0, 0.0, 0.0, 0.0

        global_step += 1
        progress_bar.update()

        # Run validation
        if global_step % args.validate_every == 0:
            model = model.eval()
            with torch.no_grad():
                validate(model, ddp_model, valid_diffusion_dataset, valid_causal_dataset, global_step, args)
            model = model.train()

        # save a backup of the model and the full training state
        if global_step % args.save_every == 0:
            save(model, optimizer, lr_scheduler, mask_scheduler, global_step, train_dataset, args)

        # save a checkpoint of the model and full training state
        if _do_checkpoint(global_step, args):
            save_checkpoint(model, optimizer, lr_scheduler, mask_scheduler, global_step, tokens_trained, train_dataset, args)

        # Exiting the training due to hitting max steps
        if global_step >= args.max_steps:
            progress_bar.close()
            return

        training_metrics.step_start()

    progress_bar.close()


def validate(model, ddp_model, valid_diffusion_dataset, valid_causal_dataset, global_step, args):

    num_steps = args.validation_steps * args.accumulate_steps

    for causal, valid_dataset in [(False, valid_diffusion_dataset), (True, valid_causal_dataset)]:
        total_loss, total_accuracy, total_z_loss, total_mask_p = 0.0, 0.0, 0.0, 0.0
        local_step = 0
        valid_steps = 0
        for batch in valid_dataset.iterate_over_all(args.max_seq_length, args.local_batch_size):
            input_ids, target_ids, doc_ids, mask_p, causal_mask = [t.cuda(non_blocking=True) for t in batch]
            input_ids, target_ids = input_ids.t(), target_ids.t()
            mask_p = mask_p.mean()

            output = ddp_model(input_ids, doc_ids, causal_mask, target_ids)
            local_step += 1

            loss, accuracy, z_loss, _ = output.loss, output.accuracy, output.z_loss, output.num_tokens
            loss, z_loss = loss.mean(), z_loss.mean()

            weight = 1.0 / num_steps
            if mask_p != 0:
                weight = weight * (mask_p / args.mask_p_max)

            # add the tracked metrics (for gradient accumulation)
            total_loss += loss.detach() / num_steps
            total_accuracy += accuracy.mean().item() / num_steps
            total_z_loss += z_loss.detach() / num_steps
            total_mask_p += mask_p / num_steps

            valid_steps += 1

            if valid_steps == num_steps:
                break

        if causal:
            log_dict = {
                "valid/clm_loss": total_loss,
                "valid/clm_z_loss": total_z_loss,
                "valid/clm_perplexity": math.exp(total_loss),
                "valid/clm_accuracy": total_accuracy * 100.0,
            }
        else:
            log_dict = {
                "valid/mlm_loss": total_loss,
                "valid/mlm_z_loss": total_z_loss,
                "valid/mlm_perplexity": math.exp(total_loss),
                "valid/mlm_accuracy": total_accuracy * 100.0,
                "valid/mlm_mask": total_mask_p,
            }
        wandb.log(log_dict, step=global_step)


def save(model, optimizer, lr_scheduler, mask_scheduler, global_step, train_dataset, args):
    path_to_save_folder = args.output_path / "final"
    path_to_save_folder.mkdir(parents=True, exist_ok=True)
    if is_main_process():
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_schedulers": lr_scheduler.state_dict(),
                "mask_scheduler": mask_scheduler.state_dict(),
                "global_step": global_step,
            },
            path_to_save_folder / "state_dict.bin"
        )


def save_checkpoint(model, optimizer, lr_scheduler, mask_scheduler, global_step, tokens_trained, train_dataset, args):
    if global_step != args.cooldown_checkpoint:
        path_to_save_folder = args.output_path / f"checkpoint_{tokens_trained / 1e9:.2f}B"
    else:
        path_to_save_folder = args.output_path / f"checkpoint_pre_cooldown_{tokens_trained / 1e9:.2f}B"
    path_to_save_folder.mkdir(parents=True, exist_ok=True)
    if is_main_process():
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_schedulers": lr_scheduler.state_dict(),
                "mask_scheduler": mask_scheduler.state_dict(),
                "global_step": global_step,
            },
            path_to_save_folder / "state_dict.bin"
        )


def load_train_dataset(args, tokenizer, global_step=0):
    valid_diffusion_dataset = ValidationMaskedDataset(args.valid_path, tokenizer, args, args.max_seq_length, args.shard_ranks)
    valid_causal_dataset = ValidationCausalDataset(args.valid_path, tokenizer, args, args.max_seq_length, args.shard_ranks)

    train_dataset = TrainDataset(args.train_path, tokenizer, args, args.max_seq_length, args.shard_ranks, args.seed, causal_ratio=args.causal_ratio, shuffle=True)
    if global_step > 0:
        num_sequences_seen = global_step * (args.global_batch_size)
        train_dataset.load_state_from_num_sequences_seen(num_sequences_seen)

    # train_diffusion_dataset = DiffusionDatasetv2(args.train_path, tokenizer, args, args.max_seq_length, args.shard_ranks, shuffle=True)
    # train_causal_dataset = CausalDatasetv2(args.train_path, tokenizer, args, args.max_seq_length, args.shard_ranks, shuffle=True)

    return train_dataset, valid_diffusion_dataset, valid_causal_dataset


if __name__ == "__main__":
    args = parse_arguments()

    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    print(f"Tokenizer loaded from {args.tokenizer_path}", flush=True)

    setup_training(args, tokenizer)
    model, ddp_model, optimizer, lr_scheduler, mask_scheduler, global_step = prepare_model_and_optimizer(args)
    train_dataset, valid_diffusion_dataset, valid_causal_dataset = load_train_dataset(args, tokenizer, global_step)

    training_loop(model, ddp_model, train_dataset, valid_diffusion_dataset, valid_causal_dataset, optimizer, lr_scheduler, mask_scheduler, global_step, args)

    save(model, optimizer, lr_scheduler, mask_scheduler, args.max_steps, train_dataset, args)
