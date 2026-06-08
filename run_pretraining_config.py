import os
import sys
import os.path
import argparse
from tqdm import tqdm
from socket import gethostname
import json
import math
from pathlib import Path
from contextlib import nullcontext
import datetime

from tokenizers import Tokenizer
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch._dynamo

from dual_language_models.model.model import Model
from dual_language_models.optimizers.kimi_muon import Muon
from dual_language_models.optimizers.normuon import NorMuonWithAuxAdam
from dual_language_models.utils import trapezoid_schedule, MaskScheduler, is_main_process, seed_everything, cosine_schedule_with_warmup, cosine_schedule_with_warmup_cooldown, flat_with_warmup_schedule, trapezoid_schedule_sqrt
from dual_language_models.pretraining.dataset import ValidationCausalDataset, ValidationMaskedDataset, FusedDatasetv2
from dual_language_models.metrics import TrainingMetrics
from dual_language_models.config import Config
from dual_language_models.arguments import parse_arguments
from dual_language_models.pretraining.distributed import setup_distributed, cleanup_distributed, setup_model_for_distributed

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.suppress_errors = True


# if int(os.environ["SLURM_PROCID"]) == 0:
#     import wandb


def setup_training(config: Config):
    if config.distributed_params.distributed:
        setup_distributed(config)
    else:
        config.training_params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config.data_params.shard_ranks = [0]
    
    if is_main_process() and config.logging_params.wandb_log:
        import wandb
        wandb.init(
            name=config.logging_params.run_name,
            project=config.logging_params.wandb_project,
            entity=config.logging_params.wandb_entity,
            group=config.logging_params.experiment
        )
        wandb.config.update(config)
        wandb.save(sys.argv[0], policy="now")


def prepare_model_and_optimizer(config):
    print("Starting to load the model", flush=True)
    model = Model(config.model_params)
    print("Model loaded", flush=True)

    if is_main_process():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if config.logging_params.wandb_log:
            wandb.config.update({"n_params": n_params})
        print(model)
        print(f"NUMBER OF PARAMETERS: {n_params}\n", flush=True)

    for p in model.parameters():
        if not p.data.is_contiguous():
            print("Warning: non-contiguous parameter data", flush=True)
            p.data = p.data.contiguous()

    model.to(config.training_params.device)
    model.create_mask(config.training_params.device)

    if config.model_params.compile_model:
        model = torch.compile(model)

    if config.distributed_params.distributed:
        ddp_model = setup_model_for_distributed(model, config)
    else:
        ddp_model = model

    matrix_params = [(n, p) for n, p in model.encoder.named_parameters() if p.ndim == 2]
    if hasattr(model.classifier, "projection"):
        matrix_params.append(("classifier.projection.weight", model.classifier.projection.weight))
    other_params = [(n, p) for n, p in model.named_parameters() if p.ndim != 2]
    other_params.append(("embedding.word_embedding.weight", model.embedding.word_embedding.weight))
    if not config.model_params.tie_weights:
        other_params.append(("classifier.emb2vocab.weight", model.classifier.emb2vocab.weight))

    muon_parameters = [p for _, p in matrix_params]
    adamw_parameters = [p for _, p in other_params]

    param_groups = [
        {"params": muon_parameters, "use_muon": True, "lr": config.training_params.muon_learning_rate, "weight_decay": config.training_params.muon_weight_decay, "momentum": config.training_params.momentum, "beta2": config.training_params.muon_beta2},
        {"params": adamw_parameters, "use_muon": False, "lr": config.training_params.adam_learning_rate, "weight_decay": config.training_params.adam_weight_decay, "eps": config.training_params.adam_eps, "betas": (config.training_params.adam_beta1, config.training_params.adam_beta2)},
    ]

    if is_main_process():
        print(f"Parameters with {config.training_params.optimizer} Optimizer:")
        for n, _ in matrix_params:
            print(n)
        print("\nParameters with AdamW Optimizer:")
        for n, _ in other_params:
            print(n)
        print(flush=True)

    if config.training_params.optimizer == "muon":
        optimizer = Muon(
            muon_params=muon_parameters,
            lr=config.training_params.learning_rate,
            wd=config.training_params.weight_decay,
            momentum=config.training_params.momentum,
            nesterov=True,
            ns_steps=5,
            adamw_params=adamw_parameters
        )
    elif config.training_params.optimizer == "normuon":
        optimizer = NorMuonWithAuxAdam(
            param_groups
        )

    if config.training_params.warmup_proportion < 1:
        warmup_steps = int(config.training_params.max_steps * config.training_params.warmup_proportion)
    else:
        warmup_steps = int(config.training_params.warmup_proportion)
    
    if config.training_params.cooldown_proportion < 1:
        cooldown_steps = int(config.training_params.max_steps * config.training_params.cooldown_proportion)
    else:
        cooldown_steps = int(config.training_params.cooldown_proportion)

    if config.training_params.scheduler == "trapezoid":
        lr_scheduler = trapezoid_schedule(
            optimizer,
            warmup_steps,
            cooldown_steps,
            config.training_params.max_steps
        )
    elif config.training_params.scheduler == "trapezoid_sqrt":
        lr_scheduler = trapezoid_schedule_sqrt(
            optimizer,
            warmup_steps,
            cooldown_steps,
            config.training_params.max_steps
        )
    elif config.training_params.scheduler == "cosine":
        lr_scheduler = cosine_schedule_with_warmup(
            optimizer,
            warmup_steps,
            config.training_params.max_steps,
            min_factor=0.1
        )
    elif config.training_params.scheduler == "cosine_cooldown":
        lr_scheduler = cosine_schedule_with_warmup_cooldown(
            optimizer,
            warmup_steps,
            cooldown_steps,
            config.training_params.max_steps,
            min_factor=0.1
        )
    elif config.training_params.scheduler == "flat":
        lr_scheduler = flat_with_warmup_schedule(
            optimizer,
            warmup_steps,
            config.training_params.max_steps
        )

    mask_scheduler = MaskScheduler(
        config.data_params.mask_p_min,
        config.data_params.mask_p_max,
        warmup_steps,
        cooldown_steps,
        config.training_params.max_steps
    )
    config.training_params.mask_p = config.data_params.mask_p_max

    global_step = 0
    if config.training_params.load_checkpoint and config.training_params.checkpoint_foldername is not None:
        path_to_checkpoint = config.training_params.checkpoint_foldername / "state_dict.bin"
        state_dict = torch.load(path_to_checkpoint, map_location=config.training_params.device)
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
def get_batch(config, dataset, global_step):
    batch = dataset.next(config.model_params.max_sequence_length, config.training_params.local_batch_size)
    if torch.cuda.is_available():
        input_ids, target_ids, doc_ids, mask_p = [t.cuda(non_blocking=True) for t in batch]
    else:
        input_ids, target_ids, doc_ids, mask_p = batch
    input_ids, target_ids, mask_p = input_ids.t(), target_ids.t(), mask_p.t()

    return input_ids, target_ids, doc_ids, mask_p


def _do_checkpoint(global_step, checkpoint_config):
    if global_step >= checkpoint_config.next_checkpoint:
        if checkpoint_config.checkpoint_style == "exp":
            checkpoint_config.next_checkpoint *= checkpoint_config.checkpoint_mult
        elif checkpoint_config.checkpoint_style == "linear":
            checkpoint_config.next_checkpoint += checkpoint_config.checkpoint_every
        return True
    elif global_step == checkpoint_config.cooldown_checkpoint:
        return True
    return False


def training_loop(model, ddp_model, train_dataset, valid_diffusion_dataset, valid_causal_dataset, optimizer, lr_scheduler, mask_scheduler, global_step, config):
    if config.checkpoint_params.checkpoint_init and global_step == 0:
        save_checkpoint(model, optimizer, lr_scheduler, mask_scheduler, global_step, 0, train_dataset, config.checkpoint_params)

    model = model.train()
    model.zero_grad(set_to_none=True)

    if is_main_process():
        training_metrics = TrainingMetrics(
            num_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
            seq_len=config.model_params.max_sequence_length,
            world_size=config.distributed_params.world_size,
            device=config.training_params.device,
            peak_flops_per_sec=989e12  # Set this if you have the peak FLOPS of your device
        )

    train_dataset.set_mode("diffusion" if config.data_params.dataset_type == "masked" else "causal")

    # initialize the dataloader and the metrics
    total_loss, total_accuracy, total_z_loss, total_mask_p, total_grad_norm = 0.0, 0.0, 0.0, 0.0, 0.0
    tokens_trained = global_step * config.training_params.tokens_per_step

    # calculate the number of forward passes to perform
    num_steps = int(config.training_params.max_steps * config.training_params.accumulate_steps)

    # Initialize the progress bar
    progress_bar = tqdm(total=config.training_params.max_steps, initial=global_step, disable=not is_main_process(), desc="Train iteration")

    # Start the timer
    if is_main_process():
        training_metrics.step_start()

    # iterate over the steps
    for local_step in range(num_steps):
        epoch = global_step // config.training_params.steps_per_epoch
        if (epoch + config.distributed_params.rank) % config.training_params.hybrid_denominator < config.training_params.hybrid_numerator:
            dataset_type = "masked"
        else:
            dataset_type = "causal"
        if dataset_type != config.data_params.dataset_type:
            config.data_params.dataset_type = dataset_type
            train_dataset.set_mode("diffusion" if config.data_params.dataset_type == "masked" else "causal")
            model.change_model_type(dataset_type, config.training_params.device)

        next_batch = get_batch(config, train_dataset, global_step)

        input_ids, target_ids, doc_ids, mask_p = next_batch

        # forward pass, do a more detailed check of the model every 100 steps
        # with ModelLogger(enable=global_step % 100 == 0, module=model):
        with ddp_model.no_sync() if (local_step + 1) % config.training_params.accumulate_steps != 0 else nullcontext():

            output = ddp_model(input_ids, doc_ids, target_ids)

            loss, accuracy, z_loss, _ = output.loss, output.accuracy, output.z_loss, output.num_tokens

            if config.data_params.dataset_type == "masked":
                weight = 1.0 / mask_p[target_ids != -100]
            else:
                weight = 1.0
            
            loss = (loss / input_ids.numel() * weight).sum() / config.training_params.accumulate_steps
            z_loss = (z_loss / input_ids.numel() * weight).sum() / config.training_params.accumulate_steps

            # backward pass through both losses
            (loss + config.training_params.z_loss_weight * z_loss).backward()

        # add the tracked metrics (for gradient accumulation)
        with torch.no_grad():
            total_loss += loss.detach()
            total_accuracy += accuracy / config.training_params.accumulate_steps
            total_z_loss += z_loss.detach()
            total_mask_p += mask_p / config.training_params.accumulate_steps

        # gradient accumulation -- if we have accumulated enough gradients, we can perform the optimizer step; otherwise, we just continue and backpropagate through the next batch
        if (local_step + 1) % config.training_params.accumulate_steps != 0:
            continue

        tokens_trained += config.training_params.tokens_per_step

        # clip the gradients
        total_grad_norm += nn.utils.clip_grad_norm_(model.parameters(), config.training_params.max_gradient) / config.training_params.accumulate_steps

        # optimizer step
        optimizer.step()
        lr_scheduler.step()
        config.training_params.mask_p = mask_scheduler.step()
        if is_main_process():
            step_timings = training_metrics.step_end(config.training_params.global_batch_size)

        with torch.no_grad():
            ratio = config.training_params.hybrid_numerator / config.training_params.hybrid_denominator
            # be careful here, not all GPUs work with the same training objective
            if config.data_params.dataset_type == "masked":
                total_mlm_loss = total_loss / ratio
                total_mlm_accuracy = total_accuracy / ratio
                total_clm_loss = torch.zeros_like(total_mlm_loss)
                total_clm_accuracy = torch.zeros_like(total_mlm_accuracy)
                total_mask_p = total_mask_p / ratio
            else:
                total_clm_loss = total_loss / (1 - ratio)
                total_clm_accuracy = total_accuracy / (1 - ratio)
                total_mlm_loss = torch.zeros_like(total_clm_loss)
                total_mlm_accuracy = torch.zeros_like(total_clm_accuracy)
                total_mask_p = torch.zeros_like(total_mask_p)

            # accumulate the metrics across GPUs
            metrics = torch.stack([total_loss, total_accuracy, total_z_loss, total_mask_p.mean(), total_mlm_loss, total_mlm_accuracy, total_clm_loss, total_clm_accuracy])
            dist.all_reduce(metrics, dist.ReduceOp.AVG)
            total_loss, total_accuracy, total_z_loss, total_mask_p, total_mlm_loss, total_mlm_accuracy, total_clm_loss, total_clm_accuracy = metrics.tolist()

            # log the metrics
            if is_main_process() and config.logging_params.wandb_log:
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
                        "stats/global_batch_size": config.training_params.global_batch_size * config.model_params.max_sequence_length,
                        "stats/local_batch_size": config.training_params.local_batch_size * config.model_params.max_sequence_length,  # Is this correct?
                        "stats/accumulate_steps": config.training_params.accumulate_steps,
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

        global_step += 1
        progress_bar.update()

        # Run validation
        if global_step % config.checkpoint_params.validate_every == 0:
            model = model.eval()
            with torch.no_grad():
                validate(model, ddp_model, valid_diffusion_dataset, valid_causal_dataset, global_step, config)
            model = model.train()

        # save a backup of the model and the full training state
        if global_step % config.checkpoint_params.save_every == 0:
            save(model, optimizer, lr_scheduler, mask_scheduler, global_step, train_dataset, config.checkpoint_params)

        # save a checkpoint of the model and full training state
        if _do_checkpoint(global_step, config.checkpoint_params):
            save_checkpoint(model, optimizer, lr_scheduler, mask_scheduler, global_step, tokens_trained, train_dataset, config.checkpoint_params)

        # Exiting the training due to hitting max steps
        if global_step >= config.training_params.max_steps:
            progress_bar.close()
            return

        if is_main_process():
            training_metrics.step_start()

    progress_bar.close()


def validate(model, ddp_model, valid_diffusion_dataset, valid_causal_dataset, global_step, config):

    total_loss, total_accuracy, total_z_loss, total_mask_p = 0.0, 0.0, 0.0, 0.0
    local_step = 0
    valid_steps = 0

    valid_dataset = valid_diffusion_dataset if config.data_params.dataset_type == "masked" else valid_causal_dataset

    for batch in valid_dataset.iterate_over_all(config.data_params.max_sequence_length, config.training_params.local_batch_size):
        input_ids, target_ids, doc_ids, mask_p = [t.cuda(non_blocking=True) for t in batch]
        input_ids, target_ids = input_ids.t(), target_ids.t()
        mask_p = mask_p.mean()

        with ddp_model.no_sync() if (local_step + 1) % config.training_params.accumulate_steps != 0 else nullcontext():

            output = ddp_model(input_ids, doc_ids, target_ids)
            local_step += 1

            loss, accuracy, z_loss, _ = output.loss, output.accuracy, output.z_loss, output.num_tokens
            loss, z_loss = loss.mean(), z_loss.mean()

            weight = 1.0 / config.training_params.accumulate_steps
            if mask_p != 0:
                weight = weight * (mask_p / config.data_params.mask_p_max)

        # add the tracked metrics (for gradient accumulation)
        total_loss += loss.detach() / config.training_params.accumulate_steps
        total_accuracy += accuracy / config.training_params.accumulate_steps
        total_z_loss += z_loss.detach() / config.training_params.accumulate_steps
        total_mask_p += mask_p / config.training_params.accumulate_steps

        # gradient accumulation -- if we have accumulated enough gradients, we can perform the optimizer step; otherwise, we just continue and backpropagate through the next batch
        if (local_step + 1) % config.training_params.accumulate_steps != 0:
            continue

        # be careful here, not all GPUs work with the same training objective
        ratio = config.training_params.hybrid_numerator / config.training_params.hybrid_denominator
        if config.data_params.dataset_type == "masked":
            total_mlm_loss = total_loss / ratio
            total_mlm_accuracy = total_accuracy / ratio
            total_clm_loss = torch.zeros_like(total_mlm_loss)
            total_clm_accuracy = torch.zeros_like(total_mlm_accuracy)
            total_mask_p = total_mask_p / ratio
        else:
            total_clm_loss = total_loss / (1 - ratio)
            total_clm_accuracy = total_accuracy / (1 - ratio)
            total_mlm_loss = torch.zeros_like(total_clm_loss)
            total_mlm_accuracy = torch.zeros_like(total_clm_accuracy)
            total_mask_p = torch.zeros_like(total_mask_p)

        # accumulate the metrics across GPUs
        metrics = torch.stack([total_loss, total_accuracy, total_z_loss, total_mask_p, total_mlm_loss, total_mlm_accuracy, total_clm_loss, total_clm_accuracy])
        dist.all_reduce(metrics, dist.ReduceOp.AVG)
        total_loss, total_accuracy, total_z_loss, total_mask_p, total_mlm_loss, total_mlm_accuracy, total_clm_loss, total_clm_accuracy = metrics.tolist()

        # log the metrics
        if is_main_process() and config.logging_params.wandb_log:
            wandb.log(
                {
                    "valid/loss": total_loss,
                    "valid/z_loss": total_z_loss,
                    "valid/perplexity": math.exp(total_loss),
                    "valid/accuracy": total_accuracy * 100.0,
                    "valid/mlm_loss": total_mlm_loss,
                    "valid/mlm_accuracy": total_mlm_accuracy * 100.0,
                    "valid/clm_loss": total_clm_loss,
                    "valid/clm_accuracy": total_clm_accuracy * 100.0,
                },
                step=global_step
            )

        # zero the metrics
        total_loss, total_accuracy, total_z_loss, total_mask_p = 0.0, 0.0, 0.0, 0.0

        valid_steps += 1

        if valid_steps == config.logging_params.validation_steps:
            break


def save(model, optimizer, lr_scheduler, mask_scheduler, global_step, train_dataset, checkpoint_config):
    path_to_save_folder = checkpoint_config.output_path / "final"
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
        # torch.save(
        #     train_dataset.get_state(),
        #     path_to_save_folder / f"dataset_info_{checkpoint_config.dataset_type}_{checkpoint_config.shard_rank}.bin"
        # )


def save_checkpoint(model, optimizer, lr_scheduler, mask_scheduler, global_step, tokens_trained, train_dataset, checkpoint_config):
    if global_step != checkpoint_config.cooldown_checkpoint:
        path_to_save_folder = checkpoint_config.output_path / f"checkpoint_{tokens_trained / 1e9:.2f}B"
    else:
        path_to_save_folder = checkpoint_config.output_path / f"checkpoint_pre_cooldown_{tokens_trained / 1e9:.2f}B"
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
        # torch.save(
        #     train_dataset.get_state(),
        #     path_to_save_folder / f"dataset_info_{checkpoint_config.dataset_type}_{checkpoint_config.shard_rank}.bin"
        # )


def load_train_dataset(data_config, seed, tokenizer):
    valid_diffusion_dataset = ValidationMaskedDataset(data_config.valid_path, tokenizer, data_config, data_config.max_sequence_length, data_config.shard_ranks, seed)
    valid_causal_dataset = ValidationCausalDataset(data_config.valid_path, tokenizer, data_config, data_config.max_sequence_length, data_config.shard_ranks, seed)

    train_dataset = FusedDatasetv2(data_config.train_path, tokenizer, data_config, data_config.max_sequence_length, data_config.shard_ranks, seed, shuffle=True)

    # train_diffusion_dataset = DiffusionDatasetv2(data_config.train_path, tokenizer, data_config, data_config.max_sequence_length, data_config.shard_ranks, shuffle=True)
    # train_causal_dataset = CausalDatasetv2(data_config.train_path, tokenizer, data_config, data_config.max_sequence_length, data_config.shard_ranks, shuffle=True)

    return train_dataset, valid_diffusion_dataset, valid_causal_dataset


if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    config = Config.from_yaml(args.config)
    config.update_from_args(args)
    config.create_directories()
    config.calculate_duration_variables()
    config.setup_checkpointing()

    tokenizer = Tokenizer.from_file(str(config.data_params.tokenizer_path))
    print(f"Tokenizer loaded from {config.data_params.tokenizer_path}", flush=True)

    setup_training(config)
    model, ddp_model, optimizer, lr_scheduler, mask_scheduler, global_step = prepare_model_and_optimizer(config)
    train_dataset, valid_diffusion_dataset, valid_causal_dataset = load_train_dataset(config.data_params, config.training_params.seed, tokenizer)

    training_loop(model, ddp_model, train_dataset, valid_diffusion_dataset, valid_causal_dataset, optimizer, lr_scheduler, mask_scheduler, global_step, config)

    save(model, optimizer, lr_scheduler, mask_scheduler, config.training_params.max_steps, train_dataset, config.checkpoint_params)

    if config.distributed_params.distributed:
        cleanup_distributed()
