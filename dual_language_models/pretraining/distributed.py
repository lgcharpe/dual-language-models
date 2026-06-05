from __future__ import annotations

import os
from socket import gethostname
import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from dual_language_models.config import Config
from dual_language_models.utils import seed_everything


def setup_distributed(config: Config):
    
    assert torch.cuda.is_available()
    config.distributed_params.n_gpus = torch.cuda.device_count()
    config.distributed_params.world_size = int(os.environ["WORLD_SIZE"])
    # config.distributed_params.rank = int(os.environ["SLURM_PROCID"])
    config.distributed_params.local_rank = int(os.environ["LOCAL_RANK"])
    config.distributed_params.rank = int(os.environ["RANK"])
    config.distributed_params.gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    assert config.distributed_params.gpus_per_node == torch.cuda.device_count()  # Might create errors on ROCm
    print(f"Hello from rank {config.distributed_params.rank} of {config.distributed_params.world_size} on {gethostname()} where there are {config.distributed_params.gpus_per_node} allocated GPUs per node.", flush=True)

    config.training_params.accumulate_steps = max(1, (config.training_params.global_batch_size // config.distributed_params.world_size) // config.training_params.local_batch_size)

    assert config.distributed_params.world_size % config.training_params.hybrid_denominator == 0
    if (config.distributed_params.rank % config.training_params.hybrid_denominator) < config.training_params.hybrid_numerator:
        config.data_params.dataset_type = "masked"
    else:
        config.data_params.dataset_type = "causal"
    print(f"Dataset type: {config.data_params.dataset_type}", flush=True)

    # config.distributed_params.local_rank = config.distributed_params.rank % config.distributed_params.gpus_per_node

    dist.init_process_group(
        backend="nccl",
        init_method='env://',
        rank=config.distributed_params.rank,
        world_size=config.distributed_params.world_size,
        timeout=datetime.timedelta(minutes=10)
    )

    seed_everything(config.training_params.seed + config.distributed_params.rank)

    num_shards_per_gpu = config.data_params.num_shards // config.distributed_params.world_size
    config.data_params.shard_ranks = [i for i in range(config.distributed_params.rank * num_shards_per_gpu, (config.distributed_params.rank + 1) * num_shards_per_gpu)]
    # config.data_params.shard_ranks = [config.data_params.shard_ranks[0]]  # Debugging OOM errors, remove this line for full dataset
    torch.cuda.set_device(config.distributed_params.local_rank)
    config.training_params.device = torch.device("cuda", config.distributed_params.local_rank)
    print(f"RCCL started on device {config.training_params.device}", flush=True)
    print(f"host: {gethostname()}, rank: {config.distributed_params.rank}, local_rank: {config.distributed_params.local_rank}")


def cleanup_distributed():
    dist.destroy_process_group()

def setup_model_for_distributed(model: torch.nn.Module, config: Config) -> DDP:
    model = model.cuda(config.training_params.device)
    model = DDP(
        model,
        device_ids=[config.distributed_params.local_rank],
        output_device=config.distributed_params.local_rank,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        find_unused_parameters=True
    )

    print(f"Model initialized on device {config.training_params.device}", flush=True)

    return model