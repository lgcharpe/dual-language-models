from __future__ import annotations

import yaml
from typing import TYPE_CHECKING, Any, Optional
from dataclasses import dataclass, field, fields
from pathlib import Path
import math

if TYPE_CHECKING:
    from argparse import Namespace

@dataclass
class ModelConfig:
    # Model architecture parameters
    model_name: str = "Base"
    attention_pre_norm_affine: bool = True
    classifier_pre_norm_affine: bool = True
    feed_forward_pre_norm_affine: bool = True
    norm_eps: float = 1e-7
    d_h: int = 64
    hidden_size: int = 192
    num_attention_heads: int = 3
    intermediate_size: int = 768
    num_layers: int = 12
    vocab_size: int = 51200
    max_sequence_length: int = 2048
    rope_theta: float = 10000.0
    tie_weights: bool = False
    dataset_type: str = "causal"  # "masked" or "causal", will be set in distributed setup based on hybrid training parameters

    # Model optimization parameters
    compile_model: bool = True

    # Tokenizer parameters
    tokenizer_path: Path = Path("tokenizers/tokenizer.json")
    n_special_tokens: int = 16

    @classmethod
    def from_dict(cls, config_dict: dict) -> ModelConfig:
        existing_keys = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in existing_keys}
        return cls(**filtered_dict)

    def update(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)


@dataclass
class DataConfig:
    train_data_path: Path = Path("data/train")
    val_data_path: Path = Path("data/val")
    num_shards: int = 1
    num_train_tokens: Optional[int] = 100_000_000
    shard_ranks: Optional[list[int]] = None
    dataset_type: str = "causal"

    @classmethod
    def from_dict(cls, config_dict: dict) -> DataConfig:
        existing_keys = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in existing_keys}
        return cls(**filtered_dict)

    def update(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)


@dataclass
class TrainingConfig:
    # Initialization and seeding parameters
    seed: int = 42
    load_checkpoint: bool = False
    checkpoint_foldername: Optional[Path] = None
    device: str = "cpu"

    # Loss parameters
    hybrid_numerator: int = 1
    hybrid_denominator: int = 2
    z_loss_weight: float = 0.0001

    # Training duration parameters
    max_steps: Optional[int] = 1_000
    number_of_tokens: Optional[int] = None
    epochs: Optional[float] = None
    tokens_per_step: Optional[int] = None

    # Batch Size
    local_batch_size: int = 16
    global_batch_size: int = 128
    accumulate_steps: int = 1

    # Learning Rate
    adam_learning_rate: float = 1e-4
    muon_learning_rate: float = 1e-2
    scheduler: str = "cosine"
    warmup_proportion: float = 0.1
    cooldown_proportion: float = 0.1

    # Weight Decay
    adam_weight_decay: float = 0.0
    muon_weight_decay: float = 0.1

    # Gradient Clipping
    max_gradient: float = 1.0

    # Masking parameters
    mask_p_max: float = 0.3
    mask_p_min: float = 0.1
    mask_random_p: float = 0.1
    mask_keep_p: float = 0.1

    # Optimizer parameters
    optimizer: str = "muon"
    adam_eps: float = 1e-8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    momentum: float = 0.95
    muon_beta2: float = 0.95

    @classmethod
    def from_dict(cls, config_dict: dict) -> TrainingConfig:
        existing_keys = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in existing_keys}
        return cls(**filtered_dict)

    def update(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)


@dataclass
class CheckpointConfig:
    output_dir: Path = Path("checkpoints")
    output_path: Optional[Path] = None
    save_every: int = 1000
    checkpoint_style: str = "linear"
    first_checkpoint: int = 1000
    checkpoint_every: int = 1000
    checkpoint_mult: float = 1.0
    checkpoint_on: str = "steps"
    checkpoint_init: bool = False
    checkpoint_before_cooldown: bool = False
    next_checkpoint: Optional[int] = None
    cooldown_checkpoint: Optional[int] = None

    @classmethod
    def from_dict(cls, config_dict: dict) -> CheckpointConfig:
        existing_keys = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in existing_keys}
        return cls(**filtered_dict)

    def update(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)


@dataclass
class LoggingConfig:
    # Run parameters
    run_name: str = "test_run"  # Becomes run name in WandB and part of the checkpoint path
    experiment: str = "test"  # Becomes WandB group name

    # WandB logging parameters
    wandb_log: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None

    # Validation parameters
    validate_every: int = 100
    validation_steps: int = 1

    @classmethod
    def from_dict(cls, config_dict: dict) -> LoggingConfig:
        existing_keys = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in existing_keys}
        return cls(**filtered_dict)

    def update(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)


@dataclass
class DistributedConfig:
    local_rank: int = 0
    world_size: int = 1
    rank: int = 0
    n_gpus: int = 1
    gpus_per_node: int = 1
    distributed: bool = False

    @classmethod
    def from_dict(cls, config_dict: dict) -> DistributedConfig:
        existing_keys = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in existing_keys}
        return cls(**filtered_dict)

@dataclass
class Config:
    model_params: ModelConfig = field(default_factory=ModelConfig)
    data_params: DataConfig = field(default_factory=DataConfig)
    training_params: TrainingConfig = field(default_factory=TrainingConfig)
    logging_params: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint_params: CheckpointConfig = field(default_factory=CheckpointConfig)
    distributed_params: DistributedConfig = field(default_factory=DistributedConfig)
    @classmethod
    def from_yaml(cls, config_path: Path) -> Config:
        config_dict = yaml.safe_load(config_path.open("r"))
        model_params = ModelConfig.from_dict(config_dict.get("model", {}))
        data_params = DataConfig.from_dict(config_dict.get("data", {}))
        training_params = TrainingConfig.from_dict(config_dict.get("training", {}))
        logging_params = LoggingConfig.from_dict(config_dict.get("logging", {}))
        checkpoint_params = CheckpointConfig.from_dict(config_dict.get("checkpointing", {}))
        distributed_params = DistributedConfig.from_dict(config_dict.get("distributed", {}))
        return cls(
            model_params=model_params,
            data_params=data_params,
            training_params=training_params,
            logging_params=logging_params,
            checkpoint_params=checkpoint_params,
            distributed_params=distributed_params,
        )

    def update_from_args(self, args: Namespace) -> None:
        args_dict = vars(args)
        self.model_params.update(**args_dict)
        self.data_params.update(**args_dict)
        self.training_params.update(**args_dict)
        self.logging_params.update(**args_dict)
        self.checkpoint_params.update(**args_dict)

    def create_directories(self) -> None:
        self.checkpoint_params.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_params.output_path = self.checkpoint_params.output_dir / self.logging_params.run_name
        self.checkpoint_params.output_path.mkdir(parents=True, exist_ok=True)

    def calculate_duration_variables(self) -> None:
        self.training_params.tokens_per_step = self.training_params.global_batch_size * self.model_params.max_sequence_length

        # Calculate max_steps and number_of_tokens if one of them is not provided
        if self.training_params.max_steps is None and self.training_params.number_of_tokens is not None:
            self.training_params.max_steps = self.training_params.number_of_tokens // self.training_params.tokens_per_step
        elif self.training_params.max_steps is not None and self.training_params.number_of_tokens is None:
            self.training_params.number_of_tokens = self.training_params.max_steps * self.training_params.tokens_per_step
        elif self.training_params.max_steps is None and self.training_params.number_of_tokens is None:
            raise ValueError("At least one of max_steps or number_of_tokens must be provided.")
        
        # Calculate epochs if not provided
        if self.training_params.epochs is None:
            if self.data_params.num_train_tokens is not None:
                self.training_params.epochs = self.training_params.number_of_tokens / self.data_params.num_train_tokens
            else:
                self.training_params.epochs = 1
        
        # Calculate number of training tokens if not provided
        if self.training_params.number_of_tokens is None:
            self.data_params.num_train_tokens = round(self.training_params.number_of_tokens / self.training_params.epochs)

    def setup_checkpointing(self) -> None:
        if self.checkpoint_params.checkpoint_on == "steps":
            self.checkpoint_params.next_checkpoint = self.checkpoint_params.first_checkpoint
        elif self.checkpoint_params.checkpoint_on == "tokens":
             self.checkpoint_params.next_checkpoint = math.ceil(self.checkpoint_params.first_checkpoint / self.training_params.tokens_per_step)

        if self.checkpoint_params.checkpoint_before_cooldown and self.training_params.cooldown_proportion > 0:
            self.checkpoint_params.cooldown_checkpoint =  self.training_params.max_steps - int(self.training_params.cooldown_proportion * self.training_params.max_steps)
        elif self.checkpoint_params.checkpoint_before_cooldown:
            self.checkpoint_params.cooldown_checkpoint = self.training_params.max_steps