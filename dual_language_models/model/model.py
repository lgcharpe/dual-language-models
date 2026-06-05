from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

import math
from functools import partial

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dual_language_models.config import ModelConfig


class ModelOutput:

    def __init__(
        self,
        logits: torch.Tensor | None = None,
        loss: torch.Tensor | float | None = None,
        perplexity: torch.Tensor | float | None = None,
        accuracy: float | None = None,
        z_loss: torch.Tensor | float | None = None,
        num_tokens: int | None = None,
        **kwargs
    ):
        self.logits = logits
        self.loss = loss
        self.perplexity = perplexity
        self.accuracy = accuracy
        self.z_loss = z_loss
        self.num_tokens = num_tokens

        for attr, value in kwargs.items():
            setattr(self, attr, value)


class CastedLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias):
        super().__init__(in_features, out_features, bias=bias)

    def reset_parameters(self) -> None:
        std: float = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), bias=self.bias.type_as(x) if self.bias is not None else None)


class MultiCastedLinearOrtho(nn.Module):

    def __init__(self, in_features, out_features, bias):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weights = nn.ParameterList()
        for out_feature in out_features:
            self.weights.append(nn.Parameter(torch.empty((out_feature, in_features))))

        if bias:
            self.bias = nn.Parameter(torch.zeros(sum(out_features)))
        else:
            self.bias = self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for i, weight in enumerate(self.weights):
            std: float = math.sqrt(2.0 / (self.in_features + self.out_features[i]))
            nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(self, x):
        return F.linear(x, torch.cat([weight for weight in self.weights], dim=0).type_as(x), bias=self.bias.type_as(x) if self.bias is not None else None)


class SwiGLU(nn.Module):
    def forward(self, gate, x):
        x = x * F.silu(gate)
        return x


class Model(nn.Module):

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.embedding: Embedding
        self.encoder: Encoder
        self.classifier: Classifier

        self.embedding = Embedding(config)
        self.encoder = Encoder(config)
        self.classifier = Classifier(config, self.embedding.word_embedding.weight)
        self.loss_fn = partial(F.cross_entropy, reduction='none')

    def change_model_type(self, model_type: str, device: torch.device):
        for layer in self.encoder.layers:
            layer.attention.is_causal = model_type == "causal"
        self.encoder._create_mask(device)

    def create_mask(self, device: torch.device):
        self.encoder._create_mask(device)

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        word_embeddings: torch.Tensor

        word_embeddings = self.embedding(input_ids)

        return word_embeddings

    def get_encodings(self, input_ids: torch.Tensor, doc_ids: torch.Tensor) -> torch.Tensor:
        word_embeddings: torch.Tensor

        word_embeddings = self.embedding(input_ids)
        encodings = self.encoder(word_embeddings, doc_ids)

        return encodings

    def forward(self, input_ids: torch.Tensor, doc_ids: torch.Tensor, labels: torch.Tensor | None = None) -> ModelOutput:
        word_embeddings: torch.Tensor
        encodings: torch.Tensor
        logits: torch.Tensor
        gold_labels: torch.Tensor
        output: ModelOutput

        word_embeddings = self.embedding(input_ids).bfloat16()
        encodings = self.encoder(word_embeddings, doc_ids).bfloat16()
        logits = self.classifier(encodings, labels).float()

        output = ModelOutput(logits=logits, loss=None, perplexity=None, z_loss=None, accuracy=None, num_tokens=None)

        if labels is not None:

            gold_labels = labels.flatten()
            gold_labels = gold_labels[gold_labels != -100]

            output.loss = self.loss_fn(logits, gold_labels)
            output.perplexity = torch.exp(output.loss)
            output.z_loss = torch.logsumexp(logits, dim=-1).pow(2)

            with torch.no_grad():
                output.accuracy = (logits.argmax(-1) == gold_labels).float().mean()

            output.num_tokens = gold_labels.size(0)

        return output


class Encoder(nn.Module):

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.layers: nn.ModuleList[Layer]

        self.layers = nn.ModuleList([Layer(config) for _ in range(config.num_layers)])

        for i, layer in enumerate(self.layers):
            for weight in layer.mlp.up_proj.weights:
                weight.data *= math.sqrt(1.0 / (2.0 * (i + 1)))
            layer.mlp.down_proj.weight.data *= math.sqrt(1.0 / (2.0 * (i + 1)))

    def _create_mask(self, device: torch.device) -> None:
        for layer in self.layers:
            layer._create_mask(device)

    def forward(self, hidden_layer: torch.Tensor, doc_ids: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            hidden_layer = layer(hidden_layer, doc_ids)

        return hidden_layer


class Layer(nn.Module):

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.attention: SelfAttention
        self.mlp: FeedForward

        self.attention = SelfAttention(config)
        self.mlp = FeedForward(config)

    def _create_mask(self, device: torch.device) -> None:
        self.attention._create_block_mask(device)

    def forward(self, hidden_layer: torch.Tensor, doc_ids: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor

        attention_layer = self.attention(hidden_layer, doc_ids)
        mlp_layer = self.mlp(hidden_layer + attention_layer)
        output = hidden_layer + attention_layer + mlp_layer

        return output


class Embedding(nn.Module):

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        assert hasattr(config, "vocab_size"), "The config must have a vocab_size attribute!"
        assert hasattr(config, "hidden_size"), "The config must have a hidden_size attribute!"

        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.initialize(config.hidden_size, config.vocab_size)

    @torch.no_grad()
    def initialize(self, hidden_size: int, vocab_size: int) -> None:
        std: float

        std = math.sqrt(2.0 / (hidden_size + vocab_size))
        nn.init.trunc_normal_(self.word_embedding.weight, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        word_embedding = self.word_embedding(input_ids)

        return word_embedding


class Classifier(nn.Module):

    def __init__(self, config: ModelConfig, embedding_weights: nn.Parameter) -> None:
        super().__init__()

        self.projection: CastedLinear
        self.emb2vocab: CastedLinear
        self.pre_norm: nn.RMSNorm

        self.pre_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps, elementwise_affine=config.classifier_pre_norm_affine)
        self.projection = CastedLinear(config.hidden_size, config.hidden_size, bias=False)
        self.emb2vocab = CastedLinear(config.hidden_size, config.vocab_size, bias=True)

        self.initialize(config.hidden_size, config.vocab_size, config.tie_weights, embedding_weights)

    @torch.no_grad()
    def initialize(self, hidden_size: int, vocab_size: int, tie_weights: bool, embedding_weights: nn.Parameter) -> None:
        proj_std: float = math.sqrt(2.0 / (hidden_size + 4*hidden_size))

        nn.init.trunc_normal_(self.projection.weight, mean=0.0, std=proj_std, a=-2*proj_std, b=2*proj_std)
        if tie_weights:
            self.emb2vocab.weight = embedding_weights
        else:
            std = math.sqrt(2.0 / (hidden_size + vocab_size))
            nn.init.trunc_normal_(self.emb2vocab.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        self.emb2vocab.bias.zero_()

    def project(self, hidden_layer: torch.Tensor) -> torch.Tensor:
        projection: torch.Tensor

        projection = self.projection(hidden_layer)
        projection = F.gelu(projection, approximate='tanh')

        return projection

    def calculate_output(self, hidden_layer: torch.Tensor) -> torch.Tensor:
        return self.emb2vocab(hidden_layer)

    def forward(self, hidden_layer: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        output: torch.Tensor

        if labels is not None:
            hidden_layer = torch.index_select(hidden_layer.flatten(0, 1), 0, torch.nonzero(labels.flatten() != -100).squeeze())

        hidden_layer = self.pre_norm(hidden_layer.float()).type_as(hidden_layer)
        hidden_layer = self.project(hidden_layer)
        output = self.calculate_output(hidden_layer)

        return output


class SelfAttention(nn.Module):

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.d_h = config.d_h
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size

        self.qkv_proj = MultiCastedLinearOrtho(self.hidden_size, [self.hidden_size, self.hidden_size, self.hidden_size], bias=False)
        self.out_proj = CastedLinear(self.d_h*self.num_attention_heads, self.hidden_size, bias=False)

        self.pre_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps, elementwise_affine=config.attention_pre_norm_affine)

        self.rope_embedding = RotaryPositionalEmbeddings(config)
        self.scale: float = 1.0 / math.sqrt(self.d_h)

        self.max_sequence_length = config.max_sequence_length
        self.is_causal = config.dataset_type == "causal"

        self.initialize()

    @staticmethod
    def causal_mask_mode(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx)

    @staticmethod
    def bidirectional_mask_mode(b, h, q_idx, kv_idx):
        return torch.ones_like(q_idx, dtype=torch.bool)

    def _create_block_mask(self, device: torch.device) -> None:
        if self.is_causal:
            self.mask = create_block_mask(
                self.causal_mask_mode,
                None, None, self.max_sequence_length, self.max_sequence_length, device=device
            )
        else:
            self.mask = create_block_mask(
                self.bidirectional_mask_mode,
                None, None, self.max_sequence_length, self.max_sequence_length, device=device
            )

    @torch.no_grad()
    def initialize(self) -> None:
        std: float = math.sqrt(2.0 / (self.hidden_size + 4*self.hidden_size))
        for weight in self.qkv_proj.weights:
            nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2*std, b=2*std)
        self.out_proj.weight.data.zero_()

    def forward(self, hidden_layer: torch.Tensor, doc_ids: torch.Tensor) -> torch.Tensor:
        hidden_layer = self.pre_norm(hidden_layer.float()).type_as(hidden_layer)

        query, key, value = self.qkv_proj(hidden_layer).chunk(3, dim=-1)  # shape: [T, B, H*D]

        query_length: int = hidden_layer.size(0)
        key_length: int = hidden_layer.size(0)
        batch_size: int = hidden_layer.size(1)

        query = query.reshape(query_length, batch_size, self.num_attention_heads, self.d_h).permute(1, 2, 0, 3)  # shape: [B, H, T, D]
        key = key.reshape(key_length, batch_size, self.num_attention_heads, self.d_h).permute(1, 2, 0, 3)  # shape: [B, H, T, D]
        value = value.reshape(key_length, batch_size, self.num_attention_heads, self.d_h).permute(1, 2, 0, 3)  # shape: [B, H, T, D]

        query, key = self.rope_embedding(query, key)

        def score_mod(score, b, h, q_idx, kv_idx):
            return torch.where(doc_ids[b, q_idx] == doc_ids[b, kv_idx], score, -float("inf"))

        output = flex_attention(query, key, value, block_mask=self.mask, score_mod=score_mod)

        output = output.permute(2, 0, 1, 3).flatten(2, 3)  # shape: [T, B, H*D]
        output = self.out_proj(output)

        return output


class FeedForward(nn.Module):

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.up_proj: CastedLinear
        self.down_proj: CastedLinear
        self.pre_norm: nn.RMSNorm
        self.activation: SwiGLU

        self.pre_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps, elementwise_affine=config.feed_forward_pre_norm_affine)
        self.up_proj = MultiCastedLinearOrtho(config.hidden_size, [config.intermediate_size, config.intermediate_size], bias=False)
        self.activation = SwiGLU()
        self.down_proj = CastedLinear(config.intermediate_size, config.hidden_size, bias=False)

        self.initialize(config.hidden_size)

    @torch.no_grad()
    def initialize(self, hidden_size: int) -> None:
        std: float = math.sqrt(2.0 / (5*hidden_size))

        for weight in self.up_proj.weights:
            nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2*std, b=2*std)
        self.down_proj.weight.data.zero_()

    def up_project(self, hidden_layer: torch.Tensor) -> torch.Tensor:
        hidden_layer = self.pre_norm(hidden_layer.float()).type_as(hidden_layer)
        return self.up_proj(hidden_layer)

    def activate(self, projection: torch.Tensor) -> torch.Tensor:
        activated_projection: torch.Tensor

        x, gate = projection.chunk(2, dim=-1)

        activated_projection = self.activation(gate, x)

        return activated_projection

    def down_project(self, activated_projection: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor

        output = self.down_proj(activated_projection)

        return output

    def forward(self, hidden_layer: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor

        output = self.up_project(hidden_layer)
        output = self.activate(output)
        output = self.down_project(output)

        return output


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_matrix: torch.Tensor,
    sin_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_rotate_half: torch.Tensor
    k_rotate_half: torch.Tensor
    q_out: torch.Tensor
    k_out: torch.Tensor

    cos_matrix = cos_matrix[:, None, :, :]
    sin_matrix = sin_matrix[:, None, :, :]

    q_rotate_half = torch.cat(
        [
            -q[:, :, :, q.size(-1) // 2:],
            q[:, :, :, :q.size(-1) // 2]
        ],
        dim=-1
    )

    k_rotate_half = torch.cat(
        [
            -k[:, :, :, k.size(-1) // 2:],
            k[:, :, :, :k.size(-1) // 2]
        ],
        dim=-1
    )

    q_out = q * cos_matrix + q_rotate_half * sin_matrix
    k_out = k * cos_matrix + k_rotate_half * sin_matrix

    return q_out.type_as(q), k_out.type_as(k)


class RotaryPositionalEmbeddings(nn.Module):

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        assert hasattr(config, "d_h"), "The config must have a d_h attribute!"
        assert hasattr(config, "max_sequence_length"), "The config must have a max_sequence_length attribute!"

        self.inv_freq: torch.Tensor
        self.cos_matrix: torch.Tensor
        self.sin_matrix: torch.Tensor
        head_size: int
        max_seq_len: int
        inv_freq: torch.Tensor
        pos: torch.Tensor
        embedding: torch.Tensor

        head_size = config.d_h
        assert head_size % 2 == 0
        max_seq_len = config.max_sequence_length

        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, head_size, 2, dtype=torch.float32) / head_size))
        pos = torch.arange(max_seq_len, dtype=torch.float32)
        embedding = torch.einsum('n, d -> nd', pos, inv_freq)
        embedding = torch.cat([embedding, embedding], dim=-1).unsqueeze(0)
        self.register_buffer("cos_matrix", embedding.cos(), persistent=False)
        self.register_buffer("sin_matrix", embedding.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len: int

        q_layer = q.float()
        k_layer = k.float()

        seq_len = q_layer.size(2)

        q_layer, k_layer = apply_rotary_pos_emb(
            q_layer, k_layer, self.cos_matrix[:, :seq_len], self.sin_matrix[:, :seq_len]
        )

        return q_layer.type_as(q), k_layer.type_as(k)
