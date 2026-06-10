from __future__ import annotations
import torch
import numpy as np
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from tokenizers import Tokenizer
    from argparse import Namespace


class ValidationDataset:
    def __init__(self, dataset: str, tokenizer, args, seq_length, ranks):
        self.dataset = dataset
        self.max_seq_length = seq_length + 1
        self.n_special_tokens = args.n_special_tokens
        self.args = args
        self.global_step = 0

        self.mask_index = tokenizer.token_to_id("<mask>")
        self.cls_index = tokenizer.token_to_id("<s>")
        self.pad_index = tokenizer.token_to_id("<pad>")

        self.doc_segments = []
        self.orders = []
        self.lens = []
        self.seed = args.seed
        for rank in ranks:
            documents = torch.load(f"{dataset}/{rank:d}.bin", weights_only=False)
            print(f"Dataset {dataset}/{rank:d}.bin loaded", flush=True)
            for i, document in enumerate(documents):
                # if i % args.document_skip != 0:
                #     continue

                document = torch.cat([torch.LongTensor([self.cls_index]), document])
                self.doc_segments += [
                    document[offset : offset + self.max_seq_length]
                    for offset in range(0, len(document), self.max_seq_length)
                    if len(document) > 0 and len(document) - offset > 1
                ]
        self.len = len(self.doc_segments)
        self.current_idx = 0

        print(f"Validation dataset loaded with {self.len} segments", flush=True)


class ValidationCausalDataset(ValidationDataset):

    def iterate_over_all(self, seq_len, batch_size):
        self.current_idx = 0  # Reset for next iteration
        while self.current_idx < self.len:
            yield self.next(seq_len, batch_size)
        # self.current_idx = 0  # Reset for next iteration

    def next(self, current_seq_len, batch_size):
        all_input_ids, all_target_ids, all_sequence_lengths = [], [], []
        for _ in range(batch_size):
            if self.current_idx > self.len:
                break
            input_ids, target_ids, sequence_lengths, _ = self._getitem()
            self.current_idx += 1

            all_input_ids.append(input_ids)
            all_target_ids.append(target_ids)
            all_sequence_lengths.append(sequence_lengths)

        input_ids = torch.stack(all_input_ids)
        target_ids = torch.stack(all_target_ids)
        sequence_lengths = torch.stack(all_sequence_lengths)
        causal_mask = torch.ones(len(input_ids), dtype=torch.bool)

        return input_ids, target_ids, sequence_lengths, torch.zeros([]), causal_mask

    def _getitem(self):
        tokens = self.doc_segments[self.current_idx]
        seq_length = min(self.max_seq_length, tokens.size(0))

        input_ids = tokens[:seq_length].long()
        target_ids = tokens[:seq_length].long()

        document_index = 0
        sequence_lengths = torch.full((seq_length,), document_index, dtype=torch.int)

        while self.max_seq_length - input_ids.size(0) > 1:
            self.current_idx += 1
            if self.current_idx >= self.len:
                break
            tokens = self.doc_segments[self.current_idx].long()
            seq_length = min(self.max_seq_length - input_ids.size(0), tokens.size(0))

            # select random offset
            offset = 0
            if seq_length < tokens.size(0):
                offset = torch.randint(0, tokens.size(0) - seq_length, size=(1,)).item()

            tokens = tokens[offset:offset + seq_length]

            input_ids = torch.cat([
                input_ids,
                tokens
            ])
            target_ids = torch.cat([
                target_ids,
                tokens
            ])
            document_index += 1
            sequence_lengths = torch.cat([
                sequence_lengths,
                torch.full((seq_length,), document_index, dtype=torch.int)
            ])

        padding_length = self.max_seq_length - input_ids.size(0)
        if padding_length > 0:
            input_ids = torch.cat([
                input_ids,
                torch.LongTensor([self.pad_index] * padding_length)
            ])
            target_ids = torch.cat([
                target_ids,
                torch.LongTensor([-100] * padding_length)
            ])
            document_index += 1
            sequence_lengths = torch.cat([
                sequence_lengths,
                torch.full((padding_length,), document_index, dtype=torch.int)
            ])

        input_ids = input_ids[:-1]
        target_ids = target_ids[1:]
        sequence_lengths = sequence_lengths[:-1]

        return input_ids, target_ids, sequence_lengths, torch.zeros([])


class ValidationMaskedDataset(ValidationDataset):

    def __init__(self, dataset: str, tokenizer, args, seq_length, ranks):
        super().__init__(dataset, tokenizer, args, seq_length, ranks)

        self.masking_strategy = SpanMaskingStrategy(args.n_special_tokens, args.mask_random_p, args.mask_keep_p, args.vocab_size, self.mask_index)

    def iterate_over_all(self, seq_len, batch_size):
        self.current_idx = 0  # Reset for next iteration
        while self.current_idx < self.len:
            yield self.next(seq_len, batch_size)

    def next(self, current_seq_len, batch_size):
        all_input_ids, all_target_ids, all_sequence_lengths, all_real_mask_p = [], [], [], []
        for _ in range(batch_size):
            if self.current_idx > self.len:
                break
            input_ids, target_ids, sequence_lengths, real_mask_p = self._getitem()
            self.current_idx += 1

            all_input_ids.append(input_ids)
            all_target_ids.append(target_ids)
            all_sequence_lengths.append(sequence_lengths)
            all_real_mask_p.append(real_mask_p)

        input_ids = torch.stack(all_input_ids)
        target_ids = torch.stack(all_target_ids)
        sequence_lengths = torch.stack(all_sequence_lengths)
        real_mask_p = torch.stack(all_real_mask_p).mean()
        causal_mask = torch.zeros(len(input_ids), dtype=torch.bool)

        return input_ids, target_ids, sequence_lengths, real_mask_p, causal_mask

    def apply_mask(self, input_ids, mask_ratios, replacement_ids):
        mask_p = self.args.mask_p_min
        mask_p = torch.topk(mask_ratios, max(1, int(mask_ratios.size(0) * mask_p + torch.rand(1).item())), largest=False).values.max().item()

        mask = mask_ratios <= mask_p
        target_ids = torch.where(mask, input_ids, -100)
        input_ids = torch.where(mask, replacement_ids, input_ids)

        real_mask_p = mask.sum() / mask_ratios.numel()

        return input_ids, target_ids, real_mask_p

    def _getitem(self):
        tokens = self.doc_segments[self.current_idx]
        seq_length = min(self.max_seq_length, tokens.size(0))
        tokens = tokens[:seq_length].long()

        mask_ratios, replacement_tokens = self.masking_strategy(tokens)
        input_ids, target_ids, real_mask_p = self.apply_mask(tokens, mask_ratios, replacement_tokens)

        document_index = 0
        sequence_lengths = torch.full((seq_length,), document_index, dtype=torch.int)

        while self.max_seq_length - input_ids.size(0) > 1:
            self.current_idx += 1
            if self.current_idx >= self.len:
                break
            tokens = self.doc_segments[self.current_idx].long()
            seq_length = min(self.max_seq_length - input_ids.size(0), tokens.size(0))

            # select random offset
            offset = 0
            if seq_length < tokens.size(0):
                offset = torch.randint(0, tokens.size(0) - seq_length, size=(1,)).item()

            tokens = tokens[offset:offset + seq_length]

            mask_ratios, replacement_tokens = self.masking_strategy(tokens)
            input_ids_, target_ids_, _ = self.apply_mask(tokens, mask_ratios, replacement_tokens)

            input_ids = torch.cat([
                input_ids,
                input_ids_,
            ])
            target_ids = torch.cat([
                target_ids,
                target_ids_
            ])

            document_index += 1
            sequence_lengths = torch.cat([
                sequence_lengths,
                torch.full((seq_length,), document_index, dtype=torch.int)
            ])

        padding_length = self.max_seq_length - input_ids.size(0)
        if padding_length > 0:
            input_ids = torch.cat([
                input_ids,
                torch.LongTensor([self.pad_index] * padding_length)
            ])
            target_ids = torch.cat([
                target_ids,
                torch.LongTensor([-100] * padding_length)
            ])
            document_index += 1
            sequence_lengths = torch.cat([
                sequence_lengths,
                torch.full((padding_length,), document_index, dtype=torch.int)
            ])

        input_ids = input_ids[:-1]
        target_ids = target_ids[1:]
        sequence_lengths = sequence_lengths[:-1]

        return input_ids, target_ids, sequence_lengths, real_mask_p

# This class is the same as TrainDataset in dataset.py, but does both dataset at once, meaning that when forming the batch, it returns datat with both
# causal and maksed/diffusion masking based on a passed ratio. By doing this we can do dual objective training on a single GPU and easily scale as well as
# choose percentages not based on the nnumber of GPUs. The goal is to use this for both cases if still efficient.
class TrainDataset:
    def __init__(self: TrainDataset, dataset: str, tokenizer: Tokenizer, args: Namespace, seq_length: int, ranks: list[int], seed: int, causal_ratio: float = 0.5, masked_objective: str = "diffusion", shuffle: bool = True) -> None:
        self.dataset = dataset
        self.seq_length = seq_length
        self.n_special_tokens = args.n_special_tokens
        self.args = args
        self.shuffle = shuffle
        self.current_idx = 0
        self.iterations = 0
        self.seed = seed
        self.causal_ratio = max(min(causal_ratio, 1.0), 0.0)
        self.masked_objective = masked_objective

        self.mask_index = tokenizer.token_to_id("<mask>")
        self.cls_index = tokenizer.token_to_id("<s>")
        self.pad_index = tokenizer.token_to_id("<pad>")

        self.tensors, self.doc_boundaries = self._build_documents(dataset, ranks)

        self.num_sequences = len(self.tensors) // self.seq_length

        self.tensors = self.tensors[:self.num_sequences * self.seq_length]
        self.doc_ids = self._build_doc_ids(self.doc_boundaries)

        self.order = np.arange(len(self.tensors) // self.seq_length)
        if self.shuffle:
            self._reshuffle()

        print(f"TrainDataset initialized with {len(self.order)} sequences", flush=True)

        self.masking_strategy = DiffusionMaskingStrategy(
            n_special_tokens=args.n_special_tokens,
            vocab_size=args.vocab_size,
            mask_token_id=self.mask_index
        ) if masked_objective == "diffusion" else SpanMaskingStrategy(
            n_special_tokens=args.n_special_tokens,
            random_p=args.mask_random_p,
            keep_p=args.mask_keep_p,
            vocab_size=args.vocab_size,
            mask_token_id=self.mask_index
        )

    def _build_documents(self: TrainDataset, dataset: str, ranks: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        all_documents = []
        boundaries = [0]
        total_tokens = 0
        for rank in ranks:
            documents = torch.load(f"{dataset}/{rank:d}.bin", weights_only=False)
            for document in documents:
                doc_with_cls = torch.cat([torch.LongTensor([self.cls_index]), document])
                total_tokens += len(doc_with_cls)
                boundaries.append(total_tokens)
                all_documents.append(doc_with_cls)
        tensors = torch.cat(all_documents)
        boundaries = torch.tensor(boundaries, dtype=torch.long)
        print(f"TrainDataset initialized with {len(tensors)} tokens and {len(boundaries) - 1} documents", flush=True)

        return tensors, boundaries

    def _build_doc_ids(self: TrainDataset, boundaries: torch.Tensor) -> torch.Tensor:
        doc_ids = torch.zeros_like(self.tensors, dtype=torch.long)
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = min(boundaries[i + 1], len(self.tensors))
            if start >= len(self.tensors):
                break
            doc_ids[start:end] = i
        return doc_ids

    def _reshuffle(self) -> None:
        """Permute sequence order (no data copying)."""
        rng = np.random.default_rng(self.seed + self.iterations)
        rng.shuffle(self.order)

    def _apply_mask(self, input_ids, target_ids, mask_ratios, replacement_ids):
        if self.masked_objective == "masked":
            mask_p = self.args.mask_p
        else:  # diffusion
            mask_p = torch.rand(1).item()
        mask_p = torch.topk(mask_ratios, max(1, int(mask_ratios.size(0) * mask_p + torch.rand(1).item())), largest=False).values.max().item()

        mask = mask_ratios <= mask_p
        target_mask = torch.cat([mask[1:], torch.ones(1, dtype=torch.bool)])
        target_ids = torch.where(target_mask, target_ids, -100)
        input_ids = torch.where(mask, replacement_ids, input_ids)

        real_mask_p = mask.sum() / mask_ratios.numel()
        if self.masked_objective == "diffusion":
            real_mask_p = torch.full_like(input_ids, real_mask_p, dtype=torch.float)
            real_mask_p = (1 - 1e-4) * real_mask_p + 1e-4

        return input_ids, target_ids, real_mask_p

    def next(self: TrainDataset, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        all_input_ids, all_target_ids, all_sequence_lengths, all_real_mask_p = [], [], [], []
        num_causal = int(batch_size * self.causal_ratio)
        mask_p_out = torch.zeros([])
        causal_mask = torch.zeros(batch_size, dtype=torch.bool)
        causal_mask[:num_causal] = True
        for i in range(batch_size):
            input_ids, target_ids, doc_ids = self.__getitem__(self.current_idx)
            self.current_idx += 1
            if self.current_idx >= len(self.order):
                if self.shuffle:
                    self.iterations += 1
                    self._reshuffle()
                    print("TrainDataset reloaded")
                self.current_idx = 0

            # Apply masking if needed
            if i >= num_causal:  # Masked/diffusion objective
                mask_ratios, replacement_tokens = self.masking_strategy(input_ids)
                input_ids, target_ids, real_mask_p = self._apply_mask(
                    input_ids, target_ids, mask_ratios, replacement_tokens
                )
                all_real_mask_p.append(real_mask_p)

            all_input_ids.append(input_ids)
            all_target_ids.append(target_ids)
            all_sequence_lengths.append(doc_ids)

        input_ids = torch.stack(all_input_ids)
        target_ids = torch.stack(all_target_ids)
        sequence_lengths = torch.stack(all_sequence_lengths)
        
        if self.causal_ratio < 1.0:
            if self.masked_objective == "diffusion":
                mask_p_out = torch.stack(all_real_mask_p)      # (batch, seq)
            else:  # "masked"
                mask_p_out = torch.stack(all_real_mask_p).mean()  # scalar

        return input_ids, target_ids, sequence_lengths, mask_p_out, causal_mask

    def __getitem__(self: TrainDataset, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = self.order[idx] * self.seq_length
        end = start + self.seq_length
        chunk = self.tensors[start:end + 1]
        if len(chunk) < self.seq_length + 1:
            chunk = torch.cat([chunk, torch.full((self.seq_length + 1 - len(chunk),), self.pad_index, dtype=torch.long)])
        
        input_ids = chunk[:-1]
        target_ids = chunk[1:]
        doc_ids = self.doc_ids[start:end]
        doc_ids = doc_ids - doc_ids[0]  # Reset doc_ids to start from 0 for each sequence
        return input_ids, target_ids, doc_ids

    def load_state(self: TrainDataset, dataset_state: dict[str, int]) -> None:
        self.current_idx = dataset_state["current_idx"]
        self.iterations = dataset_state["iterations"]
        self._reshuffle()

    def load_state_from_num_sequences_seen(self: TrainDataset, num_sequences_seen: int) -> None:
        self.iterations = num_sequences_seen // self.num_sequences
        self.current_idx = num_sequences_seen % self.num_sequences
        print(f"TrainDataset loaded state from {num_sequences_seen} sequences seen: iteration {self.iterations}, current_idx {self.current_idx}", flush=True)
        if self.shuffle:
            self._reshuffle()

    def get_state(self: TrainDataset) -> dict[str, int]:
        return {
            "current_idx": self.current_idx,
            "iterations": self.iterations,
        }
    
    def __len__(self: TrainDataset) -> int:
        return len(self.order)


class SpanMaskingStrategy:
    def __init__(self, n_special_tokens, random_p, keep_p, vocab_size, mask_token_id):
        self.n_special_tokens = n_special_tokens
        self.random_p = random_p
        self.keep_p = keep_p
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.max_span_length = 3

    def __call__(self, tokens):
        length = tokens.size(0)

        span_lengths = torch.randint(1, self.max_span_length + 1, size=(length,), dtype=torch.int)
        cumsum = torch.cumsum(span_lengths, dim=0)

        total_length = cumsum[-1].item()
        indices = torch.zeros(total_length, dtype=torch.int)
        indices[cumsum - span_lengths] = torch.arange(length, dtype=torch.int)
        indices = torch.cummax(indices, dim=0)[0]
        indices = indices[:length]

        max_index = indices[-1].item()
        span_random_numbers_1, span_random_numbers_2 = torch.rand([(max_index + 1) * 2]).chunk(2)

        mask_ratios = span_random_numbers_1[indices]

        mask_ratios[tokens < self.n_special_tokens] = float('inf')

        replacement_p = span_random_numbers_2[indices]
        random_mask = replacement_p < self.random_p

        replacement_tokens = tokens.clone()
        replacement_tokens[random_mask] = torch.randint(
            low=self.n_special_tokens,
            high=self.vocab_size,
            size=[random_mask.sum().item()],
            dtype=torch.long
        )
        replacement_tokens[replacement_p > (self.random_p + self.keep_p)] = self.mask_token_id

        return mask_ratios, replacement_tokens


class DiffusionMaskingStrategy:
    def __init__(self, n_special_tokens, vocab_size, mask_token_id):
        self.n_special_tokens = n_special_tokens
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id

    def __call__(self, tokens):
        length = tokens.size(0)

        mask_ratios = torch.rand(length)
        mask_ratios[tokens < self.n_special_tokens] = float('inf')

        replacement_tokens = tokens.clone()
        replacement_tokens.fill_(self.mask_token_id)

        return mask_ratios, replacement_tokens