from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tokenizers import Tokenizer
from tqdm import tqdm

from dual_language_models.tokenization.tokenize_shards import tokenize_shard


def _tokenize_shard_worker(
    shard_index: int,
    input_path_template: str,
    output_path_template: str,
    output_valid_path_template: str | None,
    tokenizer_path: str,
    max_size: int,
    text_column: str,
) -> int:
    # Avoid excessive nested parallelism when running multiple processes.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tokenizer = Tokenizer.from_file(tokenizer_path)

    input_path = Path(input_path_template.format(shard_index))
    output_path = Path(output_path_template.format(shard_index))
    output_valid_path = None
    if output_valid_path_template is not None:
        output_valid_path = Path(output_valid_path_template.format(shard_index))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_valid_path is not None:
        output_valid_path.parent.mkdir(parents=True, exist_ok=True)

    tokenize_shard(
        tokenizer,
        input_path,
        output_path,
        output_valid_path,
        max_size=max_size,
        verbose=False,
        text_column=text_column,
    )

    return shard_index


def tokenize_shards_parallel(
    n_shards: int,
    input_path_template: str,
    output_path_template: str,
    output_valid_path_template: str | None,
    tokenizer_path: str,
    total_size: int,
    text_column: str = "text",
    workers: int | None = None,
    skip_existing: bool = False,
) -> None:
    if n_shards <= 0:
        raise ValueError("n_shards must be greater than 0")

    if workers is None or workers <= 0:
        workers = os.cpu_count() or 1
    workers = max(1, min(workers, n_shards))

    max_size = total_size // n_shards

    shard_indices: list[int] = []
    for shard_index in range(n_shards):
        output_path = Path(output_path_template.format(shard_index))
        output_valid_path = None
        if output_valid_path_template is not None:
            output_valid_path = Path(output_valid_path_template.format(shard_index))

        if skip_existing:
            if output_valid_path is None and output_path.exists():
                continue
            if output_valid_path is not None and output_path.exists() and output_valid_path.exists():
                continue

        shard_indices.append(shard_index)

    if not shard_indices:
        print("All shard outputs already exist; nothing to do.")
        return

    if workers == 1:
        for shard_index in tqdm(shard_indices, desc="Tokenizing shards"):
            _tokenize_shard_worker(
                shard_index=shard_index,
                input_path_template=input_path_template,
                output_path_template=output_path_template,
                output_valid_path_template=output_valid_path_template,
                tokenizer_path=tokenizer_path,
                max_size=max_size,
                text_column=text_column,
            )
        return

    futures = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for shard_index in shard_indices:
            future = executor.submit(
                _tokenize_shard_worker,
                shard_index,
                input_path_template,
                output_path_template,
                output_valid_path_template,
                tokenizer_path,
                max_size,
                text_column,
            )
            futures[future] = shard_index

        failures: list[tuple[int, str]] = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Tokenizing shards"):
            shard_index = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                failures.append((shard_index, str(exc)))

    if failures:
        failures.sort(key=lambda x: x[0])
        details = "\n".join(f"- shard {shard}: {message}" for shard, message in failures[:10])
        raise RuntimeError(
            f"{len(failures)} shard(s) failed during tokenization.\n{details}"
        )
