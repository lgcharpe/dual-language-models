from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import zstandard as zstd


def _iter_jsonl_lines(input_path: Path) -> Iterator[str]:
    with input_path.open("rt", encoding="utf-8") as f:
        for line in f:
            yield line


def _iter_zst_lines(input_path: Path) -> Iterator[str]:
    with input_path.open("rb") as f:
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(f)

        buffer = b""
        while True:
            chunk = stream_reader.read(8192)
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                try:
                    yield line.decode("utf-8")
                except UnicodeDecodeError:
                    return

        if buffer.strip():
            try:
                yield buffer.decode("utf-8")
            except UnicodeDecodeError:
                pass


def _iter_text_from_arrow_or_parquet(input_path: Path, text_column: str) -> Iterator[str]:
    try:
        import pyarrow.dataset as ds
    except ImportError as exc:
        raise ImportError(
            "Reading Arrow/Parquet input requires pyarrow. Install it with: pip install pyarrow"
        ) from exc

    input_path_str = str(input_path)
    lower_path = input_path_str.lower()
    if lower_path.endswith(".parquet"):
        dataset = ds.dataset(input_path_str, format="parquet")
    elif lower_path.endswith(".arrow") or lower_path.endswith(".feather"):
        dataset = ds.dataset(input_path_str, format="ipc")
    else:
        raise ValueError(f"Unsupported tabular format for file: {input_path_str}")

    scanner = dataset.scanner(columns=[text_column])
    for batch in scanner.to_batches():
        for value in batch.column(text_column).to_pylist():
            if value is None:
                continue
            yield str(value)


def iter_input_texts(input_path: str | Path, text_column: str = "text") -> Iterator[str]:
    path = Path(input_path)
    input_path_str = str(path)
    lower_path = input_path_str.lower()

    if lower_path.endswith(".jsonl.zst"):
        for line in _iter_zst_lines(path):
            sample = json.loads(line)
            text = sample.get(text_column)
            if text is not None:
                yield str(text)
        return

    if lower_path.endswith(".jsonl"):
        for line in _iter_jsonl_lines(path):
            sample = json.loads(line)
            text = sample.get(text_column)
            if text is not None:
                yield str(text)
        return

    if lower_path.endswith(".parquet") or lower_path.endswith(".arrow") or lower_path.endswith(".feather"):
        yield from _iter_text_from_arrow_or_parquet(path, text_column=text_column)
        return

    raise ValueError(f"Unsupported input format for file: {input_path_str}")
