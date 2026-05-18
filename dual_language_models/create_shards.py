import argparse
import json
import requests
from tqdm import tqdm
import zstandard as zstd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_shards", type=int, default=256)
    parser.add_argument("--n_input_shards", type=int, default=160)
    parser.add_argument("--url_path", type=str, default="https://data.hplt-project.org/two/cleaned/eng_Latn/{}.jsonl.zst")
    parser.add_argument("--intermediate_path", type=str, default="data/hplt_2_8b_raw/{}.jsonl.zst")
    parser.add_argument("--output_path", type=str, default="data/hplt_2_32b_text_shards/{}.jsonl")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--total_size", type=int, default=2048*2048*2048*4)
    return parser.parse_args()


def iter_zst_lines(file_path, encoding='utf-8'):
    with open(file_path, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(f)

        buffer = b""
        while True:
            chunk = stream_reader.read(8192)
            if not chunk:
                break
            buffer += chunk
            while b'\n' in buffer:
                line, buffer = buffer.split(b'\n', 1)
                try:
                    yield line.decode(encoding)
                except UnicodeDecodeError:
                    return  # stop on malformed line at truncation
        # Optional: yield last line if it looks complete
        if buffer.strip():
            try:
                yield buffer.decode(encoding)
            except UnicodeDecodeError:
                pass  # ignore partial/broken line


def iter_jsonl_lines(file_path, encoding='utf-8'):
    with open(file_path, "r", encoding=encoding) as f:
        for line in f:
            yield line


def iter_text_from_arrow_or_parquet(file_path, text_column="text"):
    try:
        import pyarrow.dataset as ds
    except ImportError as exc:
        raise ImportError(
            "Reading Arrow/Parquet input requires `pyarrow`. "
            "Install it with: pip install pyarrow"
        ) from exc

    lower_path = file_path.lower()
    if lower_path.endswith(".parquet"):
        dataset = ds.dataset(file_path, format="parquet")
    elif lower_path.endswith(".arrow") or lower_path.endswith(".feather"):
        dataset = ds.dataset(file_path, format="ipc")
    else:
        raise ValueError(f"Unsupported tabular format for file: {file_path}")

    scanner = dataset.scanner(columns=[text_column])
    for batch in scanner.to_batches():
        for value in batch.column(text_column).to_pylist():
            if value is None:
                continue
            yield str(value)


def iter_input_texts(file_path, text_column="text"):
    lower_path = file_path.lower()

    if lower_path.endswith(".jsonl.zst"):
        for line in iter_zst_lines(file_path):
            sample = json.loads(line)
            text = sample.get(text_column)
            if text is not None:
                yield str(text)
        return

    if lower_path.endswith(".jsonl"):
        for line in iter_jsonl_lines(file_path):
            sample = json.loads(line)
            text = sample.get(text_column)
            if text is not None:
                yield str(text)
        return

    if lower_path.endswith(".parquet") or lower_path.endswith(".arrow") or lower_path.endswith(".feather"):
        yield from iter_text_from_arrow_or_parquet(file_path, text_column=text_column)
        return

    raise ValueError(f"Unsupported input format for file: {file_path}")


if __name__ == "__main__":
    args = parse_args()

    for i in tqdm(range(args.n_input_shards), desc="Downloading input shards"):
        url_path = args.url_path.format(i + 1)
        output_path = args.intermediate_path.format(i)

        bytes_to_download = args.total_size // args.n_input_shards * 8  # assuming 8 bytes per word

        headers = {
            "Range": f"bytes=0-{bytes_to_download - 1}"
        }

        response = requests.get(url_path, headers=headers, stream=True)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    # Open documents
    input_files = [
        iter_input_texts(args.intermediate_path.format(i), text_column=args.text_column)
        for i in range(args.n_input_shards)
    ]
    output_files = [open(args.output_path.format(i), "w") for i in range(args.n_shards)]

    # Shard files
    shard_lengths = [0 for _ in range(args.n_shards)]
    max_shard_length = args.total_size // args.n_shards

    input_index, output_index = 0, 0
    while input_files and any(length < max_shard_length for length in shard_lengths):
        try:
            line = next(input_files[input_index])
        except StopIteration:
            input_files.pop(input_index)
            if not input_files:
                break
            input_index %= len(input_files)
            continue

        if not line:
            input_index = (input_index + 1) % len(input_files)
            continue

        text = line.strip()
        if not text:
            input_index = (input_index + 1) % len(input_files)
            continue

        while shard_lengths[output_index] >= max_shard_length:
            output_index = (output_index + 1) % len(output_files)

        output_files[output_index].write(f"{json.dumps(text, ensure_ascii=False)}\n")
        shard_lengths[output_index] += len(text.split())

        input_index = (input_index + 1) % len(input_files)
        output_index = (output_index + 1) % len(output_files)

    if any(length < max_shard_length for length in shard_lengths):
        print("Warning: input data was exhausted before all output shards reached target size.")

    # Close files
    for f in input_files:
        f.close()
    for f in output_files:
        f.close()
