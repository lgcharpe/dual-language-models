import argparse
import json
import os
import requests
from tqdm import tqdm
from dual_language_models.data_utils import iter_input_texts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_shards", type=int, default=256)
    parser.add_argument("--n_input_shards", type=int, default=160)
    parser.add_argument("--url_path", type=str, default="https://data.hplt-project.org/two/cleaned/eng_Latn/{}.jsonl.zst")
    parser.add_argument("--intermediate_path", type=str, default="data/hplt_2_8b_raw/{}.jsonl.zst")
    parser.add_argument("--output_path", type=str, default="data/hplt_2_32b_text_shards/{}.jsonl")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--use_local_files", action="store_true", help="Use local intermediate files and skip URL download")
    parser.add_argument("--total_size", type=int, default=2048*2048*2048*4)
    return parser.parse_args()


def is_url(value):
    return value.startswith("http://") or value.startswith("https://")


if __name__ == "__main__":
    args = parse_args()

    use_local_files = args.use_local_files or not is_url(args.url_path)

    if not use_local_files:
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
    else:
        missing_paths = [
            args.intermediate_path.format(i)
            for i in range(args.n_input_shards)
            if not os.path.exists(args.intermediate_path.format(i))
        ]
        if missing_paths:
            first_missing = missing_paths[0]
            raise FileNotFoundError(
                f"Local input file not found: {first_missing}. "
                "Provide existing files via --intermediate_path or disable --use_local_files."
            )

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
