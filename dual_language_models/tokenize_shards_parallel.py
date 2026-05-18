import argparse

from dual_language_models.tokenization.parallel_tokenize_shards import tokenize_shards_parallel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_shards", type=int, default=256)
    parser.add_argument("--input_path", type=str, default="data/hplt_2_32b_text_shards/{}.jsonl")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--output_path", type=str, default="data/hplt_2_32b_token_shards/{}.bin")
    parser.add_argument("--output_valid_path", type=str, default="data/hplt_2_32b_valid_token_shards/{}.bin")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizers/tokenizer.json")
    parser.add_argument("--total_size", type=int, default=32 * 1024 * 1024 * 1024)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes; 0 means auto",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip shards whose output files already exist",
    )
    parser.add_argument(
        "--no_valid_split",
        action="store_true",
        help="Do not write validation shard files",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_valid_path = None if args.no_valid_split else args.output_valid_path

    tokenize_shards_parallel(
        n_shards=args.n_shards,
        input_path_template=args.input_path,
        output_path_template=args.output_path,
        output_valid_path_template=output_valid_path,
        tokenizer_path=args.tokenizer_path,
        total_size=args.total_size,
        text_column=args.text_column,
        workers=args.workers,
        skip_existing=args.skip_existing,
    )

    print("Parallel tokenization complete.")


if __name__ == "__main__":
    main()
