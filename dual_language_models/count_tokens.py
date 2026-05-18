import argparse

from dual_language_models.tokenization.token_count import count_tokens


def parse_args():
    parser = argparse.ArgumentParser(description="Count token ids in tokenized shard files")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to a tokenized shard file or a directory containing shard files",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.bin",
        help="Glob pattern when --path is a directory",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search directory recursively",
    )
    parser.add_argument(
        "--per_file",
        action="store_true",
        help="Print per-file token counts",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    total, per_file = count_tokens(
        path=args.path,
        pattern=args.pattern,
        recursive=args.recursive,
    )

    if args.per_file:
        for file_path, count in per_file.items():
            print(f"{file_path}\t{count}")

    print(f"TOTAL_TOKENS\t{total}")
    print(f"NUM_FILES\t{len(per_file)}")


if __name__ == "__main__":
    main()
