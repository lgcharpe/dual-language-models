<h2 align="center"><b><h3>Dual Language Models:<br>Balancing Training Efficiency and Overfitting Resilience</h3></b></h2><br>


<p align="center">
  <b>David Samuel</b> and <b>Lucas Georges Gabriel Charpentier</b>
</p>

<p align="center">
  <i>
    University of Oslo<br>
    Language Technology Group<br>
  </i>
</p>
<br>

<p align="center">
  <a href="https://arxiv.org/abs/2410.24159"><b>Paper</b></a><br>
</p>

_______

<br>

### Abstract

This paper combines autoregressive and masked-diffusion training objectives without any architectural modifications, resulting in flexible language models that outperform single-objective models. Autoregressive modeling has been a popular approach, partly because of its training efficiency; however, that comes at the cost of sensitivity to overfitting. On the other hand, masked-diffusion models are less efficient to train while being more resilient to overfitting. In this work, we demonstrate that dual-objective training achieves the best of both worlds. To derive the optimal ratio between both objectives, we train and evaluate 50 language models under varying levels of data repetition. We show that it is optimal to combine both objectives under all evaluated settings and that the optimal ratio is similar whether targeting autoregressive or masked-diffusion downstream performance.

_______

<br>

This is the official repository for Dual Language Models.

_______

<br>

### Please cite the following publication
```bibtex
@misc{samuel2025duallanguagemodelsbalancing,
      title={Dual Language Models: Balancing Training Efficiency and Overfitting Resilience}, 
      author={David Samuel and Lucas Georges Gabriel Charpentier},
      year={2025},
      eprint={2512.14549},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.14549}, 
}
```

### Data Pipeline Usage

The preprocessing and tokenization scripts support text shards stored as:
- `.jsonl`
- `.jsonl.zst`
- `.parquet`
- `.arrow` / `.feather`

For tabular formats (`.parquet`, `.arrow`, `.feather`), use `--text_column` to select the column that contains text (default: `text`).

#### 1) Create text shards

Download from URL (default behavior):
```bash
python -m dual_language_models.create_shards \
  --n_input_shards 160 \
  --intermediate_path data/hplt_2_8b_raw/{}.jsonl.zst \
  --output_path data/hplt_2_32b_text_shards/{}.jsonl
```

Use local Parquet files directly (skip download):
```bash
python -m dual_language_models.create_shards \
  --use_local_files \
  --n_input_shards 160 \
  --intermediate_path data/local_raw/{}.parquet \
  --text_column text \
  --output_path data/hplt_2_32b_text_shards/{}.jsonl
```

Use local Arrow files directly:
```bash
python -m dual_language_models.create_shards \
  --use_local_files \
  --n_input_shards 160 \
  --intermediate_path data/local_raw/{}.arrow \
  --text_column text \
  --output_path data/hplt_2_32b_text_shards/{}.jsonl
```

#### 2) Train tokenizer from shards

From JSONL shards:
```bash
python -m dual_language_models.create_tokenizer \
  --num_shards 256 \
  --shard_path data/hplt_2_32b_text_shards/{}.jsonl
```

From Parquet shards:
```bash
python -m dual_language_models.create_tokenizer \
  --num_shards 256 \
  --shard_path data/local_text_shards/{}.parquet \
  --text_column text
```

From Arrow shards:
```bash
python -m dual_language_models.create_tokenizer \
  --num_shards 256 \
  --shard_path data/local_text_shards/{}.arrow \
  --text_column text
```

#### 3) Tokenize text shards

From JSONL shards:
```bash
python -m dual_language_models.tokenize_shards \
  --n_shards 256 \
  --input_path data/hplt_2_32b_text_shards/{}.jsonl \
  --output_path data/hplt_2_32b_token_shards/{}.bin \
  --output_valid_path data/hplt_2_32b_valid_token_shards/{}.bin
```

From Parquet shards:
```bash
python -m dual_language_models.tokenize_shards \
  --n_shards 256 \
  --input_path data/local_text_shards/{}.parquet \
  --text_column text \
  --output_path data/hplt_2_32b_token_shards/{}.bin \
  --output_valid_path data/hplt_2_32b_valid_token_shards/{}.bin
```

From Arrow shards:
```bash
python -m dual_language_models.tokenize_shards \
  --n_shards 256 \
  --input_path data/local_text_shards/{}.arrow \
  --text_column text \
  --output_path data/hplt_2_32b_token_shards/{}.bin \
  --output_valid_path data/hplt_2_32b_valid_token_shards/{}.bin
```

Note: Arrow/Parquet support requires `pyarrow`.

#### Common errors

1. Missing `pyarrow`

If you read `.parquet`, `.arrow`, or `.feather` shards without `pyarrow`, scripts fail with an import error.

Install it with:
```bash
uv add pyarrow
```

2. Wrong `--text_column`

If the text column name does not exist in your JSON records or tabular file schema, no usable documents are read (or a column error is raised for tabular formats).

Verify your column name and pass it explicitly:
```bash
python -m dual_language_models.create_tokenizer \
  --shard_path data/local_text_shards/{}.parquet \
  --text_column text
```
