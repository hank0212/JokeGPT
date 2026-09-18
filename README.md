# JokeGPT

[![CI](https://github.com/hank0212/JokeGPT/actions/workflows/ci.yml/badge.svg)](https://github.com/hank0212/JokeGPT/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A character-level GPT written from scratch in PyTorch — no Hugging Face, no
`nn.Transformer`. Pre-trained on general prose, then fine-tuned on ~480k jokes.

Every piece is implemented directly: the attention head, the causal mask,
multi-head attention, the pre-norm residual block, the training loop, and the
sampler. Originally the final project for UCLA **CS M148 (Data Science)**, since
restructured into a runnable package.

```console
$ python scripts/generate.py --prompt "why did the chicken" --top-k 20
why did the chicken prace. spoiler follow, tomorrow was passived, dr. haml, tak
```

That output is from the pre-trained base model, which knows English but not
comedy. See [Known limitations](#known-limitations) — the fine-tuned weights are
not committed, but step 4 below reproduces them.

## The model

|  |  |
|---|---|
| Architecture | Decoder-only transformer, pre-norm |
| Parameters | 0.82M |
| Layers / heads / embedding | 4 / 4 / 128 |
| Context window | 128 characters |
| Vocabulary | 34 characters (`\n !',.;?` + `a-z`) |
| Optimizer | AdamW |

It is deliberately tiny. The point is that it is legible end to end.

## Two-stage training

The project follows the standard pre-train → fine-tune recipe at miniature
scale — the same shape as GPT-3 → ChatGPT, about five orders of magnitude down:

| Stage | Corpus | What it learns | Script |
|---|---|---|---|
| **Pre-train** | TinyStories / MiniPile prose | English spelling, grammar, sentence shape | `scripts/pretrain.py` |
| **Fine-tune** | ~480k jokes from three datasets | Setup–punchline rhythm, joke vocabulary | `scripts/finetune.py` |

Pre-training corpora are too large to hold in memory, so batches are drawn by
memory-mapping the file and decoding a random window (`MemmapCorpus`).
Fine-tuning encodes the whole joke corpus into one tensor and slices it
(`TensorCorpus`).

## Quick start

```bash
git clone https://github.com/hank0212/JokeGPT.git
cd JokeGPT
pip install -r requirements-dev.txt
pip install -e .

# Generate from the checkpoint committed in this repo
python scripts/generate.py --prompt "why did the chicken" --top-k 20

pytest
```

CPU is fine for generation. Training defaults to CUDA when available; override
with `--device cpu`.

## Full pipeline

```bash
# 1. Fetch the joke datasets (~100 MB)
python scripts/download_data.py

# 2. Clean and merge them into one corpus
python scripts/prepare_jokes.py --data-root data/raw --out data/jokes.txt

# 3. Pre-train a base model
#    (optional — checkpoints/model_128_4_4.pth is already a pre-trained base)
python scripts/download_data.py --tinystories        # ~2 GB
python scripts/pretrain.py \
    --train-file data/raw/TinyStories-train.txt \
    --val-file   data/raw/TinyStories-valid.txt \
    --out        checkpoints/base.pth

# 4. Fine-tune on jokes
python scripts/finetune.py \
    --base   checkpoints/model_128_4_4.pth \
    --corpus data/jokes.txt \
    --out    checkpoints/jokegpt.pth

# 5. Sample from the result
python scripts/generate.py --checkpoint checkpoints/jokegpt.pth --prompt "my wife"
```

Every script takes `--help`.

> **Fine-tuning must reuse the pre-training vocabulary.** Step 2 also writes a
> vocab derived from the joke corpus; do *not* pass it to `finetune.py` against
> the committed checkpoint. Omitting `--vocab` uses the correct built-in default.

## Repository layout

```
src/jokegpt/
  config.py                 ModelConfig / TrainConfig (frozen dataclasses)
  tokenizer.py              CharTokenizer — character <-> id mapping
  model.py                  Head, MultiHeadAttention, FeedForward, Block, GPTLanguageModel
  generate.py               Sampling with temperature and top-k
  data/
    cleaning.py             clean_text / is_valid_line — corpus normalisation
    jokes.py                Loaders for the three joke datasets
    batching.py             TensorCorpus (in-memory) and MemmapCorpus (streamed)
  training/
    trainer.py              Training loop and loss estimation
    checkpoint.py           Save/load, architecture recovered from the file
    logging_backends.py     Console logging; optional Comet via env vars

scripts/                    download_data, prepare_jokes, pretrain, finetune, generate
tests/                      77 tests, 96% coverage
notebooks/                  The two original Colab notebooks, kept for reference
checkpoints/                model_128_4_4.pth (pre-trained base) + vocab.txt
```

## Datasets

| Dataset | Size | Fields used |
|---|---|---|
| [taivop/joke-dataset](https://github.com/taivop/joke-dataset) | ~200k | reddit `title` + `body`, stupidstuff `body`, wocka `body` |
| [amoudgl/short-jokes-dataset](https://github.com/amoudgl/short-jokes-dataset) | ~230k | `Joke` |
| [shuttie/dadjokes](https://huggingface.co/datasets/shuttie/dadjokes) | ~53k | `question` + `response` |
| [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | ~2 GB | raw text (pre-training) |

None are vendored here; `scripts/download_data.py` fetches them.

## The tokenizer

`clean_text` lowercases everything, strips URLs, emails and any token containing
a digit, and keeps only `[a-z.,!?;' \n]`. Sorted, that is exactly 34 symbols —
matching the 34-row output layer of the committed checkpoint:

```
'\n !\',.;?abcdefghijklmnopqrstuvwxyz'
```

The original vocabulary file lived in Google Drive and was never committed. It
is reconstructed in `tokenizer.DEFAULT_VOCAB` and saved at
`checkpoints/vocab.txt`; round-tripping the checkpoint produces well-formed
English words, which confirms it is the mapping the model was trained with.

## Notes on the port

This code was extracted from the two Colab notebooks in `notebooks/`. The
architecture and training mathematics are unchanged — the checkpoint loads and
generates identically. What changed:

- **Configuration is injected, not global.** The notebook model classes read
  `n_embd`, `block_size`, `dropout` and `vocab_size` from the notebook
  namespace, so they could only be built in that one session, and loading a
  checkpoint under mismatched globals failed confusingly. They now take a
  `ModelConfig`, and `load_checkpoint` derives the architecture from the
  checkpoint's own stored hyperparameters.
- **`estimate_loss` reported wrong numbers.** The fine-tuning notebook allocated
  `torch.zeros(eval_iters)` with `eval_iters = 2000`, filled only the first 100
  entries, then averaged the whole buffer — scaling every reported loss down by
  roughly 20x. Fixed, with a regression test.
- **No secrets in source.** Both notebooks had a Comet ML API key hardcoded in
  the training cell. Credentials now come from environment variables
  (`.env.example`), and console logging is the default so no account is needed.
- **Four copies of `clean_text` became one**, with regexes compiled once instead
  of per line.
- **Paths are arguments**, not `%cd` into Google Drive.
- **Sampling gained `temperature` and `top_k`**, a noticeable quality
  improvement over the notebook's raw-softmax sampling.
- **Tests added**: 77 covering 96% of the package, run on Python 3.10–3.12 in CI.

## Known limitations

- **The committed checkpoint is the pre-trained base, not the joke model.** Its
  output is prose, not punchlines. The fine-tuned weights were never committed;
  step 4 above reproduces them.
- **The cleaning step damages the training data.** `clean_text` keeps only the
  span between the first and last period, intended to drop partial sentences.
  On jokes it misfires: any joke whose setup ends in a period loses the setup.

  ```
  "I told my wife she was drawing her eyebrows too high.
   She looked surprised."          ->  "she looked surprised"
  ```

  Jokes whose setup ends in `?` are unaffected, which is why dad jokes survive
  and one-liners do not. The default is preserved for checkpoint fidelity; pass
  `--no-trim-sentences` to `prepare_jokes.py` for a corpus that keeps setups.
  Retraining on that corpus is the single highest-value next step here.
- **A 128-character context and 0.82M parameters** cannot reliably hold a setup
  and a punchline together. Output is locally fluent and globally incoherent.
- **Character-level modelling** spends capacity learning spelling that a subword
  tokenizer would provide for free.

## License

[MIT](LICENSE). The joke datasets carry their own licenses.
