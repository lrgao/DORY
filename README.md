# Prompt Recovery Pipeline

This is the cleaned, end-to-end version of the prompt recovery project. The old numbered scripts are no longer the main interface. This folder is organized as a single pipeline with clear source code, data, scripts, and outputs.

## Directory Layout

```text
DORY/
├── data/                         # Datasets and few-shot examples
│   └── alpaca/
├── outputs/                      # Pipeline outputs
├── scripts/
│   └── run_pipeline.py           # One-command end-to-end runner
├── src/
│   └── prompt_recovery/          # Core prompt recovery package
├── requirements.txt
└── README.md
```

## Setup

```bash
cd DORY
pip install -r requirements.txt
python3 -m nltk.downloader punkt
```

The runner uses an OpenAI-compatible chat-completions API. You can either use
the defaults in `scripts/run_pipeline.py`, or override the API configuration
with environment variables or command-line arguments.

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="your_base_url"
```

Stanford CoreNLP is required for hint generation. The default path is:

```text
src/prompt_recovery/stanford-corenlp-4.5.5
```

## Run The Full Pipeline

The only required argument is the input file. The input can be a JSON array or
JSONL file. The runner uses the few-shot examples in `data/alpaca/`, the default
Stanford CoreNLP path, and the default model/API settings unless you override
them.

```bash
python3 scripts/run_pipeline.py \
  --input data/alpaca/icl_sample.json
```

By default, outputs are written to:

```text
outputs/<input_file_stem>_pipeline/
```

For example, the command above writes the final output to:

```text
outputs/icl_sample_pipeline/recovered_prompts.jsonl
```

For a quick debug run, process only the first few records:

```bash
python3 scripts/run_pipeline.py \
  --input data/alpaca/icl_sample.json \
  --limit 5
```

To choose a custom output directory:

```bash
python3 scripts/run_pipeline.py \
  --input data/alpaca/icl_sample.json \
  --output-dir outputs/my_run
```

The runner also keeps named intermediate files, including:

- `original_generations.jsonl`: original prompt generations and logprobs
- `original_features.jsonl`: token, word, and sentence entropy features
- `topic_predictions.jsonl`: predicted topic sentences
- `original_hints.jsonl`: hint words extracted from original generations
- `draft_prompts.jsonl`: draft recovered prompts
- `draft_features.jsonl`: entropy features from draft-prompt generations
- `clues.jsonl`: hint/noise comparison results
- `recovered_prompts.jsonl`: final recovered prompts

## Input Format

The main input file can be a JSON array or JSONL. Each record should contain at least:

```json
{
  "id": 13422,
  "instruction": "Write a Python function ...",
  "input": "",
  "output": ""
}
```

The default few-shot examples are included under `data/alpaca/`. To use your own data:

```bash
python3 scripts/run_pipeline.py \
  --input data/your_data.json \
  --topic-examples data/your_topic_examples.json \
  --draft-examples data/your_draft_examples.json \
  --final-examples data/your_final_examples.json \
  --output-dir outputs/your_run
```

If your input prompt field is not named `instruction`, pass the source field:

```bash
python3 scripts/run_pipeline.py \
  --input data/your_data.jsonl \
  --prompt-field your_prompt_field
```

## Optional Overrides

The default settings match the stage-by-stage debug script:

- `--model qwen-flash`
- `--bert-model bert-base-uncased`
- `--stanford-corenlp-path src/prompt_recovery/stanford-corenlp-4.5.5`
- `--max-hint-words 10`

Override them only when you need a different model, API endpoint, BERT model,
CoreNLP installation, or output location.

## Source Modules

- `src/prompt_recovery/io_utils.py`: JSON and JSONL I/O
- `src/prompt_recovery/llm.py`: OpenAI-compatible API calls
- `src/prompt_recovery/probs.py`: conversion from logprobs to entropy features
- `src/prompt_recovery/prompts.py`: few-shot ICL prompt construction
- `src/prompt_recovery/similarity.py`: BERT similarity for topic-related sentence selection
- `src/prompt_recovery/hints.py`: Stanford CoreNLP POS tagging, hint word extraction, and noise word extraction
- `scripts/run_pipeline.py`: end-to-end orchestration

## Output Fields

- `llm_gen_text`: text generated from the original prompt
- `word_list`, `word_pes_list`: merged words and word-level entropy values
- `item_sents`, `item_sents2words`, `item_sents2entropies_list`: sentence-level features
- `topic_sent`: predicted topic sentence
- `top_words`: hint words extracted from the generated text
- `draftprompt`: first-pass recovered prompt
- `reconstruct_instruction_text`: text generated from the draft prompt
- `reconstruct_instruction_llmout_info`: draft-generation logprob and entropy features
- `removewords`, `addwords`: noise/addition clues from the second pass
- `Recovered_prompt`: final recovered prompt
