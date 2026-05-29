#!/usr/bin/env python3
"""Run the complete prompt recovery pipeline with one command."""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = lambda x, **_: x

DEFAULT_EXAMPLE_IDS = ["13422", "4077", "24852", "10851", "29593"]


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def default_output_dir(input_path):
    return PROJECT_ROOT / "outputs" / f"{input_path.stem}_pipeline"


def save_stage(records, output_dir, name):
    from prompt_recovery.io_utils import write_jsonl

    path = output_dir / f"{name}.jsonl"
    write_jsonl(records, path)
    print(f"[saved] {path}")
    return path


def set_nested(item, dotted_key, value):
    keys = dotted_key.split(".")
    target = item
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    target[keys[-1]] = value


def call_llm(records, client, args, prompt_field, output_field, response_dump_field=None, logprobs=False, label="LLM"):
    from prompt_recovery.llm import chat_text

    for item in tqdm(records, desc=label):
        response = chat_text(
            client,
            args.model,
            item[prompt_field],
            logprobs=logprobs,
            max_tokens=args.max_tokens,
            top_logprobs=args.top_logprobs,
        )
        set_nested(item, output_field, response.choices[0].message.content)
        if response_dump_field:
            set_nested(item, response_dump_field, response.model_dump())
    return records


def attach_topic_prompts(records, examples, text_field):
    from prompt_recovery.prompts import build_topic_icl

    icl = build_topic_icl(examples, text_field="llm_gen_text", topic_field="topic_sent")
    for item in records:
        text = item.get(text_field)
        if text is None and text_field == "reconstruct_instruction_llmout":
            text = item.get("reconstruct_instruction_text")
        if text is None:
            raise KeyError(text_field)
        item["icl_prompt"] = f'{icl}Text:"{text}"\nTopic sentence:'
    return records


def attach_draft_prompts(records, examples):
    from prompt_recovery.prompts import build_draft_icl

    icl = build_draft_icl(examples, text_field="llm_gen_text", prompt_field="instruction")
    for item in records:
        item["icl_prompt"] = f'{icl}LLM-Generated Text: "{item["llm_gen_text"]}"\nPrompt:'
    return records


def attach_final_prompts(records, examples):
    from prompt_recovery.prompts import build_final_icl
    from prompt_recovery.text_utils import parse_word_list

    icl = build_final_icl(examples)
    for item in records:
        hints = ",".join(parse_word_list(item.get("top_words")))
        noises = '","'.join(parse_word_list(item.get("removewords")))
        draft = item.get("reconstruct_instruction", item.get("draftprompt", ""))
        item["icl_prompt"] = (
            f'{icl}Outputs: "{item["llm_gen_text"]}"\n'
            f'Draft: "{draft}"\n'
            f'Hint: "{hints}"\n'
            f'Noise: "{noises}"\n'
            "Recovered prompt: "
        )
    return records


def maybe_limit(records, limit):
    return records[:limit] if limit else records


def run(args):
    from prompt_recovery.hints import add_noise_words, add_top_words, make_stanford_nlp
    from prompt_recovery.io_utils import load_records
    from prompt_recovery.llm import make_client
    from prompt_recovery.probs import add_generation_probs
    from prompt_recovery.prompts import select_examples
    from prompt_recovery.similarity import load_bert, select_topic_sentence

    input_path = project_path(args.input)
    output_dir = project_path(args.output_dir) if args.output_dir else default_output_dir(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    topic_examples_path = project_path(args.topic_examples)
    draft_examples_path = project_path(args.draft_examples)
    final_examples_path = project_path(args.final_examples)

    records = maybe_limit(load_records(input_path), args.limit)
    topic_examples = select_examples(topic_examples_path, args.example_ids)
    draft_examples = select_examples(draft_examples_path, args.example_ids)
    final_examples = select_examples(final_examples_path, args.example_ids)

    client = make_client(args.base_url, args.api_key)

    print("\n[stage] Generate outputs from original prompts")
    records = call_llm(
        records,
        client,
        args,
        prompt_field=args.prompt_field,
        output_field="llm_gen_text",
        response_dump_field="llmout_info",
        logprobs=True,
        label="original generation",
    )
    save_stage(records, output_dir, "original_generations")

    print("\n[stage] Extract entropy features")
    records = [add_generation_probs(item, "llmout_info", "llm_gen_text") for item in tqdm(records, desc="original probs")]
    save_stage(records, output_dir, "original_features")

    print("\n[stage] Predict topic sentences")
    records = attach_topic_prompts(records, topic_examples, text_field="llm_gen_text")
    save_stage(records, output_dir, "topic_prompts")
    records = call_llm(records, client, args, "icl_prompt", "topic_sent", label="topic sentence")
    save_stage(records, output_dir, "topic_predictions")

    print("\n[stage] Select topic-related generated sentence")
    tokenizer, bert_model, device = load_bert(args.bert_model, args.device)
    stanford_path = project_path(args.stanford_corenlp_path) if args.stanford_corenlp_path else None
    stanford_nlp = make_stanford_nlp(stanford_path)
    records = [
        select_topic_sentence(item, "topic_sent", None, tokenizer, bert_model, device)
        for item in tqdm(records, desc="select original sentence")
    ]
    save_stage(records, output_dir, "selected_topic_sentences")

    try:
        print("\n[stage] Extract original hint words")
        records = [
            add_top_words(item, None, args.max_hint_words, stanford_nlp=stanford_nlp)
            for item in tqdm(records, desc="original hints")
        ]
        save_stage(records, output_dir, "original_hints")

        print("\n[stage] Recover draft prompts")
        records = attach_draft_prompts(records, draft_examples)
        save_stage(records, output_dir, "draft_prompt_requests")
        records = call_llm(records, client, args, "icl_prompt", "draftprompt", label="draft prompt")
        save_stage(records, output_dir, "draft_prompts")

        print("\n[stage] Generate outputs from draft prompts")
        records = call_llm(
            records,
            client,
            args,
            prompt_field="draftprompt",
            output_field="reconstruct_instruction_text",
            response_dump_field="reconstruct_instruction_llmout",
            logprobs=True,
            label="draft generation",
        )
        save_stage(records, output_dir, "draft_generations")

        print("\n[stage] Extract entropy features for draft outputs")
        records = [
            add_generation_probs(
                item,
                response_field="reconstruct_instruction_llmout",
                text_field="reconstruct_instruction_llmout",
                info_prefix="reconstruct_instruction_llmout_info",
            )
            for item in tqdm(records, desc="draft probs")
        ]
        save_stage(records, output_dir, "draft_features")

        print("\n[stage] Predict topic sentences for draft outputs")
        records = attach_topic_prompts(records, topic_examples, text_field="reconstruct_instruction_llmout")
        save_stage(records, output_dir, "draft_topic_prompt_requests")
        records = call_llm(
            records,
            client,
            args,
            prompt_field="icl_prompt",
            output_field="reconstruct_instruction_llmout_info.topic_sent",
            label="draft topic sentence",
        )
        save_stage(records, output_dir, "draft_topic_predictions")

        print("\n[stage] Select topic-related draft-output sentence")
        records = [
            select_topic_sentence(
                item,
                topic_field="topic_sent",
                info_field="reconstruct_instruction_llmout_info",
                tokenizer=tokenizer,
                model=bert_model,
                device=device,
            )
            for item in tqdm(records, desc="select draft sentence")
        ]
        save_stage(records, output_dir, "selected_draft_topic_sentences")

        print("\n[stage] Extract draft-output hint words")
        records = [
            add_top_words(
                item,
                "reconstruct_instruction_llmout_info",
                args.max_hint_words,
                stanford_nlp=stanford_nlp,
            )
            for item in tqdm(records, desc="draft hints")
        ]
        save_stage(records, output_dir, "draft_hints")

        print("\n[stage] Extract noise/add words")
        clue_records = [add_noise_words(item) for item in tqdm(records, desc="noise words")]
        save_stage(clue_records, output_dir, "clues")

        print("\n[stage] Final prompt recovery")
        clue_records = attach_final_prompts(clue_records, final_examples)
        save_stage(clue_records, output_dir, "final_prompt_requests")
        clue_records = call_llm(clue_records, client, args, "icl_prompt", "Recovered_prompt", label="final recovery")
        final_path = save_stage(clue_records, output_dir, "recovered_prompts")
    finally:
        stanford_nlp.close()

    summary_path = output_dir / "summary.json"
    summary = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "num_records": len(clue_records),
        "final_output": str(final_path),
        "model": args.model,
        "bert_model": args.bert_model,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] Final results: {final_path}")
    print(f"[done] Summary: {summary_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Run prompt recovery end to end.")
    parser.add_argument("--input", required=True, help="Input JSON/JSONL file.")
    parser.add_argument(
        "--output-dir",
        help="Directory for all intermediate and final outputs. Defaults to outputs/<input_stem>_pipeline.",
    )
    parser.add_argument("--topic-examples", default="data/alpaca/icl_sample1_topic-sent.json.topwords-num.json")
    parser.add_argument("--draft-examples", default="data/alpaca/icl_sample.gen_probs.probs.json")
    parser.add_argument("--final-examples", default="data/alpaca/icl_sample_delete-words.json")
    parser.add_argument("--example-ids", nargs="*", default=DEFAULT_EXAMPLE_IDS)
    parser.add_argument("--model", default="qwen-flash")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--prompt-field", default="instruction", help="Input field containing the original prompt.")
    parser.add_argument("--bert-model", default="bert-base-uncased")
    parser.add_argument(
        "--stanford-corenlp-path",
        default="src/prompt_recovery/stanford-corenlp-4.5.5",
        help="Path to a Stanford CoreNLP distribution for Stanford POS tagging during hint extraction.",
    )
    parser.add_argument("--device", help="BERT device, e.g. cuda or cpu. Defaults to cuda when available.")
    parser.add_argument("--max-hint-words", type=int, default=10)
    parser.add_argument("--limit", type=int, help="Optional small-run limit for debugging.")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
