import re
from copy import deepcopy


def contains_alpha(text):
    return bool(re.search(r"[a-zA-Z]", text or ""))


def group_tokens_into_words(tokens, probs):
    words, word_probs = [], []
    for token, prob in zip(tokens, probs):
        if " " not in token:
            if not words or not contains_alpha(token):
                words.append([token])
                word_probs.append([prob])
            else:
                words[-1].append(token)
                word_probs[-1].append(prob)
        else:
            words.append([token])
            word_probs.append([prob])
    return words, word_probs


def split_words_to_sentences(words):
    sentences, word_to_sent = [], {}
    current = ""
    sent_id = 0
    for idx, word in enumerate(words):
        is_last = idx == len(words) - 1
        if word.endswith((".", "!", "?", ":", "\n")) or is_last:
            current += word
            sentences.append(current)
            word_to_sent[idx] = sent_id
            current = ""
            sent_id += 1
        elif "\n\n" in word:
            parts = re.split(r"(\n\n)", word)
            current += "".join(parts[:2])
            sentences.append(current)
            word_to_sent[idx] = sent_id
            current = parts[2] + " " if len(parts) > 2 else ""
            sent_id += 1
        else:
            current += word + " "
            word_to_sent[idx] = sent_id
    return sentences, word_to_sent


def group_by_sentence(word_to_sent, values):
    grouped = []
    bucket = []
    for idx in range(len(word_to_sent)):
        if idx > 0 and word_to_sent[idx] != word_to_sent[idx - 1]:
            grouped.append(deepcopy(bucket))
            bucket = []
        bucket.append(values[idx])
    if bucket:
        grouped.append(deepcopy(bucket))
    return grouped


def clean_word(word):
    return re.sub(r"[^a-zA-Z0-9'-]", "", word.replace('"', "").replace(":", "").replace("\n", ""))


def parse_word_list(value):
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        parsed = eval(value, {"__builtins__": {}})
        return list(parsed) if isinstance(parsed, (list, tuple)) else []
    except Exception:
        return []

