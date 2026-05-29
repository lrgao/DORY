from copy import deepcopy

import torch
from transformers import BertModel, BertTokenizer

from .text_utils import contains_alpha


def load_bert(model_name_or_path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
    model = BertModel.from_pretrained(model_name_or_path).to(device)
    model.eval()
    return tokenizer, model, device


def sentence_embedding(text, tokenizer, model, device):
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    chunks = [input_ids[:, i : i + 512] for i in range(0, input_ids.shape[1], 512)]
    outputs = []
    with torch.no_grad():
        for chunk in chunks:
            outputs.append(model(chunk).last_hidden_state)
    return torch.mean(torch.cat(outputs, dim=1), dim=1).squeeze()


def cosine_sim(text_a, text_b, tokenizer, model, device):
    vec_a = sentence_embedding(text_a, tokenizer, model, device)
    vec_b = sentence_embedding(text_b, tokenizer, model, device)
    return torch.nn.functional.cosine_similarity(vec_a, vec_b, dim=0).cpu().item()


def select_topic_sentence(item, topic_field="topic_sent", info_field=None, tokenizer=None, model=None, device=None):
    source = item if info_field is None else item[info_field]
    sents = source["item_sents"]
    words = source["item_sents2words"]
    entropies = source["item_sents2entropies_list"]
    topic = source.get(topic_field, item.get(topic_field, ""))
    scores = [cosine_sim(sent, topic, tokenizer, model, device) for sent in sents]
    source["item-topic_sents_sim"] = scores

    ranked = sorted(zip(scores, entropies, words, sents), key=lambda x: x[0], reverse=True)
    for _, sent_entropy, sent_words, sent in ranked:
        if len(sent) > 10 and contains_alpha(sent):
            source["top2_sents"] = deepcopy([sent])
            source["top2_sents2words"] = deepcopy([sent_words])
            source["top2_sents2wordsentropies"] = deepcopy([sent_entropy])
            break
    if "top2_sents" not in source:
        source["top2_sents"] = deepcopy(sents)
        source["top2_sents2words"] = deepcopy(words)
        source["top2_sents2wordsentropies"] = deepcopy(entropies)
    return item

