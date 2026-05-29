import os

from openai import OpenAI


def make_client(base_url=None, api_key=None):
    api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL")
    if not api_key:
        raise RuntimeError("Missing API key. Set OPENAI_API_KEY or pass --api-key.")
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def chat_text(client, model, prompt, logprobs=False, max_tokens=4096, top_logprobs=5):
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "max_tokens": max_tokens,
    }
    if logprobs:
        kwargs.update({"logprobs": True, "top_logprobs": top_logprobs})
    return client.chat.completions.create(**kwargs)

