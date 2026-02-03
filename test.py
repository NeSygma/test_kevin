import requests
import json
from typing import List

LMSTUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"

MODEL_NAME = "vibethinker-1.5b"
MAX_TOKENS = 8192          # large budget, model will stop earlier
TOKEN_CHUNK = 20           # client-side chunking
TEMPERATURE = 0.2
TIMEOUT = 600              # seconds (important for long generations)


def emit_chunk(tokens: List[str], final: bool = False):
    """
    Hook for downstream logic:
    - logging
    - verifier
    - UI streaming
    - agent control
    """
    tag = "[FINAL]" if final else "[CHUNK]"
    text = "".join(tokens)
    print(f"\n{tag}\n{text}")


payload = {
    "model": MODEL_NAME,
    "messages": [
        {
            "role": "system",
            "content": (
                "You are a careful, concise reasoner.\n"
                "Explain concepts clearly and avoid unnecessary verbosity.\n"
                "Do not include meta-commentary or disclaimers.\n"
                "Write in plain, direct language."
            ),
        },
        {
            "role": "user",
            "content": "Facts: Linda is described as a social activist.\n(a) Linda is a bank teller.\n(b) Linda is a bank teller and active in the feminist movement.\nWhich is more probable?",
        },
    ],
    "stream": True,
    "max_tokens": MAX_TOKENS,
    "temperature": TEMPERATURE,
}

buffer: List[str] = []

with requests.post(
    LMSTUDIO_URL,
    json=payload,
    stream=True,
    timeout=TIMEOUT,
) as response:

    response.raise_for_status()

    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue

        # LM Studio uses OpenAI-style SSE
        if not raw_line.startswith("data:"):
            continue

        data = raw_line[len("data:"):].strip()

        if data == "[DONE]":
            break

        try:
            msg = json.loads(data)
        except json.JSONDecodeError:
            # Defensive: skip malformed partial lines
            continue

        delta = msg["choices"][0].get("delta", {})
        token = delta.get("content")

        if token:
            buffer.append(token)

            if len(buffer) >= TOKEN_CHUNK:
                emit_chunk(buffer)
                buffer.clear()

# Flush remaining tokens
if buffer:
    emit_chunk(buffer, final=True)
